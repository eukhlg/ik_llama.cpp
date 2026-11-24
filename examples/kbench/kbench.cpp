// llama-kbench (roadmap 9.5 / E6): matmul kernel reference at the model's shapes and quant types.
//
// For each case (dense matmul, output head, MoE up/gate, MoE down) and token count it times the
// kernel in two placements:
//   dram - weights cycle through a working set >= --gb GiB, so every op streams its weights from DRAM
//   l3   - every op re-reads the same weights (only for cases that fit in a few MB of L3)
// and reports time per op and logical weight throughput (bytes of the weights an op reads, MoE:
// distinct selected experts only). The l3 rate is an empirical upper reference, not a compute ceiling
// (it can itself be limited by cache bandwidth, dequantisation or SIMD throughput).
// Each graph node gets its own activation tensor, so consecutive matmuls are not fused into chains.
//
// Run pinned to one NUMA node, e.g.:
//   numactl --cpunodebind=0 --membind=0 llama-kbench -t 12 --cpus 0-11
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct bench_case {
    std::string name;
    std::string kind;     // dense | moe_up_gate | moe_down
    ggml_type   type;
    int64_t     K, N;     // weight: K inputs -> N outputs
    int64_t     n_exp;    // experts (MoE)
    int64_t     n_used;   // experts per token (MoE)
    bool        l3_ok;    // run the l3 placement too
};

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::vector<int32_t> parse_cpus(const std::string & s) {
    std::vector<int32_t> v;
    size_t p = 0;
    while (p < s.size()) {
        size_t c = s.find(',', p);
        std::string part = s.substr(p, c == std::string::npos ? std::string::npos : c - p);
        int a, b;
        if (sscanf(part.c_str(), "%d-%d", &a, &b) == 2) { for (int i = a; i <= b; ++i) v.push_back(i); }
        else if (sscanf(part.c_str(), "%d", &a) == 1) v.push_back(a);
        if (c == std::string::npos) break;
        p = c + 1;
    }
    return v;
}

static std::vector<int> parse_ints(const std::string & s) {
    std::vector<int> v;
    for (auto c : parse_cpus(s)) v.push_back(c);
    return v;
}

// fill a weight tensor (2D or 3D) with quantized random data, chunk by chunk
static void fill_weight(ggml_tensor * t, std::mt19937 & rng) {
    const int64_t K = t->ne[0];
    const int64_t nrows = ggml_nrows(t);
    const int64_t chunk = 256;   // rows per chunk; multiple of every row interleaving (4/8/16)
    std::normal_distribution<float> nd(0.0f, 0.02f);
    std::vector<float> buf(chunk * K);
    const size_t row_size = ggml_row_size(t->type, K);
    for (int64_t r0 = 0; r0 < nrows; r0 += chunk) {
        const int64_t nr = std::min(chunk, nrows - r0);
        for (int64_t i = 0; i < nr * K; ++i) buf[i] = nd(rng);
        ggml_quantize_chunk(t->type, buf.data(), (char *) t->data + r0 * row_size, 0, nr, K, nullptr, nullptr);
    }
}

static void fill_f32(ggml_tensor * t, std::mt19937 & rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    float * d = (float *) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); ++i) d[i] = nd(rng);
}

// random distinct experts per token
static void fill_ids(ggml_tensor * ids, int64_t n_exp, std::mt19937 & rng) {
    std::vector<int32_t> pool(n_exp);
    for (int64_t t = 0; t < ids->ne[1]; ++t) {
        for (int64_t e = 0; e < n_exp; ++e) pool[e] = (int32_t) e;
        std::shuffle(pool.begin(), pool.end(), rng);
        for (int64_t u = 0; u < ids->ne[0]; ++u) ((int32_t *) ((char *) ids->data + t * ids->nb[1]))[u] = pool[u];
    }
}

static int count_distinct(const ggml_tensor * ids, int64_t n_exp) {
    std::vector<char> seen(n_exp, 0);
    int d = 0;
    for (int64_t t = 0; t < ids->ne[1]; ++t)
        for (int64_t u = 0; u < ids->ne[0]; ++u) {
            const int32_t e = ((const int32_t *) ((const char *) ids->data + t * ids->nb[1]))[u];
            if (!seen[e]) { seen[e] = 1; ++d; }
        }
    return d;
}

struct result { double us_best, us_med, gbs_best, gbs_med, bytes_per_op; int n_ops, copies; };

static bool run_case(const bench_case & bc, int ntok, bool l3, double gb, int reps, int nth,
                     const std::vector<int32_t> & cpus, result & res) {
    const bool moe = bc.kind != "dense";
    const int64_t n_exp = moe ? bc.n_exp : 1;
    const size_t w_bytes = ggml_row_size(bc.type, bc.K) * bc.N * n_exp;      // one weight copy (one matrix)
    const int n_w_per_copy = bc.kind == "moe_up_gate" ? 2 : 1;               // up + gate
    const size_t copy_bytes = w_bytes * n_w_per_copy;
    int copies = l3 ? 1 : (int) std::max<double>(4, std::ceil(gb * 1073741824.0 / copy_bytes));
    const int n_ops = std::max(l3 ? 64 : 32, copies);

    // activations per op: dense [K, ntok]; moe up/gate [K, 1, ntok]; moe down [K, n_used, ntok]
    const int64_t act_elems = bc.kind == "moe_down" ? bc.K * bc.n_used * ntok : bc.K * ntok;
    size_t mem = (size_t) copies * (copy_bytes + 1024) + (size_t) n_ops * (act_elems * 4 + 1024)
               + (size_t) n_ops * (bc.n_used * ntok * 4 + 1024)
               + ggml_graph_overhead_custom(4 * n_ops + 16, false) + (size_t) n_ops * 4 * ggml_tensor_overhead() + (64u << 20);
    // output tensors
    const int64_t out_elems = moe ? bc.N * bc.n_used * ntok : bc.N * ntok;
    mem += (size_t) n_ops * (out_elems * 4 + 1024);

    ggml_init_params ip = { mem, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { fprintf(stderr, "ggml_init failed (%.1f GiB)\n", mem / 1073741824.0); return false; }

    std::mt19937 rng(1234);
    std::vector<ggml_tensor *> w0(copies), w1(copies, nullptr);
    for (int c = 0; c < copies; ++c) {
        w0[c] = moe ? ggml_new_tensor_3d(ctx, bc.type, bc.K, bc.N, n_exp) : ggml_new_tensor_2d(ctx, bc.type, bc.K, bc.N);
        if (n_w_per_copy == 2) w1[c] = ggml_new_tensor_3d(ctx, bc.type, bc.K, bc.N, n_exp);
        if (c == 0) {
            fill_weight(w0[0], rng);
            if (w1[0]) fill_weight(w1[0], rng);
        } else {
            memcpy(w0[c]->data, w0[0]->data, ggml_nbytes(w0[0]));
            if (w1[c]) memcpy(w1[c]->data, w1[0]->data, ggml_nbytes(w1[0]));
        }
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4 * n_ops + 16, false);
    double bytes_total = 0;
    ggml_tensor * fixed_ids = nullptr;
    for (int i = 0; i < n_ops; ++i) {
        const int c = i % copies;
        ggml_tensor * out = nullptr;
        if (!moe) {
            ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, bc.K, ntok);
            fill_f32(b, rng);
            out = ggml_mul_mat(ctx, w0[c], b);
            bytes_total += (double) ggml_nbytes(w0[c]);
        } else {
            ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, bc.n_used, ntok);
            if (l3) {
                // l3: every token and every op uses the same 8 experts, so the working set stays small
                if (!fixed_ids) { fill_ids(ids, n_exp, rng); for (int64_t t = 1; t < ntok; ++t) memcpy((char *) ids->data + t * ids->nb[1], ids->data, ids->nb[1]); }
                else memcpy(ids->data, fixed_ids->data, ggml_nbytes(ids));
            } else {
                fill_ids(ids, n_exp, rng);
            }
            if (!fixed_ids) fixed_ids = ids;
            const int d = count_distinct(ids, n_exp);
            if (bc.kind == "moe_up_gate") {
                ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, bc.K, 1, ntok);
                fill_f32(b, rng);
                out = ggml_moe_up_gate(ctx, w0[c], w1[c], b, ids, GGML_UNARY_OP_SILU);
                bytes_total += (double) d * (w0[c]->nb[2] + w1[c]->nb[2]);
            } else {
                ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, bc.K, bc.n_used, ntok);
                fill_f32(b, rng);
                out = ggml_mul_mat_id(ctx, w0[c], b, ids);
                bytes_total += (double) d * w0[c]->nb[2];
            }
        }
        ggml_build_forward_expand(gf, out);
    }

    ggml_cplan plan = ggml_graph_plan(gf, nth);
    std::vector<uint8_t> work(plan.work_size + 64);
    plan.work_data = work.data();
    if (!cpus.empty()) { plan.cpu_affinity = cpus.data(); plan.n_cpu_affinity = (int32_t) cpus.size(); }

    ggml_graph_compute(gf, &plan);                 // warm-up (and page-in)
    std::vector<double> t(reps);
    for (int r = 0; r < reps; ++r) {
        const double t0 = now_s();
        ggml_graph_compute(gf, &plan);
        t[r] = now_s() - t0;
    }
    std::sort(t.begin(), t.end());
    const double best = t[0], med = t[reps / 2];
    res.n_ops = n_ops; res.copies = copies;
    res.bytes_per_op = bytes_total / n_ops;
    res.us_best = best / n_ops * 1e6; res.us_med = med / n_ops * 1e6;
    res.gbs_best = bytes_total / best / 1e9; res.gbs_med = bytes_total / med / 1e9;
    ggml_free(ctx);
    return true;
}

int main(int argc, char ** argv) {
    int nth = 12, reps = 7;
    double gb = 1.0;
    std::string cpus_s, tokens_s = "1,2,3,4", modes = "dram,l3", only;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); } return std::string(argv[++i]); };
        if (a == "-t") nth = std::stoi(next());
        else if (a == "--cpus") cpus_s = next();
        else if (a == "--tokens") tokens_s = next();
        else if (a == "--gb") gb = std::stod(next());
        else if (a == "--reps") reps = std::max(3, std::stoi(next()));
        else if (a == "--modes") modes = next();
        else if (a == "--only") only = next();
        else { fprintf(stderr, "usage: %s [-t N] [--cpus 0-11] [--tokens 1,2,3,4] [--gb 1.0] [--reps 7] [--modes dram,l3] [--only name]\n", argv[0]); return 1; }
    }
    const std::vector<int32_t> cpus = parse_cpus(cpus_s);
    const std::vector<int> tokens = parse_ints(tokens_s);
    const bool do_dram = modes.find("dram") != std::string::npos, do_l3 = modes.find("l3") != std::string::npos;

    ggml_init_params ip0 = { 16u << 20, nullptr, false };   // initialise ggml tables once
    ggml_free(ggml_init(ip0));

    // shapes of Qwen3.6-35B-A3B (n_embd 2048, delta-net inner 4096, 256 experts of 512, 8 used, vocab 248320)
    // types as after -rtr (K-quants repacked to _R4) and -mtprot iq4_xs for the MTP output head
    const std::vector<bench_case> cases = {
        { "dense_2048x8192_q4kr4",  "dense",       GGML_TYPE_Q4_K_R4, 2048,   8192,   0, 0, true  },
        { "dense_2048x4096_q4kr4",  "dense",       GGML_TYPE_Q4_K_R4, 2048,   4096,   0, 0, true  },
        { "dense_4096x2048_q4kr4",  "dense",       GGML_TYPE_Q4_K_R4, 4096,   2048,   0, 0, true  },
        { "dense_2048x8192_q6kr4",  "dense",       GGML_TYPE_Q6_K_R4, 2048,   8192,   0, 0, true  },
        { "head_2048x248320_q6kr4", "dense",       GGML_TYPE_Q6_K_R4, 2048, 248320,   0, 0, false },
        { "head_2048x248320_iq4xs", "dense",       GGML_TYPE_IQ4_XS,  2048, 248320,   0, 0, false },
        { "moe_up_gate_q4kr4",      "moe_up_gate", GGML_TYPE_Q4_K_R4, 2048,    512, 256, 8, true  },
        { "moe_down_q4kr4",         "moe_down",    GGML_TYPE_Q4_K_R4,  512,   2048, 256, 8, true  },
    };

    printf("# llama-kbench: %d threads, cpus %s, DRAM working set >= %.2f GiB, %d reps (best / median)\n",
           nth, cpus_s.empty() ? "unpinned" : cpus_s.c_str(), gb, reps);
    printf("%-24s %-5s %4s %9s %7s %10s %10s %9s %9s\n", "case", "mode", "tok", "MB/op", "ops", "us_best", "us_med", "GBs_best", "GBs_med");
    for (const auto & bc : cases) {
        if (!only.empty() && bc.name.find(only) == std::string::npos) continue;
        for (int ntok : tokens) {
            for (int m = 0; m < 2; ++m) {
                const bool l3 = m == 1;
                if ((l3 && (!do_l3 || !bc.l3_ok)) || (!l3 && !do_dram)) continue;
                result r{};
                if (!run_case(bc, ntok, l3, gb, reps, nth, cpus, r)) return 1;
                printf("%-24s %-5s %4d %9.2f %7d %10.1f %10.1f %9.1f %9.1f\n", bc.name.c_str(), l3 ? "l3" : "dram", ntok,
                       r.bytes_per_op / 1e6, r.n_ops, r.us_best, r.us_med, r.gbs_best, r.gbs_med);
                fflush(stdout);
            }
        }
    }
    return 0;
}
