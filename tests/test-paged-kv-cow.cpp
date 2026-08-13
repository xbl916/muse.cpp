#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>

static bool decode_one(llama_context * ctx, llama_batch & batch, llama_token token, llama_pos pos, llama_seq_id seq_id) {
    common_batch_clear(batch);
    common_batch_add(batch, token, pos, { seq_id }, true);
    return llama_decode(ctx, batch) == 0;
}

static std::vector<float> logits_cur(const llama_model * model, llama_context * ctx) {
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * logits = llama_get_logits_ith(ctx, 0);
    return std::vector<float>(logits, logits + n_vocab);
}

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());

    float result = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        result = std::max(result, std::fabs(a[i] - b[i]));
    }
    return result;
}

int main(int argc, char ** argv) {
    common_params params;

    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    const char * test_paged = std::getenv("LLAMA_TEST_PAGED_KV");

    params.kv_unified = true;
    params.paged_kv = !test_paged || std::atoi(test_paged) != 0;
    params.n_parallel = 2;
    params.n_parallel_max = 2;
    params.n_ctx = 512;
    params.max_model_len = params.paged_kv ? 256 : 0;
    params.n_batch = 64;
    params.n_ubatch = 64;
    params.kv_block_size = 32;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    common_init_result_ptr init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (!model || !ctx) {
        std::fprintf(stderr, "failed to initialize model\n");
        return 1;
    }

    llama_batch batch = llama_batch_init(64, 0, 1);
    for (llama_pos pos = 0; pos < 31; ++pos) {
        common_batch_add(batch, 1, pos, { 0 }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;
    if (llama_decode(ctx, batch) != 0) {
        std::fprintf(stderr, "failed to decode prefix\n");
        llama_batch_free(batch);
        return 1;
    }
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_cp(mem, 0, 1, 0, -1);

    if (!decode_one(ctx, batch, 2, 31, 1)) {
        std::fprintf(stderr, "failed to decode copied sequence\n");
        llama_batch_free(batch);
        return 1;
    }
    const std::vector<float> logits_copied = logits_cur(model, ctx);

    if (!decode_one(ctx, batch, 2, 31, 0)) {
        std::fprintf(stderr, "failed to decode original sequence\n");
        llama_batch_free(batch);
        return 1;
    }
    const std::vector<float> logits_original = logits_cur(model, ctx);

    const float diff = max_abs_diff(logits_copied, logits_original);
    if (diff > 1.0e-3f) {
        std::fprintf(stderr, "logits mismatch after copy-on-write: %g\n", diff);
        llama_batch_free(batch);
        return 1;
    }

    if (!decode_one(ctx, batch, 3, 32, 1)) {
        std::fprintf(stderr, "failed to allocate the next block\n");
        llama_batch_free(batch);
        return 1;
    }

    std::fprintf(stderr, "paged KV copy-on-write passed, max logits diff=%g\n", diff);
    llama_batch_free(batch);
    return 0;
}
