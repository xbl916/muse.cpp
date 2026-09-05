#include "common.h"
#include "speculative.h"
#include "llama-ext.h"
#include "json.hpp"
#include <fstream>
#include <stdexcept>
#include <filesystem>

static void save(const std::string & path, const float * data, size_t n) {
    if (!data) throw std::runtime_error("Missing tensor: " + path);
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(data), n * sizeof(float));
    if (!out) throw std::runtime_error("Cannot write " + path);
}

int main(int argc, char ** argv) {
    if (argc != 4) return 1;
    common_params params;
    params.model.path = argv[1];
    params.n_gpu_layers = 99;
    params.split_mode = LLAMA_SPLIT_MODE_TENSOR;
    params.tensor_split[0] = params.tensor_split[1] = 1;
    params.n_ctx = 4096;
    params.n_batch = 1024;
    params.n_ubatch = 512;
    params.n_parallel = 1;
    params.paged_kv = true;
    params.kv_unified = true;
    params.kv_prefix_cache = false;
    params.fit_params = false;
    params.warmup = false;
    params.cpuparams.n_threads = 16;
    params.cpuparams_batch.n_threads = 16;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.cache_type_k = params.cache_type_v = GGML_TYPE_Q8_0;
    params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
    params.speculative.draft.n_max = 1;
    params.speculative.draft.n_min = 0;
    params.speculative.draft.p_min = 0;
    params.speculative.draft.cache_type_k = GGML_TYPE_Q8_0;
    params.speculative.draft.cache_type_v = GGML_TYPE_Q8_0;
    params.speculative.draft.backend_sampling = false;
    ggml_backend_load_all();
    auto init = common_init_from_params(params);
    auto * model = init->model();
    auto * tgt = init->context();
    if (!model || !tgt) return 2;
    auto dinit = common_speculative_init_from_params(params, model, tgt);
    auto * dft = dinit->context();
    if (!dft) return 3;
    params.speculative.draft.ctx_tgt = tgt;
    params.speculative.draft.ctx_dft = dft;
    common_speculative_ptr spec(common_speculative_init(params.speculative, 1));
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int nh = llama_model_n_embd(model);
    std::ifstream input(argv[2]);
    nlohmann::json records;
    input >> records;
    std::filesystem::create_directories(argv[3]);
    llama_batch batch = llama_batch_init(1024, 0, 1);
    for (const auto & record : records) {
        const auto tokens = record.at("prefix").get<llama_tokens>();
        const auto name = record.at("name").get<std::string>();
        const std::string base = std::string(argv[3]) + "/" + name;
        llama_memory_clear(llama_get_memory(tgt), true);
        llama_memory_clear(llama_get_memory(dft), true);
        common_speculative_reset_seq(spec.get(), 0);
        // Teacher-force the exact target prefix before the anchor token.
        for (size_t off = 0; off < tokens.size() - 1;) {
            common_batch_clear(batch);
            const size_t end = std::min(off + 512, tokens.size() - 1);
            for (size_t i = off; i < end; ++i) {
                common_batch_add(batch, tokens[i], i, {0}, i + 1 == end);
            }
            if (llama_decode(tgt, batch) || !common_speculative_process(spec.get(), batch)) return 4;
            off = end;
        }
        if (!llama_mtp_hidden_state_ready(dft)) {
            save(base + ".target_hidden.f32", llama_get_embeddings_nextn_ith(tgt, batch.n_tokens - 1), nh);
        }
        llama_tokens draft;
        auto & dp = common_speculative_get_draft_params(spec.get(), 0);
        dp = {};
        dp.drafting = true;
        dp.n_max = 1;
        dp.n_past = tokens.size() - 1;
        dp.id_last = tokens.back();
        dp.prompt = &tokens;
        dp.result = &draft;
        common_speculative_draft(spec.get());
        save(base + ".mtp_logits.f32", llama_get_logits_ith(dft, -1), nv);
        if (!llama_mtp_hidden_state_ready(dft)) {
            save(base + ".mtp_hidden.f32", llama_get_embeddings_nextn_ith(dft, -1), nh);
        }
        common_batch_clear(batch);
        common_batch_add(batch, tokens.back(), tokens.size() - 1, {0}, true);
        if (llama_decode(tgt, batch)) return 5;
        save(base + ".target_logits.f32", llama_get_logits_ith(tgt, -1), nv);
        fprintf(stderr, "CAPTURE %s prefix=%zu draft=%d\n", name.c_str(), tokens.size(), draft.empty() ? -1 : draft[0]);
    }
    llama_batch_free(batch);
}
