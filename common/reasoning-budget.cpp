#include "reasoning-budget.h"
#include "common.h"
#include "trie.h"
#include "unicode.h"

#include "log.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

struct token_matcher {
    std::vector<llama_tokens> seqs;
    common_aho_corasick ac;
    size_t state = 0;

    token_matcher(const std::vector<llama_tokens> & seqs) : seqs(collect(seqs)), ac(build_trie(this->seqs)) {}

    static std::vector<llama_tokens> collect(const std::vector<llama_tokens> & seqs) {
        std::vector<llama_tokens> res;
        for (const auto & seq : seqs) {
            if (!seq.empty() && std::find(res.begin(), res.end(), seq) == res.end()) {
                res.push_back(seq);
            }
        }
        return res;
    }

    static common_trie build_trie(const std::vector<llama_tokens> & seqs) {
        common_trie t;
        for (const auto & seq : seqs) {
            t.insert(std::vector<uint32_t>(seq.begin(), seq.end()));
        }
        return t;
    }

    // returns the index into seqs of the longest sequence ending at this token, or -1
    int32_t advance(llama_token token) {
        state = ac.next(state, (uint32_t) token);
        const int32_t p = ac.match_pattern(state);
        if (p >= 0) {
            state = 0;
        }
        return p;
    }

    void reset() { state = 0; }
};

struct common_reasoning_budget_ctx {
    const llama_vocab * vocab;
    common_reasoning_budget_piece_cb piece_cb;
    void * piece_user_data;

    token_matcher start_matcher;
    token_matcher end_matcher;
    llama_tokens forced_tokens;

    int32_t budget;           // maximum tokens in reasoning block
    int32_t remaining;        // tokens remaining in budget

    common_reasoning_budget_state state;

    // for forcing
    size_t force_pos;         // next position in forced_tokens to force

    int32_t end_match;        // index into end_matcher.seqs of the sequence that transitioned to DONE, -1 if none

    int32_t soft_threshold;   // consumed-token count at which the soft hint becomes eligible
    llama_tokens soft_tokens;
    llama_tokens handoff_tokens;
    bool soft_fired;
    size_t soft_force_pos;
    int32_t soft_boundary_grace_total;
    int32_t soft_boundary_grace_used;

    int32_t converge_configured;

    int32_t grace_total;
    int32_t grace_used;
    bool previous_line_boundary;
    bool previous_safe_boundary;
    bool boundary_in_code_fence;
    std::string boundary_line;
    std::string boundary_last_nonempty_line;
};

static bool common_reasoning_budget_piece(
        const common_reasoning_budget_ctx * ctx, llama_token token, std::string & piece) {
    if (ctx->piece_cb != nullptr) {
        piece = ctx->piece_cb(ctx->piece_user_data, token);
        return true;
    }
    if (ctx->vocab != nullptr) {
        piece = common_token_to_piece(ctx->vocab, token, false);
        return true;
    }
    return false;
}

bool common_reasoning_budget_is_line_boundary(const std::string & piece) {
    return piece.find('\n') != std::string::npos;
}

static std::string common_reasoning_budget_trim(const std::string & text) {
    size_t begin = 0;
    size_t end = text.size();
    while (begin < end && std::isspace((unsigned char) text[begin])) {
        begin++;
    }
    while (end > begin && std::isspace((unsigned char) text[end - 1])) {
        end--;
    }
    return text.substr(begin, end - begin);
}

static bool common_reasoning_budget_has_suffix(const std::string & text, const char * suffix) {
    const size_t size = strlen(suffix);
    return text.size() >= size && text.compare(text.size() - size, size, suffix) == 0;
}

static bool common_reasoning_budget_has_odd_code_fences(const std::string & line) {
    size_t count = 0;
    size_t pos = 0;
    while ((pos = line.find("```", pos)) != std::string::npos) {
        count++;
        pos += 3;
    }
    return count % 2 != 0;
}

static bool common_reasoning_budget_has_unclosed_delimiter(const std::string & line) {
    int round = 0;
    int square = 0;
    int curly = 0;
    for (const unsigned char ch : line) {
        switch (ch) {
            case '(': round++; break;
            case ')': round--; break;
            case '[': square++; break;
            case ']': square--; break;
            case '{': curly++; break;
            case '}': curly--; break;
            default: break;
        }
    }
    return round > 0 || square > 0 || curly > 0;
}

static bool common_reasoning_budget_has_safe_line_end(const std::string & line) {
    const std::string trimmed = common_reasoning_budget_trim(line);
    if (trimmed.empty() || common_reasoning_budget_has_unclosed_delimiter(trimmed)) {
        return false;
    }

    static const std::string unsafe_ascii = "[({=,:,+-*/\\|&";
    return unsafe_ascii.find(trimmed.back()) == std::string::npos;
}

static bool common_reasoning_budget_has_sentence_end(const std::string & line) {
    const std::string trimmed = common_reasoning_budget_trim(line);
    if (trimmed.empty()) {
        return false;
    }
    const char last = trimmed.back();
    if (last == '.' || last == '!' || last == '?' || last == ';') {
        return true;
    }
    return common_reasoning_budget_has_suffix(trimmed, "。") ||
            common_reasoning_budget_has_suffix(trimmed, "！") ||
            common_reasoning_budget_has_suffix(trimmed, "？") ||
            common_reasoning_budget_has_suffix(trimmed, "；");
}

static void common_reasoning_budget_advance_boundary(
        common_reasoning_budget_ctx * ctx, const std::string & piece) {
    ctx->previous_safe_boundary = false;
    for (const char ch : piece) {
        if (ch != '\n') {
            ctx->boundary_line.push_back(ch);
            if (ctx->boundary_line.size() > 4096) {
                ctx->boundary_line.erase(0, ctx->boundary_line.size() - 4096);
            }
            continue;
        }

        const std::string completed = common_reasoning_budget_trim(ctx->boundary_line);
        if (common_reasoning_budget_has_odd_code_fences(completed)) {
            ctx->boundary_in_code_fence = !ctx->boundary_in_code_fence;
        }
        if (!completed.empty()) {
            ctx->boundary_last_nonempty_line = completed;
        }

        const bool paragraph_boundary = completed.empty();
        const std::string & candidate = paragraph_boundary ?
                ctx->boundary_last_nonempty_line : completed;
        ctx->previous_safe_boundary = !ctx->boundary_in_code_fence &&
                common_reasoning_budget_has_safe_line_end(candidate) &&
                (paragraph_boundary || common_reasoning_budget_has_sentence_end(candidate));
        ctx->boundary_line.clear();
    }
}

static void common_reasoning_budget_reset_boundary(common_reasoning_budget_ctx * ctx) {
    ctx->previous_line_boundary = false;
    ctx->previous_safe_boundary = false;
    ctx->boundary_in_code_fence = false;
    ctx->boundary_line.clear();
    ctx->boundary_last_nonempty_line.clear();
}

static const char * common_reasoning_budget_name(const struct llama_sampler * /*smpl*/) {
    return "reasoning-budget";
}

static void common_reasoning_budget_begin_forcing(
        common_reasoning_budget_ctx * ctx,
        bool                          utf8_complete) {
    ctx->end_matcher.reset();
    if (utf8_complete) {
        ctx->state = REASONING_BUDGET_FORCING;
        ctx->force_pos = 0;
    } else {
        ctx->state = REASONING_BUDGET_WAITING_UTF8;
    }
}

static void common_reasoning_budget_reach_limit(
        common_reasoning_budget_ctx * ctx,
        bool                          utf8_complete,
        bool                          line_boundary,
        const char                  * limit_name) {
    if (ctx->grace_total > 0 && !line_boundary) {
        ctx->state = REASONING_BUDGET_HARD_PENDING;
        ctx->grace_used = 0;
        COM_INF("reasoning %s reached, waiting up to %d tokens for a line boundary\n",
                limit_name, ctx->grace_total);
    } else {
        common_reasoning_budget_begin_forcing(ctx, utf8_complete);
        COM_INF("reasoning %s reached%s, forcing end sequence\n",
                limit_name, line_boundary ? " at line boundary" : "");
    }
}

static void common_reasoning_budget_begin_two_stage_handoff(common_reasoning_budget_ctx * ctx) {
    ctx->state = REASONING_BUDGET_HANDOFF_FORCING;
    ctx->soft_force_pos = 0;
    ctx->end_matcher.reset();
    COM_INF("reasoning starting deterministic two-stage handoff: transition=%zu tokens, native_end=%zu tokens\n",
            ctx->soft_tokens.size(), ctx->handoff_tokens.size() - ctx->soft_tokens.size());
}

static void common_reasoning_budget_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    switch (ctx->state) {
        case REASONING_BUDGET_IDLE:
        {
            if (ctx->start_matcher.advance(token) >= 0) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                ctx->soft_fired = false;
                ctx->soft_force_pos = 0;
                ctx->soft_boundary_grace_used = 0;
                ctx->grace_used = 0;
                common_reasoning_budget_reset_boundary(ctx);
                COM_TRC("activated, budget=%d tokens\n", ctx->budget);

                if (ctx->remaining <= 0) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    COM_TRC("%s", "budget=0, forcing immediately\n");
                }
            }
            break;
        }
        case REASONING_BUDGET_COUNTING:
        case REASONING_BUDGET_SOFT_PENDING:
        case REASONING_BUDGET_HARD_PENDING:
        case REASONING_BUDGET_WAITING_UTF8:
        {
            const int32_t match = ctx->end_matcher.advance(token);
            if (match >= 0) {
                ctx->state = REASONING_BUDGET_DONE;
                ctx->end_match = match;
                COM_INF("%s", "reasoning ended naturally before two-stage handoff\n");
                break;
            }

            bool utf8_complete = true;
            bool line_boundary = true;
            std::string piece;
            const bool has_piece = common_reasoning_budget_piece(ctx, token, piece);
            if (has_piece) {
                utf8_complete = common_utf8_is_complete(piece);
                line_boundary = common_reasoning_budget_is_line_boundary(piece);
                common_reasoning_budget_advance_boundary(ctx, piece);
            }
            ctx->previous_line_boundary = line_boundary;
            const bool safe_boundary = !has_piece || ctx->previous_safe_boundary;

            if (ctx->state == REASONING_BUDGET_WAITING_UTF8) {
                if (utf8_complete) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    ctx->end_matcher.reset();
                    COM_TRC("%s", "UTF-8 complete, now forcing end sequence\n");
                }
                break;
            }

            if (ctx->state == REASONING_BUDGET_HARD_PENDING) {
                ctx->grace_used++;
                if (line_boundary || ctx->grace_used >= ctx->grace_total) {
                    common_reasoning_budget_begin_forcing(ctx, utf8_complete);
                    COM_INF("reasoning hard-boundary wait ended after %d tokens (%s), forcing end sequence\n",
                            ctx->grace_used, line_boundary ? "line boundary" : "limit reached");
                }
                break;
            }

            // COUNTING and SOFT_PENDING continue consuming the nominal budget.
            ctx->remaining--;
            if (ctx->remaining <= 0) {
                common_reasoning_budget_reach_limit(
                        ctx, utf8_complete, line_boundary, "hard budget");
                break;
            }

            if (ctx->state == REASONING_BUDGET_COUNTING &&
                    !ctx->soft_fired &&
                    ctx->soft_threshold > 0 &&
                    (ctx->budget - ctx->remaining) >= ctx->soft_threshold) {
                ctx->soft_fired = true;
                ctx->soft_boundary_grace_used = 0;
                if (utf8_complete && (safe_boundary || ctx->soft_boundary_grace_total <= 0 || !has_piece)) {
                    if (ctx->converge_configured >= 0) {
                        COM_INF("reasoning handoff threshold reached at %d/%d tokens (%s); starting two-stage transition\n",
                                ctx->soft_threshold, ctx->budget,
                                safe_boundary ? "safe text boundary" : "boundary wait disabled");
                        common_reasoning_budget_begin_two_stage_handoff(ctx);
                    } else {
                        ctx->state = REASONING_BUDGET_SOFT_FORCING;
                        ctx->soft_force_pos = 0;
                        COM_INF("reasoning soft threshold reached at %d/%d tokens, injecting legacy wrap-up hint (%s)\n",
                                ctx->soft_threshold, ctx->budget,
                                safe_boundary ? "safe text boundary" : "boundary wait disabled");
                    }
                } else {
                    ctx->state = REASONING_BUDGET_SOFT_PENDING;
                    COM_INF("reasoning handoff threshold reached at %d/%d tokens, waiting up to %d tokens for a safe text boundary\n",
                            ctx->soft_threshold, ctx->budget, ctx->soft_boundary_grace_total);
                }
                break;
            }

            if (ctx->state == REASONING_BUDGET_SOFT_PENDING) {
                ctx->soft_boundary_grace_used++;
                if (utf8_complete && (safe_boundary ||
                            ctx->soft_boundary_grace_used >= ctx->soft_boundary_grace_total)) {
                    if (ctx->converge_configured >= 0) {
                        COM_INF("reasoning starting two-stage transition after waiting %d tokens (%s)\n",
                                ctx->soft_boundary_grace_used,
                                safe_boundary ? "safe text boundary" : "boundary wait limit reached");
                        common_reasoning_budget_begin_two_stage_handoff(ctx);
                    } else {
                        ctx->state = REASONING_BUDGET_SOFT_FORCING;
                        ctx->soft_force_pos = 0;
                        COM_INF("reasoning injecting legacy wrap-up hint after waiting %d tokens (%s)\n",
                                ctx->soft_boundary_grace_used,
                                safe_boundary ? "safe text boundary" : "boundary wait limit reached");
                    }
                }
            }
            break;
        }
        case REASONING_BUDGET_SOFT_FORCING:
        {
            std::string piece;
            if (common_reasoning_budget_piece(ctx, token, piece)) {
                ctx->previous_line_boundary = common_reasoning_budget_is_line_boundary(piece);
            }
            ctx->soft_force_pos++;
            if (ctx->soft_force_pos >= ctx->soft_tokens.size()) {
                ctx->state = REASONING_BUDGET_COUNTING;
                COM_TRC("%s", "legacy wrap-up hint complete, resuming budget countdown\n");
            }
            break;
        }
        case REASONING_BUDGET_HANDOFF_FORCING:
        {
            const int32_t match = ctx->end_matcher.advance(token);
            ctx->soft_force_pos++;
            if (ctx->soft_force_pos >= ctx->handoff_tokens.size()) {
                ctx->state = REASONING_BUDGET_DONE;
                ctx->end_match = match;
                COM_INF("%s", "reasoning two-stage handoff complete; native reasoning closed, continuing with final answer\n");
            }
            break;
        }
        case REASONING_BUDGET_FORCING:
        {
            // track the end sequence within forced_tokens so it is also reported on DONE
            const int32_t match = ctx->end_matcher.advance(token);
            ctx->force_pos++;
            if (ctx->force_pos >= ctx->forced_tokens.size()) {
                ctx->state = REASONING_BUDGET_DONE;
                ctx->end_match = match;
                COM_INF("%s", "reasoning ended by forced end sequence\n");
            }
            break;
        }
        case REASONING_BUDGET_DONE:
            // Re-arm on a new start tag: some models emit multiple <think> blocks
            // per response, and each should get a fresh budget window.
            if (ctx->start_matcher.advance(token) >= 0) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                ctx->end_matcher.reset();
                ctx->end_match = -1;
                ctx->soft_fired = false;
                ctx->soft_force_pos = 0;
                ctx->soft_boundary_grace_used = 0;
                ctx->grace_used = 0;
                common_reasoning_budget_reset_boundary(ctx);
                COM_TRC("re-activated on new start tag, budget=%d tokens\n", ctx->budget);

                if (ctx->remaining <= 0) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    COM_TRC("%s", "budget=0, forcing immediately\n");
                }
            }
            break;
    }
}

static void common_reasoning_budget_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    if (ctx->state == REASONING_BUDGET_HANDOFF_FORCING) {
        if (ctx->soft_force_pos >= ctx->handoff_tokens.size()) {
            return;
        }
        const llama_token forced = ctx->handoff_tokens[ctx->soft_force_pos];
        for (size_t i = 0; i < cur_p->size; i++) {
            if (cur_p->data[i].id != forced) {
                cur_p->data[i].logit = -INFINITY;
            }
        }
        return;
    }

    if (ctx->state == REASONING_BUDGET_SOFT_FORCING) {
        if (ctx->soft_force_pos >= ctx->soft_tokens.size()) {
            return;
        }

        const llama_token forced = ctx->soft_tokens[ctx->soft_force_pos];
        for (size_t i = 0; i < cur_p->size; i++) {
            if (cur_p->data[i].id != forced) {
                cur_p->data[i].logit = -INFINITY;
            }
        }
        return;
    }

    if (ctx->state != REASONING_BUDGET_FORCING) {
        // passthrough — don't modify logits
        return;
    }

    if (ctx->force_pos >= ctx->forced_tokens.size()) {
        return;
    }

    const llama_token forced = ctx->forced_tokens[ctx->force_pos];

    // set all logits to -inf except the forced token
    for (size_t i = 0; i < cur_p->size; i++) {
        if (cur_p->data[i].id != forced) {
            cur_p->data[i].logit = -INFINITY;
        }
    }
}

static void common_reasoning_budget_reset(struct llama_sampler * smpl) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;
    ctx->state = REASONING_BUDGET_IDLE;
    ctx->remaining = ctx->budget;
    ctx->start_matcher.reset();
    ctx->end_matcher.reset();
    ctx->force_pos = 0;
    ctx->end_match = -1;
    ctx->soft_fired = false;
    ctx->soft_force_pos = 0;
    ctx->soft_boundary_grace_used = 0;
    ctx->grace_used = 0;
    common_reasoning_budget_reset_boundary(ctx);
}

static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab * vocab, common_reasoning_budget_piece_cb piece_cb, void * piece_user_data,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs, const llama_tokens & forced_tokens,
        int32_t budget, common_reasoning_budget_state initial_state,
        float soft_ratio, const llama_tokens & soft_tokens, int32_t grace_tokens,
        int32_t soft_boundary_grace_tokens, int32_t converge_tokens, float converge_max_bias,
        int32_t converge_bias_delay_tokens);

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl);

static void common_reasoning_budget_free(struct llama_sampler * smpl) {
    delete (common_reasoning_budget_ctx *) smpl->ctx;
}

static struct llama_sampler_i common_reasoning_budget_i = {
    /* .name              = */ common_reasoning_budget_name,
    /* .accept            = */ common_reasoning_budget_accept,
    /* .apply             = */ common_reasoning_budget_apply,
    /* .reset             = */ common_reasoning_budget_reset,
    /* .clone             = */ common_reasoning_budget_clone,
    /* .free              = */ common_reasoning_budget_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
    /* .backend_reset     = */ nullptr,
    /* .copy_state        = */ nullptr,
};

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx(*ctx)
    );
}

static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab        * vocab,
        common_reasoning_budget_piece_cb  piece_cb,
        void                            * piece_user_data,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs,
        const llama_tokens              & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state,
        float                             soft_ratio,
        const llama_tokens              & soft_tokens,
        int32_t                           grace_tokens,
        int32_t                           soft_boundary_grace_tokens,
        int32_t                           converge_tokens,
        float                             converge_max_bias,
        int32_t                           converge_bias_delay_tokens) {
    (void) converge_max_bias;
    (void) converge_bias_delay_tokens;

    // promote COUNTING with budget <= 0 to FORCING
    if (initial_state == REASONING_BUDGET_COUNTING && budget <= 0) {
        initial_state = REASONING_BUDGET_FORCING;
    }

    int32_t soft_threshold = 0;
    if (budget > 0 && budget != INT_MAX && soft_ratio > 0.0f && soft_ratio < 1.0f &&
            (converge_tokens >= 0 || !soft_tokens.empty())) {
        soft_threshold = (int32_t) std::ceil((double) budget * soft_ratio);
    }

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx {
            /* .vocab         = */ vocab,
            /* .piece_cb      = */ piece_cb,
            /* .piece_user_data = */ piece_user_data,
            /* .start_matcher = */ token_matcher(start_seqs),
            /* .end_matcher   = */ token_matcher(end_seqs),
            /* .forced_tokens = */ forced_tokens,
            /* .budget        = */ budget,
            /* .remaining     = */ budget,
            /* .state         = */ initial_state,
            /* .force_pos     = */ 0,
            /* .end_match     = */ -1,
            /* .soft_threshold = */ soft_threshold,
            /* .soft_tokens    = */ soft_tokens,
            /* .handoff_tokens = */ [&]() {
                llama_tokens tokens = soft_tokens;
                if (!end_seqs.empty()) {
                    tokens.insert(tokens.end(), end_seqs.front().begin(), end_seqs.front().end());
                }
                return tokens;
            }(),
            /* .soft_fired     = */ false,
            /* .soft_force_pos = */ 0,
            /* .soft_boundary_grace_total = */ std::max(0, soft_boundary_grace_tokens),
            /* .soft_boundary_grace_used  = */ 0,
            /* .converge_configured = */ converge_tokens,
            /* .grace_total    = */ std::max(0, grace_tokens),
            /* .grace_used     = */ 0,
            /* .previous_line_boundary = */ false,
            /* .previous_safe_boundary = */ false,
            /* .boundary_in_code_fence = */ false,
            /* .boundary_line = */ {},
            /* .boundary_last_nonempty_line = */ {},
        }
    );
}

struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab        * vocab,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs,
        const llama_tokens              & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state,
        float                             soft_ratio,
        const llama_tokens              & soft_tokens,
        int32_t                           grace_tokens,
        int32_t                           soft_boundary_grace_tokens,
        int32_t                           converge_tokens,
        float                             converge_max_bias,
        int32_t                           converge_bias_delay_tokens) {
    return common_reasoning_budget_init_state(
            vocab, nullptr, nullptr, start_seqs, end_seqs, forced_tokens, budget, initial_state,
            soft_ratio, soft_tokens, grace_tokens, soft_boundary_grace_tokens,
            converge_tokens, converge_max_bias, converge_bias_delay_tokens);
}

struct llama_sampler * common_reasoning_budget_init_pieces(
        common_reasoning_budget_piece_cb   piece_cb,
        void                             * piece_user_data,
        const std::vector<llama_tokens>  & start_seqs,
        const std::vector<llama_tokens>  & end_seqs,
        const llama_tokens               & forced_tokens,
        int32_t                            budget,
        common_reasoning_budget_state      initial_state,
        float                              soft_ratio,
        const llama_tokens               & soft_tokens,
        int32_t                            grace_tokens,
        int32_t                            soft_boundary_grace_tokens,
        int32_t                            converge_tokens,
        float                              converge_max_bias,
        int32_t                            converge_bias_delay_tokens) {
    return common_reasoning_budget_init_state(
            nullptr, piece_cb, piece_user_data, start_seqs, end_seqs, forced_tokens, budget, initial_state,
            soft_ratio, soft_tokens, grace_tokens, soft_boundary_grace_tokens,
            converge_tokens, converge_max_bias, converge_bias_delay_tokens);
}

common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl) {
    if (!smpl) {
        return REASONING_BUDGET_IDLE;
    }
    return ((const common_reasoning_budget_ctx *)smpl->ctx)->state;
}

const llama_tokens * common_reasoning_budget_get_end_match(const struct llama_sampler * smpl) {
    if (!smpl) {
        return nullptr;
    }

    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;
    if (ctx->end_match < 0) {
        return nullptr;
    }

    return &ctx->end_matcher.seqs[ctx->end_match];
}

bool common_reasoning_budget_force(struct llama_sampler * smpl) {
    if (!smpl) {
        return false;
    }

    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    if (ctx->state != REASONING_BUDGET_COUNTING &&
            ctx->state != REASONING_BUDGET_SOFT_PENDING &&
            ctx->state != REASONING_BUDGET_SOFT_FORCING &&
            ctx->state != REASONING_BUDGET_HANDOFF_FORCING &&
            ctx->state != REASONING_BUDGET_HARD_PENDING) {
        return false;
    }

    ctx->state = REASONING_BUDGET_FORCING;
    ctx->force_pos = 0;
    ctx->end_matcher.reset();
    COM_TRC("%s", "forced into forcing state (manual transition)\n");

    return true;
}
