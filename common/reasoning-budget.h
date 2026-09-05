#pragma once

#include "llama.h"

#include "common.h"

#include <cstdint>
#include <string>
#include <vector>

enum common_reasoning_budget_state {
    REASONING_BUDGET_IDLE,         // waiting for start sequence
    REASONING_BUDGET_COUNTING,     // counting down tokens
    REASONING_BUDGET_FORCING,      // forcing budget message + end sequence
    REASONING_BUDGET_WAITING_UTF8, // budget exhausted, waiting for UTF-8 completion
    REASONING_BUDGET_DONE,         // passthrough forever
    REASONING_BUDGET_SOFT_PENDING, // convergence threshold reached, waiting for a safe text boundary
    REASONING_BUDGET_SOFT_FORCING, // forcing a legacy soft hint
    REASONING_BUDGET_HANDOFF_FORCING, // forcing the transition sentence and native reasoning end tag
    REASONING_BUDGET_HARD_PENDING, // convergence/hard limit reached, waiting for a line boundary
};

// Creates a reasoning budget sampler that limits token generation inside a
// reasoning block (e.g. between <think> and </think>).
//
// Main state path: IDLE -> COUNTING -> WAITING_UTF8 -> FORCING -> DONE
//   IDLE:         passthrough, watching for a start sequence
//   COUNTING:     counting down remaining tokens, watching for a natural end sequence
//   SOFT_PENDING: convergence threshold reached, waiting for a safe text boundary or bounded wait
//   SOFT_FORCING: forces a legacy soft hint
//   HANDOFF_FORCING: forces a short transition sentence followed by the native end tag
//   HARD_PENDING: convergence/hard limit reached, waiting for a line boundary or wait limit
//   WAITING_UTF8: budget exhausted, allowing tokens to complete a UTF-8 sequence
//   FORCING:      forces forced_tokens token-by-token (all other logits -> -inf)
//   DONE:         passthrough, watching for another reasoning start sequence
//
// Parameters:
//   vocab          - vocabulary (used for UTF-8 boundary detection; can be nullptr)
//   start_seqs     - token sequences, any of which activates counting
//   end_seqs       - token sequences, any of which naturally deactivates
//   forced_tokens  - token sequence forced when budget expires
//   budget         - max tokens allowed in the reasoning block
//   initial_state  - initial state
//   soft_ratio     - fraction of budget at which soft_tokens are injected; -1 disables
//   soft_tokens    - two-stage transition sentence or legacy wrap-up hint
//   grace_tokens   - maximum extra tokens after budget exhaustion
//   soft_boundary_grace_tokens - maximum tokens to wait for a safe text boundary before the soft hint
//   converge_tokens - non-negative enables deterministic two-stage handoff; -1 keeps legacy behavior
//   converge_max_bias - retained for command-line and request compatibility
//   converge_bias_delay_tokens - retained for command-line and request compatibility
//
struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab        * vocab,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs,
        const llama_tokens              & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state = REASONING_BUDGET_IDLE,
        float                             soft_ratio = -1.0f,
        const llama_tokens              & soft_tokens = {},
        int32_t                           grace_tokens = 0,
        int32_t                           soft_boundary_grace_tokens = 64,
        int32_t                           converge_tokens = -1,
        float                             converge_max_bias = 0.0f,
        int32_t                           converge_bias_delay_tokens = 0);

// Test seam for exercising token-piece boundaries without constructing a vocabulary.
using common_reasoning_budget_piece_cb = std::string (*)(void * user_data, llama_token token);

struct llama_sampler * common_reasoning_budget_init_pieces(
        common_reasoning_budget_piece_cb   piece_cb,
        void                             * piece_user_data,
        const std::vector<llama_tokens>  & start_seqs,
        const std::vector<llama_tokens>  & end_seqs,
        const llama_tokens               & forced_tokens,
        int32_t                            budget,
        common_reasoning_budget_state      initial_state = REASONING_BUDGET_IDLE,
        float                              soft_ratio = -1.0f,
        const llama_tokens               & soft_tokens = {},
        int32_t                            grace_tokens = 0,
        int32_t                            soft_boundary_grace_tokens = 64,
        int32_t                            converge_tokens = -1,
        float                              converge_max_bias = 0.0f,
        int32_t                            converge_bias_delay_tokens = 0);

bool common_reasoning_budget_is_line_boundary(const std::string & piece);

common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl);

// The end sequence that transitioned the sampler to DONE, or nullptr if none
// was recorded. Cleared when a new start sequence re-arms the sampler.
const llama_tokens * common_reasoning_budget_get_end_match(const struct llama_sampler * smpl);

// Manually transition the reasoning budget sampler into the FORCING state.
// Returns true if the transition occurred.
bool common_reasoning_budget_force(struct llama_sampler * smpl);
