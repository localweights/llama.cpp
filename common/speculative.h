#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

// comma separated list of all types
std::string common_speculative_type_name_str();

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt);

void common_speculative_free(common_speculative * spec);

// optionally call once at the beginning of a new generation.
// last_row: row index in the trunk's last t_h_pre_norm tensor for this slot's last prefill token.
// Pass -1 (default) for single-slot case (uses last row automatically).
void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt, int32_t last_row = -1);

// sample up to n_draft tokens and add them to the batch using the draft model
llama_tokens common_speculative_draft(
                     common_speculative * spec,
        const common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last);

// informs the speculative decoder that n_accepted tokens were accepted by the target model.
// last_accepted_row: the row index within the trunk's last t_h_pre_norm tensor that corresponds
// to the last accepted token (used by MTP to correctly index the hidden state for the next draft).
// For n_accepted==0, pass the row of the original (non-draft) token; defaults to -1 (auto = last row).
void common_speculative_accept(common_speculative * spec, uint16_t n_accepted, int32_t last_accepted_row = -1);

int32_t common_speculative_n_max(const common_speculative * spec, const common_params_speculative & params);
int32_t common_speculative_n_min(const common_speculative * spec, const common_params_speculative & params);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

// clear internal KV-cache state of all speculative impls (e.g. ctx_mtp).
// must be called on slot eviction / prompt_clear to prevent stale-KV bugs
// on the next request assigned to the same slot.
void common_speculative_reset_kv(common_speculative * spec);

// Phase D.1: Direct-commit query.
//
// After a tree-mode draft() call, returns the number of accepted tokens that
// were committed directly to the trunk slot_seq KV (>= 0), or -1 if the last
// draft was not tree-committed (linear mode or tree verify failed).
//
// When >= 0:
//   - The accepted tokens are in the draft_tokens returned by common_speculative_draft().
//   - Their KV (at n_past..n_past+n_committed) is already in slot_seq's KV.
//   - The caller MUST NOT re-submit those tokens to trunk; instead decode only
//     the correction token (from common_speculative_tree_correction()) at
//     n_past+n_committed+1 (which IS slot_seq's current pos_next after commit).
//   - After using this result, it is automatically reset on the next draft() call.
int32_t     common_speculative_tree_n_committed(const common_speculative * spec);

// Returns the correction token from the last tree verify, or -1 if not available.
// Valid only when common_speculative_tree_n_committed() >= 0.
llama_token common_speculative_tree_correction(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;
