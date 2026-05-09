#pragma once

// EAGLE-2 Phase A: parallel MTP forward proof
// tree-spec.h — data structures and public interface for tree drafting
//
// Phase A scope: single-depth expansion only.  At each draft step feed N
// top-K candidates from the previous logit row into ctx_mtp simultaneously,
// collect logits, but still verify only the leftmost (path-0) branch so the
// accept rate is identical to the linear chain.  This proves n_tokens>1 MTP
// forward works.

#include "llama.h"
#include "common.h"

#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

struct mtp_tree_node {
    llama_token token         = 0;
    int32_t     parent_idx    = -1;   // -1 = root sentinel
    int32_t     depth         = -1;   // -1 = root sentinel
    float       log_prob      = 0.0f; // log P(token | parent) from softmax
    float       cum_log_prob  = 0.0f; // cumulative log-prob from root
    int32_t     kv_seq_id     = 0;    // for future Phase C trunk verification
    bool        verified      = false;
    llama_token verify_token  = 0;    // trunk argmax at this pos (set during accept)
};

// Configuration mirroring common_params_speculative_mtp tree fields.
// Passed in from speculative.cpp so tree-spec.cpp has no dep on common.h
// params hierarchy.
struct mtp_tree_config {
    int32_t branching  = 1;    // top-K per expansion step
    int32_t max_depth  = 6;    // unused in Phase A (single depth)
    int32_t max_nodes  = 40;   // hard node budget
    float   p_min      = 0.0f; // log-prob floor; 0 = disabled
};

// Per-leaf hidden state storage: one vector<float> per node index.
// Index 0 = root (populated with last_h_vec from speculative.cpp).
using mtp_h_vecs = std::vector<std::vector<float>>;

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

// mtp_tree_draft() — Phase A parallel MTP forward.
//
// Expands ONE depth level (depth=0→depth=1) using branching=K parallel MTP
// decodes in a single llama_decode(ctx_mtp, batch) call.
//
// Inputs:
//   ctx_mtp       — MTP draft context
//   batch         — pre-allocated batch (capacity >= max_nodes); modified in place
//   cfg           — tree configuration
//   root_h_vec    — hidden state for root (from last_h_pre_norm)
//   n_embd        — embedding dim
//   id_last       — last accepted token (root cond_tok)
//   pos_start     — position for depth-0 expansion in ctx_mtp KV
//
// Outputs:
//   nodes_out     — flat node array built by this call (root sentinel at [0])
//   h_vecs_out    — hidden states for each node (index mirrors nodes_out)
//   draft_tokens  — leftmost-path tokens (path-0) for linear-compatible accept
//
// Returns true on success, false if llama_decode failed.
bool mtp_tree_draft(
        llama_context                 * ctx_mtp,
        llama_batch                   & batch,
        const mtp_tree_config         & cfg,
        const std::vector<float>      & root_h_vec,
        int32_t                         n_embd,
        llama_token                     id_last,
        llama_pos                       pos_start,
        std::vector<mtp_tree_node>    & nodes_out,
        mtp_h_vecs                    & h_vecs_out,
        llama_tokens                  & draft_tokens);
