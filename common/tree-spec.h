#pragma once

// EAGLE-2 Phase B: multi-depth tree expansion
// tree-spec.h — data structures and public interface for tree drafting
//
// Phase B: build a static fixed-topology tree of width `branching` (K) and
// depth `max_depth` (D).  At each depth d, expand every leaf at depth d-1
// into K children via a single parallel llama_decode call.  Total nodes:
// 1 + K + K² + … + K^D.  Verification still uses path-0 only (Phase C wires
// real trunk verification).  KV management is "cold" per depth (Phase D fix).

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

// mtp_tree_draft() — Phase B multi-depth tree expansion.
//
// Expands cfg.max_depth levels (root → K → K² → … → K^D children) using
// one llama_decode(ctx_mtp, batch) call per depth level.
//
// Inputs:
//   ctx_mtp       — MTP draft context
//   batch         — pre-allocated batch (capacity >= max_nodes); modified in place
//   cfg           — tree configuration (branching, max_depth, max_nodes, p_min)
//   root_h_vec    — hidden state for root (from last_h_pre_norm)
//   n_embd        — embedding dim
//   id_last       — last accepted token (root cond_tok)
//   pos_start     — base position in ctx_mtp KV; depth d uses pos_start+d+1
//
// Outputs:
//   nodes_out     — flat node array (root sentinel at [0], depth 0 at [1..K], …)
//   h_vecs_out    — MTP output hidden states per node (index mirrors nodes_out)
//   draft_tokens  — leftmost-path tokens (path-0) for linear-compatible accept
//
// Returns true on success, false if root llama_decode failed.
// (Partial failure mid-tree still returns true with a shorter draft.)
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
