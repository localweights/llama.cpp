#pragma once

// EAGLE-2 Phase C: trunk verification with per-path seq_ids
// tree-spec.h — data structures and public interface for tree drafting
//
// Phase A: parallel MTP forward (single-depth expansion).
// Phase B: multi-depth tree expansion.
// Phase C: trunk verification via per-path seq_ids.  Each root-to-leaf path
//           is assigned a unique seq_id in ctx_tgt.  Prompt KV is copied to
//           each path seq_id via llama_memory_seq_cp.  All tree nodes are
//           decoded in a single llama_decode(ctx_tgt) call.  The longest
//           accepted path is found via a greedy tree walk.  Auxiliary seq_ids
//           are cleaned up afterwards.  draft() returns the accepted path.

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
    int32_t     kv_seq_id     = 0;    // seq_id assigned for trunk verification (Phase C)
    bool        verified      = false;
    llama_token verify_token  = 0;    // trunk argmax at this pos (set during accept walk)
    std::vector<int32_t> children;    // child node indices (populated in Phase C)
    int32_t     batch_idx     = -1;   // index in the trunk verification batch (-1 = root)
};

// Configuration mirroring common_params_speculative_mtp tree fields.
// Passed in from speculative.cpp so tree-spec.cpp has no dep on common.h
// params hierarchy.
struct mtp_tree_config {
    int32_t branching  = 1;    // top-K per expansion step
    int32_t max_depth  = 6;    // max tree depth
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

// mtp_tree_verify() — Phase C: trunk verification of all tree paths.
//
// Given the tree built by mtp_tree_draft(), runs a single trunk forward pass
// over all tree nodes (with per-path seq_ids for correct ancestor-only
// attention), then walks the tree to find the longest accepted path.
//
// The approach uses explicit KV copy per path (llama_memory_seq_cp) to
// duplicate the prompt KV for each root-to-leaf path, then submits all tree
// nodes to ctx_tgt in one llama_decode call.  After the accept walk, all
// auxiliary seq_ids are removed from ctx_tgt's KV.
//
// A "root probe" token (id_last at pos n_past-1) is prepended to the batch
// to obtain trunk logits for depth-0 child selection.
//
// Inputs:
//   ctx_tgt       — trunk context (must have n_seq_max >= slot_seq + n_paths + 1)
//   tgt_batch     — pre-allocated batch for trunk (capacity >= max_nodes+1); modified
//   nodes         — tree from mtp_tree_draft() (mutated: kv_seq_id, children, batch_idx)
//   n_past        — number of tokens already in ctx_tgt KV (= trunk position base)
//   id_last       — last accepted token (root cond_tok); added to batch for root logits
//   slot_seq      — trunk seq_id for this slot (prompt KV is under this seq_id)
//   base_seq      — first seq_id available for tree paths; must be > slot_seq
//   n_seq_max     — llama_n_seq_max(ctx_tgt); checked at runtime
//
// Outputs:
//   accepted_out  — the accepted+correction tokens (length >= 1: at least the correction)
//
// Returns number of draft tokens accepted (0 = only correction, no draft accepted).
// accepted_out always contains accepted_tokens + 1 correction token.
//
// After return, all path seq_ids are removed from ctx_tgt's KV.  The caller
// should re-submit accepted_out through the normal trunk decode path so the
// tokens are written back under the slot seq_id.
int32_t mtp_tree_verify(
        llama_context               * ctx_tgt,
        llama_batch                 & tgt_batch,
        std::vector<mtp_tree_node>  & nodes,
        llama_pos                     n_past,
        llama_token                   id_last,
        llama_seq_id                  slot_seq,
        llama_seq_id                  base_seq,
        uint32_t                      n_seq_max,
        llama_tokens                & accepted_out);
