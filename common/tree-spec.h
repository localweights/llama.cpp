#pragma once

// EAGLE-2 Phase D: eliminate double trunk decode + per-leaf MTP KV management
// tree-spec.h — data structures and public interface for tree drafting
//
// Phase A: parallel MTP forward (single-depth expansion).
// Phase B: multi-depth tree expansion.
// Phase C: trunk verification via per-path seq_ids.
// Phase D (new):
//   D.1 — after mtp_tree_verify() finds the accepted path, the path's KV
//          (positions n_past .. n_past+n_accepted inclusive) is copied from
//          the path seq_id back to slot_seq.  The caller (server) only needs
//          to run a single-token trunk decode (for the correction token at
//          n_past+n_accepted+1) instead of re-decoding the full accepted prefix.
//   D.2 — per-leaf MTP seq_ids in ctx_mtp.  Previously all leaves at each
//          depth shared seq_id=0 and the KV was wiped between depths ("cold"
//          MTP).  Phase D assigns a unique seq_id per leaf path so each leaf's
//          depth-d forward sees its complete ancestor KV up to depth d-1.
//          Requires ctx_mtp n_seq_max >= tree_max_nodes.

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
    int32_t     mtp_seq_id    = 0;    // seq_id assigned within ctx_mtp for this path (Phase D.2)
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

// Phase D.2: MTP seq_id assigned to this node within ctx_mtp.
// Root sentinel = 0 (the "root" forward run uses seq_id=0).
// Leaf node at depth d gets its own seq_id so it can accumulate KV
// for positions 0..d without collision against sibling paths.
// (Added to mtp_tree_node below via field mtp_seq_id.)

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

// mtp_tree_draft() — Phase D: multi-depth tree expansion with per-leaf MTP KV.
//
// Expands cfg.max_depth levels (root → K → K² → … → K^D children) using
// one llama_decode(ctx_mtp, batch) call per depth level.
//
// Phase D.2: each leaf path gets its own seq_id in ctx_mtp so its MTP forward
// at depth d attends to its full ancestor KV at depths 0..d-1, rather than
// operating "cold" with only depth d's input.  Requires ctx_mtp.n_seq_max
// >= number of leaf paths (K^D).  The seq_ids used are 0..n_paths-1.
//
// Inputs:
//   ctx_mtp       — MTP draft context (must have n_seq_max >= K^D)
//   batch         — pre-allocated batch (capacity >= max_nodes * n_seq_max_mtp); modified in place
//   cfg           — tree configuration (branching, max_depth, max_nodes, p_min)
//   root_h_vec    — hidden state for root (from last_h_pre_norm)
//   n_embd        — embedding dim
//   id_last       — last accepted token (root cond_tok)
//   pos_start     — base position in ctx_mtp KV; depth d uses pos_start+d
//   n_seq_max_mtp — llama_n_seq_max(ctx_mtp); checked at runtime
//
// Outputs:
//   nodes_out     — flat node array (root sentinel at [0], depth 0 at [1..K], …)
//                   node.mtp_seq_id is set to the per-leaf MTP seq_id (Phase D.2)
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
        uint32_t                        n_seq_max_mtp,
        std::vector<mtp_tree_node>    & nodes_out,
        mtp_h_vecs                    & h_vecs_out,
        llama_tokens                  & draft_tokens);

// mtp_tree_verify() — Phase D: trunk verification + direct KV commit.
//
// Given the tree built by mtp_tree_draft(), runs a single trunk forward pass
// over all tree nodes (with per-path seq_ids for correct ancestor-only
// attention), then walks the tree to find the longest accepted path.
//
// Phase D.1 (direct commit): after finding the accepted path of length n_acc,
// the accepted positions (n_past .. n_past+n_acc inclusive) from the path's
// seq_id are copied directly to slot_seq via llama_memory_seq_cp.  This means
// the caller does NOT need to re-submit those tokens to ctx_tgt — only the
// next single decode (at n_past+n_acc+1, for the correction token) is needed.
// Set commit_to_slot = true to enable this (requires that the path seq_id's
// KV positions are valid at n_past..n_past+n_acc).  After copy, all path
// seq_ids are cleaned up.
//
// Inputs:
//   ctx_tgt        — trunk context (n_seq_max >= slot_seq + n_paths + 1)
//   tgt_batch      — pre-allocated batch for trunk (capacity >= max_nodes+1)
//   nodes          — tree from mtp_tree_draft() (mutated: kv_seq_id, children, batch_idx)
//   n_past         — number of tokens already in ctx_tgt KV
//   id_last        — last accepted token (root cond_tok)
//   slot_seq       — trunk seq_id for this slot (prompt KV is under this seq_id)
//   base_seq       — first seq_id available for tree paths; must be > slot_seq
//   n_seq_max      — llama_n_seq_max(ctx_tgt); checked at runtime
//   commit_to_slot — if true (Phase D.1), copy accepted path KV into slot_seq
//
// Outputs:
//   accepted_out   — accepted tokens (length n_acc) — does NOT include correction.
//                    correction_out is returned separately (Phase D.1 callers skip
//                    re-decode and get the correction token from here).
//   correction_out — the correction token (trunk argmax after last accepted, or
//                    the argmax at root if n_acc==0).  Always valid.
//
// Returns number of draft tokens accepted (0 = only correction, no draft accepted).
//
// After return:
//   - If commit_to_slot=true: slot_seq contains KV at n_past..n_past+n_acc.
//     The caller should decode only the correction token at n_past+n_acc+1.
//   - If commit_to_slot=false: all path seq_ids removed; caller re-submits full
//     accepted_out through normal trunk decode (Phase C behavior).
int32_t mtp_tree_verify(
        llama_context               * ctx_tgt,
        llama_batch                 & tgt_batch,
        std::vector<mtp_tree_node>  & nodes,
        llama_pos                     n_past,
        llama_token                   id_last,
        llama_seq_id                  slot_seq,
        llama_seq_id                  base_seq,
        uint32_t                      n_seq_max,
        bool                          commit_to_slot,
        llama_tokens                & accepted_out,
        llama_token                 & correction_out);
