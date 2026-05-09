// EAGLE-2 Phase B: multi-depth tree expansion
// tree-spec.cpp — multi-depth tree drafting orchestrator
//
// Phase B extends Phase A to depth > 1.  A static fixed-topology tree of
// width `branching` (K) and depth `max_depth` (D) is built:
//
//   depth 0: K children of root  (same as Phase A)
//   depth 1: K children of each depth-0 node  →  K² nodes total at depth 1
//   ...
//   depth D-1: K^D leaves total
//
// At each depth d the expansion:
//   1. Collects all leaves at depth d-1.
//   2. Clears ctx_mtp KV beyond position (pos_start + d) so each depth
//      starts "cold" (no KV collision from prior depths).
//   3. Submits all leaves in ONE llama_decode batch.
//   4. For each leaf: extracts top-K logits → appends K child nodes.
//
// NOTE: KV management is intentionally simple.  MTP attention is "cold" at
// each depth (no accumulated KV context across depths).  Phase D will implement
// proper per-path seq_id KV management.
//
// Phase B accept: still path-0 only (leftmost chain root → child[0] → …).
// Phase C will wire real trunk verification.

#include "tree-spec.h"

#include "log.h"
#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <numeric>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Compute log-softmax numerically stable, return top-K (tok, log_p) pairs
// sorted by log_p descending.
static std::vector<std::pair<llama_token, float>> argtop_k_logprobs(
        const float * logits,
        int32_t       n_vocab,
        int32_t       k) {
    // Partial sort to find top-k using a min-heap of size k
    std::vector<std::pair<float, int32_t>> heap; // (logit, tok)
    heap.reserve(k + 1);

    for (int32_t i = 0; i < n_vocab; ++i) {
        heap.push_back({logits[i], i});
        if ((int32_t)heap.size() > k) {
            std::push_heap(heap.begin(), heap.end(),
                           [](const auto & a, const auto & b){ return a.first > b.first; });
            std::pop_heap(heap.begin(), heap.end(),
                          [](const auto & a, const auto & b){ return a.first > b.first; });
            heap.pop_back();
        }
    }

    // Sort descending
    std::sort(heap.begin(), heap.end(),
              [](const auto & a, const auto & b){ return a.first > b.first; });

    // Convert to log-probs via log-softmax over top-k (approximation)
    float max_logit = heap[0].first;
    float sum_exp = 0.0f;
    for (auto & p : heap) sum_exp += std::exp(p.first - max_logit);
    float log_sum_exp = max_logit + std::log(sum_exp);

    std::vector<std::pair<llama_token, float>> result;
    result.reserve(heap.size());
    for (auto & p : heap) {
        float log_p = p.first - log_sum_exp;
        result.push_back({(llama_token)p.second, log_p});
    }
    return result;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool mtp_tree_draft(
        llama_context              * ctx_mtp,
        llama_batch                & batch,
        const mtp_tree_config      & cfg,
        const std::vector<float>   & root_h_vec,
        int32_t                      n_embd,
        llama_token                  id_last,
        llama_pos                    pos_start,
        std::vector<mtp_tree_node> & nodes_out,
        mtp_h_vecs                 & h_vecs_out,
        llama_tokens               & draft_tokens) {

    const int32_t K     = std::max(1, cfg.branching);
    const int32_t D     = std::max(1, cfg.max_depth);
    const int32_t Keff  = std::min(K, cfg.max_nodes - 1); // budget-limited K
    if (Keff <= 0) {
        LOG_WRN("%s: branching/budget too small (K=%d), nothing to draft\n", __func__, K);
        return false;
    }

    const llama_model * model_mtp = llama_get_model(ctx_mtp);
    const llama_vocab * vocab     = llama_model_get_vocab(model_mtp);
    const int32_t       n_vocab   = llama_vocab_n_tokens(vocab);
    const size_t        row_bytes = (size_t)n_embd * sizeof(float);

    GGML_ASSERT((int32_t)root_h_vec.size() == n_embd);

    // ------------------------------------------------------------------
    // Initialize node array.
    // nodes_out[0] = root sentinel (depth = -1, holds id_last).
    // We will append children at each depth level.
    // ------------------------------------------------------------------
    nodes_out.clear();
    h_vecs_out.clear();

    // Root sentinel: we need its MTP output h-state to seed depth-0.
    // Do a single root forward to get root logits AND root MTP output h-state.
    {
        std::memcpy(batch.embd, root_h_vec.data(), row_bytes);
        batch.token[0]      = id_last;
        batch.pos[0]        = pos_start;
        batch.n_seq_id[0]   = 1;
        batch.seq_id[0][0]  = 0;
        batch.logits[0]     = 1;
        batch.n_tokens      = 1;

        llama_synchronize(ctx_mtp);
        const int32_t rc = llama_decode(ctx_mtp, batch);
        if (rc != 0) {
            LOG_WRN("%s: root llama_decode rc=%d\n", __func__, rc);
            return false;
        }
    }

    // Extract root MTP output h-state.
    std::vector<float> root_mtp_out(n_embd, 0.0f);
    {
        ggml_tensor * t_out = llama_context_get_t_mtp_out(ctx_mtp);
        if (t_out) {
            llama_synchronize(ctx_mtp);
            ggml_backend_tensor_get(t_out, root_mtp_out.data(), 0, row_bytes);
        }
    }

    // Extract root logits and top-K candidates for depth-0.
    const float * root_logits = llama_get_logits_ith(ctx_mtp, 0);
    if (!root_logits) {
        LOG_WRN("%s: null root_logits\n", __func__);
        return false;
    }
    auto root_topk = argtop_k_logprobs(root_logits, n_vocab, Keff);

    // Root sentinel node
    {
        mtp_tree_node root;
        root.token        = id_last;
        root.parent_idx   = -1;
        root.depth        = -1;
        root.log_prob     = 0.0f;
        root.cum_log_prob = 0.0f;
        nodes_out.push_back(root);
        h_vecs_out.push_back(root_mtp_out);
    }

    // ------------------------------------------------------------------
    // Depth expansion loop: d = 0 .. D-1
    // At each iteration we:
    //   - collect all leaves at depth (d-1) from nodes_out
    //   - build a batch of all those leaves using their h-states from h_vecs_out
    //   - run one llama_decode
    //   - append K children per leaf into nodes_out / h_vecs_out
    // ------------------------------------------------------------------

    for (int32_t d = 0; d < D; ++d) {
        // Collect indices of nodes at depth (d-1).
        // Depth -1 = root sentinel, depth 0..D-1 = expansion levels.
        const int32_t parent_depth = d - 1;
        std::vector<int32_t> leaves;
        for (int32_t ni = 0; ni < (int32_t)nodes_out.size(); ++ni) {
            if (nodes_out[ni].depth == parent_depth) {
                leaves.push_back(ni);
            }
        }

        // Budget check: if adding K*|leaves| nodes would exceed max_nodes, truncate.
        const int32_t remaining = cfg.max_nodes - (int32_t)nodes_out.size();
        if (remaining <= 0) {
            LOG_DBG("%s: node budget exhausted at depth %d (nodes=%d)\n",
                    __func__, d, (int)nodes_out.size());
            break;
        }
        const int32_t max_leaves = remaining / Keff;
        if (max_leaves == 0) break;
        if ((int32_t)leaves.size() > max_leaves) {
            leaves.resize(max_leaves);
        }

        const int32_t n_leaves  = (int32_t)leaves.size();
        const llama_pos pos_d   = pos_start + d + 1; // position for this depth

        // Clear ctx_mtp KV at pos_d and beyond to avoid collisions from prior
        // iterations writing to the same absolute position.
        // This makes MTP attention "cold" at each depth — acceptable for Phase B.
        llama_memory_seq_rm(llama_get_memory(ctx_mtp), 0, pos_d, -1);

        // Build the batch: one slot per leaf, embedding = that leaf's h-state.
        batch.n_tokens = n_leaves;
        for (int32_t li = 0; li < n_leaves; ++li) {
            const int32_t ni = leaves[li];
            // Use the parent node's MTP output h-state as embedding for this depth.
            // h_vecs_out[ni] contains the MTP out h-state generated when that
            // node was processed (or root_mtp_out for the root sentinel).
            const std::vector<float> & h_src = h_vecs_out[ni];
            if ((int32_t)h_src.size() == n_embd) {
                std::memcpy(batch.embd + (size_t)li * n_embd, h_src.data(), row_bytes);
            } else {
                // Fallback: zero-fill if h-state wasn't captured
                std::memset(batch.embd + (size_t)li * n_embd, 0, row_bytes);
            }
            batch.token[li]     = nodes_out[ni].token;
            batch.pos[li]       = pos_d;
            batch.n_seq_id[li]  = 1;
            batch.seq_id[li][0] = 0;
            batch.logits[li]    = 1;
        }

        // Run the parallel MTP forward for this depth.
        llama_synchronize(ctx_mtp);
        {
            const int32_t rc = llama_decode(ctx_mtp, batch);
            if (rc != 0) {
                LOG_WRN("%s: depth-%d llama_decode rc=%d (n_leaves=%d, pos=%d)\n",
                        __func__, d, rc, n_leaves, (int)pos_d);
                // Partial tree is still usable — stop expansion here.
                break;
            }
        }

        // Extract logits + h-states; append children.
        ggml_tensor * t_out = llama_context_get_t_mtp_out(ctx_mtp);
        LOG_DBG("%s: depth-%d expansion: %d leaves → %d children each\n",
                __func__, d, n_leaves, Keff);

        for (int32_t li = 0; li < n_leaves; ++li) {
            const int32_t parent_ni = leaves[li];

            // Extract MTP out h-state for this leaf's decode slot.
            std::vector<float> child_h(n_embd, 0.0f);
            if (t_out) {
                llama_synchronize(ctx_mtp);
                ggml_backend_tensor_get(t_out, child_h.data(),
                                        (size_t)li * row_bytes, row_bytes);
            }

            // Get logits for this slot.
            const float * slot_logits = llama_get_logits_ith(ctx_mtp, li);
            if (!slot_logits) {
                LOG_WRN("%s: null logits for leaf %d at depth %d\n", __func__, li, d);
                continue;
            }

            auto topk = argtop_k_logprobs(slot_logits, n_vocab, Keff);

            // Log top-1 for sanity.
            LOG_DBG("%s:   leaf[%d] node=%d tok=%d -> child[0]=%d (logp=%.3f)\n",
                    __func__, li, parent_ni, (int)nodes_out[parent_ni].token,
                    (int)topk[0].first, (double)topk[0].second);

            for (int32_t ki = 0; ki < (int32_t)topk.size(); ++ki) {
                mtp_tree_node child;
                child.token        = topk[ki].first;
                child.parent_idx   = parent_ni;
                child.depth        = d;
                child.log_prob     = topk[ki].second;
                child.cum_log_prob = nodes_out[parent_ni].cum_log_prob + topk[ki].second;

                nodes_out.push_back(child);
                // All children at this depth share the leaf's MTP out h-state
                // (they all feed the same parent, same decode output).
                h_vecs_out.push_back(child_h);
            }
        }
    }

    // ------------------------------------------------------------------
    // Build path-0 draft tokens: leftmost branch from each depth.
    // root (depth=-1) → first child at depth=0 → first child at depth=1 → …
    // "first child" = lowest node index at that depth that is a child of the
    // current path node.
    // ------------------------------------------------------------------
    draft_tokens.clear();
    int32_t cur = 0; // root sentinel index
    for (int32_t d = 0; d < D; ++d) {
        // Find first child of cur at depth d
        int32_t first_child = -1;
        for (int32_t ni = 0; ni < (int32_t)nodes_out.size(); ++ni) {
            if (nodes_out[ni].depth == d && nodes_out[ni].parent_idx == cur) {
                first_child = ni;
                break; // nodes appended in order, so first match = leftmost
            }
        }
        if (first_child < 0) break;
        draft_tokens.push_back(nodes_out[first_child].token);
        cur = first_child;
    }

    LOG_INF("%s: tree branching=%d max_depth=%d total_nodes=%d drafted %d path-0 tokens\n",
            __func__, Keff, D, (int)nodes_out.size(), (int)draft_tokens.size());

    return true;
}
