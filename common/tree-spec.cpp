// EAGLE-2 Phase A: parallel MTP forward proof
// tree-spec.cpp — single-depth tree drafting orchestrator
//
// Phase A scope: prove that llama_decode(ctx_mtp, batch) with n_tokens>1 works.
// We expand ONE depth level — take top-K tokens from the root hidden state,
// feed all K in a single MTP forward pass, collect their logits.
// Verification and accept still use only path-0 (leftmost branch), so the
// accept rate is identical to the linear chain.  The key validation is that
// the parallel decode produces sensible, non-garbage logits for each candidate.

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
    // Partial sort to find top-k
    // Use a min-heap of size k for efficiency on large vocab
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

    const int32_t K = std::min(cfg.branching, cfg.max_nodes - 1);
    if (K <= 0) {
        LOG_WRN("%s: branching/budget too small (K=%d), nothing to draft\n", __func__, K);
        return false;
    }

    const llama_model * model_mtp = llama_get_model(ctx_mtp);
    const llama_vocab * vocab     = llama_model_get_vocab(model_mtp);
    const int32_t       n_vocab   = llama_vocab_n_tokens(vocab);
    const size_t        row_bytes = (size_t)n_embd * sizeof(float);

    // ------------------------------------------------------------------
    // Phase A: single depth expansion.
    // We expand the root node (depth -1 → depth 0) by feeding K candidate
    // tokens from the root's logits all in one MTP forward pass.
    //
    // Step 1: Get top-K candidates from a SINGLE root forward pass.
    //         This is the same as the linear AR step-0 but we want the
    //         top-K logits, not just the top-1.
    // ------------------------------------------------------------------

    // --- Step 1: single-token forward to get root logits ---
    GGML_ASSERT((int32_t)root_h_vec.size() == n_embd);

    // Populate batch slot 0 for root
    std::memcpy(batch.embd, root_h_vec.data(), row_bytes);
    batch.token[0]      = id_last;
    batch.pos[0]        = pos_start;
    batch.n_seq_id[0]   = 1;
    batch.seq_id[0][0]  = 0;
    batch.logits[0]     = 1;
    batch.n_tokens      = 1;

    llama_synchronize(ctx_mtp);
    {
        const int32_t rc = llama_decode(ctx_mtp, batch);
        if (rc != 0) {
            LOG_WRN("%s: root llama_decode rc=%d\n", __func__, rc);
            return false;
        }
    }

    // Extract top-K from root logits (row 0)
    const float * root_logits = llama_get_logits_ith(ctx_mtp, 0);
    if (!root_logits) {
        LOG_WRN("%s: null root_logits\n", __func__);
        return false;
    }

    auto topk = argtop_k_logprobs(root_logits, n_vocab, K);

    // Extract root t_mtp_out hidden state (we'll need it to populate each
    // leaf in the parallel expansion step).
    ggml_tensor * t_mtp_out_root = llama_context_get_t_mtp_out(ctx_mtp);
    std::vector<float> root_mtp_out(n_embd, 0.0f);
    if (t_mtp_out_root) {
        llama_synchronize(ctx_mtp);
        // The tensor is [n_embd, 1] (single token), row 0.
        ggml_backend_tensor_get(t_mtp_out_root, root_mtp_out.data(), 0, row_bytes);
    }

    // Build node array: index 0 = root sentinel, 1..K = depth-0 candidates.
    nodes_out.clear();
    h_vecs_out.clear();

    // root sentinel
    {
        mtp_tree_node root;
        root.token        = id_last;
        root.parent_idx   = -1;
        root.depth        = -1;
        root.log_prob     = 0.0f;
        root.cum_log_prob = 0.0f;
        nodes_out.push_back(root);
        h_vecs_out.push_back(root_mtp_out); // root's output h-state
    }

    // --- Step 2: parallel forward for all K depth-0 candidates ---
    // Each candidate uses root_mtp_out as its hidden state embedding.
    // We feed all K in one batch at position pos_start (same position,
    // MTP KV cleared before this to avoid aliasing — handled by caller
    // via the existing draft() KV cleanup logic).

    // Clear the MTP KV entry written by the root step so we can reuse pos.
    // After step 1, pos_start is in the KV.  Remove it so the parallel step
    // can write K entries at pos_start without collision.  We use pos_start+1
    // for the parallel candidates to keep things clean.
    const llama_pos pos_leaf = pos_start + 1;

    // We don't need to clear pos_start from the KV for ctx_mtp because
    // the root step IS valid — we just advance the position for leaves.
    // The linear AR chain would have done root at pos_start, leaf at pos_start+1.
    // We do the same: K leaves all at pos_leaf.  Because ctx_mtp is KV seq=0
    // single-slot and pos_leaf == pos_start+1, these K writes collide in the KV
    // but since we clear ctx_mtp KV after acceptance (existing accept() logic)
    // and the MTP model's attention only conditions on its own past lightly,
    // this is acceptable for Phase A proof.  Phase D will fix proper KV mgmt.
    //
    // For Phase A we just feed them all at the same pos and ignore KV quality.
    // The goal is to show the parallel decode produces sensible logits.

    // Populate batch with K leaves
    batch.n_tokens = K;
    for (int32_t i = 0; i < K; ++i) {
        const llama_token cand_tok = topk[i].first;
        const float       log_p    = topk[i].second;

        // embedding = root's MTP output hidden state (same for all leaves)
        std::memcpy(batch.embd + (size_t)i * n_embd,
                    root_mtp_out.data(), row_bytes);
        batch.token[i]     = cand_tok;
        batch.pos[i]       = pos_leaf;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1;

        // Register as a depth-0 node
        mtp_tree_node node;
        node.token        = cand_tok;
        node.parent_idx   = 0; // root
        node.depth        = 0;
        node.log_prob     = log_p;
        node.cum_log_prob = log_p;
        nodes_out.push_back(node);
        h_vecs_out.push_back({}); // filled after parallel decode below
    }

    // Clear ctx_mtp KV at pos_leaf before writing K tokens there.
    llama_memory_seq_rm(llama_get_memory(ctx_mtp), 0, pos_leaf, -1);

    // --- Parallel MTP forward: all K candidates in one call ---
    llama_synchronize(ctx_mtp);
    {
        const int32_t rc = llama_decode(ctx_mtp, batch);
        if (rc != 0) {
            LOG_WRN("%s: parallel llama_decode rc=%d (K=%d, pos=%d)\n",
                    __func__, rc, K, (int)pos_leaf);
            return false;
        }
    }

    // Extract logits and hidden states for each candidate.
    // Log the top-1 token predicted by each branch (sanity check).
    LOG_DBG("%s: tree depth-0 expansion (K=%d, pos=%d):\n", __func__, K, (int)pos_leaf);

    ggml_tensor * t_mtp_out = llama_context_get_t_mtp_out(ctx_mtp);

    for (int32_t i = 0; i < K; ++i) {
        const int32_t node_idx = i + 1; // nodes_out[0] = root sentinel

        // Log the top-1 of each branch's logits
        const float * leaf_logits = llama_get_logits_ith(ctx_mtp, i);
        if (leaf_logits) {
            // Find argmax for logging
            int32_t argmax = 0;
            float   vmax   = leaf_logits[0];
            for (int32_t v = 1; v < n_vocab; ++v) {
                if (leaf_logits[v] > vmax) { vmax = leaf_logits[v]; argmax = v; }
            }
            LOG_DBG("%s:   branch[%d] cand_tok=%d -> next_argmax=%d (logit=%.3f)\n",
                    __func__, i, (int)topk[i].first, argmax, (double)vmax);
        }

        // Extract hidden state for this leaf (for future depth-1 expansion)
        if (t_mtp_out) {
            llama_synchronize(ctx_mtp);
            h_vecs_out[node_idx].resize(n_embd);
            ggml_backend_tensor_get(t_mtp_out, h_vecs_out[node_idx].data(),
                                    (size_t)i * row_bytes, row_bytes);
        }
    }

    // --- Draft tokens: use path-0 (leftmost/highest-prob branch) ---
    // Phase A: accept only the linear path-0 so accept rate == linear baseline.
    // The parallel forward was for proof-of-concept; the accepted sequence is
    // just the top-1 token from the root forward (same as linear chain).
    draft_tokens.clear();
    draft_tokens.push_back(topk[0].first); // depth-0 best token

    LOG_INF("%s: tree branching=%d drafted %d tokens (path-0), candidates logged above\n",
            __func__, K, (int)draft_tokens.size());

    return true;
}
