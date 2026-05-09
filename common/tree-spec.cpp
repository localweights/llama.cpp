// EAGLE-2 Phase D: eliminate double trunk decode + per-leaf MTP KV management
// tree-spec.cpp — multi-depth tree drafting + trunk verification
//
// Phase D.1: Direct KV commit after mtp_tree_verify.
//   After finding the accepted path, copy the path's KV (n_past..n_past+n_acc)
//   from path seq_id into slot_seq, so the caller only needs to decode one
//   correction token rather than re-decoding the full accepted prefix.
//
// Phase D.2: Per-leaf MTP seq_ids in ctx_mtp.
//   Instead of wiping ctx_mtp KV between depths and using seq_id=0 for all
//   leaves, each root-to-leaf path gets a unique seq_id in ctx_mtp.  Before
//   expanding depth d+1 children of leaf L, we copy L's MTP KV to each child's
//   seq_id, so each child sees the full ancestor history.
//   ctx_mtp must have n_seq_max >= number of leaf paths.
//
// Phase B: static fixed-topology tree of width K and depth D.
// Phase C: trunk verification via per-path seq_ids (single llama_decode).
// Phase D: see above.

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
        uint32_t                     n_seq_max_mtp,
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

    // Phase D.2: determine whether per-leaf MTP seq_ids are available.
    // We need up to K^D seq_ids (one per leaf path).  If n_seq_max_mtp < that,
    // fall back to the Phase B "cold" approach (all seq_id=0).
    //
    // Maximum number of leaf paths for a K-ary tree of depth D:
    //   n_paths_max = min(K^D, max_nodes / K) ... exact depends on tree shape.
    // Conservative upper bound: cfg.max_nodes (one seq per node).
    const bool use_per_leaf_kv = (n_seq_max_mtp >= (uint32_t)std::min(cfg.max_nodes, 255));
    LOG_DBG("%s: n_seq_max_mtp=%u, use_per_leaf_kv=%d\n",
            __func__, n_seq_max_mtp, (int)use_per_leaf_kv);

    llama_memory_t mem_mtp = llama_get_memory(ctx_mtp);

    // ------------------------------------------------------------------
    // Initialize node array.
    // nodes_out[0] = root sentinel (depth = -1, holds id_last).
    // We will append children at each depth level.
    // ------------------------------------------------------------------
    nodes_out.clear();
    h_vecs_out.clear();

    // Clear all ctx_mtp KV from pos_start onwards before starting.
    // This ensures a clean slate regardless of previous draft cycles.
    for (llama_seq_id sid = 0; sid < (llama_seq_id)n_seq_max_mtp; ++sid) {
        llama_memory_seq_rm(mem_mtp, sid, pos_start, -1);
    }

    // Root sentinel: we need its MTP output h-state to seed depth-0.
    // Do a single root forward to get root logits AND root MTP output h-state.
    // Root always uses seq_id=0.
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

    // Root sentinel node: mtp_seq_id=0 (same seq as root forward)
    {
        mtp_tree_node root;
        root.token        = id_last;
        root.parent_idx   = -1;
        root.depth        = -1;
        root.log_prob     = 0.0f;
        root.cum_log_prob = 0.0f;
        root.mtp_seq_id   = 0;
        nodes_out.push_back(root);
        h_vecs_out.push_back(root_mtp_out);
    }

    // ------------------------------------------------------------------
    // Depth expansion loop: d = 0 .. D-1
    //
    // Phase D.2 per-leaf KV strategy:
    //   Before expanding depth d, each leaf L at depth (d-1) has its MTP KV
    //   at positions pos_start..pos_start+d-1 under seq_id L.mtp_seq_id.
    //   When we expand L's K children, we assign each child a new seq_id and
    //   copy L's MTP KV to each child's seq_id before the decode.
    //   The decode batch for depth d thus has:
    //     - one entry per leaf, using the child's seq_id
    //     - pos = pos_start + d
    //   After decode, the child's KV at pos_start+d is set.
    //
    // Phase B fallback (use_per_leaf_kv=false):
    //   All entries use seq_id=0; KV wiped at pos_d before each depth.
    //   Same behavior as Phase B/C.
    //
    // seq_id assignment:
    //   Root = 0 (already decoded).
    //   Depth-0 children: seq_ids 1..K (or same 0 for fallback).
    //   Depth-1 children: new seq_ids K+1.. etc.
    //   We use next_mtp_seq (starting at 1) and increment for each new leaf.
    // ------------------------------------------------------------------

    llama_seq_id next_mtp_seq = 1; // next available seq_id in ctx_mtp (0 is root)

    for (int32_t d = 0; d < D; ++d) {
        // Collect indices of nodes at depth (d-1).
        const int32_t parent_depth = d - 1;
        std::vector<int32_t> leaves;
        for (int32_t ni = 0; ni < (int32_t)nodes_out.size(); ++ni) {
            if (nodes_out[ni].depth == parent_depth) {
                leaves.push_back(ni);
            }
        }

        // Budget check
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
        const llama_pos pos_d   = pos_start + d; // position for this depth
        // Root sentinel was at pos_start, so depth-0 children are at pos_start+1... wait:
        // Root is at pos_start. depth-0 is the first set of children.
        // Actually in the original code: root forward is at pos_start, depth-0 at pos_start+1, etc.
        // Let's keep the same convention: depth d children are at pos_start + d + 1.
        const llama_pos pos_children = pos_start + d + 1;

        if (!use_per_leaf_kv) {
            // Phase B fallback: wipe pos_d+1 and beyond from seq_id=0.
            llama_memory_seq_rm(mem_mtp, 0, pos_children, -1);
        }

        // Build the batch: one slot per leaf.
        // Phase D.2: each leaf child gets a new seq_id; copy parent's MTP KV first.
        // Phase B fallback: all slots use seq_id=0.

        // For Phase D.2: pre-assign seq_ids for the children we're about to create.
        // We need Keff * n_leaves new seq_ids; check budget.
        std::vector<llama_seq_id> leaf_child_base(n_leaves, 0); // base seq_id for first child of each leaf

        if (use_per_leaf_kv) {
            // Check seq_id budget
            const int32_t n_new_seqs = Keff * n_leaves;
            if ((int32_t)next_mtp_seq + n_new_seqs > (int32_t)n_seq_max_mtp) {
                LOG_WRN("%s: MTP seq_id budget exceeded at depth %d "
                        "(need %d, have %d remaining). Falling back to cold KV.\n",
                        __func__, d, n_new_seqs, (int)(n_seq_max_mtp - next_mtp_seq));
                // Fall back to cold for this and deeper depths
                for (int32_t li = 0; li < n_leaves; ++li) {
                    leaf_child_base[li] = 0;
                }
                llama_memory_seq_rm(mem_mtp, 0, pos_children, -1);
            } else {
                for (int32_t li = 0; li < n_leaves; ++li) {
                    leaf_child_base[li] = next_mtp_seq;
                    next_mtp_seq += Keff;
                }
            }
        }

        // Build batch: one entry per leaf (using its first child's seq_id for the decode,
        // then we'll use the output for all Keff children).
        batch.n_tokens = n_leaves;
        for (int32_t li = 0; li < n_leaves; ++li) {
            const int32_t ni = leaves[li];
            const std::vector<float> & h_src = h_vecs_out[ni];

            if ((int32_t)h_src.size() == n_embd) {
                std::memcpy(batch.embd + (size_t)li * n_embd, h_src.data(), row_bytes);
            } else {
                std::memset(batch.embd + (size_t)li * n_embd, 0, row_bytes);
            }
            batch.token[li]     = nodes_out[ni].token;
            batch.pos[li]       = pos_children;
            batch.n_seq_id[li]  = 1;
            batch.logits[li]    = 1;

            if (use_per_leaf_kv && leaf_child_base[li] > 0) {
                // Use first child's seq_id for this leaf's decode.
                // The seq_id must have a copy of the parent's MTP KV so attention
                // sees the full history.
                const llama_seq_id parent_mtp_seq = nodes_out[ni].mtp_seq_id;
                const llama_seq_id child_seq_base = leaf_child_base[li];
                // Copy parent KV to all Keff child seq_ids.
                for (int32_t ki = 0; ki < Keff; ++ki) {
                    llama_memory_seq_cp(mem_mtp, parent_mtp_seq,
                                        child_seq_base + ki, 0, -1);
                }
                // Run this leaf's decode under the first child's seq_id.
                batch.seq_id[li][0] = child_seq_base;
            } else {
                batch.seq_id[li][0] = 0;
            }
        }

        // Run the parallel MTP forward for this depth.
        llama_synchronize(ctx_mtp);
        {
            const int32_t rc = llama_decode(ctx_mtp, batch);
            if (rc != 0) {
                LOG_WRN("%s: depth-%d llama_decode rc=%d (n_leaves=%d, pos=%d)\n",
                        __func__, d, rc, n_leaves, (int)pos_children);
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

            LOG_DBG("%s:   leaf[%d] node=%d tok=%d -> child[0]=%d (logp=%.3f)\n",
                    __func__, li, parent_ni, (int)nodes_out[parent_ni].token,
                    (int)topk[0].first, (double)topk[0].second);

            // The decode ran under leaf_child_base[li] (seq_id for ki=0).
            // ki=0 already has the KV written; ki=1..Keff-1 were pre-copied from parent
            // but the NEW position (pos_children) needs to be assigned to each child.
            // Since all Keff children share the same parent KV history and diverge at
            // pos_children, we:
            //   - ki=0: KV already written by llama_decode (seq = leaf_child_base[li]+0)
            //   - ki=1..Keff-1: copy ki=0's new position into each sibling seq_id.
            if (use_per_leaf_kv && leaf_child_base[li] > 0) {
                const llama_seq_id base = leaf_child_base[li];
                // KV at pos_children is under base (ki=0).
                // Copy that single position to ki=1..Keff-1.
                for (int32_t ki = 1; ki < Keff; ++ki) {
                    llama_memory_seq_cp(mem_mtp, base, base + ki,
                                        pos_children, pos_children + 1);
                }
            }

            for (int32_t ki = 0; ki < (int32_t)topk.size(); ++ki) {
                mtp_tree_node child;
                child.token        = topk[ki].first;
                child.parent_idx   = parent_ni;
                child.depth        = d;
                child.log_prob     = topk[ki].second;
                child.cum_log_prob = nodes_out[parent_ni].cum_log_prob + topk[ki].second;

                if (use_per_leaf_kv && leaf_child_base[li] > 0) {
                    child.mtp_seq_id = leaf_child_base[li] + ki;
                } else {
                    child.mtp_seq_id = 0;
                }

                nodes_out.push_back(child);
                // All children at this depth share the same MTP out h-state
                // (same forward pass output — they diverge in future depths).
                h_vecs_out.push_back(child_h);
            }
        }
    }

    // ------------------------------------------------------------------
    // Build path-0 draft tokens: leftmost branch from each depth.
    // ------------------------------------------------------------------
    draft_tokens.clear();
    int32_t cur = 0; // root sentinel index
    for (int32_t d = 0; d < D; ++d) {
        int32_t first_child = -1;
        for (int32_t ni = 0; ni < (int32_t)nodes_out.size(); ++ni) {
            if (nodes_out[ni].depth == d && nodes_out[ni].parent_idx == cur) {
                first_child = ni;
                break;
            }
        }
        if (first_child < 0) break;
        draft_tokens.push_back(nodes_out[first_child].token);
        cur = first_child;
    }

    LOG_INF("%s: tree branching=%d max_depth=%d total_nodes=%d drafted %d path-0 tokens "
            "(per_leaf_kv=%d, mtp_seqs_used=%d)\n",
            __func__, Keff, D, (int)nodes_out.size(), (int)draft_tokens.size(),
            (int)use_per_leaf_kv, (int)next_mtp_seq);

    return true;
}

// ---------------------------------------------------------------------------
// Phase C: trunk verification
// ---------------------------------------------------------------------------

// argmax over a float array of length n.
static int32_t argmax_f32(const float * v, int32_t n) {
    int32_t best = 0;
    for (int32_t i = 1; i < n; ++i) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

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
        llama_token                 & correction_out) {

    accepted_out.clear();
    correction_out = -1;

    const int32_t n_nodes = (int32_t)nodes.size();
    if (n_nodes <= 1) {
        // Only root sentinel — nothing to verify.
        LOG_WRN("%s: tree has no draft nodes (n_nodes=%d)\n", __func__, n_nodes);
        correction_out = -1;
        return 0;
    }

    const llama_model * model_tgt = llama_get_model(ctx_tgt);
    const llama_vocab * vocab     = llama_model_get_vocab(model_tgt);
    const int32_t       n_vocab   = llama_vocab_n_tokens(vocab);

    // ------------------------------------------------------------------
    // Step 1: Build children index for each node and reset Phase-C fields.
    // ------------------------------------------------------------------
    for (int32_t ni = 0; ni < n_nodes; ++ni) {
        nodes[ni].children.clear();
        nodes[ni].kv_seq_id = slot_seq;
        nodes[ni].batch_idx = -1;
        nodes[ni].verified  = false;
    }
    for (int32_t ni = 1; ni < n_nodes; ++ni) {
        const int32_t pi = nodes[ni].parent_idx;
        if (pi >= 0 && pi < n_nodes) {
            nodes[pi].children.push_back(ni);
        }
    }

    // ------------------------------------------------------------------
    // Step 2: Assign seq_ids per root-to-leaf path via DFS.
    //
    // The root's first child inherits base_seq; each subsequent branch at
    // any fork point gets a new seq_id (next_seq_id++).
    //
    // Internal nodes are assigned the seq_id of their first descendant leaf
    // (they're on that path).  This means they're written under only one
    // path's seq_id — which is fine because they also get copied via
    // llama_memory_seq_cp below.
    //
    // Used seq_ids: all unique seq_ids that appear on non-root nodes.
    // These are the seq_ids we copy prompt KV to and clean up afterwards.
    // ------------------------------------------------------------------
    struct dfs_entry { int32_t node_idx; llama_seq_id seq_id; };

    llama_seq_id next_seq_id = base_seq;

    // Collect unique seq_ids used by non-root nodes (for KV copy + cleanup).
    std::vector<llama_seq_id> used_seq_ids;
    // Use a small set via linear scan (n_paths is small, <= 64).
    auto add_seq = [&](llama_seq_id sid) {
        for (auto s : used_seq_ids) if (s == sid) return;
        used_seq_ids.push_back(sid);
    };

    // DFS.  Root (ni=0) gets slot_seq (not added to used_seq_ids — it's the prompt seq).
    // Nodes with depth >= 0 get their assigned path seq_id.
    {
        std::vector<dfs_entry> stack;
        // Push root's children directly, giving first child base_seq.
        const auto & root_ch = nodes[0].children;
        for (int32_t ci_idx = 0; ci_idx < (int32_t)root_ch.size(); ++ci_idx) {
            llama_seq_id sid = (ci_idx == 0) ? next_seq_id : ++next_seq_id;
            stack.push_back({root_ch[ci_idx], sid});
        }
        if (!root_ch.empty()) ++next_seq_id; // advance past base_seq if we used it

        while (!stack.empty()) {
            auto [ni, sid] = stack.back();
            stack.pop_back();

            nodes[ni].kv_seq_id = sid;
            add_seq(sid);

            const auto & ch = nodes[ni].children;
            for (int32_t ci_idx = 0; ci_idx < (int32_t)ch.size(); ++ci_idx) {
                llama_seq_id child_sid = (ci_idx == 0) ? sid : next_seq_id++;
                stack.push_back({ch[ci_idx], child_sid});
            }
        }
    }

    const int32_t n_paths = (int32_t)used_seq_ids.size();
    LOG_DBG("%s: %d non-root nodes, %d path seq_ids [%d..%d], n_seq_max=%u\n",
            __func__, n_nodes - 1, n_paths, (int)base_seq, (int)(next_seq_id-1), n_seq_max);

    // Runtime check: highest seq_id used must be < n_seq_max.
    if ((uint32_t)next_seq_id > n_seq_max) {
        LOG_ERR("%s: seq_id overflow (need up to %d, n_seq_max=%u). "
                "Reduce tree size or set --parallel higher. Falling back.\n",
                __func__, (int)next_seq_id, n_seq_max);
        correction_out = -1;
        return 0;
    }

    // ------------------------------------------------------------------
    // Step 3: Copy prompt KV from slot_seq to each path seq_id.
    //
    // The trunk KV for positions [0, n_past) belongs to slot_seq.
    // Each tree path needs its own copy so that its tokens at positions
    // [n_past, n_past+depth] attend only to that path's ancestors.
    // ------------------------------------------------------------------
    llama_memory_t mem_tgt = llama_get_memory(ctx_tgt);

    // Use p1=-1 (end of buffer) to satisfy the "full buffer" requirement of
    // llama_kv_cache::seq_cp for cross-stream copies.  Cells at positions
    // >= n_past are empty (not assigned to slot_seq) so they won't be added
    // to the destination seq_ids.
    for (llama_seq_id sid : used_seq_ids) {
        llama_memory_seq_cp(mem_tgt, slot_seq, sid, 0, -1);
    }
    LOG_DBG("%s: copied prompt KV (slot_seq=%d, full buf) → %d path seq_ids\n",
            __func__, (int)slot_seq, n_paths);

    // ------------------------------------------------------------------
    // Step 4: Build the trunk verification batch.
    //
    // Batch layout (batch_idx = 0..N-1):
    //   [0]:   root probe — id_last at pos (n_past-1), seq_id=slot_seq, logits=1.
    //          Gives us trunk's prediction for depth-0 (the first draft token).
    //   [1..]: all non-root nodes in BFS order (depth 0 first, then depth 1, …).
    //          Each entry: token=node.token, pos=n_past+depth, seq_id=node.kv_seq_id.
    //
    // The root probe re-decodes the last prompt token.  This is cheap (1 token)
    // and gives trunk's next-token distribution at the root position.
    //
    // NOTE: n_past-1 is already in the KV for slot_seq.  Re-decoding it under
    // slot_seq does NOT write a duplicate KV entry because the KV cache uses
    // position as the key — an existing cell at the same (seq, pos) is reused.
    // ------------------------------------------------------------------
    int32_t max_depth = 0;
    for (int32_t ni = 1; ni < n_nodes; ++ni) {
        max_depth = std::max(max_depth, nodes[ni].depth);
    }

    int32_t batch_idx = 0;

    // Root probe entries.
    //
    // id_last is the "sampled" token from the previous trunk decode step.
    // It has NOT yet been decoded through ctx_tgt (the server will add it in
    // the next normal trunk verification batch).  We decode it here at position
    // n_past under EACH PATH SEQ_ID to:
    //   1. Fill KV position n_past for each path (ensures consecutive positions).
    //   2. Obtain trunk's prediction for depth-0 via the first path's entry (logits=1).
    //
    // We do NOT add a root probe under slot_seq.  This avoids contaminating
    // the slot's KV and ensures the server's normal batch can write id_last at
    // n_past cleanly (no collision, no X < Y failure).
    //
    // Root sentinel (nodes[0]) points to the first path root probe entry.
    //
    // Batch layout:
    //   [0]:       id_last @ n_past, path_seq[0], logits=1  (root probe with logits)
    //   [1..P-1]:  id_last @ n_past, path_seq[1..P-1], logits=0  (continuity fill)
    //   [P..]:     BFS tree nodes (depth-0 at n_past+1, depth-1 at n_past+2, etc.)
    //
    // After verify, path seq_ids are removed entirely (0..-1), including n_past.
    // slot_seq is untouched during this verify decode.

    for (int32_t pi = 0; pi < n_paths; ++pi) {
        const llama_seq_id sid = used_seq_ids[pi];
        const bool want_logits = (pi == 0);  // first path provides root logits
        tgt_batch.token[batch_idx]      = id_last;
        tgt_batch.pos[batch_idx]        = n_past;
        tgt_batch.n_seq_id[batch_idx]   = 1;
        tgt_batch.seq_id[batch_idx][0]  = sid;
        tgt_batch.logits[batch_idx]     = want_logits ? 1 : 0;
        if (want_logits) {
            nodes[0].batch_idx = batch_idx;  // root sentinel → first path probe
        }
        ++batch_idx;
    }

    // Non-root nodes in BFS order.
    // Depth-0 nodes are at position n_past + 1 (since id_last is at n_past).
    // Depth-d nodes are at position n_past + 1 + d.
    for (int32_t d = 0; d <= max_depth; ++d) {
        for (int32_t ni = 1; ni < n_nodes; ++ni) {
            if (nodes[ni].depth != d) continue;

            nodes[ni].batch_idx = batch_idx;
            tgt_batch.token[batch_idx]      = nodes[ni].token;
            tgt_batch.pos[batch_idx]        = n_past + 1 + d;
            tgt_batch.n_seq_id[batch_idx]   = 1;
            tgt_batch.seq_id[batch_idx][0]  = nodes[ni].kv_seq_id;
            tgt_batch.logits[batch_idx]     = 1;
            ++batch_idx;
        }
    }
    tgt_batch.n_tokens = batch_idx;

    LOG_DBG("%s: trunk batch: %d tokens (%d path probes + %d tree nodes), max_depth=%d\n",
            __func__, batch_idx, n_paths, batch_idx - n_paths, max_depth);

    // ------------------------------------------------------------------
    // Step 5: Run trunk forward pass.
    // ------------------------------------------------------------------
    llama_synchronize(ctx_tgt);
    const int32_t dec_rc = llama_decode(ctx_tgt, tgt_batch);
    if (dec_rc != 0) {
        LOG_ERR("%s: llama_decode(ctx_tgt) rc=%d — tree verify failed, cleaning up\n",
                __func__, dec_rc);
        for (llama_seq_id sid : used_seq_ids) {
            llama_memory_seq_rm(mem_tgt, sid, 0, -1);
        }
        correction_out = -1;
        return 0;
    }

    // ------------------------------------------------------------------
    // Step 6: Accept walk — greedy tree traversal from root.
    //
    // accepted_out = [accepted_tok_0, ..., accepted_tok_{k-1}] (length = n_accepted)
    // correction_out = the correction token (always set)
    //
    // Phase D.1 change: correction is returned separately.  The caller (server)
    // needs only correction_out to generate the next token; it does not need to
    // re-run the full accepted prefix through the trunk.
    // ------------------------------------------------------------------
    int32_t n_accepted   = 0;
    int32_t cur_node     = 0; // root sentinel
    int32_t accepted_seq = -1; // seq_id of the accepted path (for D.1 commit)

    for (;;) {
        const int32_t cur_bidx = nodes[cur_node].batch_idx;
        if (cur_bidx < 0) {
            LOG_ERR("%s: node %d has batch_idx=-1 (unexpected)\n", __func__, cur_node);
            // Emit fallback correction (id_last) so caller isn't stuck.
            correction_out = id_last;
            break;
        }
        const float * logits = llama_get_logits_ith(ctx_tgt, cur_bidx);
        if (!logits) {
            LOG_ERR("%s: null logits at batch_idx=%d\n", __func__, cur_bidx);
            correction_out = id_last;
            break;
        }
        const llama_token trunk_tok = (llama_token)argmax_f32(logits, n_vocab);

        const auto & ch = nodes[cur_node].children;
        LOG_DBG("%s: node=%d depth=%d batch=%d trunk_tok=%d n_children=%d\n",
                __func__, cur_node, nodes[cur_node].depth, cur_bidx,
                (int)trunk_tok, (int)ch.size());

        // Find matching child.
        int32_t match = -1;
        for (int32_t ci : ch) {
            if (nodes[ci].token == trunk_tok) {
                match = ci;
                break;
            }
        }

        if (match >= 0) {
            // Accept this child's token.
            accepted_out.push_back(trunk_tok);
            nodes[match].verified = true;
            accepted_seq = nodes[match].kv_seq_id; // track which path seq we're on
            cur_node = match;
            ++n_accepted;
        } else {
            // Rejection: trunk_tok is the correction.
            correction_out = trunk_tok;
            break;
        }

        // If we've accepted a leaf (no children), get correction from leaf logits.
        if (nodes[cur_node].children.empty()) {
            const int32_t leaf_bidx = nodes[cur_node].batch_idx;
            if (leaf_bidx >= 0) {
                const float * leaf_logits = llama_get_logits_ith(ctx_tgt, leaf_bidx);
                if (leaf_logits) {
                    correction_out = (llama_token)argmax_f32(leaf_logits, n_vocab);
                    LOG_DBG("%s: leaf correction=%d\n", __func__, (int)correction_out);
                } else {
                    correction_out = id_last; // fallback
                }
            } else {
                correction_out = id_last; // fallback
            }
            break;
        }
    }

    LOG_INF("%s: tree verify: %d draft tokens accepted, correction=%d\n",
            __func__, n_accepted, (int)correction_out);

    // ------------------------------------------------------------------
    // Step 7: Phase D.1 — direct commit.
    //
    // If commit_to_slot=true and we have an accepted path (n_accepted > 0),
    // copy the accepted positions from the path's seq_id into slot_seq.
    // This includes:
    //   - n_past         (id_last via root probe under accepted_seq)
    //   - n_past+1       (first accepted token at depth 0)
    //   - ...
    //   - n_past+n_accepted  (last accepted token)
    //
    // After copy, the caller only needs to decode the correction token at
    // n_past+n_accepted+1, rather than the full [id_last, tok0, ..., tok_n_acc].
    //
    // If commit_to_slot=false (Phase C compat) or n_accepted==0, skip commit.
    // (When n_accepted==0 the caller still does the normal single-token decode.)
    // ------------------------------------------------------------------
    if (commit_to_slot && n_accepted > 0 && accepted_seq >= 0) {
        // Copy positions n_past .. n_past+n_accepted (inclusive) from accepted_seq to slot_seq.
        // p1 is exclusive, so p1 = n_past + n_accepted + 1.
        llama_memory_seq_cp(mem_tgt, (llama_seq_id)accepted_seq, slot_seq,
                            n_past, n_past + n_accepted + 1);
        LOG_DBG("%s: D.1 commit: copied pos [%d..%d] from seq %d to slot_seq %d\n",
                __func__, (int)n_past, (int)(n_past + n_accepted),
                (int)accepted_seq, (int)slot_seq);
    }

    // ------------------------------------------------------------------
    // Step 8: Cleanup — remove all path seq_ids from ctx_tgt's KV.
    // ------------------------------------------------------------------
    for (llama_seq_id sid : used_seq_ids) {
        llama_memory_seq_rm(mem_tgt, sid, 0, -1);
    }
    LOG_DBG("%s: removed %d path seq_ids from ctx_tgt KV (slot_seq=%d%s)\n",
            __func__, n_paths, (int)slot_seq,
            (commit_to_slot && n_accepted > 0) ? " — committed accepted path" : "");

    return n_accepted;
}
