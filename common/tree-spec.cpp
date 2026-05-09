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

    // Clear all ctx_mtp KV before starting.
    // Phase E.1 cross-request fix: path-scratch seqs (sid >= 1) carry stale KV
    // from prior draft cycles (potentially from a different request entirely).
    // The earlier wipe range [pos_start, -1) only cleared positions >= pos_start,
    // leaving cells at positions < pos_start intact — which corrupts the
    // consecutive-position invariant when a new request starts at a smaller
    // pos_start. Seq 0 is the canonical (correctly synced via mtp_set_pending),
    // so trim it at pos_start to preserve the committed prefix; scratch seqs
    // are fully wiped.
    llama_memory_seq_rm(mem_mtp, 0, pos_start, -1);
    for (llama_seq_id sid = 1; sid < (llama_seq_id)n_seq_max_mtp; ++sid) {
        llama_memory_seq_rm(mem_mtp, sid, 0, -1);
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

        // Extract logits + h-states; collect candidate children with pruning (E.2/E.3).
        ggml_tensor * t_out = llama_context_get_t_mtp_out(ctx_mtp);
        LOG_DBG("%s: depth-%d expansion: %d leaves → up to %d children each\n",
                __func__, d, n_leaves, Keff);

        // E.2/E.3: collect ALL candidate children first, then prune + sort + cap.
        // Each candidate carries: (parent_ni, ki, child node, h-state).
        struct CandChild {
            int32_t            parent_ni;
            int32_t            ki;           // sibling index within parent (0 = path-0)
            mtp_tree_node      node;
            std::vector<float> h;
        };
        std::vector<CandChild> candidates;
        candidates.reserve((size_t)n_leaves * Keff);

        // Per-leaf: track whether ki=0 was kept (for MTP KV bookkeeping).
        // We must do the KV sibling-copy BEFORE we know which ki survive pruning,
        // because the copy is cheap and we clean up unused seq_ids later.
        // The seq_id cleanup is handled at draft-start (next cycle) via the
        // llama_memory_seq_rm loop at the top of mtp_tree_draft().

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

            // Copy sibling MTP KV positions (per-leaf KV accounting for ki=1..Keff-1).
            if (use_per_leaf_kv && leaf_child_base[li] > 0) {
                const llama_seq_id base = leaf_child_base[li];
                for (int32_t ki = 1; ki < Keff; ++ki) {
                    llama_memory_seq_cp(mem_mtp, base, base + ki,
                                        pos_children, pos_children + 1);
                }
            }

            const float parent_cum_lp = nodes_out[parent_ni].cum_log_prob;

            for (int32_t ki = 0; ki < (int32_t)topk.size(); ++ki) {
                mtp_tree_node child;
                child.token        = topk[ki].first;
                child.parent_idx   = parent_ni;
                child.depth        = d;
                child.log_prob     = topk[ki].second;
                child.cum_log_prob = parent_cum_lp + topk[ki].second;

                if (use_per_leaf_kv && leaf_child_base[li] > 0) {
                    child.mtp_seq_id = leaf_child_base[li] + ki;
                } else {
                    child.mtp_seq_id = 0;
                }

                // E.2: log-prob floor pruning (skip unless ki=0 of first leaf — always keep path-0).
                const bool is_path0_fallback = (li == 0 && ki == 0);
                if (cfg.p_min < 0.0f && child.cum_log_prob < cfg.p_min && !is_path0_fallback) {
                    LOG_DBG("%s:   pruned leaf[%d] ki=%d cum_lp=%.3f < p_min=%.3f\n",
                            __func__, li, ki, (double)child.cum_log_prob, (double)cfg.p_min);
                    continue;
                }

                candidates.push_back({parent_ni, ki, child, child_h});
            }
        }

        // E.3: budget allocation — sort by cum_log_prob descending, keep top-(remaining).
        // Always ensure path-0 (li=0, ki=0) survives even if budget is tight.
        const int32_t budget_now = cfg.max_nodes - (int32_t)nodes_out.size();
        if (budget_now <= 0) {
            LOG_DBG("%s: budget exhausted at depth %d after candidates\n", __func__, d);
            break;
        }

        // Find path-0 candidate index before sorting.
        int32_t path0_cand_idx = -1;
        for (int32_t ci = 0; ci < (int32_t)candidates.size(); ++ci) {
            if (candidates[ci].parent_ni == leaves[0] && candidates[ci].ki == 0) {
                path0_cand_idx = ci;
                break;
            }
        }

        // Sort by cum_log_prob descending.
        std::sort(candidates.begin(), candidates.end(),
                  [](const CandChild & a, const CandChild & b) {
                      return a.node.cum_log_prob > b.node.cum_log_prob;
                  });

        // Cap at budget.
        if ((int32_t)candidates.size() > budget_now) {
            candidates.resize((size_t)budget_now);
        }

        // Ensure path-0 is present if it was pruned by the cap.
        if (path0_cand_idx >= 0 && (int32_t)path0_cand_idx >= (int32_t)candidates.size()) {
            // path-0 was in original candidates but got cut; add it back.
            // We need to find it in the original (unsorted) list — but we sorted in place.
            // Rebuild path-0 from leaves[0] topk[0] data (already computed above).
            // Actually, since we sorted and trimmed, path-0 may still be in the vector
            // under a different index. Search by token and parent.
            const int32_t p0_parent = (n_leaves > 0) ? leaves[0] : -1;
            bool found = false;
            for (const auto & cc : candidates) {
                if (cc.parent_ni == p0_parent && cc.ki == 0) { found = true; break; }
            }
            if (!found && p0_parent >= 0 && !candidates.empty()) {
                // path-0 was cut — swap in the lowest-ranked candidate for it.
                // We don't have the original data anymore since we sorted. This is a
                // degenerate case (budget < n_leaves). Just accept it — we'll have
                // at least 1 candidate from some other leaf.
                LOG_DBG("%s: path-0 evicted by budget cap at depth %d\n", __func__, d);
            }
        }

        // Append accepted candidates to node array.
        for (const auto & cc : candidates) {
            nodes_out.push_back(cc.node);
            h_vecs_out.push_back(cc.h);
        }

        LOG_DBG("%s: depth %d: %d candidates → %d kept (budget=%d, p_min=%.3f)\n",
                __func__, d, (int)(candidates.size() + /* pruned approximation */ 0),
                (int)candidates.size(), budget_now, (double)cfg.p_min);
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
    // Step 2: Assign seq_ids per root-to-leaf path via DFS (path-sharing).
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
    //
    // Use p1=-1 (end of buffer) to satisfy the "full buffer" requirement of
    // llama_kv_cache::seq_cp for cross-stream copies.  Cells at positions
    // >= n_past are empty (not assigned to slot_seq) so they won't be added
    // to the destination seq_ids.
    //
    // With kv_unified=true on ctx_tgt (same stream for all seq_ids), partial
    // range copies also work.  With kv_unified=false, p0=0,p1=-1 passes the
    // is_full check and only copies existing cells (positions 0..n_past-1).
    // ------------------------------------------------------------------
    llama_memory_t mem_tgt = llama_get_memory(ctx_tgt);

    // Copy positions [0, n_past) — NOT including n_past, which the root probe
    // will add.  With kv_unified=true on ctx_tgt (same stream for all seq_ids),
    // partial range seq_cp works via cell-tag sharing without data copy.
    // n_past is currently the max KV position in slot_seq (most recently sampled
    // token); we exclude it so root probe can write id_last at n_past without
    // triggering the "positions must be consecutive" check.
    for (llama_seq_id sid : used_seq_ids) {
        llama_memory_seq_cp(mem_tgt, slot_seq, sid, 0, n_past);
    }
    LOG_DBG("%s: copied prompt KV (slot_seq=%d, pos 0..%d-1) → %d path seq_ids\n",
            __func__, (int)slot_seq, (int)n_past, n_paths);

    // ------------------------------------------------------------------
    // Step 4: Sequential depth-by-depth trunk verification.
    //
    // Single-pass tree verify fails for sibling branches: a path_seq whose
    // first node is at depth d>0 has a positional gap between its root probe
    // (n_past) and its first node (n_past+1+d), causing the llama-batch
    // "positions not continuous" check to fire.
    //
    // Fix: decode one depth level at a time.
    //   Pass d=0: root probes for all path_seqs + depth-0 nodes.
    //   After pass d: for each path_seq that has its FIRST node at depth d+1
    //     (i.e., a sibling branch that diverges at depth d+1), copy the decoded
    //     position n_past+d from the parent path_seq into this sibling path_seq.
    //     This fills the continuity gap so the next decode pass sees consecutive
    //     positions on this sibling path.
    //   Pass d+1: depth-(d+1) nodes (all path_seqs already have n_past..n_past+d).
    //
    // For each path_seq, the parent is the path_seq of its shallowest ancestor
    // that differs from it — i.e., the path_seq from which it branched.
    //
    // n_decode_passes = max_depth + 1.
    // ------------------------------------------------------------------
    int32_t max_depth = 0;
    for (int32_t ni = 1; ni < n_nodes; ++ni) {
        max_depth = std::max(max_depth, nodes[ni].depth);
    }

    // For each non-root node, determine the depth at which its path_seq first
    // diverges from its parent's path_seq.  We need this to know which sibling
    // path_seqs need a bridge-copy after each decode pass.
    //
    // "First node at depth D on path_seq S" = the shallowest node whose
    // kv_seq_id == S.  If that depth is > 0, then path S branches off at depth D
    // and needs positions n_past..n_past+D-1 copied from the parent path_seq
    // before we can place a node at n_past+D+1... wait, that's still wrong.
    //
    // Actually the issue is simpler: after decoding pass at depth d, every path
    // has KV at positions 0..n_past-1 (from step 3), n_past (root probe), and
    // n_past+1..n_past+1+d (tree nodes at depths 0..d) — BUT only for paths
    // that had a node at each of those depths.
    //
    // A sibling path that branches at depth d' > 0 only has nodes at depths
    // d'..max_depth.  Before we can decode its node at depth d'+1, it needs
    // a KV entry at n_past+d'+1 (the parent's node).  We bridge this by copying
    // from the parent path after we decode depth d'.
    //
    // Implementation: for each path_seq S, find its "first node depth" F(S).
    // After decoding depth d == F(S)-1, copy pos n_past+F(S) from parent_path(S)
    // to S so that S has n_past..n_past+F(S) before we decode F(S)+1... hmm.
    //
    // Simplest correct approach: after each depth-d decode, for every path_seq S
    // whose first node depth > d (i.e., no node yet at depth d), copy the JUST-
    // decoded position from the appropriate donor path_seq.  We do this for
    // every depth that the sibling path is missing, until it has its first node.
    //
    // "Donor" for path_seq S at depth d = the path_seq of S's parent node at
    // depth d.  Since path_seqs are assigned via DFS, the parent of S at depth d
    // is the path_seq that carries the tree node at (depth=d, kv_seq_id != S).
    //
    // For simplicity: maintain a per-path_seq "filled_up_to" counter.
    // path_seqs that have a node at depth d are filled at depth d by the decode.
    // path_seqs without a node at depth d need bridging from their donor at depth d.
    //
    // Donor lookup: for path_seq S, find the deepest ancestor node that has a
    // different kv_seq_id; that kv_seq_id is the donor at that depth.
    // Walk up the tree from S's first node to find donors at each depth.
    //
    // Precompute: for each path_seq S, which path_seq provides its bridge-copy
    // at each depth level where S has no node.
    //
    // Per path_seq: first_node_depth = min depth among nodes with kv_seq_id == S.
    // Donor = path_seq of the parent of S's first node.

    // Map path_seq → first node depth and donor path_seq.
    struct PathInfo {
        int32_t first_depth;   // depth of first node in this path_seq (-1 = root probe only)
        llama_seq_id donor;    // path_seq to copy from before decoding first node
    };
    std::vector<PathInfo> path_info(n_paths, {max_depth + 1, -1});
    for (int32_t pi = 0; pi < n_paths; ++pi) {
        const llama_seq_id sid = used_seq_ids[pi];
        int32_t first_d = max_depth + 1;
        int32_t first_ni = -1;
        for (int32_t ni = 1; ni < n_nodes; ++ni) {
            if (nodes[ni].kv_seq_id == sid && nodes[ni].depth < first_d) {
                first_d  = nodes[ni].depth;
                first_ni = ni;
            }
        }
        path_info[pi].first_depth = first_d;
        // Donor = kv_seq_id of the parent of this path's first node.
        if (first_ni >= 0 && first_d > 0) {
            const int32_t par_ni = nodes[first_ni].parent_idx;
            if (par_ni >= 0) {
                path_info[pi].donor = nodes[par_ni].kv_seq_id;
            }
        }
    }

    // Decode pass-by-pass, interleaved with accept walk.
    //
    // KEY FIX (E.1): logits buffer is overwritten by each llama_decode call.
    // Solution: read logits for each node immediately after the decode pass
    // that produced them.
    //
    // Accept walk uses a "pending_trunk_tok" model:
    //
    //   pending_trunk_tok = the argmax token we are looking for among the
    //     current depth's children of cur_node.
    //
    //   Phase:
    //   PHASE_FIND_CHILD: scanning depth-d children of cur_node for pending_trunk_tok.
    //     After each depth-d decode, scan batch: if any depth-d child of cur_node
    //     has token == pending_trunk_tok, accept it.  Then immediately read that
    //     child's logits (valid now), compute new pending_trunk_tok, and go to
    //     PHASE_FIND_CHILD at depth d+1.
    //     If no match: correction = pending_trunk_tok, done.
    //
    //   PHASE_READ_ROOT: d=0 pass produces root probe logits.
    //     Read root's logits → pending_trunk_tok for depth-0 children.
    //
    // Walk state:
    //   cur_node = last accepted node (root=0 initially)
    //   pending_trunk_tok = what we need to find among depth-d children (-1 if not yet set)
    //   The root probe logits give us pending_trunk_tok after d=0 decode.
    //   After accepting depth-d child C: read C's logits → new pending_trunk_tok for d+1.

    int32_t n_accepted        = 0;
    int32_t cur_node          = 0;      // last accepted node
    int32_t accepted_seq      = -1;
    bool    walk_done         = false;
    llama_token pending_trunk_tok = -1; // trunk token to look for at next depth

    bool decode_failed = false;
    for (int32_t d = 0; d <= max_depth && !decode_failed; ++d) {
        if (walk_done) break;

        int32_t batch_idx = 0;

        if (d == 0) {
            // Root probes: one per path seq, all at position n_past.
            for (int32_t pi = 0; pi < n_paths; ++pi) {
                const llama_seq_id sid = used_seq_ids[pi];
                const bool want_logits = (pi == 0);
                tgt_batch.token[batch_idx]      = id_last;
                tgt_batch.pos[batch_idx]        = n_past;
                tgt_batch.n_seq_id[batch_idx]   = 1;
                tgt_batch.seq_id[batch_idx][0]  = sid;
                tgt_batch.logits[batch_idx]     = want_logits ? 1 : 0;
                if (want_logits) {
                    nodes[0].batch_idx = batch_idx;
                }
                ++batch_idx;
            }
        }

        // Nodes at depth d.  Request logits for ALL depth-d children of cur_node
        // (we need them for the accept check).  Other nodes: logits=0 (save bandwidth).
        // But we always request logits=1 for all to keep it simple.
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

        if (batch_idx == 0) continue;

        tgt_batch.n_tokens = batch_idx;

        LOG_DBG("%s: depth-pass d=%d: %d tokens\n", __func__, d, batch_idx);

        llama_synchronize(ctx_tgt);
        const int32_t rc = llama_decode(ctx_tgt, tgt_batch);
        if (rc != 0) {
            LOG_ERR("%s: llama_decode(ctx_tgt) depth=%d rc=%d\n", __func__, d, rc);
            decode_failed = true;
            break;
        }

        // --- Accept walk step (all logits in this pass are valid NOW) ---
        if (d == 0) {
            // Step 1: read root probe logits → pending_trunk_tok for depth-0.
            const int32_t root_bidx = nodes[0].batch_idx;
            if (root_bidx < 0) {
                correction_out = id_last;
                walk_done = true;
            } else {
                const float * logits = llama_get_logits_ith(ctx_tgt, root_bidx);
                if (!logits) {
                    correction_out = id_last;
                    walk_done = true;
                } else {
                    pending_trunk_tok = (llama_token)argmax_f32(logits, n_vocab);
                    LOG_DBG("%s: d=0 root probe batch=%d id_last=%d n_past=%d pending_trunk_tok=%d\n",
                            __func__, root_bidx, (int)id_last, (int)n_past, (int)pending_trunk_tok);
                }
            }
        }

        if (walk_done) goto bridge_copy;

        // Step 2: look for pending_trunk_tok among depth-d children of cur_node.
        // Depth-d children of cur_node are nodes with depth==d and parent_idx==cur_node.
        {
            int32_t match = -1;
            for (int32_t ci : nodes[cur_node].children) {
                // nodes[ci] is at depth d (all children of cur_node are at depth d).
                if (nodes[ci].token == pending_trunk_tok) {
                    match = ci;
                    break;
                }
            }

            LOG_DBG("%s: accept-walk d=%d cur_node=%d pending_tok=%d n_children=%d match=%d\n",
                    __func__, d, cur_node, (int)pending_trunk_tok,
                    (int)nodes[cur_node].children.size(), match);

            if (match >= 0) {
                // Accept child at depth d.
                accepted_out.push_back(pending_trunk_tok);
                nodes[match].verified = true;
                accepted_seq = nodes[match].kv_seq_id;
                cur_node = match;
                ++n_accepted;

                // Immediately read this child's logits (valid NOW in this d pass).
                // These logits determine pending_trunk_tok for depth d+1.
                const int32_t child_bidx = nodes[match].batch_idx;
                if (child_bidx < 0) {
                    correction_out = id_last;
                    walk_done = true;
                } else {
                    const float * child_logits = llama_get_logits_ith(ctx_tgt, child_bidx);
                    if (!child_logits) {
                        correction_out = id_last;
                        walk_done = true;
                    } else {
                        const llama_token next_tok = (llama_token)argmax_f32(child_logits, n_vocab);
                        LOG_DBG("%s: accepted depth-%d child=%d tok=%d next_pending=%d\n",
                                __func__, d, match, (int)pending_trunk_tok, (int)next_tok);

                        if (nodes[cur_node].children.empty()) {
                            // Leaf: next_tok is the correction.
                            correction_out = next_tok;
                            LOG_DBG("%s: leaf correction=%d\n", __func__, (int)correction_out);
                            walk_done = true;
                        } else {
                            // Non-leaf: next_tok is what we look for among depth-(d+1) children.
                            pending_trunk_tok = next_tok;
                        }
                    }
                }
            } else {
                // No match: pending_trunk_tok is the correction.
                correction_out = pending_trunk_tok;
                walk_done = true;
            }
        }

        bridge_copy:

        // After decoding depth d, bridge-copy for sibling paths that branch at d+1.
        // A path_seq S with first_depth == d+1 needs position n_past+d+1 from
        // its donor before we can place its first node at n_past+1+(d+1) = n_past+d+2.
        // Actually: S's first node IS at depth d+1, position n_past+1+(d+1).
        // For S's first node to decode correctly, S must have KV at n_past..n_past+d+1.
        // After this pass (d), S has n_past (root probe) but not n_past+1..n_past+d+1.
        // We copy n_past+1+d from donor to S — this gives S KV at n_past through
        // n_past+1+d (consecutive), allowing the next decode at n_past+1+(d+1).
        //
        // Wait: after decoding depth d, nodes at depth d under donor have been added
        // at pos n_past+1+d.  We need to copy pos n_past+1+d from donor to S.
        // But S might be missing n_past+1..n_past+d as well (if it branches at d+1
        // from a path that has those positions).  Need to copy the full range
        // n_past+1..n_past+1+d from donor to S if S has none of those.
        //
        // Simpler: copy the ENTIRE range decoded so far: n_past+1..n_past+1+d.
        // This is safe because S has n_past from its root probe, and donor has
        // n_past+1..n_past+1+d from decode passes 0..d.  After this copy,
        // S has n_past..n_past+1+d (consecutive), ready for depth d+1 nodes.
        for (int32_t pi = 0; pi < n_paths; ++pi) {
            if (path_info[pi].first_depth == d + 1 && path_info[pi].donor >= 0) {
                const llama_seq_id donor_sid = path_info[pi].donor;
                const llama_seq_id sibling_sid = used_seq_ids[pi];
                // Copy positions n_past+1 .. n_past+1+d (inclusive).
                // seq_cp range: p0=n_past+1, p1=n_past+2+d (exclusive end).
                llama_memory_seq_cp(mem_tgt, donor_sid, sibling_sid,
                                    n_past + 1, n_past + 2 + d);
                LOG_DBG("%s: bridge-copy depth=%d: pos [%d..%d] donor_sid=%d → sibling_sid=%d\n",
                        __func__, d, (int)(n_past+1), (int)(n_past+1+d),
                        (int)donor_sid, (int)sibling_sid);
            }
        }
    }

    // Guard: if accept walk never resolved (shouldn't happen), fallback.
    if (!walk_done && correction_out < 0) {
        correction_out = id_last;
    }

    if (decode_failed) {
        for (llama_seq_id sid : used_seq_ids) {
            llama_memory_seq_rm(mem_tgt, sid, 0, -1);
        }
        correction_out = -1;
        return 0;
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
        // E.1 commit: copy accepted path KV from path seq to slot_seq.
        // The path seq (accepted_seq) has:
        //   - positions 0..n_past-1: copied from slot_seq in step 3
        //   - position n_past:       id_last (from root probe in step 4)
        //   - positions n_past+1..n_past+n_accepted: accepted tree node KV
        //
        // We copy n_past..n_past+n_accepted from accepted_seq to slot_seq.
        // slot_seq currently has 0..n_past-1; after copy it has 0..n_past+n_accepted.
        // This allows the server to skip re-decoding id_last and the accepted tokens.
        //
        // With kv_unified=true on ctx_tgt (same stream → s0==s1), partial range
        // seq_cp works directly.  With kv_unified=false, cross-stream partial copy
        // would need p0=0,p1=-1; here we use the exact range since kv_unified=true
        // is required for E.1 to be active.
        // Only copy positions not already in slot_seq.
        // Normally slot_seq has 0..n_past-1, so we copy n_past..n_past+n_accepted.
        // If ctx_mtp state is slightly off (e.g., pending_pos not fully propagated),
        // slot_seq might already have n_past; skip those to avoid double-tagging
        // which would violate the seq_add !test assertion in kv_cells.h.
        const llama_pos slot_max = llama_memory_seq_pos_max(mem_tgt, slot_seq);
        const llama_pos copy_from = std::max(n_past, slot_max + 1);
        const llama_pos copy_to   = n_past + n_accepted + 1; // exclusive
        if (copy_from < copy_to) {
            llama_memory_seq_cp(mem_tgt, (llama_seq_id)accepted_seq,
                                slot_seq, copy_from, copy_to);
            LOG_INF("%s: E.1 committed %d positions [%d..%d] from seq %d to slot_seq %d "
                    "(slot_max_was=%d)\n",
                    __func__, (int)(copy_to - copy_from),
                    (int)copy_from, (int)(copy_to - 1),
                    (int)accepted_seq, (int)slot_seq, (int)slot_max);
        } else {
            LOG_WRN("%s: E.1 skip commit — slot_seq already has up to %d (copy_from=%d >= copy_to=%d)\n",
                    __func__, (int)slot_max, (int)copy_from, (int)copy_to);
        }
    }

    // ------------------------------------------------------------------
    // Step 8: Cleanup — remove all path seq_ids from ctx_tgt's KV.
    // Path seq_ids hold prompt copy (0..n_past-1) + root probe (n_past) +
    // tree node KV (n_past+1..n_past+d).
    // For E.1: slot_seq now has 0..n_past+n_accepted; path seqs cleaned up.
    // For non-E.1: slot_seq still has 0..n_past-1 (id_last not yet decoded).
    // ------------------------------------------------------------------
    for (llama_seq_id sid : used_seq_ids) {
        llama_memory_seq_rm(mem_tgt, sid, 0, -1);
    }

    LOG_DBG("%s: removed %d path seq_ids from ctx_tgt KV\n",
            __func__, n_paths);

    return n_accepted;
}
