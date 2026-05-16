#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"
#include "llama-mtp.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <fstream>
#include <map>
#include <vector>

struct llama_model;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    ggml_tensor * get_t_h_pre_norm() const;
    ggml_tensor * get_t_mtp_out()    const;

    // Translate batch token index to output row index in t_h_pre_norm/logits.
    // Returns -1 if the token was not configured to output.
    int32_t get_output_row(int32_t batch_token_idx) const {
        try {
            return (int32_t) output_resolve_row(batch_token_idx);
        } catch (const std::exception & e) {
            LLAMA_LOG_ERROR("get_output_row(%d): %s (n_outputs=%d, output_ids.size=%zu)\n",
                    batch_token_idx, e.what(), n_outputs, output_ids.size());
            return -1;
        } catch (...) {
            LLAMA_LOG_ERROR("get_output_row(%d): unknown exception (n_outputs=%d, output_ids.size=%zu)\n",
                    batch_token_idx, n_outputs, output_ids.size());
            return -1;
        }
    }

    void set_mtp(llama_seq_id seq_id, llama_context * ctx_mtp_in);
    // E.1: advance the hook's pending_pos to new_pending_pos with given h-state.
    void mtp_set_pending(llama_seq_id seq_id, llama_pos new_pending_pos,
                         const float * h_vec, int32_t n_embd_in);
    // Returns the ctx_mtp registered for seq_id, or nullptr if none.
    llama_context * get_mtp(llama_seq_id seq_id) const {
        auto it = mtp_map.find(seq_id);
        return (it != mtp_map.end()) ? it->second.ctx_mtp : nullptr;
    }

    // Phase E.1 slot-reuse fix: clear prefill-hook pending state for seq_id
    // when its KV is trimmed at trim_from. If pending_pos >= trim_from the
    // stashed h-state is stale and must not be used to chain into the next
    // ubatch (would create a ctx_mtp KV gap → llama_decode rc=-1).
    void mtp_invalidate_pending(llama_seq_id seq_id, llama_pos trim_from) {
        auto it = mtp_map.find(seq_id);
        if (it == mtp_map.end()) return;
        auto & slot = it->second;
        if (slot.pending_pos < 0 || slot.pending_pos >= trim_from) {
            slot.pending_pos = -1;
            if (!slot.pending_h.empty()) {
                std::fill(slot.pending_h.begin(), slot.pending_h.end(), 0.0f);
            }
        }
    }

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret,
                              bool   apply_mctx = true);

    llm_graph_params graph_params_mtp(
            llm_graph_result * res,
            const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx) const;

    // Gemma 4 MTP: greedy multi-step draft using nested gemma4_assistant + target KV.
    int32_t decode_mtp(
            llama_seq_id seq_id,
            llama_pos attn_pos,
            llama_token last_token,
            float * h_prev,
            int32_t n_steps,
            llama_token * out_drafts,
            float * out_logits,
            float * out_h_prev_last);

    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);

    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

    // tap-layer hidden-state dump — public so the C-API wrapper can call it
    void init_tap_layers(const char * dir, const std::vector<int> & layers, int n_embd,
                         int n_seq_max, bool merge_on_close);
    void set_cb_eval(ggml_backend_sched_eval_callback cb, void * ud);

private:
    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    llm_graph_cb graph_get_cb() const;

    void handle_mtp_for_ubatch(
            int32_t                n_tokens,
            const llama_token    * tokens,
            const llama_pos      * positions,
            const llama_seq_id   * seq_ids,    // seq_id for each token in the ubatch
            struct ggml_tensor   * t_h_pre_norm);

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    llama_mtp_map mtp_map; // seq_id → per-slot MTP hook state (empty == MTP disabled)

    std::unique_ptr<llama_memory_i> memory;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    bool sched_need_reserve = true;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    llm_graph_result_ptr gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused

    // tap-layer hidden-state dump
    // Populated by init_tap_layers(); zero overhead when tap_out_dir == nullptr.
    const char *             tap_out_dir        = nullptr;
    std::vector<int>         tap_layers;
    // tap_files[layer_idx][seq_idx] — one file per (layer, seq) pair.
    // When n_seq_max == 1 the filename is h_l<L>.bin (backward compat).
    // When n_seq_max > 1  the filenames are h_l<L>.s0.bin .. h_l<L>.s<N-1>.bin.
    std::vector<std::vector<std::ofstream>> tap_files;
    int                      tap_n_embd         = 0;
    int                      tap_n_seq_max      = 0; // set by init_tap_layers
    bool                     tap_merge_on_close = false; // concat s*.bin → h_l<L>.bin on destroy

    // Per-layer staging buffers: filled eagerly during graph execution via eval callback
    // before ggml's memory allocator can reuse the tensor buffer.
    // tap_staged[layer_idx] holds f16 data for the most recent ubatch.
    struct tap_staged_t {
        std::vector<ggml_fp16_t> data;
        uint32_t                 n_tokens = 0;
    };
    std::vector<tap_staged_t> tap_staged;

    // User-data bundle passed to tap eval callback; kept as a member so its
    // lifetime covers the graph_compute() call that uses it.
    struct tap_eval_ud_t {
        llama_context *                  lctx    = nullptr;
        ggml_backend_sched_eval_callback user_cb = nullptr;
        void *                           user_ud = nullptr;
    };
    tap_eval_ud_t tap_eval_ud;

    // Called from process_ubatch after graph_compute returns GGML_STATUS_SUCCESS.
    void write_tap_layers_post_compute(ggml_cgraph * gf, const llama_ubatch & ubatch);
};
