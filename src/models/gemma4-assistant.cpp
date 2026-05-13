#include "models.h"

// Gemma 4 MTP assistant — standalone arch port.
//
// This file registers the gemma4_assistant architecture so the GGUF drafter
// can be loaded without "unknown architecture" errors.  The full speculative-
// decode graph builder (llm_build_gemma4_mtp) requires LLM_GRAPH_TYPE_MTP and
// related infrastructure that is deferred to a follow-up.  For now
// build_arch_graph() throws, which is consistent with the upstream fork's
// behaviour: the drafter is never used as the primary model via -m.

// ---------------------------------------------------------------------------
// hparams
// ---------------------------------------------------------------------------

void llama_model_gemma4_assistant::load_arch_hparams(llama_model_loader & ml) {
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer);

    uint32_t n_kv_shared_layers = 0;
    ml.get_key(LLM_KV_ATTENTION_SHARED_KV_LAYERS, n_kv_shared_layers, false);

    hparams.n_layer_kv_from_start = hparams.n_layer - (int32_t) n_kv_shared_layers;
    hparams.f_attention_scale     = 1.0f;

    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,          hparams.rope_freq_base_train_swa, false);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);

    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA,   hparams.n_embd_head_k_swa);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA, hparams.n_embd_head_v_swa);

    // Gemma4 assistant-specific keys
    ml.get_key(LLM_KV_GEMMA4_ASSISTANT_N_EMBD_BACKBONE,        hparams.n_embd_backbone,        false);
    ml.get_key(LLM_KV_GEMMA4_ASSISTANT_N_CENTROIDS,            hparams.n_centroids,            false);
    ml.get_key(LLM_KV_GEMMA4_ASSISTANT_CENTROID_TOP_K,         hparams.centroid_top_k,         false);
    ml.get_key(LLM_KV_GEMMA4_ASSISTANT_ATTENTION_K_EQ_V,       hparams.attention_k_eq_v,       false);
    ml.get_key(LLM_KV_GEMMA4_ASSISTANT_USE_ORDERED_EMBEDDINGS, hparams.use_ordered_embeddings, false);

    type = LLM_TYPE_UNKNOWN;
}

// ---------------------------------------------------------------------------
// tensors
// ---------------------------------------------------------------------------

void llama_model_gemma4_assistant::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const uint32_t n_bb = hparams.n_embd_backbone;
    if (n_bb == 0) {
        throw std::runtime_error("gemma4_assistant: n_embd_backbone must be set in GGUF metadata");
    }
    if (n_embd_head_k != n_embd_head_v) {
        throw std::runtime_error("gemma4_assistant: n_embd_head_k must equal n_embd_head_v");
    }
    if (hparams.n_embd_head_k_swa != hparams.n_embd_head_v_swa) {
        throw std::runtime_error("gemma4_assistant: n_embd_head_k_swa must equal n_embd_head_v_swa");
    }

    // Token embedding: tied lm_head, inner hidden size = hparams.n_embd (e.g. 1024)
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // MTP projection matrices
    mtp_pre_projection  = create_tensor(tn(LLM_TENSOR_MTP_PRE_PROJECTION,  "weight"), {2 * (int64_t) n_bb, n_embd}, 0);
    mtp_post_projection = create_tensor(tn(LLM_TENSOR_MTP_POST_PROJECTION, "weight"), {n_embd, (int64_t) n_bb}, 0);

    if (hparams.use_ordered_embeddings) {
        const uint32_t n_c = hparams.n_centroids;
        if (n_c == 0) {
            throw std::runtime_error("gemma4_assistant: use_ordered_embeddings requires n_centroids > 0");
        }
        mtp_centroids      = create_tensor(tn(LLM_TENSOR_MTP_CENTROIDS,      "weight"), {(int64_t) n_c, n_embd}, 0);
        mtp_token_ordering = create_tensor(tn(LLM_TENSOR_MTP_TOKEN_ORDERING, "weight"), {n_vocab}, TENSOR_NOT_REQUIRED);
    }

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    int rope_freqs_flag = 0;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];
        const int64_t n_head_i      = hparams.n_head(i);
        const int64_t n_embd_head_i = hparams.n_embd_head_k(i);

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_i * n_head_i}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_i * n_head_i, n_embd}, 0);

        layer.attn_q_norm    = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM,    "weight", i), {n_embd_head_i}, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        layer.out_scale = create_tensor(tn(LLM_TENSOR_LAYER_OUT_SCALE, "weight", i), {1}, TENSOR_NOT_REQUIRED);

        if (!hparams.is_swa(i)) {
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_embd_head_i / 2}, rope_freqs_flag);
            rope_freqs_flag  = TENSOR_DUPLICATED;
        }

        const int64_t n_ff_cur = hparams.n_ff(i);

        layer.ffn_norm      = create_tensor(tn(LLM_TENSOR_FFN_NORM,      "weight", i), {n_embd},              0);
        layer.ffn_gate      = create_tensor(tn(LLM_TENSOR_FFN_GATE,      "weight", i), {n_embd,   n_ff_cur},  0);
        layer.ffn_up        = create_tensor(tn(LLM_TENSOR_FFN_UP,        "weight", i), {n_embd,   n_ff_cur},  0);
        layer.ffn_down      = create_tensor(tn(LLM_TENSOR_FFN_DOWN,      "weight", i), {n_ff_cur, n_embd},    0);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd},              0);
    }
}

// ---------------------------------------------------------------------------
// graph — drafter cannot be used as primary model
// ---------------------------------------------------------------------------

std::unique_ptr<llm_graph_context> llama_model_gemma4_assistant::build_arch_graph(
        const llm_graph_params & /* params */) const {
    throw std::runtime_error(
        "gemma4_assistant cannot be used as the primary model (-m). "
        "Load the Gemma 4 target with -m, then attach the assistant drafter "
        "via llama_model_load_mtp_from_file() (spec-decode loop not yet wired).");
}
