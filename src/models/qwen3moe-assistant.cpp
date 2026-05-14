#include "models.h"

#include <cmath>

#include "llama-model.h"
#include "llama-arch.h"

// Include the model declarations needed for the llama_model_mapping function
#include "llama-model.h"

// Qwen3MoE MTP assistant graph builder.
// The assistant model cannot be used as a primary model (-m) — build_arch_graph()
// is only valid when called from llama_context::decode_mtp() via the nested
// mtp_assistant pointer on the target llama_model.

// ---------------------------------------------------------------------------
// hparams
// ---------------------------------------------------------------------------

void llama_model_qwen3moe_assistant::load_arch_hparams(llama_model_loader & ml) {
    hparams.swa_type           = LLAMA_SWA_TYPE_NONE;
    hparams.f_attention_scale  = 1.0f;

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // n_embd_head_k / n_embd_head_v are derived from per-layer attn shape;
    // the base loader populates them. requires_target_arch is metadata that
    // the dispatcher checks separately — not needed in hparams.

    // Qwen3MoE assistant-specific: n_embd_backbone (target's n_embd).
    ml.get_key(LLM_KV_QWEN3MOE_ASSISTANT_N_EMBD_BACKBONE, hparams.n_embd_backbone, false);

    type = LLM_TYPE_UNKNOWN;
}

// ---------------------------------------------------------------------------
// tensors
// ---------------------------------------------------------------------------

void llama_model_qwen3moe_assistant::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const uint32_t n_bb = hparams.n_embd_backbone;
    if (n_bb == 0) {
        throw std::runtime_error("qwen3moe_assistant: n_embd_backbone must be set in GGUF metadata");
    }

    // Token embedding: tied lm_head, inner hidden size = hparams.n_embd (e.g. 1024)
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // MTP projection matrices
    mtp_pre_projection  = create_tensor(tn(LLM_TENSOR_MTP_PRE_PROJECTION,  "weight"), {2 * (int64_t) n_bb, n_embd}, 0);
    mtp_post_projection = create_tensor(tn(LLM_TENSOR_MTP_POST_PROJECTION, "weight"), {n_embd, (int64_t) n_bb}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    // RoPE freqs are optional for Qwen3 NEOX path — let llama.cpp compute them
    // from rope.freq_base if the GGUF doesn't ship a precomputed tensor.
    int rope_freqs_flag = TENSOR_NOT_REQUIRED;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];
        const int64_t n_head_i      = hparams.n_head(i);
        const int64_t n_embd_head_i = hparams.n_embd_head_k(i);

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_i * n_head_i}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_i * n_head_i, n_embd}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_head_i * n_head_kv}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_head_i * n_head_kv}, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_i}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_i}, 0);

        const int64_t n_ff_cur = hparams.n_ff(i);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff_cur}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff_cur}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff_cur, n_embd},   0);

        if (!hparams.is_swa(i)) {
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_embd_head_i / 2}, rope_freqs_flag);
            rope_freqs_flag  = TENSOR_DUPLICATED;
        }
    }
}

// ---------------------------------------------------------------------------
// graph — drafter cannot be used as primary model
// ---------------------------------------------------------------------------

std::unique_ptr<llm_graph_context> llama_model_qwen3moe_assistant::build_arch_graph(
        const llm_graph_params & /* params */) const {
    throw std::runtime_error(
        "qwen3moe_assistant cannot be used as the primary model (-m). "
        "Load the Qwen3MoE target with -m, then attach the assistant drafter "
        "via llama_model_load_mtp_from_file().");
}

// ---------------------------------------------------------------------------
// llm_build_qwen3moe_mtp — the actual draft graph builder
// ---------------------------------------------------------------------------

static llm_graph_params graph_params_for_mtp(llm_graph_params p, const llama_model & mtp_model) {
    p.arch    = mtp_model.arch;
    p.hparams = mtp_model.hparams;
    p.gtype   = LLM_GRAPH_TYPE_MTP;
    return p;
}

llm_build_qwen3moe_mtp::llm_build_qwen3moe_mtp(
        const llama_model & target_model,
        const llama_model & mtp_model,
        const llm_graph_params & params) :
        llm_graph_context(graph_params_for_mtp(params, mtp_model)),
        target(target_model),
        mtp(mtp_model) {
    const int64_t n_bb = mtp.hparams.n_embd_backbone;
    GGML_ASSERT(n_bb > 0);
    GGML_ASSERT(mtp.mtp_pre_projection != nullptr && mtp.mtp_post_projection != nullptr);

    ggml_tensor * inp_tok = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_input(inp_tok);
    cb(inp_tok, "mtp_inp_last_token", -1);

    ggml_tensor * inp_h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_bb, 1);
    ggml_set_input(inp_h);
    cb(inp_h, "mtp_inp_h_prev", -1);

    {
        auto inp_wrap = std::make_unique<llm_graph_input_mtp>();
        inp_wrap->inp_last_token = inp_tok;
        inp_wrap->inp_h_prev     = inp_h;
        res->add_input(std::move(inp_wrap));
    }

    ggml_tensor * tok_e = ggml_get_rows(ctx0, target.tok_embd, inp_tok);
    cb(tok_e, "mtp_tgt_tok_embd", -1);

    // Qwen3MoE scales token embeddings by sqrt(hidden_size).
    tok_e = ggml_scale(ctx0, tok_e, sqrtf((float) n_bb));
    cb(tok_e, "mtp_tgt_tok_embd_scaled", -1);

    ggml_tensor * inp_cat = ggml_concat(ctx0, tok_e, inp_h, 0);
    cb(inp_cat, "mtp_concat", -1);

    ggml_tensor * inpL = build_lora_mm(mtp.mtp_pre_projection, inp_cat);
    cb(inpL, "mtp_pre_proj_out", -1);

    ggml_build_forward_expand(gf, inpL);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * cur = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_v(il));

        const int64_t n_head = hparams.n_head(il);

        const float freq_base_l  = mtp.get_rope_freq_base(cparams, il);
        const float freq_scale_l = mtp.get_rope_freq_scale(cparams, il);
        const int   n_rot_l      = hparams.n_rot(il);

        cur = build_norm(inpL, mtp.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * freq_factors = nullptr;
        if (!hparams.is_swa(il)) {
            freq_factors = mtp.layers[il].rope_freqs;
        }

        ggml_tensor * Qcur = build_lora_mm(mtp.layers[il].wq, cur);
        cb(Qcur, "Qcur", il);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

        Qcur = build_norm(Qcur, mtp.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur_normed", il);

        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig,
                             freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur_pos", il);

        const int32_t il_kv = static_cast<int32_t>(target.hparams.n_layer - 1); // Use last target layer only

        const int64_t kv_embd_head_v = target.hparams.n_embd_head_v(il_kv);
        const int64_t kv_n_head_v    = target.hparams.n_head_kv(il_kv);

        const float kq_scale = 1.0f / sqrtf((float) n_embd_head);

        cur = build_attn_mtp_plain(inp_attn, mtp.layers[il].wo, nullptr, Qcur, nullptr, nullptr, nullptr,
                kq_scale, il, il_kv, kv_embd_head_v, kv_n_head_v, false);

        cur = build_norm(cur, mtp.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * attn_out = ggml_add(ctx0, cur, inpL);
        cb(attn_out, "attn_out", il);

        GGML_ASSERT(mtp.layers[il].ffn_gate_inp == nullptr && "qwen3moe_assistant MTP does not support MoE FFN");

        cur = build_norm(attn_out, mtp.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                mtp.layers[il].ffn_up,   nullptr, nullptr,
                mtp.layers[il].ffn_gate, nullptr, nullptr,
                mtp.layers[il].ffn_down, nullptr, nullptr,
                nullptr,
                LLM_FFN_GELU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, attn_out);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, mtp.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    ggml_tensor * h_inner = cur;

    ggml_tensor * backbone = build_lora_mm(mtp.mtp_post_projection, h_inner);
    cb(backbone, "mtp_post_proj_out", -1);
    res->t_embd = backbone;
    // Add backbone to forward graph so alloc_graph binds it to a buffer.
    // decode_mtp reads it after compute as next-step h_prev. Without this,
    // t_embd->buffer is null and the post-compute tensor_get hits
    // "tensor buffer not set" in ggml_backend.cpp:342.
    ggml_build_forward_expand(gf, backbone);

    cur = build_lora_mm(mtp.output, h_inner);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}