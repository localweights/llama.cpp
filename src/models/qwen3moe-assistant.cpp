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

    // MoE drafter hparams — absent in old dense GGUFs (default to 0 = dense mode).
    ml.get_key(LLM_KV_EXPERT_COUNT,               hparams.n_expert,      false);
    ml.get_key(LLM_KV_EXPERT_USED_COUNT,          hparams.n_expert_used, false);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp,      false);

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

    // MTP projection matrices — Eagle3 uses 3*n_bb cols, Eagle2 uses 2*n_bb
    mtp_pre_projection = create_tensor(tn(LLM_TENSOR_MTP_PRE_PROJECTION, "weight"), {3 * (int64_t) n_bb, n_embd}, TENSOR_NOT_REQUIRED);
    if (!mtp_pre_projection) {
        mtp_pre_projection = create_tensor(tn(LLM_TENSOR_MTP_PRE_PROJECTION, "weight"), {2 * (int64_t) n_bb, n_embd}, 0);
    }
    mtp_post_projection = create_tensor(tn(LLM_TENSOR_MTP_POST_PROJECTION, "weight"), {n_embd, (int64_t) n_bb}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

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

        if (hparams.n_expert > 0) {
            // MoE drafter path
            const int64_t n_exp     = (int64_t) hparams.n_expert;
            const int64_t n_ff_exp  = (int64_t) hparams.n_ff_exp;
            layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_exp},              TENSOR_NOT_REQUIRED);
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_exp},    TENSOR_NOT_REQUIRED);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp, n_exp},    TENSOR_NOT_REQUIRED);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_exp},    TENSOR_NOT_REQUIRED);
        } else {
            // Dense drafter path (original)
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff_cur}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff_cur}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff_cur, n_embd},   0);
        }

        // RoPE freqs optional (TENSOR_NOT_REQUIRED). Qwen3 NEOX RoPE builds
        // them on-the-fly from rope.freq_base when nullptr.
        layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i),
                                         {n_embd_head_i / 2},
                                         llama_model_loader::TENSOR_NOT_REQUIRED);
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

    // Eagle3 detection: pre_projection input cols = 3*n_bb (h_low||h_mid||h_high)
    const int64_t pre_proj_cols = mtp.mtp_pre_projection->ne[0];
    const bool is_eagle3 = (pre_proj_cols == 3 * n_bb);

    ggml_tensor * inp_tok = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_input(inp_tok);
    cb(inp_tok, "mtp_inp_last_token", -1);

    ggml_tensor * inp_h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, pre_proj_cols, 1);
    ggml_set_input(inp_h);
    cb(inp_h, "mtp_inp_h_prev", -1);

    {
        auto inp_wrap = std::make_unique<llm_graph_input_mtp>();
        inp_wrap->inp_last_token = inp_tok;
        inp_wrap->inp_h_prev     = inp_h;
        res->add_input(std::move(inp_wrap));
    }

    ggml_tensor * tok_e = ggml_get_rows(ctx0, mtp.tok_embd, inp_tok);
    cb(tok_e, "mtp_self_tok_embd", -1);

    ggml_tensor * inpL;
    if (is_eagle3) {
        // Eagle3: fused = h_proj(cat(h_low, h_mid, h_high)) + tok_embd(token)
        // inp_h is (3*n_bb, 1), pre_projection is (3*n_bb → n_embd_inner)
        ggml_tensor * h_fused = build_lora_mm(mtp.mtp_pre_projection, inp_h);
        cb(h_fused, "mtp_h_fused", -1);
        inpL = ggml_add(ctx0, h_fused, tok_e);
        cb(inpL, "mtp_pre_proj_out", -1);
    } else {
        // Eagle2: inpL = pre_proj(concat(tok_embd_padded, h_prev))
        const int64_t n_inner = (int64_t) hparams.n_embd;
        if (n_inner < (int64_t) n_bb) {
            ggml_tensor * pad = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, (int64_t) n_bb - n_inner, 1);
            pad = ggml_scale(ctx0, pad, 0.0f);
            tok_e = ggml_concat(ctx0, tok_e, pad, 0);
            cb(tok_e, "mtp_tok_embd_padded", -1);
        }
        ggml_tensor * inp_cat = ggml_concat(ctx0, tok_e, inp_h, 0);
        cb(inp_cat, "mtp_concat", -1);
        inpL = build_lora_mm(mtp.mtp_pre_projection, inp_cat);
        cb(inpL, "mtp_pre_proj_out", -1);
    }

    ggml_build_forward_expand(gf, inpL);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();
    // build_attn_inp_kv creates self_k_idxs / self_v_idxs / self_kq_mask
    // tensors that are populated in set_input but only referenced inside
    // build_attn (not build_attn_mtp_plain). Without an explicit graph-output
    // reference, sched_alloc_graph skips them and their buffers stay null,
    // causing set_input to abort at ggml-backend.cpp:194. Force them into
    // the forward graph so they get allocated.
    if (inp_attn->self_k_idxs)     ggml_build_forward_expand(gf, inp_attn->self_k_idxs);
    if (inp_attn->self_v_idxs)     ggml_build_forward_expand(gf, inp_attn->self_v_idxs);
    if (inp_attn->self_kq_mask)    ggml_build_forward_expand(gf, inp_attn->self_kq_mask);
    if (inp_attn->self_kq_mask_cnv) ggml_build_forward_expand(gf, inp_attn->self_kq_mask_cnv);

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

        ggml_tensor * freq_factors = mtp.layers[il].rope_freqs;  // nullptr if absent

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

        // No post-attention RMSNorm: trainer's EagleBlock (eagle_head.py) does
        // direct residual add. Previous version applied attn_q_norm (dim 128)
        // to cur (dim n_embd_inner) which is a shape mismatch and silently
        // distorts the activation.
        ggml_tensor * attn_out = ggml_add(ctx0, cur, inpL);
        cb(attn_out, "attn_out", il);

        cur = build_norm(attn_out, mtp.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if (mtp.layers[il].ffn_gate_inp != nullptr) {
            const int64_t n_exp      = (int64_t) mtp.hparams.n_expert;
            const int64_t n_exp_used = (int64_t) mtp.hparams.n_expert_used;
            cur = build_moe_ffn(cur,
                    mtp.layers[il].ffn_gate_inp,
                    mtp.layers[il].ffn_up_exps,
                    mtp.layers[il].ffn_gate_exps,
                    mtp.layers[il].ffn_down_exps,
                    nullptr,
                    n_exp, n_exp_used,
                    LLM_FFN_SILU, true,
                    0.0f,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il,
                    nullptr, nullptr,
                    nullptr, nullptr, nullptr);
            cb(cur, "ffn_moe_out", il);
        } else {
            cur = build_ffn(cur,
                    mtp.layers[il].ffn_up,   nullptr, nullptr,
                    mtp.layers[il].ffn_gate, nullptr, nullptr,
                    mtp.layers[il].ffn_down, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_GELU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        }

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
