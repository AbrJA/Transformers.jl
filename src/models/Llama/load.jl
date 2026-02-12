using ..TransformerLayers
using ..TransformerLayers: CompositeEmbedding, SelfAttention, MultiheadQKVAttenOp, CausalRoPEMultiheadQKVAttenOp, @fluxshow, @fluxlayershow
using ChainRulesCore
using Functors
using Static

using NeuralAttentionlib
using NeuralAttentionlib: WithScore

include("attention.jl")

struct LLamaGated{G,D}
    gate::G
    dense::D
end
Flux.@layer LLamaGated


(m::LLamaGated)(x) = m.gate(x) .* m.dense(x)

struct HGFLlamaModel <: HGFPreTrained{:llama,:model}
    embed
    decoder
end
@fluxshow HGFLlamaModel

(model::HGFLlamaModel)(nt::NamedTuple) = model.decoder(model.embed(nt))

struct HGFLlamaForCausalLM <: HGFPreTrained{:llama,:forcausallm}
    model::HGFLlamaModel
    cls
end
@fluxlayershow HGFLlamaForCausalLM

(model::HGFLlamaForCausalLM)(nt::NamedTuple) = model.cls(model.model(nt))

const HGFLlamaPreTrainedModel = HGFPreTrained{:llama}

basemodelkey(::Type{<:HGFPreTrained{:llama}}) = :model

function load_model(_type::Type{HGFLlamaModel}, cfg, state_dict, prefix)
    embed = load_model(_type, CompositeEmbedding, cfg, state_dict, prefix)
    decoder = load_model(_type, TransformerBlock, cfg, state_dict, prefix)
    return HGFLlamaModel(embed, decoder)
end

function load_model(_type::Type{HGFLlamaForCausalLM}, cfg, state_dict, prefix)
    model = load_model(HGFLlamaModel, cfg, state_dict, joinname(prefix, "model"))
    if cfg[:tie_word_embeddings]
        embedding = model.embed.token.embeddings
    else
        vocab_size, dims, factor = cfg[:vocab_size], cfg[:hidden_size], Float32(cfg[:initializer_range])
        embedding = getweight(weight_init(vocab_size, dims, factor), TransformerLayers.Embed,
            state_dict, joinname(prefix, "lm_head.weight"))
    end
    lmhead = TransformerLayers.EmbedDecoder(TransformerLayers.Embed(embedding))
    return HGFLlamaForCausalLM(model, TransformerLayers.Branch{(:logit,),(:hidden_state,)}(lmhead))
end

function load_model(_type::Type{<:HGFLlamaPreTrainedModel}, ::Type{<:CompositeEmbedding}, cfg, state_dict, prefix)
    vocab_size, dims, pad_id = cfg[:vocab_size], cfg[:hidden_size], cfg[:pad_token_id]
    factor = Float32(cfg[:initializer_range])
    token_weight = getweight(TransformerLayers.Embed, state_dict, joinname(prefix, "embed_tokens.weight")) do
        weight = weight_init(vocab_size, dims, factor)()
        if !isnothing(pad_id)
            weight[:, pad_id+1] .= 0
        end
        return weight
    end
    embed = CompositeEmbedding(token=TransformerLayers.Embed(token_weight))
    return embed
end

function load_model(::Type{<:HGFLlamaPreTrainedModel}, ::Type{<:TransformerLayers.RMSLayerNorm}, cfg, state_dict, prefix)
    dims = cfg[:hidden_size]
    ln_ϵ = Float32(cfg[:rms_norm_eps])
    ln_init = one_init(dims)
    ln_weight = getweight(ln_init, Array, state_dict, joinname(prefix, "weight"))
    return TransformerLayers.RMSLayerNorm(ln_weight, ln_ϵ)
end

function load_model(_type::Type{<:HGFLlamaPreTrainedModel}, ::Type{<:SelfAttention}, cfg, state_dict, prefix)
    head, dims = cfg[:num_attention_heads], cfg[:hidden_size]
    @assert dims % head == 0 "The hidden size is not a multiple of the number of attention heads."
    head_dims = div(dims, head)
    kv_head = something(cfg[:num_key_value_heads], head)
    grouped_attn = kv_head != head
    @assert head % kv_head == 0 "The number of query is not dividable by the number of key/value groups"
    return_score = cfg[:output_attentions]
    factor = Float32(cfg[:initializer_range])
    rotary_pe_base = Float64(cfg[:rope_theta])
    @assert isnothing(cfg[:rope_scaling]) "Scaling Rotary Embedding is not support yet"
    q_weight = getweight(weight_init(dims, dims, factor), Array,
        state_dict, joinname(prefix, "q_proj.weight"))
    k_weight = getweight(weight_init(dims, kv_head * head_dims, factor), Array,
        state_dict, joinname(prefix, "k_proj.weight"))
    v_weight = getweight(weight_init(dims, kv_head * head_dims, factor), Array,
        state_dict, joinname(prefix, "v_proj.weight"))
    o_weight = getweight(weight_init(dims, dims, factor), Array, state_dict, joinname(prefix, "o_proj.weight"))
    qkv_proj = TransformerLayers.Fork(TransformerLayers.Dense(q_weight), TransformerLayers.Dense(k_weight), TransformerLayers.Dense(v_weight))
    o_proj = TransformerLayers.Dense(o_weight)
    if grouped_attn
        op = CausalLlamaRoPEGroupedQueryAttenOp(rotary_pe_base, head, kv_head)
    else
        op = CausalGPTNeoXRoPEMultiheadQKVAttenOp(rotary_pe_base, head_dims, head)
    end
    return_score && (op = WithScore(op))
    return SelfAttention(op, qkv_proj, o_proj)
end

function load_model(
    _type::Type{<:HGFLlamaPreTrainedModel}, ::Type{<:TransformerLayers.Chain{<:Tuple{TransformerLayers.Dense,TransformerLayers.Dense}}},
    cfg, state_dict, prefix
)
    dims = cfg[:hidden_size]
    ff_dims = cfg[:intermediate_size]
    factor = Float32(cfg[:initializer_range])
    act = ACT2FN[Symbol(cfg[:hidden_act])]
    gate_weight = getweight(weight_init(dims, ff_dims, factor), Array,
        state_dict, joinname(prefix, "gate_proj.weight"))
    wi_weight = getweight(weight_init(dims, ff_dims, factor), Array,
        state_dict, joinname(prefix, "up_proj.weight"))
    wo_weight = getweight(weight_init(ff_dims, dims, factor), Array,
        state_dict, joinname(prefix, "down_proj.weight"))
    return TransformerLayers.Chain(LLamaGated(TransformerLayers.Dense(act, gate_weight),
            TransformerLayers.Dense(wi_weight)),
        TransformerLayers.Dense(wo_weight))
end

function load_model(_type::Type{<:HGFLlamaPreTrainedModel}, ::Type{<:TransformerBlock}, cfg, state_dict, prefix)
    n = cfg[:num_hidden_layers]
    collect_output = cfg[:output_attentions] || cfg[:output_hidden_states]
    blocks = []
    for i = 1:n
        lprefix = joinname(prefix, :layers, i - 1)
        sa = load_model(_type, SelfAttention, cfg, state_dict, joinname(lprefix, "self_attn"))
        sa_ln = load_model(_type, TransformerLayers.RMSLayerNorm, cfg, state_dict, joinname(lprefix, "input_layernorm"))
        sa = TransformerLayers.PreNormResidual(sa, sa_ln)
        ff = load_model(_type, TransformerLayers.Chain{Tuple{TransformerLayers.Dense,TransformerLayers.Dense}}, cfg, state_dict, joinname(lprefix, "mlp"))
        ff_ln = load_model(_type, TransformerLayers.RMSLayerNorm, cfg, state_dict, joinname(lprefix, "post_attention_layernorm"))
        ff = TransformerLayers.PreNormResidual(ff, ff_ln)
        block = TransformerBlock(sa, ff)
        push!(blocks, block)
    end
    collect_f = collect_output ? TransformerLayers.collect_outputs : nothing
    trf = Transformer(Tuple(blocks), collect_f)
    final_ln = load_model(_type, TransformerLayers.RMSLayerNorm, cfg, state_dict, joinname(prefix, "norm"))
    return TransformerLayers.Chain(trf, final_ln)
end


function get_state_dict(m::HGFLlamaModel, state_dict, prefix)
    get_state_dict(HGFLlamaModel, m.embed, state_dict, prefix)
    get_state_dict(HGFLlamaModel, m.decoder[1], state_dict, prefix)
    get_state_dict(HGFLlamaModel, m.decoder[2], state_dict, joinname(prefix, "norm"))
    return state_dict
end

function get_state_dict(m::HGFLlamaForCausalLM, state_dict, prefix)
    get_state_dict(m.model, state_dict, joinname(prefix, "model"))
    get_state_dict(HGFLlamaModel, m.cls.layer.embed, state_dict, joinname(prefix, "lm_head"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFLlamaPreTrainedModel}, m::CompositeEmbedding, state_dict, prefix)
    get_state_dict(p, m.token, state_dict, joinname(prefix, "embed_tokens"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFLlamaPreTrainedModel}, m::SelfAttention, state_dict, prefix)
    get_state_dict(p, m.qkv_proj.layers[1], state_dict, joinname(prefix, "q_proj"))
    get_state_dict(p, m.qkv_proj.layers[2], state_dict, joinname(prefix, "k_proj"))
    get_state_dict(p, m.qkv_proj.layers[3], state_dict, joinname(prefix, "v_proj"))
    get_state_dict(p, m.o_proj, state_dict, joinname(prefix, "o_proj"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFLlamaPreTrainedModel}, m::TransformerLayers.Chain{<:Tuple{Any,TransformerLayers.Dense}},
    state_dict, prefix)
    get_state_dict(p, m[1].gate, state_dict, joinname(prefix, "gate_proj"))
    get_state_dict(p, m[1].dense, state_dict, joinname(prefix, "up_proj"))
    get_state_dict(p, m[2], state_dict, joinname(prefix, "down_proj"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFLlamaPreTrainedModel}, m::PreNormTransformerBlock, state_dict, prefix)
    get_state_dict(p, m.attention.layer, state_dict, joinname(prefix, "self_attn"))
    get_state_dict(p, m.attention.norm, state_dict, joinname(prefix, "input_layernorm"))
    get_state_dict(p, m.feedforward.layer, state_dict, joinname(prefix, "mlp"))
    get_state_dict(p, m.feedforward.norm, state_dict, joinname(prefix, "post_attention_layernorm"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFLlamaPreTrainedModel}, m::Transformer, state_dict, prefix)
    for (i, t) in enumerate(m.blocks)
        get_state_dict(p, t, state_dict, joinname(prefix, :layers, i - 1))
    end
    return state_dict
end
