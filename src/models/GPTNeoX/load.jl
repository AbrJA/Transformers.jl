using ..TransformerLayers
using ..TransformerLayers: CompositeEmbedding, SelfAttention, @fluxshow, @fluxlayershow
using ChainRulesCore
using Functors
using Static

using NeuralAttentionlib
using NeuralAttentionlib: WithScore
using ..HuggingFaceModels: _load_layernorm, _load_dense, weight_init, zero_init, getweight, joinname

include("./attention.jl")

struct ParallelPreNorm2TransformerBlock{A,AN,F,FN} <: TransformerLayers.AbstractTransformerBlock
    attention::A
    attention_norm::AN
    feedforward::F
    feedforward_norm::FN
end
Flux.@layer ParallelPreNorm2TransformerBlock

function (b::ParallelPreNorm2TransformerBlock)(nt::NamedTuple)
    nt_a = TransformerLayers.apply_on_namedtuple(b.attention_norm, nt)
    nt_f = TransformerLayers.apply_on_namedtuple(b.feedforward_norm, nt)
    a = TransformerLayers.apply_on_namedtuple(b.attention, nt_a)
    f = TransformerLayers.apply_on_namedtuple(b.feedforward, nt_f)
    hidden_state = a.hidden_state + f.hidden_state + nt.hidden_state
    return TransformerLayers.return_hidden_state(a, hidden_state)
end

function _materialize(x)
    y = similar(x)
    y .= x
    return y
end

function ChainRulesCore.rrule(::typeof(_materialize), x)
    pullback(Ȳ) = (NoTangent(), Ȳ)
    return _materialize(x), pullback
end

struct GPTNeoXSplit{D}
    head::Int
    layer::D
end
Flux.@layer GPTNeoXSplit trainable = (layer,)

function (s::GPTNeoXSplit)(x)
    bs = Base.tail(size(x))
    y = s.layer(x)
    y_head = reshape(y, (:, s.head, bs...))
    q_head, k_head, v_head = TransformerLayers.__nsplit(static(3), identity, y_head)
    q = reshape(_materialize(q_head), (:, bs...))
    k = reshape(_materialize(k_head), (:, bs...))
    v = reshape(_materialize(v_head), (:, bs...))
    return (q, k, v)
end

function Base.show(io::IO, layer::GPTNeoXSplit)
    print(io, "GPTNeoXSplit(")
    show(io, layer.head)
    print(io, ", ")
    show(io, layer.layer)
    print(io, ')')
end


struct HGFGPTNeoXModel{E,D} <: HGFPreTrained{:gpt_neox,:model}
    embed::E
    decoder::D
end
@functor HGFGPTNeoXModel
@fluxshow HGFGPTNeoXModel

(model::HGFGPTNeoXModel)(nt::NamedTuple) = model.decoder(model.embed(nt))

struct HGFGPTNeoXForCausalLM{M,C} <: HGFPreTrained{:gpt_neox,:forcausallm}
    model::M
    cls::C
end
@fluxlayershow HGFGPTNeoXForCausalLM

(model::HGFGPTNeoXForCausalLM)(nt::NamedTuple) = model.cls(model.model(nt))

const HGFGPTNeoXPreTrainedModel = HGFPreTrained{:gpt_neox}

basemodelkey(::Type{<:HGFPreTrained{:gpt_neox}}) = :gpt_neox

function load_model(_type::Type{HGFGPTNeoXModel}, cfg, state_dict, prefix)
    embed = load_model(_type, CompositeEmbedding, cfg, state_dict, prefix)
    decoder = load_model(_type, TransformerBlock, cfg, state_dict, prefix)
    return HGFGPTNeoXModel(embed, decoder)
end

function load_model(_type::Type{HGFGPTNeoXForCausalLM}, cfg, state_dict, prefix)
    model = load_model(HGFGPTNeoXModel, cfg, state_dict, joinname(prefix, "gpt_neox"))
    if cfg[:tie_word_embeddings]
        embedding = model.embed.token.embeddings
    else
        vocab_size, dims, factor = cfg[:vocab_size], cfg[:hidden_size], Float32(cfg[:initializer_range])
        embedding = getweight(weight_init(vocab_size, dims, factor), TransformerLayers.Embed,
            state_dict, joinname(prefix, "embed_out.weight"))
    end
    lmhead = TransformerLayers.EmbedDecoder(TransformerLayers.Embed(embedding))
    return HGFGPTNeoXForCausalLM(model, TransformerLayers.Branch{(:logit,),(:hidden_state,)}(lmhead))
end

function load_model(_type::Type{<:HGFGPTNeoXPreTrainedModel}, ::Type{<:CompositeEmbedding}, cfg, state_dict, prefix)
    vocab_size, dims = cfg[:vocab_size], cfg[:hidden_size]
    factor = Float32(cfg[:initializer_range])
    token_weight = getweight(weight_init(vocab_size, dims, factor), TransformerLayers.Embed,
        state_dict, joinname(prefix, "embed_in.weight"))
    embed = CompositeEmbedding(token=TransformerLayers.Embed(token_weight))
    return embed
end

function load_model(_type::Type{<:HGFGPTNeoXPreTrainedModel}, ::Type{<:TransformerLayers.LayerNorm}, cfg, state_dict, prefix)
    dims = cfg[:hidden_size]
    ln_ϵ = Float32(cfg[:layer_norm_eps])
    old_weight_name = joinname(prefix, "gamma")
    old_bias_name = joinname(prefix, "beta")
    weight_name = haskey(state_dict, old_weight_name) ? old_weight_name : joinname(prefix, "weight")
    bias_name = haskey(state_dict, old_bias_name) ? old_bias_name : joinname(prefix, "bias")
    ln_weight = getweight(one_init(dims), Array, state_dict, weight_name)
    ln_bias = getweight(zero_init(dims), Array, state_dict, bias_name)
    return TransformerLayers.LayerNorm(ln_weight, ln_bias, ln_ϵ)
end

function load_model(_type::Type{<:HGFGPTNeoXPreTrainedModel}, ::Type{<:SelfAttention}, cfg, state_dict, prefix)
    head, dims = cfg[:num_attention_heads], cfg[:hidden_size]
    @assert dims % head == 0 "The hidden size is not a multiple of the number of attention heads."
    head_size = div(dims, head)
    return_score = cfg[:output_attentions]
    factor = Float32(cfg[:initializer_range])
    rotary_dim = floor(Int, cfg[:rotary_pct] * head_size)
    rotary_pe_base = Float64(cfg[:rotary_emb_base])
    qkv_weight = getweight(weight_init(dims, 3dims, factor), Array,
        state_dict, joinname(prefix, "query_key_value.weight"))
    qkv_bias = getweight(zero_init(3dims), Array, state_dict, joinname(prefix, "query_key_value.bias"))
    qkv_proj = GPTNeoXSplit(head, TransformerLayers.Dense(qkv_weight, qkv_bias))
    o_weight = getweight(weight_init(dims, dims, factor), Array, state_dict, joinname(prefix, "dense.weight"))
    o_bias = getweight(zero_init(dims), Array, state_dict, joinname(prefix, "dense.bias"))
    o_proj = TransformerLayers.Dense(o_weight, o_bias)
    op = CausalGPTNeoXRoPEMultiheadQKVAttenOp(rotary_pe_base, rotary_dim, head)
    return_score && (op = WithScore(op))
    return SelfAttention(op, qkv_proj, o_proj)
end

function load_model(
    _type::Type{<:HGFGPTNeoXPreTrainedModel}, ::Type{<:TransformerLayers.Chain{<:Tuple{TransformerLayers.Dense,TransformerLayers.Dense}}},
    cfg, state_dict, prefix
)
    dims, ff_dims = cfg[:hidden_size], cfg[:intermediate_size]
    factor = Float32(cfg[:initializer_range])
    act = ACT2FN[Symbol(cfg[:hidden_act])]
    wi_weight = getweight(weight_init(dims, ff_dims, factor), Array,
        state_dict, joinname(prefix, "dense_h_to_4h.weight"))
    wi_bias = getweight(zero_init(ff_dims), Array, state_dict, joinname(prefix, "dense_h_to_4h.bias"))
    wo_weight = getweight(weight_init(ff_dims, dims, factor), Array,
        state_dict, joinname(prefix, "dense_4h_to_h.weight"))
    wo_bias = getweight(zero_init(dims), Array, state_dict, joinname(prefix, "dense_4h_to_h.bias"))
    return TransformerLayers.Chain(TransformerLayers.Dense(act, wi_weight, wi_bias), TransformerLayers.Dense(wo_weight, wo_bias))
end

function load_model(_type::Type{<:HGFGPTNeoXPreTrainedModel}, ::Type{<:TransformerBlock}, cfg, state_dict, prefix)
    n = cfg[:num_hidden_layers]
    parallel_residual = cfg[:use_parallel_residual]
    collect_output = cfg[:output_attentions] || cfg[:output_hidden_states]
    blocks = []
    for i = 1:n
        lprefix = joinname(prefix, :layers, i - 1)
        sa = load_model(_type, SelfAttention, cfg, state_dict, joinname(lprefix, "attention"))
        sa_ln = load_model(_type, TransformerLayers.LayerNorm, cfg, state_dict, joinname(lprefix, "input_layernorm"))
        ff = load_model(_type, TransformerLayers.Chain{Tuple{TransformerLayers.Dense,TransformerLayers.Dense}}, cfg,
            state_dict, joinname(lprefix, "mlp"))
        ff_ln = load_model(_type, TransformerLayers.LayerNorm, cfg, state_dict, joinname(lprefix, "post_attention_layernorm"))
        block = parallel_residual ?
                ParallelPreNorm2TransformerBlock(sa, sa_ln, ff, ff_ln) :
                TransformerBlock(TransformerLayers.PreNormResidual(sa, sa_ln), TransformerLayers.PreNormResidual(ff, ff_ln))
        push!(blocks, block)
    end
    collect_f = collect_output ? TransformerLayers.collect_outputs : nothing
    trf = Transformer(Tuple(blocks), collect_f)
    final_ln = load_model(_type, TransformerLayers.LayerNorm, cfg, state_dict, joinname(prefix, "final_layer_norm"))
    return TransformerLayers.Chain(trf, final_ln)
end

function get_state_dict(m::HGFGPTNeoXModel, state_dict, prefix)
    get_state_dict(HGFGPTNeoXModel, m.embed, state_dict, prefix)
    get_state_dict(HGFGPTNeoXModel, m.decoder[1], state_dict, prefix)
    get_state_dict(HGFGPTNeoXModel, m.decoder[2], state_dict, joinname(prefix, "final_layer_norm"))
    return state_dict
end

function get_state_dict(m::HGFGPTNeoXForCausalLM, state_dict, prefix)
    get_state_dict(m.model, state_dict, joinname(prefix, "gpt_neox"))
    get_state_dict(HGFGPTNeoXModel, m.cls.layer.embed, state_dict, joinname(prefix, "embed_out"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFGPTNeoXPreTrainedModel}, m::CompositeEmbedding, state_dict, prefix)
    get_state_dict(p, m.token, state_dict, joinname(prefix, "embed_in"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFGPTNeoXPreTrainedModel}, m::SelfAttention, state_dict, prefix)
    get_state_dict(p, m.qkv_proj.layer, state_dict, joinname(prefix, "query_key_value"))
    get_state_dict(p, m.o_proj, state_dict, joinname(prefix, "dense"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFGPTNeoXPreTrainedModel}, m::TransformerLayers.Chain{<:Tuple{TransformerLayers.Dense,TransformerLayers.Dense}},
    state_dict, prefix)
    get_state_dict(p, m[1], state_dict, joinname(prefix, "dense_h_to_4h"))
    get_state_dict(p, m[2], state_dict, joinname(prefix, "dense_4h_to_h"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFGPTNeoXPreTrainedModel}, m::ParallelPreNorm2TransformerBlock, state_dict, prefix)
    get_state_dict(p, m.attention, state_dict, joinname(prefix, "attention"))
    get_state_dict(p, m.attention_norm, state_dict, joinname(prefix, "input_layernorm"))
    get_state_dict(p, m.feedforward, state_dict, joinname(prefix, "mlp"))
    get_state_dict(p, m.feedforward_norm, state_dict, joinname(prefix, "post_attention_layernorm"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFGPTNeoXPreTrainedModel}, m::Transformer, state_dict, prefix)
    for (i, t) in enumerate(m.blocks)
        get_state_dict(p, t, state_dict, joinname(prefix, :layers, i - 1))
    end
    return state_dict
end
