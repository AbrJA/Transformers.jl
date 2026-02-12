using ..TransformerLayers
using ..TransformerLayers: CompositeEmbedding, @fluxshow, @fluxlayershow
using Functors
using Flux


struct T5Gated{G,L}
    gate::G
    linear::L
end
Flux.@layer T5Gated


(g::T5Gated)(x) = g.gate(x) .* g.linear(x)

# T5 Model
struct HGFT5Model <: HGFPreTrained{:t5,:model}
    embed
    seq2seq
end
@fluxshow HGFT5Model

function (model::HGFT5Model)(nt::NamedTuple)
    embs = model.embed(nt)
    outputs = model.seq2seq(embs)
    encoder_output = Base.structdiff(outputs.encoder_output, NamedTuple{(:position_bias,)})
    decoder_output = Base.structdiff(outputs.decoder_output, NamedTuple{(:position_bias,)})
    return merge(outputs, (; encoder_output, decoder_output))
end

# T5 For Conditional Generation
struct HGFT5ForConditionalGeneration <: HGFPreTrained{:t5,:forconditionalgeneration}
    model::HGFT5Model
    cls
end
@fluxlayershow HGFT5ForConditionalGeneration

(model::HGFT5ForConditionalGeneration)(nt::NamedTuple) = model.cls(model.model(nt))

# T5 Encoder Model
struct HGFT5EncoderModel <: HGFPreTrained{:t5,:encodermodel}
    embed
    encoder
end
@fluxshow HGFT5EncoderModel

function (model::HGFT5EncoderModel)(nt::NamedTuple)
    outputs = model.encoder(model.embed(nt))
    return Base.structdiff(outputs, NamedTuple{(:position_bias,)})
end

const HGFT5PreTrainedModel = HGFPreTrained{:t5}


get_model_type(::Val{:t5}) = merge(_get_model_type(:t5), (withlmheadmodel=HGFT5ForConditionalGeneration,))

basemodelkey(::Type{<:HGFPreTrained{:t5}}) = :transformer
isbasemodel(::Type{<:HGFT5Model}) = true
isbasemodel(::Type{<:HGFT5ForConditionalGeneration}) = true
isbasemodel(::Type{<:HGFT5EncoderModel}) = true


function load_model(_type::Type{HGFT5Model}, cfg, state_dict, prefix)
    embed = load_model(_type, CompositeEmbedding, cfg, state_dict, joinname(prefix, "shared"))
    seq2seq = load_model(_type, Seq2Seq, cfg, state_dict, prefix)
    return HGFT5Model(TransformerLayers.Parallel{(:encoder_input, :decoder_input)}(embed), seq2seq)
end

function load_model(::Type{HGFT5ForConditionalGeneration}, cfg, state_dict, prefix)
    model = load_model(HGFT5Model, cfg, state_dict, prefix)
    if cfg[:tie_word_embeddings]
        embedding = model.embed.layer.token.embeddings
        scale = convert(eltype(embedding), inv(sqrt(size(embedding, 1))))
    else
        vocab_size, dims, factor = cfg[:vocab_size], cfg[:d_model], Float32(cfg[:initializer_factor])
        embedding = getweight(weight_init(vocab_size, dims, factor), TransformerLayers.Embed,
            state_dict, joinname(prefix, "lm_head.weight"))
        scale = nothing
    end
    lm_head = TransformerLayers.EmbedDecoder(TransformerLayers.Embed(scale, embedding))
    return HGFT5ForConditionalGeneration(model, TransformerLayers.Branch{(:logit,),(:hidden_state,)}(lm_head))
end

function load_model(_type::Type{<:HGFT5EncoderModel}, cfg, state_dict, prefix)
    embed = load_model(HGFT5Model, CompositeEmbedding, cfg, state_dict, joinname(prefix, "shared"))
    encoder = load_model(HGFT5Model, TransformerBlock, cfg, state_dict, joinname(prefix, "encoder"))
    return HGFT5EncoderModel(embed, encoder)
end

function load_model(::Type{<:HGFT5PreTrainedModel}, ::Type{<:CompositeEmbedding}, cfg, state_dict, prefix)
    vocab_size, dims, factor = cfg[:vocab_size], cfg[:d_model], Float32(cfg[:initializer_factor])
    weight = getweight(weight_init(vocab_size, dims, factor), TransformerLayers.Embed, state_dict, joinname(prefix, "weight"))
    embed = CompositeEmbedding(token=TransformerLayers.Embed(nothing, weight))
    return embed
end

function load_model(_type::Type{<:HGFT5PreTrainedModel}, ::Type{<:Seq2Seq}, cfg, state_dict, prefix)
    encoder = load_model(_type, TransformerBlock, cfg, state_dict, joinname(prefix, "encoder"))
    decoder = load_model(_type, TransformerDecoderBlock, cfg, state_dict, joinname(prefix, "decoder"))
    return Seq2Seq(encoder, decoder)
end

function load_model(::Type{<:HGFT5PreTrainedModel}, ::Type{<:TransformerLayers.RMSLayerNorm}, cfg, state_dict, prefix)
    dims = cfg[:d_model]
    ln_ϵ = Float32(cfg[:layer_norm_epsilon])
    ln_init = one_init(dims)
    ln_weight = getweight(ln_init, Array, state_dict, joinname(prefix, "weight"))
    return TransformerLayers.RMSLayerNorm(ln_weight, ln_ϵ)
end

t5_collect_outputs(prev, output) = merge(output, TransformerLayers.collect_outputs(prev, Base.structdiff(output, NamedTuple{(:position_bias,)})))

function load_model(
    ::Type{<:HGFT5PreTrainedModel}, ::Type{<:TransformerLayers.SelfAttention{A}}, cfg, state_dict, prefix
) where {A<:Union{T5RPEMultiheadQKVAttenOp,T5RPECausalMultiheadQKVAttenOp,
        T5BiasedMultiheadQKVAttenOp,T5BiasedCausalMultiheadQKVAttenOp}}
    dims, head, kv_dims = cfg[:d_model], cfg[:num_heads], cfg[:d_kv]
    rpe_nbucket, rpe_max_dist = cfg[:relative_attention_num_buckets], cfg[:relative_attention_max_distance]
    p = Float64(cfg[:dropout_rate])
    p = iszero(p) ? nothing : p
    factor = Float32(cfg[:initializer_factor])
    return_score = cfg[:output_attentions]
    q_init = weight_init(dims, head * kv_dims, factor / sqrt(dims * kv_dims))
    kv_init = weight_init(dims, head * kv_dims, factor / sqrt(dims))
    o_init = weight_init(dims, head * kv_dims, factor / sqrt(head * kv_dims))
    q_weight = getweight(q_init, Array, state_dict, joinname(prefix, "q.weight"))
    k_weight = getweight(kv_init, Array, state_dict, joinname(prefix, "k.weight"))
    v_weight = getweight(kv_init, Array, state_dict, joinname(prefix, "v.weight"))
    o_weight = getweight(o_init, Array, state_dict, joinname(prefix, "o.weight"))
    qkv_proj = TransformerLayers.Fork(TransformerLayers.Dense(q_weight), TransformerLayers.Dense(k_weight), TransformerLayers.Dense(v_weight))
    o_proj = TransformerLayers.Dense(o_weight)
    if A <: Union{T5RPEMultiheadQKVAttenOp,T5RPECausalMultiheadQKVAttenOp}
        rpe_weight = getweight(weight_init(rpe_nbucket, head, factor / sqrt(dims)), TransformerLayers.Embed,
            state_dict, joinname(prefix, "relative_attention_bias.weight"))
        if A <: T5RPEMultiheadQKVAttenOp
            op = T5RPEMultiheadQKVAttenOp(head, rpe_nbucket, rpe_max_dist, rpe_weight, p)
        else
            op = T5RPECausalMultiheadQKVAttenOp(head, rpe_nbucket, rpe_max_dist, rpe_weight, p)
        end
    else
        if A <: T5BiasedMultiheadQKVAttenOp
            op = T5BiasedMultiheadQKVAttenOp(head, p)
        else
            op = T5BiasedCausalMultiheadQKVAttenOp(head, p)
        end
    end
    return_score && (op = WithScore(op))
    return TransformerLayers.SelfAttention(op, qkv_proj, o_proj)
end

function load_model(::Type{<:HGFT5PreTrainedModel}, ::Type{<:TransformerLayers.CrossAttention}, cfg, state_dict, prefix)
    dims, head, kv_dims = cfg[:d_model], cfg[:num_heads], cfg[:d_kv]
    p = Float64(cfg[:dropout_rate])
    p = iszero(p) ? nothing : p
    factor = Float32(cfg[:initializer_factor])
    return_score = cfg[:output_attentions]
    q_init = weight_init(dims, head * kv_dims, factor / sqrt(dims * kv_dims))
    kv_init = weight_init(dims, head * kv_dims, factor / sqrt(dims))
    o_init = weight_init(dims, head * kv_dims, factor / sqrt(head * kv_dims))
    q_weight = getweight(q_init, Array, state_dict, joinname(prefix, "q.weight"))
    k_weight = getweight(kv_init, Array, state_dict, joinname(prefix, "k.weight"))
    v_weight = getweight(kv_init, Array, state_dict, joinname(prefix, "v.weight"))
    o_weight = getweight(o_init, Array, state_dict, joinname(prefix, "o.weight"))
    q_proj = TransformerLayers.Dense(q_weight)
    kv_proj = TransformerLayers.Fork(TransformerLayers.Dense(k_weight), TransformerLayers.Dense(v_weight))
    o_proj = TransformerLayers.Dense(o_weight)
    op = T5MultiheadQKVAttenOp(head, p)
    return_score && (op = WithScore(op))
    return TransformerLayers.CrossAttention(op, q_proj, kv_proj, o_proj)
end

function load_model(
    ::Type{<:HGFT5PreTrainedModel}, ::Type{TransformerLayers.Chain{Tuple{TransformerLayers.Dense,TransformerLayers.Dense}}},
    cfg, state_dict, prefix
)
    dims, ff_dims = cfg[:d_model], cfg[:d_ff]
    factor = Float32(cfg[:initializer_factor])
    p = Float64(cfg[:dropout_rate])
    p = iszero(p) ? nothing : p
    isgated = cfg[:is_gated_act]
    act = ACT2FN[Symbol(cfg[:dense_act_fn])]
    wi_init = weight_init(dims, ff_dims, factor / sqrt(dims))
    wo_init = weight_init(ff_dims, dims, factor / sqrt(ff_dims))
    if isgated
        wi0_weight = getweight(wi_init, Array, state_dict, joinname(prefix, "wi0.weight"))
        wi1_weight = getweight(wi_init, Array, state_dict, joinname(prefix, "wi1.weight"))
        wi = T5Gated(TransformerLayers.Dense(act, wi0_weight), TransformerLayers.Dense(wi1_weight))
    else
        wi_weight = getweight(wi_init, Array, state_dict, joinname(prefix, "wi.weight"))
        wi = TransformerLayers.Dense(act, wi_weight)
    end
    wo_weight = getweight(wo_init, Array, state_dict, joinname(prefix, "wo.weight"))
    return TransformerLayers.Chain(TransformerLayers.DropoutLayer(wi, p), TransformerLayers.Dense(wo_weight))
end

function load_model(_type::Type{<:HGFT5PreTrainedModel}, ::Type{<:TransformerBlock}, cfg, state_dict, prefix)
    n = cfg[:num_layers]
    p = Float64(cfg[:dropout_rate])
    p = iszero(p) ? nothing : p
    collect_output = cfg[:output_attentions] || cfg[:output_hidden_states]
    blocks = []
    for i = 1:n
        lprefix = joinname(prefix, :block, i - 1, :layer)
        sa_ln = load_model(_type, TransformerLayers.RMSLayerNorm, cfg, state_dict, joinname(lprefix, "0.layer_norm"))
        op_type = isone(i) ? T5RPEMultiheadQKVAttenOp : T5BiasedMultiheadQKVAttenOp
        sa = load_model(_type, TransformerLayers.SelfAttention{op_type}, cfg, state_dict, joinname(lprefix, "0.SelfAttention"))
        sa = TransformerLayers.PreNormResidual(TransformerLayers.DropoutLayer(sa, p), sa_ln)
        ff_ln = load_model(_type, TransformerLayers.RMSLayerNorm, cfg, state_dict, joinname(lprefix, "1.layer_norm"))
        ff = load_model(_type, TransformerLayers.Chain{Tuple{TransformerLayers.Dense,TransformerLayers.Dense}},
            cfg, state_dict, joinname(lprefix, "1.DenseReluDense"))
        ff = TransformerLayers.PreNormResidual(TransformerLayers.DropoutLayer(ff, p), ff_ln)
        block = TransformerBlock(sa, ff)
        push!(blocks, block)
    end
    collect_f = collect_output ? t5_collect_outputs : nothing
    trf = Transformer(Tuple(blocks), collect_f)
    final_ln = load_model(_type, TransformerLayers.RMSLayerNorm, cfg, state_dict, joinname(prefix, "final_layer_norm"))
    return TransformerLayers.Chain(trf, TransformerLayers.DropoutLayer(final_ln, p))
end

function load_model(_type::Type{<:HGFT5PreTrainedModel}, ::Type{<:TransformerDecoderBlock}, cfg, state_dict, prefix)
    n = cfg[:num_layers]
    p = Float64(cfg[:dropout_rate])
    p = iszero(p) ? nothing : p
    collect_output = cfg[:output_attentions] || cfg[:output_hidden_states]
    blocks = []
    for i = 1:n
        lprefix = joinname(prefix, :block, i - 1, :layer)
        sa_ln = load_model(_type, TransformerLayers.RMSLayerNorm, cfg, state_dict, joinname(lprefix, "0.layer_norm"))
        op_type = isone(i) ? T5RPECausalMultiheadQKVAttenOp : T5BiasedCausalMultiheadQKVAttenOp
        sa = load_model(_type, TransformerLayers.SelfAttention{op_type}, cfg, state_dict, joinname(lprefix, "0.SelfAttention"))
        sa = TransformerLayers.PreNormResidual(TransformerLayers.DropoutLayer(sa, p), sa_ln)
        ca_ln = load_model(_type, TransformerLayers.RMSLayerNorm, cfg, state_dict, joinname(lprefix, "1.layer_norm"))
        ca = load_model(_type, TransformerLayers.CrossAttention, cfg, state_dict, joinname(lprefix, "1.EncDecAttention"))
        ca = TransformerLayers.PreNormResidual(TransformerLayers.DropoutLayer(ca, p), ca_ln)
        ff_ln = load_model(_type, TransformerLayers.RMSLayerNorm, cfg, state_dict, joinname(lprefix, "2.layer_norm"))
        ff = load_model(_type, TransformerLayers.Chain{Tuple{TransformerLayers.Dense,TransformerLayers.Dense}},
            cfg, state_dict, joinname(lprefix, "2.DenseReluDense"))
        ff = TransformerLayers.PreNormResidual(TransformerLayers.DropoutLayer(ff, p), ff_ln)
        block = TransformerDecoderBlock(sa, ca, ff)
        push!(blocks, block)
    end
    collect_f = collect_output ? t5_collect_outputs : nothing
    trf = Transformer(Tuple(blocks), collect_f)
    final_ln = load_model(_type, TransformerLayers.RMSLayerNorm, cfg, state_dict, joinname(prefix, "final_layer_norm"))
    return TransformerLayers.Chain(trf, TransformerLayers.DropoutLayer(final_ln, p))
end


function get_state_dict(m::HGFT5Model, state_dict, prefix)
    get_state_dict(HGFT5Model, m.embed, state_dict, joinname(prefix, "shared"))
    get_state_dict(HGFT5Model, m.seq2seq, state_dict, prefix)
    return state_dict
end

function get_state_dict(m::HGFT5ForConditionalGeneration, state_dict, prefix)
    get_state_dict(m.model, state_dict, prefix)
    embedding = m.cls.layer.embed.embeddings
    state_dict[joinname(prefix, "lm_head.weight")] = embedding'
    return state_dict
end

function get_state_dict(m::HGFT5EncoderModel, state_dict, prefix)
    get_state_dict(HGFT5Model, m.embed, state_dict, joinname(prefix, "shared"))
    get_state_dict(HGFT5Model, m.encoder[1], state_dict, joinname(prefix, "encoder"))
    get_state_dict(HGFT5Model, m.encoder[2], state_dict, joinname(prefix, "encoder.final_layer_norm"))
    return state_dict
end

get_state_dict(p::Type{<:HGFT5PreTrainedModel}, m::CompositeEmbedding, state_dict, prefix) =
    get_state_dict(p, m.token, state_dict, prefix)

function get_state_dict(p::Type{<:HGFT5PreTrainedModel}, m::Seq2Seq, state_dict, prefix)
    get_state_dict(p, m.encoder[1], state_dict, joinname(prefix, "encoder"))
    get_state_dict(p, m.encoder[2], state_dict, joinname(prefix, "encoder.final_layer_norm"))
    get_state_dict(p, m.decoder[1], state_dict, joinname(prefix, "decoder"))
    get_state_dict(p, m.decoder[2], state_dict, joinname(prefix, "decoder.final_layer_norm"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFT5PreTrainedModel}, m::TransformerLayers.SelfAttention, state_dict, prefix)
    get_state_dict(p, m.qkv_proj.layers[1], state_dict, joinname(prefix, "q"))
    get_state_dict(p, m.qkv_proj.layers[2], state_dict, joinname(prefix, "k"))
    get_state_dict(p, m.qkv_proj.layers[3], state_dict, joinname(prefix, "v"))
    get_state_dict(p, m.o_proj, state_dict, joinname(prefix, "o"))
    if m.attention_op isa Union{
        T5RPEMultiheadQKVAttenOp,T5RPECausalMultiheadQKVAttenOp,
        T5RPEMultiheadQKVAttenOpWithScore,T5RPECausalMultiheadQKVAttenOpWithScore
    }
        state_dict[joinname(prefix, "relative_attention_bias.weight")] = m.attention_op.position_embedding'
    end
    return state_dict
end

function get_state_dict(p::Type{<:HGFT5PreTrainedModel}, m::TransformerLayers.CrossAttention, state_dict, prefix)
    get_state_dict(p, m.q_proj, state_dict, joinname(prefix, "q"))
    get_state_dict(p, m.kv_proj.layers[1], state_dict, joinname(prefix, "k"))
    get_state_dict(p, m.kv_proj.layers[2], state_dict, joinname(prefix, "v"))
    get_state_dict(p, m.o_proj, state_dict, joinname(prefix, "o"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFT5PreTrainedModel}, m::TransformerLayers.Chain{<:Tuple{Any,TransformerLayers.Dense}}, state_dict, prefix)
    if m[1] isa T5Gated
        get_state_dict(p, m[1].layer.gate, state_dict, joinname(prefix, "wi0"))
        get_state_dict(p, m[1].layer.linear, state_dict, joinname(prefix, "wi1"))
    else
        get_state_dict(p, m[1], state_dict, joinname(prefix, "wi"))
    end
    get_state_dict(p, m[2], state_dict, joinname(prefix, "wo"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFT5PreTrainedModel}, m::Transformer, state_dict, prefix)
    for (i, t) in enumerate(m.blocks)
        get_state_dict(p, t, state_dict, joinname(prefix, :block, i - 1, :layer))
    end
    return state_dict
end

function get_state_dict(p::Type{<:HGFT5PreTrainedModel}, m::TransformerBlock, state_dict, prefix)
    get_state_dict(p, m.attention.layer, state_dict, joinname(prefix, "0.SelfAttention"))
    get_state_dict(p, m.attention.norm, state_dict, joinname(prefix, "0.layer_norm"))
    get_state_dict(p, m.feedforward.layer, state_dict, joinname(prefix, "1.DenseReluDense"))
    get_state_dict(p, m.feedforward.norm, state_dict, joinname(prefix, "1.layer_norm"))
    return state_dict
end

function get_state_dict(p::Type{<:HGFT5PreTrainedModel}, m::TransformerDecoderBlock, state_dict, prefix)
    get_state_dict(p, m.attention.layer, state_dict, joinname(prefix, "0.SelfAttention"))
    get_state_dict(p, m.attention.norm, state_dict, joinname(prefix, "0.layer_norm"))
    get_state_dict(p, m.crossattention.layer, state_dict, joinname(prefix, "1.EncDecAttention"))
    get_state_dict(p, m.crossattention.norm, state_dict, joinname(prefix, "1.layer_norm"))
    get_state_dict(p, m.feedforward.layer, state_dict, joinname(prefix, "2.DenseReluDense"))
    get_state_dict(p, m.feedforward.norm, state_dict, joinname(prefix, "2.layer_norm"))
    return state_dict
end
