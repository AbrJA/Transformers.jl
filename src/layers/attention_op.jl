using StructWalk
using ChainRulesCore
using NeuralAttentionlib
using NeuralAttentionlib: $, AbstractAttenOp, MultiheadQKVAttenOpWithScore, MultiheadQKVAttenOp,
    CausalMultiheadQKVAttenOp, CausalMultiheadQKVAttenOpWithScore,
    GroupedQueryAttenOp, GroupedQueryAttenOpWithScore,
    CausalGroupedQueryAttenOp, CausalGroupedQueryAttenOpWithScore,
    alibi_position_embedding, with_rotary_position_embedding,
    dot_product_score, scaled_dot_product_score,
    masked_score, normalized_score, dropout_score, weighted_sum_mixing,
    generic_multihead_qkv_attention, generic_grouped_query_attention,
    multihead_qkv_attention,
    CausalMask, BatchedMask, LocalMask


WithScore(op::AbstractAttenOp) = NeuralAttentionlib.WithScore(op)
WithScore(x) = postwalk(LayerStyle, x) do xi
    xi isa AbstractAttenOp ? NeuralAttentionlib.WithScore(xi) : xi
end

set_dropout(op::MultiheadQKVAttenOp, p) = MultiheadQKVAttenOp(op.head, p)
set_dropout(op::CausalMultiheadQKVAttenOp, p) = CausalMultiheadQKVAttenOp(op.head, p)
set_dropout(op::NeuralAttentionlib.WithScore, p) = NeuralAttentionlib.WithScore(set_dropout(getfield(op, :op), p))

function apply_attention_op(op, nt::NamedTuple)
    qkv = nt.hidden_state
    ChainRulesCore.ignore_derivatives() do
        qkv isa NTuple{3, Any} ||
            error("Expect hidden_state to be a tuple of 3 arrays, but get $(typeof(qkv)).")
        nothing
    end
    q, k, v = qkv
    mask = ChainRulesCore.ignore_derivatives(()->get(nt, :attention_mask, nothing))
    a = op(q, k, v, mask)
    return return_hidden_state(nt, a)
end

# dot attention

dot_attention_score(mask, p) =
    dropout_score(p) $ normalized_score(softmax) $
    masked_score(NeuralAttentionlib.GenericMaskOp(), mask) $ dot_product_score

ChainRulesCore.@non_differentiable dot_attention_score(arg...)

function multihead_qkv_dot_attention(head, q, k, v, mask = nothing, p = nothing)
    return generic_multihead_qkv_attention(weighted_sum_mixing, dot_attention_score(mask, p), head, q, k, v)
end
function multihead_qkv_dot_attention(
    ::typeof(NeuralAttentionlib.score_returning),
    head, q, k, v, mask = nothing, p = nothing)
    return generic_multihead_qkv_attention(
        NeuralAttentionlib.score_returning(weighted_sum_mixing),
        dot_attention_score(mask, p), head, q, k, v)
end

# RoPE

rope_attention_score(dim, mask, p) =
    dropout_score(p) $
    normalized_score(softmax) $
    masked_score(NeuralAttentionlib.GenericMaskOp(), mask) $
    scaled_dot_product_score $
    with_rotary_position_embedding(dim)

ChainRulesCore.@non_differentiable rope_attention_score(arg...)

function rope_multihead_qkv_attention(dim, head, q, k, v, mask = nothing, p = nothing)
    return generic_multihead_qkv_attention(
        weighted_sum_mixing, rope_attention_score(dim, mask, p),
        head, q, k, v)
end
function rope_multihead_qkv_attention(
    ::typeof(NeuralAttentionlib.score_returning),
    dim, head, q, k, v, mask = nothing, p = nothing
)
    return generic_multihead_qkv_attention(
        NeuralAttentionlib.score_returning(weighted_sum_mixing),
        rope_attention_score(dim, mask, p),
        head, q, k, v, position_embedding)
end

# ALiBi

alibi_attention_score(mask, p) =
    dropout_score(p) $
    normalized_score(softmax) $
    masked_score(NeuralAttentionlib.GenericMaskOp(), mask) $
    alibi_position_embedding(mask) $
    scaled_dot_product_score

ChainRulesCore.@non_differentiable alibi_attention_score(arg...)

function alibi_multihead_qkv_attention(head, q, k, v, mask = nothing, p = nothing)
    return generic_multihead_qkv_attention(
        weighted_sum_mixing, alibi_attention_score(mask, p),
        head, q, k, v)
end
function alibi_multihead_qkv_attention(
    ::typeof(NeuralAttentionlib.score_returning),
    head, q, k, v, mask = nothing, p = nothing
)
    return generic_multihead_qkv_attention(
        NeuralAttentionlib.score_returning(weighted_sum_mixing),
        alibi_attention_score(mask, p),
        head, q, k, v, position_embedding)
end

# FlexAttenOp

struct FlexAttenOp{Mode, Causal, Scaled, Arg, F} <: AbstractAttenOp
    arg::Arg
    head::Int
    p::F
    FlexAttenOp{Mode, Causal, Scaled, Arg, F}(arg::Arg, head::Int, p::F) where {Mode, Causal, Scaled, Arg, F} = new{Mode, Causal, Scaled, Arg, F}(arg, head, p)
end

# Use a non-parameterized function to break recursion
function create_flex_atten_op(Mode, Causal, Scaled, arg, head::Int, p)
    return FlexAttenOp{Mode, Causal, Scaled, typeof(arg), typeof(p)}(arg, head, p)
end

function (::Type{FlexAttenOp{Mode, Causal, Scaled}})(arg, head::Int, p) where {Mode, Causal, Scaled}
    return create_flex_atten_op(Mode, Causal, Scaled, arg, head, p)
end

function NeuralAttentionlib.get_attention_func(::FlexAttenOp{Mode, Causal, Scaled}) where {Mode, Causal, Scaled}
    if Mode === :RoPE
        return rope_multihead_qkv_attention
    elseif Mode === :ALiBi
        return alibi_multihead_qkv_attention
    elseif Mode === :Local
        return Scaled ? multihead_qkv_attention : multihead_qkv_dot_attention
    else # Standard
        return Scaled ? multihead_qkv_attention : multihead_qkv_dot_attention
    end
end

function NeuralAttentionlib.get_attention_func_args(op::FlexAttenOp{Mode, Causal, Scaled}, q, k, v, mask = nothing) where {Mode, Causal, Scaled}
    m = mask
    if Causal
        m = CausalMask() & m
    end
    if Mode === :Local
        m = LocalMask(op.arg) & m
    end

    m = BatchedMask(m)

    if Mode === :RoPE
        return (op.arg, op.head, q, k, v, m, op.p)
    else
        return (op.head, q, k, v, m, op.p)
    end
end

set_dropout(op::FlexAttenOp{Mode, Causal, Scaled, Arg, F}, p) where {Mode, Causal, Scaled, Arg, F} =
    FlexAttenOp{Mode, Causal, Scaled, Arg, typeof(p)}(op.arg, op.head, p)

# Aliases

const MultiheadQKVDotAttenOp = FlexAttenOp{:Standard, false, false, Nothing}
# MultiheadQKVDotAttenOp(head::Int, p = nothing) is handled by generic FlexAttenOp constructor

const MultiheadQKVDotAttenOpWithScore{F} = NeuralAttentionlib.WithScore{MultiheadQKVDotAttenOp{F}}

const CausalMultiheadQKVDotAttenOp = FlexAttenOp{:Standard, true, false, Nothing}
# CausalMultiheadQKVDotAttenOp(head::Int, p = nothing) is handled by generic FlexAttenOp constructor

const CausalMultiheadQKVDotAttenOpWithScore{F} = NeuralAttentionlib.WithScore{CausalMultiheadQKVDotAttenOp{F}}

const LocalMultiheadQKVAttenOp = FlexAttenOp{:Local, false, true, Int}
# LocalMultiheadQKVAttenOp(size::Int, head::Int, p = nothing) is handled by generic FlexAttenOp constructor

const LocalMultiheadQKVAttenOpWithScore{F} = NeuralAttentionlib.WithScore{LocalMultiheadQKVAttenOp{F}}

const LocalCausalMultiheadQKVAttenOp = FlexAttenOp{:Local, true, true, Int}
# LocalCausalMultiheadQKVAttenOp(size::Int, head::Int, p = nothing) is handled by generic FlexAttenOp constructor

const LocalCausalMultiheadQKVAttenOpWithScore{F} = NeuralAttentionlib.WithScore{LocalCausalMultiheadQKVAttenOp{F}}

const LocalMultiheadQKVDotAttenOp = FlexAttenOp{:Local, false, false, Int}
# LocalMultiheadQKVDotAttenOp(size::Int, head::Int, p = nothing) is handled by generic FlexAttenOp constructor

const LocalMultiheadQKVDotAttenOpWithScore{F} = NeuralAttentionlib.WithScore{LocalMultiheadQKVDotAttenOp{F}}

const LocalCausalMultiheadQKVDotAttenOp = FlexAttenOp{:Local, true, false, Int}
# LocalCausalMultiheadQKVDotAttenOp(size::Int, head::Int, p = nothing) is handled by generic FlexAttenOp constructor

const LocalCausalMultiheadQKVDotAttenOpWithScore{F} = NeuralAttentionlib.WithScore{LocalCausalMultiheadQKVDotAttenOp{F}}

const RoPEMultiheadQKVAttenOp = FlexAttenOp{:RoPE, false, true}
# RoPEMultiheadQKVAttenOp(dim::Int, head::Int, p = nothing) is handled by generic FlexAttenOp constructor

RoPEMultiheadQKVAttenOp(head::Int) = FlexAttenOp{:RoPE, false, true, Nothing, Nothing}(nothing, head, nothing)
const RoPEMultiheadQKVAttenOpWithScore{D, F} = NeuralAttentionlib.WithScore{RoPEMultiheadQKVAttenOp{D, F}}

const CausalRoPEMultiheadQKVAttenOp = FlexAttenOp{:RoPE, true, true}
# CausalRoPEMultiheadQKVAttenOp(dim::Int, head::Int, p = nothing) is handled by generic FlexAttenOp constructor

CausalRoPEMultiheadQKVAttenOp(head::Int) = FlexAttenOp{:RoPE, true, true, Nothing, Nothing}(nothing, head, nothing)
const CausalRoPEMultiheadQKVAttenOpWithScore{D, F} = NeuralAttentionlib.WithScore{CausalRoPEMultiheadQKVAttenOp{D, F}}

const ALiBiMultiheadQKVAttenOp = FlexAttenOp{:ALiBi, false, true, Nothing}
# ALiBiMultiheadQKVAttenOp(head::Int, p = nothing) is handled by generic FlexAttenOp constructor

const ALiBiMultiheadQKVAttenOpWithScore{F} = NeuralAttentionlib.WithScore{ALiBiMultiheadQKVAttenOp{F}}

const CausalALiBiMultiheadQKVAttenOp = FlexAttenOp{:ALiBi, true, true, Nothing}
# CausalALiBiMultiheadQKVAttenOp(head::Int, p = nothing) is handled by generic FlexAttenOp constructor

const CausalALiBiMultiheadQKVAttenOpWithScore{F} = NeuralAttentionlib.WithScore{CausalALiBiMultiheadQKVAttenOp{F}}

# layer api
# Imported ops
for op in :[
    MultiheadQKVAttenOp, MultiheadQKVAttenOpWithScore,
    CausalMultiheadQKVAttenOp, CausalMultiheadQKVAttenOpWithScore,
    GroupedQueryAttenOp, GroupedQueryAttenOpWithScore,
    CausalGroupedQueryAttenOp, CausalGroupedQueryAttenOpWithScore,
].args
    @eval begin
        argument_names(::$op) = (:hidden_state, :attention_mask)
        apply_on_namedtuple(op::$op, nt::NamedTuple) = apply_attention_op(op, nt)
    end
end

# FlexAttenOp
argument_names(::FlexAttenOp) = (:hidden_state, :attention_mask)
apply_on_namedtuple(op::FlexAttenOp, nt::NamedTuple) = apply_attention_op(op, nt)
