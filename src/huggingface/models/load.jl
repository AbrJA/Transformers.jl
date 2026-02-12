using ..TransformerLayers: lengthselect, skipfirsttoken, skiplasttoken, safe_logitcrossentropy
import ..TransformerLayers

using Flux
using Flux: Losses
using NNlib

using StructWalk
using ChainRulesCore
using DataStructures: OrderedDict
using NeuralAttentionlib: NoMask

using LinearAlgebra

struct FirstTokenPooler end
(m::FirstTokenPooler)(x) = selectdim(x, 2, 1)

abstract type HGFPreTrainedModel end
TransformerLayers.@fluxshow HGFPreTrainedModel

abstract type HGFPreTrained{model_type,task} <: HGFPreTrainedModel end

getmodeltype(m::HGFPreTrained) = getmodeltype(typeof(m))
getmodeltask(m::HGFPreTrained) = getmodeltask(typeof(m))
getmodeltype(::Type{<:HGFPreTrained{MT}}) where MT = MT
getmodeltask(::Type{<:HGFPreTrained{MT,T}}) where {MT,T} = T

function hgf_model_forward end
function hgf_model_loss end

function (model::HGFPreTrained)(nt::NamedTuple)
    if hasmethod(hgf_model_loss, (typeof(model),)) && haskey(nt, :label)
        return hgf_model_loss(model)(model, hgf_model_forward(model, nt))
    else
        return hgf_model_forward(model, nt)
    end
end

for (task, lfunc) in (
    (:forcausallm, :causal_lm_loss),
)
    @eval begin
        @inline hgf_model_loss(::HGFPreTrained{MT,$(QuoteNode(task))}) where MT = $lfunc
    end
end

# Note: @hgfdef macro has been completely removed.
# All models now use explicit struct definitions.
# See individual model files in src/huggingface/implementation/*/load.jl


isbasemodel(_) = false
isbasemodel(::Type{<:HGFPreTrained{T,:model}}) where T = true

# Sadly a really inaccurate gelu but needed to match the value with the python models
quick_gelu(x) = x * sigmoid_fast(NNlib.oftf(x, 1.702) * x)
function quick_gelu_forward_backward(x)
    λ = NNlib.oftf(x, 1.702)
    λx = λ * x
    σλx = sigmoid_fast(λx)
    backward = muladd(TransformerLayers._deriv_σ(σλx), λx, σλx)
    return x * σλx, backward
end
TransformerLayers.act_pullback(::typeof(quick_gelu)) = quick_gelu_forward_backward
TransformerLayers.require_x(::typeof(quick_gelu)) = true

const ACT2FN = @alias (
    [gelu, gelu_new, gelu_fast, gelu_python, gelu_pytorch_tanh, gelu_accurate]=gelu,
    [swish, silu]=swish,
    quick_gelu=quick_gelu,
    leaky_relu=leakyrelu,
    relu=relu,
    mish=mish,
    selu=selu,
    sigmoid=sigmoid_fast,
    tanh=tanh_fast,
)

joinname(prefix, name) = isempty(prefix) ? name : join((prefix, name), '.')
joinname(prefix, n1, n2...) = joinname(prefix, join((n1, n2...), '.'))

haskeystartswith(dict, prefix) = any(startswith("$prefix."), keys(dict))

function _normal0(std, s...) # normal(mean = 0, std)
    weight = randn(Float32, s...)
    if !isone(std)
        weight .*= std
    end
    return weight
end
zero_init(dims) = () -> zeros(Float32, dims)
one_init(dims) = () -> ones(Float32, dims)
bias_init(d, factor=true) = bias_init_f() = _normal0(factor, d)
weight_init(din, dout, factor=true) = weight_init_f() = _normal0(factor, dout, din)
filter_init(kh, kw, in, out, factor=true) = filter_init_f() = _normal0(factor, out, in, kw, kh)


