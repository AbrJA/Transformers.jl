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

function _hgfmodelstruct(model_type, type_name, task_name, field_names, expr=nothing)
    sbody = []
    tnames = []
    fbody = :nt
    for fname in field_names
        tname = Symbol(uppercase(String(fname)))
        push!(tnames, tname)
        push!(sbody, :($fname::$tname))
        fbody = :(model.$fname($fbody))
    end
    sname = Symbol("HGF", type_name, task_name)
    name = Expr(:<:,
        Expr(:curly, sname, tnames...),
        Expr(:curly, :HGFPreTrained, QuoteNode(model_type), QuoteNode(Symbol(lowercase(String(task_name))))))
    st = Expr(:struct, false, name, Expr(:block, sbody...))
    if !isnothing(expr)
        fbody = expr.args
    else
        fbody = (fbody,)
    end
    func = :(@inline $(@__MODULE__).hgf_model_forward(model::$sname, nt::NamedTuple) = $(fbody...))
    return Expr(:block, st, :($(@__MODULE__).Flux.@layer $sname), func)
end

function _extractfields(ex, field_names=Symbol[])
    if ex isa Expr
        for arg in ex.args
            arg isa Expr && !Meta.isexpr(arg, :.) && _extractfields(arg, field_names)
        end
        for arg in ex.args
            if Meta.isexpr(arg, :.) && arg.args[1] == :model
                sym = arg.args[2].value
                sym in field_names || push!(field_names, sym)
            end
        end
    end
    if Meta.isexpr(ex, :.) && ex.args[1] == :model
        sym = ex.args[2].value
        sym in field_names || push!(field_names, sym)
    end
    return field_names
end

function _modeldef(model_type, type_name, ex)
    ex isa Symbol &&
        return _hgfmodelstruct(model_type, type_name, ex, (:model, :cls))
    if Meta.isexpr(ex, :call, 3) && first(ex.args) == :(=>)
        task_name = ex.args[2]
        ex = ex.args[3]
        if Meta.isexpr(ex, :tuple)
            all(Base.Fix2(isa, Symbol), ex.args) &&
                return _hgfmodelstruct(model_type, type_name, task_name, ex.args)
        elseif Meta.isexpr(ex, :block)
            field_names = _extractfields(ex)
            st = _hgfmodelstruct(model_type, type_name, task_name, field_names, ex)
            return st
        end
    end
    error("Unknown pattern: $ex")
end

macro hgfdef(type_name, ex)
    model_type = QuoteNode(Symbol(lowercase(String(type_name))))
    return var"@hgfdef"(__source__, __module__, model_type, type_name, ex)
end
macro hgfdef(model_type, type_name, ex)
    if model_type isa Symbol
        model_type = QuoteNode(model_type)
    end
    @assert model_type isa QuoteNode "model_type is not a Symbol"
    @assert type_name isa Symbol
    @assert Meta.isexpr(ex, :tuple) "supported models should be put in a tuple"
    exprs = []
    for task in ex.args
        st = _modeldef(model_type.value, type_name, task)
        append!(exprs, st.args)
    end
    return esc(Expr(:block, :(const $(Symbol(:HGF, type_name, :PreTrainedModel)) = HGFPreTrained{$model_type}), exprs...))
end

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


