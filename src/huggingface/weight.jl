using DataStructures: OrderedDict
using JSON3

using SafeTensors

"""
  `load_state_dict(model_name; local_files_only = false, force_format = :auto, cache = true)`

Load the `state_dict` from the given `model_name` from huggingface hub. By default, this function would check if
 `model_name` exists on huggingface hub, download the model file (and cache it if `cache` is set), and then load
 and return the `state_dict`. If `local_files_only = false`, it would check whether the model file is up-to-date and
 update if not (and thus require network access every time it is called). By setting `local_files_only = true`, it
 would try to find the files from the cache directly and error out if not found. For managing the caches, see the
 `HuggingFaceApi.jl` package. If `force_format` is `:auto` it will automatically selects the format from which the
 weights will be loaded. If `force_format` is `:pickle` or `:safetensor`, it will prefer relevant file.
"""
const LOAD_PICKLE = Ref{Function}((args...; kws...) -> error("Pickle.jl not loaded"))
load_pickle_weight(args...; kws...) = LOAD_PICKLE[](args...; kws...)

const SAVE_PICKLE = Ref{Function}((args...; kws...) -> error("Pickle.jl not loaded"))
save_pickle_weight(args...; kws...) = SAVE_PICKLE[](args...; kws...)

const IS_PICKLE = Ref{Function}(x -> false)
is_pickle(x) = IS_PICKLE[](x)

const UNWRAP_PICKLE = Ref{Function}(identity)
unwrap_pickle(x) = UNWRAP_PICKLE[](x)

function load_state_dict(model_name; lazy=false, mmap=true, possible_files=nothing, force_format=:auto, kw...)
    possible_files = ensure_possible_files(possible_files, model_name; kw...)
    weight_format = force_format == :auto ? detect_weight_format(model_name; possible_files, kw...) : force_format
    status = WeightStatus{weight_format}(model_name; possible_files, kw...)
    if status isa HasWeightIn
        return load_state_dict_from(status; lazy, mmap, kw...)
    else
        error("The repository does not contain the weights stored in $(weight_format) format")
    end
end

function detect_weight_format(model_name; possible_files=nothing, kws...)
    HasWeightIn{:safetensor}(model_name; possible_files, kws...) && return :safetensor
    HasWeightIn{:pickle}(model_name; possible_files, kws...) && return :pickle
    error("The repository does not contain the weights stored in supported formats (safetensors or pytorch pickle)")
end

abstract type WeightStatus{format} end
abstract type HasWeightIn{format} <: WeightStatus{format} end
struct HasIndexMap{format} <: HasWeightIn{format}
    indexmap::Dict{String,Union{Nothing,Set{String}}}
end
struct HasSingleFile{format} <: HasWeightIn{format}
    file::String
end
struct NoWeightIn{format} <: WeightStatus{format} end

indexmapname(::Type{WeightStatus{:pickle}}) = PYTORCH_WEIGHTS_INDEX_NAME
indexmapname(::Type{WeightStatus{:safetensor}}) = SAFETENSOR_WEIGHTS_INDEX_NAME
singlefilename(::Type{WeightStatus{:pickle}}) = PYTORCH_WEIGHTS_NAME
singlefilename(::Type{WeightStatus{:safetensor}}) = SAFETENSOR_WEIGHTS_NAME
filelist(S::Type{WeightStatus{format}}) where format = (indexmapname(S), singlefilename(S))

function HasWeightIn{format}(model_name; possible_files=nothing, kw...) where format
    possible_files = ensure_possible_files(possible_files, model_name; kw...)
    return any(in(possible_files), filelist(WeightStatus{format})) ? true : false
end
function WeightStatus{format}(model_name; possible_files=nothing, kw...) where format
    possible_files = ensure_possible_files(possible_files, model_name; kw...)
    if indexmapname(WeightStatus{format}) in possible_files
        weightmap = JSON3.read(read(hgf_file(model_name, indexmapname(WeightStatus{format}); kw...))).weight_map
        indexmap = Dict{String,Set{String}}()
        for (weight, filename) in weightmap
            keyset = get!(() -> Set{String}(), indexmap, hgf_file(model_name, filename; kw...))
            push!(keyset, weight)
        end
        return HasIndexMap{format}(indexmap)
    elseif singlefilename(WeightStatus{format}) in possible_files
        return HasSingleFile{format}(hgf_file(model_name, singlefilename(WeightStatus{format}); kw...))
    else
        return NoWeightIn{format}()
    end
end

function _state_dict_add!(state_dict, key, val; file=nothing)
    if haskey(state_dict, key)
        @warn """weight "$key" is overwritten by weight $(isnothing(file) ? "" : "from file \"$file\" ")with the same name."""
    end
    state_dict[key] = val
end
function _debug_key_misalign(file, notfound, unexpected)
    if !(isempty(unexpected) && isempty(notfound))
        @debug(
            "$file contains/missing some weights",
            var"expected to contain but not found" = notfound,
            var"unexpected but appear in file" = unexpected,
        )
    end
end

load_state_dict_from(status::HasSingleFile; lazy=false, mmap=true, keyset=nothing, kw...) =
    load_state_dict_from!(status, OrderedDict{Any,Any}(); lazy, mmap, kw...)
function load_state_dict_from!(status::HasSingleFile{:pickle}, state_dict; lazy=false, mmap=true, keyset=nothing, kw...)
    file = status.file
    loaded_state_dict = load_pickle_weight(file; lazy, mmap)
    for (key, val) in loaded_state_dict
        _state_dict_add!(state_dict, key, val; file)
    end
    if !isnothing(keyset)
        loaded_keys = keys(loaded_state_dict)
        unexpected = setdiff(loaded_keys, keyset)
        notfound = setdiff(keyset, loaded_keys)
        _debug_key_misalign(file, notfound, unexpected)
    end
    return state_dict
end
function load_state_dict_from!(status::HasSingleFile{:safetensor}, state_dict; lazy=false, mmap=true, keyset=nothing, kw...)
    file = status.file
    safetensor = SafeTensors.deserialize(file; mmap)
    stored2shared = Dict{String,Set{String}}()
    # https://github.com/huggingface/safetensors/blob/b947b59079a6197d7930dfb535818ac4896113e8/bindings/python/py_src/safetensors/torch.py#L155-L165
    # huggingface seems to store shared tensor name in the metadata, so we tried to restore the information
    #  by checking if the metadata contain the tensor names.
    for (metakey, metaval) in safetensor.metadata # removed => kept
        if haskey(safetensor, metaval)
            sharednames = get!(() -> Set{String}(), stored2shared, metaval)
            push!(sharednames, metakey)
        end
    end
    for (key, val) in safetensor
        val = lazy ? val : collect(val)
        _state_dict_add!(state_dict, key, val; file)
        if haskey(stored2shared, key)
            for sharedname in stored2shared[key]
                _state_dict_add!(state_dict, sharedname, val; file)
            end
        end
    end
    if !isnothing(keyset)
        loaded_keys = union(keys(safetensor), values(stored2shared)...)
        unexpected = setdiff(loaded_keys, keyset)
        notfound = setdiff(keyset, loaded_keys)
        _debug_key_misalign(file, notfound, unexpected)
    end
    return state_dict
end
load_state_dict_from(status::HasIndexMap; lazy=false, mmap=true, kw...) =
    load_state_dict_from!(status, OrderedDict{Any,Any}(); lazy, mmap, kw...)
function load_state_dict_from!(status::HasIndexMap{format}, state_dict; lazy=false, mmap=true, kw...) where format
    for (file, keyset) in status.indexmap
        load_state_dict_from!(HasSingleFile{format}(file), state_dict; lazy, mmap, keyset, kw...)
    end
    return state_dict
end


"""
  `state_dict_to_namedtuple(state_dict)`

convert state_dict into nested `NamedTuple`.
"""
function state_dict_to_namedtuple(state_dict)
    dict = OrderedDict{String,Any}()
    for (k, v) in state_dict
        keys = split(k, '.')
        d = dict
        for key in keys[1:end-1]
            d = get!(OrderedDict{String,Any}, d, key)
        end
        d[keys[end]] = v
    end
    return _dict2nt(dict)
end

function _dict2nt(d::AbstractDict)
    keys_str = collect(keys(d))
    if all(k -> occursin(r"^\d+$", k), keys_str)
        # It's a vector/tuple
        indices = parse.(Int, keys_str)
        perm = sortperm(indices)
        return ntuple(i -> _dict2nt(d[keys_str[perm[i]]]), length(indices))
    else
        # It's a named tuple
        ks = Tuple(Symbol.(keys_str))
        vs = ntuple(i -> _dict2nt(d[keys_str[i]]), length(ks))
        return NamedTuple{ks}(vs)
    end
end

_dict2nt(x) = x

_reverseperm(x) = reverse(ntuple(identity, Val(ndims(x))))
_reversedims(x) = PermutedDimsArray(x, _reverseperm(x))
reversedims(x) = _reversedims(x)
reversedims(x::PermutedDimsArray{T,N,perm}) where {T,N,perm} = perm == _reversedims(x) ? parent(x) : _reversedims(x)
collect32(x) = collect(Float32, x)

getweight(init, ::Type, ::Symbol) = init()
getweight(init, x, sym::Symbol) = getproperty(x, sym)

getweight(init, ::Type{<:Array}, state_dict, name) = _getweight(collect32, init, state_dict, name)
getweight(init, ::Type{<:Array}, state_dict::OrderedDict{String}, name) = getweight(init, state_dict, name)
getweight(init, ::Type{<:TransformerLayers.Embed}, state_dict, name) = _getweight(collect32 ∘ adjoint, init, state_dict, name)
getweight(init, ::Type{<:TransformerLayers.Embed}, state_dict::OrderedDict{String}, name) = _getweight(adjoint, init, state_dict, name)
getweight(init, ::Type{<:Flux.CrossCor}, state_dict, name) = _getweight(collect32 ∘ reversedims, init, state_dict, name)
getweight(init, ::Type{<:Flux.CrossCor}, state_dict::OrderedDict{String}, name) = _getweight(reversedims, init, state_dict, name)

getweight(init, state_dict, name) = _getweight(identity, init, state_dict, name)
function _getweight(process, init, state_dict, name)
    if haskey(state_dict, name)
        state = state_dict[name]
        if is_pickle(state)
            weight = unwrap_pickle(state)
            weight = process(weight)
        else
            weight = process(state)
        end
    else
        @debug "$name not found, initialized."
        weight = init()
    end
    return weight
end

get_state_dict(_, m::TransformerLayers.Embed, state_dict, prefix) = get_state_dict(m, state_dict, prefix)
function get_state_dict(m::TransformerLayers.Embed, state_dict, prefix)
    state_dict[joinname(prefix, "weight")] = m.embeddings'
    return state_dict
end

get_state_dict(_, m::TransformerLayers.EmbedDecoder, state_dict, prefix) = get_state_dict(m, state_dict, prefix)
function get_state_dict(m::TransformerLayers.EmbedDecoder, state_dict, prefix)
    if !isnothing(m.bias)
        state_dict[joinname(prefix, "bias")] = m.bias
    end
    get_state_dict(m.embed, state_dict, prefix)
    return state_dict
end

get_state_dict(_, m::TransformerLayers.FixedLenPositionEmbed, state_dict, prefix) = get_state_dict(m, state_dict, prefix)
function get_state_dict(m::TransformerLayers.FixedLenPositionEmbed, state_dict, prefix)
    state_dict[joinname(prefix, "weight")] = m.embeddings'
    return state_dict
end

get_state_dict(_, m::TransformerLayers.Dense, state_dict, prefix) = get_state_dict(m, state_dict, prefix)
function get_state_dict(m::TransformerLayers.Dense, state_dict, prefix)
    state_dict[joinname(prefix, "weight")] = m.W
    !isnothing(m.b) && (state_dict[joinname(prefix, "bias")] = m.b)
    return state_dict
end

get_state_dict(_, m::Flux.CrossCor, state_dict, prefix) = get_state_dict(m, state_dict, prefix)
function get_state_dict(m::Flux.CrossCor, state_dict, prefix)
    state_dict[joinname(prefix, "weight")] = reversedims(m.weight)
    !isnothing(m.bias) && (state_dict[joinname(prefix, "bias")] = m.bias)
    return state_dict
end

get_state_dict(_, m::TransformerLayers.LayerNorm, state_dict, prefix) = get_state_dict(m, state_dict, prefix)
function get_state_dict(m::TransformerLayers.LayerNorm, state_dict, prefix)
    state_dict[joinname(prefix, "weight")] = m.α
    state_dict[joinname(prefix, "bias")] = m.β
    return state_dict
end

get_state_dict(_, m::TransformerLayers.RMSLayerNorm, state_dict, prefix) = get_state_dict(m, state_dict, prefix)
function get_state_dict(m::TransformerLayers.RMSLayerNorm, state_dict, prefix)
    state_dict[joinname(prefix, "weight")] = m.α
    return state_dict
end

get_state_dict(p, m::TransformerLayers.RenameArgs, state_dict, prefix) = get_state_dict(p, m.layer, state_dict, prefix)
get_state_dict(p, m::TransformerLayers.Branch, state_dict, prefix) = get_state_dict(p, m.layer, state_dict, prefix)
get_state_dict(p, m::TransformerLayers.Parallel, state_dict, prefix) = get_state_dict(p, m.layer, state_dict, prefix)
get_state_dict(p, m::TransformerLayers.DropoutLayer, state_dict, prefix) = get_state_dict(p, m.layer, state_dict, prefix)

load_model(_type::Type, cfg) = load_model(_type, cfg, OrderedDict{String,Any}())
load_model(_type::Type, cfg, state_dict) = load_model(_type, cfg, state_dict, "")

get_state_dict(m) = get_state_dict(m, OrderedDict{String,Any}())
get_state_dict(m, state_dict) = get_state_dict(m, state_dict, "")


_load_embed(state_dict, prefix, vocab_size, dims, factor, pad_idx0=nothing) =
    _load_embed(state_dict, prefix, weight_init(vocab_size, dims, factor), pad_idx0)
function _load_embed(state_dict, prefix, w_init, pad_idx0=nothing)
    embedding = getweight(TransformerLayers.Embed, state_dict, joinname(prefix, "weight")) do
        weight = w_init()
        if !isnothing(pad_idx0)
            weight[:, pad_idx0+1] .= 0
        end
        return weight
    end
    return TransformerLayers.Embed(embedding)
end

function _load_layernorm(state_dict, prefix, dims, ln_ϵ=1e-5)
    old_weight_name = joinname(prefix, "gamma")
    old_bias_name = joinname(prefix, "beta")
    weight_name = haskey(state_dict, old_weight_name) ? old_weight_name : joinname(prefix, "weight")
    bias_name = haskey(state_dict, old_bias_name) ? old_bias_name : joinname(prefix, "bias")
    ln_weight = getweight(one_init(dims), Array, state_dict, weight_name)
    ln_bias = getweight(zero_init(dims), Array, state_dict, bias_name)
    return TransformerLayers.LayerNorm(ln_weight, ln_bias, ln_ϵ)
end

_load_dense(state_dict, prefix, din, dout, factor, bias, act=nothing) =
    _load_dense(state_dict, prefix, weight_init(din, dout, factor), bias ? zero_init(dout) : nothing, act)
function _load_dense(state_dict, prefix, w_init, b_init, act=nothing)
    weight = getweight(w_init, Array, state_dict, joinname(prefix, "weight"))
    if isnothing(b_init)
        bias = nothing
    else
        bias = getweight(b_init, Array, state_dict, joinname(prefix, "bias"))
    end
    return TransformerLayers.Dense(act, weight, bias)
end
