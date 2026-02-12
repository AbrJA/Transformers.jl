module HuggingFaceModels

using ..TransformerLayers
using ..TransformerTokenizers
using ..Interfaces

using HuggingFaceApi
using Flux

export @hgf_str,
    load_config,
    load_model,
    load_tokenizer,
    load_state_dict,
    load_hgf_pretrained,
    save_model

import ..Interfaces: load_model, load_config, load_tokenizer, save_model

include("huggingface/utils.jl")
include("huggingface/download.jl")
include("huggingface/weight.jl")
include("huggingface/configs/config.jl")
include("huggingface/models/models.jl")
include("huggingface/tokenizer/tokenizer.jl")


"""
    `hgf"<model-name>:<item>"`

Get `item` from `model-name`. This will ensure the required data are downloaded. `item` can be "config",
 "tokenizer", and model related like "Model", or "ForMaskedLM", etc. Use [`get_model_type`](@ref) to see what
 model/task are supported. If `item` is omitted, return a `Tuple` of `<model-name>:tokenizer` and `<model-name>:model`.
"""
macro hgf_str(name)
    :(load_hgf_pretrained($(esc(name))))
end

"""
  `load_hgf_pretrained(name)`

The underlying function of [`@hgf_str`](@ref).
"""
function load_hgf_pretrained(name; kw...)
    name_item = rsplit(name, ':'; limit=2)
    all = length(name_item) == 1
    model_name, item = if all
        name, "model"
    else
        Iterators.map(String, name_item)
    end
    item = lowercase(item)

    cfg = load_config(model_name; kw...)
    item == "config" && return cfg

    (item == "tokenizer" || all) &&
        (tkr = load_tokenizer(model_name; config=cfg, kw...))
    item == "tokenizer" && return tkr

    model = load_model(cfg.model_type, model_name, item; config=cfg, kw...)

    if all
        return tkr, model
    else
        return model
    end
end

end
