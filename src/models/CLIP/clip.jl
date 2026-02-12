module CLIP

using Flux
using ..Transformers: Transformers
using ..Interfaces
using ..TransformerLayers
using ..HuggingFaceModels

using ..HuggingFaceModels: HGFConfig, HGFPreTrained, haskeystartswith, joinname, getweight, weight_init, zero_init, one_init, load_error, ACT2FN, @hgfcfg, _get_model_type

import ..Interfaces: load_model
import ..HuggingFaceModels: basemodelkey, get_state_dict, get_model_type, isbasemodel

export HGFCLIPTextModel, HGFCLIPVisionModel, HGFCLIPModel, HGFCLIPTextModelWithProjection, HGFCLIPVisionModelWithProjection

include("config.jl")
include("model.jl")
include("load.jl")

end
