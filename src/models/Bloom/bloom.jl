module Bloom

using Flux
using ..Transformers: Transformers
using ..Transformers: Transformers
using ..TransformerInterfaces
using ..TransformerLayers
using ..HuggingFaceModels

using ..HuggingFaceModels: HGFConfig, HGFPreTrained, haskeystartswith, joinname, getweight, weight_init, zero_init, one_init, load_error, ACT2FN, @hgfcfg

import ..TransformerInterfaces: load_model
import ..HuggingFaceModels: basemodelkey, get_state_dict, isbasemodel

export HGFBloomModel, HGFBloomForCausalLM

include("config.jl")
include("load.jl")

end
