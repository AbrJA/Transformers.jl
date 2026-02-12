module Bloom

using Flux
using ..Transformers: Transformers
using ..Interfaces
using ..TransformerLayers
using ..HuggingFaceModels

using ..HuggingFaceModels: HGFConfig, HGFPreTrained, isbasemodel, haskeystartswith, joinname, getweight, weight_init, zero_init, one_init, load_error, ACT2FN, @hgfcfg

import ..Interfaces: load_model
import ..HuggingFaceModels: basemodelkey, get_state_dict

export HGFBloomModel, HGFBloomForCausalLM

include("config.jl")
include("load.jl")

end
