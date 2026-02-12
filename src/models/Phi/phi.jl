module Phi

using Flux
using ..Transformers: Transformers
using ..TransformerInterfaces
using ..TransformerLayers
using ..HuggingFaceModels
using ..GPTNeoX: CausalGPTNeoXRoPEMultiheadQKVAttenOp
using ..TransformerInterfaces

using ..HuggingFaceModels: HGFConfig, HGFPreTrained, isbasemodel, haskeystartswith, joinname, getweight, weight_init, zero_init, one_init, load_error, ACT2FN, @hgfcfg

import ..TransformerInterfaces: load_model
import ..HuggingFaceModels: basemodelkey, get_state_dict

export HGFPhiModel, HGFPhiForCausalLM

include("config.jl")
include("load.jl")

end
