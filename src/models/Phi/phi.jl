module Phi

using Flux
using ..Transformers: Transformers
using ..Interfaces
using ..TransformerLayers
using ..HuggingFaceModels
using ..GPTNeoX: CausalGPTNeoXRoPEMultiheadQKVAttenOp
using ..Llama: CausalLlamaRoPEGroupedQueryAttenOp

using ..HuggingFaceModels: HGFConfig, HGFPreTrained, isbasemodel, haskeystartswith, joinname, getweight, weight_init, zero_init, one_init, load_error, ACT2FN, @hgfcfg

import ..Interfaces: load_model
import ..HuggingFaceModels: basemodelkey, get_state_dict

export HGFPhiModel, HGFPhiForCausalLM

include("config.jl")
include("load.jl")

end
