module Roberta

using Flux
using ..Transformers: Transformers
using ..Interfaces
using ..TransformerLayers
using ..HuggingFaceModels
using ..Bert

using ..HuggingFaceModels: HGFConfig, HGFPreTrained, isbasemodel, haskeystartswith, joinname, getweight, weight_init, zero_init, one_init, load_error, ACT2FN, @hgfcfg, _get_model_type

import ..Interfaces: load_model
import ..HuggingFaceModels: basemodelkey, get_state_dict, get_model_type

export HGFRobertaModel, HGFRobertaForMaskedLM, HGFRobertaForCausalLM, HGFRobertaForSequenceClassification, HGFRobertaForTokenClassification, HGFRobertaForQuestionAnswering

include("config.jl")
include("load.jl")

end
