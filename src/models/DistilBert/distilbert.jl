module DistilBert

using Flux
using ..Transformers: Transformers
using ..TransformerInterfaces
using ..TransformerLayers
using ..HuggingFaceModels

using ..HuggingFaceModels: HGFConfig, HGFPreTrained, haskeystartswith, joinname, getweight, weight_init, zero_init, one_init, load_error, ACT2FN, @hgfcfg

import ..TransformerInterfaces: load_model
import ..HuggingFaceModels: basemodelkey, get_state_dict, get_model_type, isbasemodel

export HGFDistilBertModel, HGFDistilBertForCausalLM, HGFDistilBertForMaskedLM, HGFDistilBertForSequenceClassification, HGFDistilBertForTokenClassification, HGFDistilBertForQuestionAnswering

include("config.jl")
include("load.jl")
include("tokenizer.jl")

end
