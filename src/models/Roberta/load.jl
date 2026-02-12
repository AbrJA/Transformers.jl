using ..TransformerLayers
using ..TransformerLayers: CompositeEmbedding, SelfAttention, MultiheadQKVAttenOp, @fluxshow, @fluxlayershow
using ChainRulesCore
using Functors
using FillArrays
using NNlib
using Static
using Flux

using NeuralAttentionlib
using NeuralAttentionlib: $, WithScore

# RoBERTa Model
struct HGFRobertaModel{E, En, P} <: HGFPreTrained{:roberta, :model}
    embed::E
    encoder::En
    pooler::P
end
@fluxshow HGFRobertaModel

function (model::HGFRobertaModel)(nt::NamedTuple)
    outputs = model.encoder(model.embed(nt))
    if isnothing(model.pooler)
        return outputs
    else
        return model.pooler(outputs)
    end
end

# RoBERTa For Masked LM
struct HGFRobertaForMaskedLM{M, C} <: HGFPreTrained{:roberta, :formaskedlm}
    model::M
    cls::C
end
@fluxlayershow HGFRobertaForMaskedLM

(model::HGFRobertaForMaskedLM)(nt::NamedTuple) = model.cls(model.model(nt))

# RoBERTa For Causal LM
struct HGFRobertaForCausalLM{M, C} <: HGFPreTrained{:roberta, :forcausallm}
    model::M
    cls::C
end
@fluxlayershow HGFRobertaForCausalLM

(model::HGFRobertaForCausalLM)(nt::NamedTuple) = model.cls(model.model(nt))

# RoBERTa For Sequence Classification
struct HGFRobertaForSequenceClassification{M, C} <: HGFPreTrained{:roberta, :forsequenceclassification}
    model::M
    cls::C
end
@fluxlayershow HGFRobertaForSequenceClassification

(model::HGFRobertaForSequenceClassification)(nt::NamedTuple) = model.cls(model.model(nt))

# RoBERTa For Token Classification
struct HGFRobertaForTokenClassification{M, C} <: HGFPreTrained{:roberta, :fortokenclassification}
    model::M
    cls::C
end
@fluxlayershow HGFRobertaForTokenClassification

(model::HGFRobertaForTokenClassification)(nt::NamedTuple) = model.cls(model.model(nt))

# RoBERTa For Question Answering
struct HGFRobertaForQuestionAnswering{M, C} <: HGFPreTrained{:roberta, :forquestionanswering}
    model::M
    cls::C
end
@fluxlayershow HGFRobertaForQuestionAnswering

(model::HGFRobertaForQuestionAnswering)(nt::NamedTuple) = model.cls(model.model(nt))

const HGFRobertaPreTrainedModel = HGFPreTrained{:roberta}


basemodelkey(::Type{<:HGFPreTrained{:roberta}}) = :roberta

roberta_pe_indices(pad_id, x) = Base.OneTo{Int32}(size(x, 2)) .+ Int32(pad_id + 1)

function load_model(::Type{HGFRobertaModel}, cfg, state_dict, prefix)
    model = load_model(HGFBertModel, cfg, state_dict, prefix)
    pad_id = cfg[:pad_token_id]
    token = model.embed[1].token
    position = model.embed[1].position
    segment = model.embed[1].segment
    ln = model.embed[2]
    embed = TransformerLayers.Chain(CompositeEmbedding(
        token = token,
        position = (position.apply, position.embed, roberta_pe_indices $ pad_id),
        segment = segment
    ), ln)
    return HGFRobertaModel(embed, model.encoder, model.pooler)
end

function load_model(
    _type::Type{<:Union{
        HGFRobertaForCausalLM, HGFRobertaForMaskedLM, HGFRobertaForSequenceClassification,
        HGFRobertaForTokenClassification, HGFRobertaForQuestionAnswering,
    }},
    ::Type{HGFRobertaModel}, cfg, state_dict, prefix)
    embed = load_model(HGFBertModel, CompositeEmbedding, cfg, state_dict, joinname(prefix, "embeddings"))
    pad_id = cfg[:pad_token_id]
    token = embed[1].token
    position = embed[1].position
    segment = embed[1].segment
    ln = embed[2]
    embed = TransformerLayers.Chain(CompositeEmbedding(
        token = token,
        position = (position.apply, position.embed, roberta_pe_indices $ pad_id),
        segment = segment
    ), ln)
    encoder = load_model(HGFBertModel, TransformerBlock,  cfg, state_dict, joinname(prefix, "encoder"))
    return HGFRobertaModel(embed, encoder, nothing)
end

function load_model(_type::Type{<:Union{HGFRobertaForCausalLM, HGFRobertaForMaskedLM}}, cfg, state_dict, prefix)
    model = load_model(_type, HGFRobertaModel, cfg, state_dict, joinname(prefix, "roberta"))
    dims, vocab_size = cfg[:hidden_size], cfg[:vocab_size]
    factor = Float32(cfg[:initializer_range])
    head_weight = getweight(weight_init(dims, dims, factor), Array, state_dict, joinname(prefix, "lm_head.dense.weight"))
    head_bias = getweight(zero_init(dims), Array, state_dict, joinname(prefix, "lm_head.dense.bias"))
    head_ln = load_model(HGFBertModel, TransformerLayers.LayerNorm, cfg, state_dict, joinname(prefix, "lm_head.layer_norm"))
    if cfg[:tie_word_embeddings]
        embedding = model.embed[1].token.embeddings
    else
        embedding = getweight(TransformerLayers.Embed, state_dict, joinname(prefix, "lm_head.decoder.weight")) do
            weight = weight_init(vocab_size, dims, factor)()
            weight[:, pad_id+1] .= 0
            return weight
        end
    end
    bias = getweight(zero_init(vocab_size), Array, state_dict, joinname(prefix, "lm_head.bias"))
    lmhead = TransformerLayers.Chain(TransformerLayers.Dense(NNlib.gelu, head_weight, head_bias), head_ln,
                          TransformerLayers.EmbedDecoder(TransformerLayers.Embed(embedding), bias))
    return _type(model, TransformerLayers.Branch{(:logit,), (:hidden_state,)}(lmhead))
end

function load_model(_type::Type{HGFRobertaForSequenceClassification}, cfg, state_dict, prefix)
    model = load_model(_type, HGFRobertaModel, cfg, state_dict, joinname(prefix, "roberta"))
    dims, nlabel = cfg[:hidden_size], cfg[:num_labels]
    factor = Float32(cfg[:initializer_range])
    p = cfg[:classifier_dropout]
    dense_weight = getweight(weight_init(dims, dims, factor), Array,
                            state_dict, joinname(prefix, "classifier.dense.weight"))
    dense_bias = getweight(zero_init(dims), Array, state_dict, joinname(prefix, "classifier.dense.bias"))
    proj_weight = getweight(weight_init(dims, nlabel, factor), Array,
                            state_dict, joinname(prefix, "classifier.out_proj.weight"))
    proj_bias = getweight(zero_init(nlabel), Array, state_dict, joinname(prefix, "classifier.out_proj.bias"))
    head = TransformerLayers.Chain(
        TransformerLayers.DropoutLayer(TransformerLayers.FirstTokenPooler(), p),
        TransformerLayers.DropoutLayer(TransformerLayers.Dense(NNlib.tanh_fast, dense_weight, dense_bias), p),
        TransformerLayers.Dense(proj_weight, proj_bias))
    cls = TransformerLayers.Branch{(:logit,), (:hidden_state,)}(head)
    return HGFRobertaForSequenceClassification(model, cls)
end

function load_model(_type::Type{HGFRobertaForTokenClassification}, cfg, state_dict, prefix)
    model = load_model(_type, HGFRobertaModel, cfg, state_dict, joinname(prefix, "roberta"))
    dims, nlabel = cfg[:hidden_size], cfg[:num_labels]
    factor = Float32(cfg[:initializer_range])
    p = cfg[:classifier_dropout]
    weight = getweight(weight_init(dims, nlabel, factor), Array, state_dict, joinname(prefix, "classifier.weight"))
    bias = getweight(zero_init(nlabel), Array, state_dict, joinname(prefix, "classifier.bias"))
    head = TransformerLayers.Chain(TransformerLayers.DropoutLayer(identity, p), TransformerLayers.Dense(weight, bias))
    cls = TransformerLayers.Branch{(:logit,), (:hidden_state,)}(head)
    return HGFRobertaForTokenClassification(model, cls)
end

function load_model(_type::Type{HGFRobertaForQuestionAnswering}, cfg, state_dict, prefix)
    model = load_model(_type, HGFRobertaModel, cfg, state_dict, joinname(prefix, "roberta"))
    dims, nlabel = cfg[:hidden_size], cfg[:num_labels]
    factor = Float32(cfg[:initializer_range])
    weight = getweight(weight_init(dims, nlabel, factor), Array, state_dict, joinname(prefix, "qa_outputs.weight"))
    bias = getweight(zero_init(nlabel), Array, state_dict, joinname(prefix, "qa_outputs.bias"))
    cls = Bert.BertQA(TransformerLayers.Dense(weight, bias))
    return HGFRobertaForQuestionAnswering(model, cls)
end


function get_state_dict(m::HGFRobertaModel, state_dict, prefix)
    get_state_dict(HGFBertModel, m.embed[1], state_dict, joinname(prefix, "embeddings"))
    get_state_dict(HGFBertModel, m.embed[2], state_dict, joinname(prefix, "embeddings.LayerNorm"))
    get_state_dict(HGFBertModel, m.encoder, state_dict, joinname(prefix, "encoder"))
    if !isnothing(m.pooler)
        get_state_dict(HGFBertModel, m.pooler.layer.dense, state_dict, joinname(prefix, "pooler.dense"))
    end
    return state_dict
end

function get_state_dict(m::Union{HGFRobertaForCausalLM, HGFRobertaForMaskedLM}, state_dict, prefix)
    get_state_dict(m.model, state_dict, joinname(prefix, "roberta"))
    get_state_dict(HGFBertModel, m.cls.layer[1], state_dict, joinname(prefix, "lm_head.dense"))
    get_state_dict(HGFBertModel, m.cls.layer[2], state_dict, joinname(prefix, "lm_head.layer_norm"))
    get_state_dict(HGFBertModel, m.cls.layer[3], state_dict, joinname(prefix, "lm_head"))
    return state_dict
end

function get_state_dict(m::HGFRobertaForSequenceClassification, state_dict, prefix)
    get_state_dict(m.model, state_dict, joinname(prefix, "roberta"))
    get_state_dict(m, m.cls.layer[2], state_dict, joinname(prefix, "classifier.dense"))
    get_state_dict(m, m.cls.layer[3], state_dict, joinname(prefix, "classifier.out_proj"))
    return state_dict
end

function get_state_dict(m::HGFRobertaForTokenClassification, state_dict, prefix)
    get_state_dict(m.model, state_dict, joinname(prefix, "roberta"))
    get_state_dict(m, m.cls.layer[2], state_dict, joinname(prefix, "classifier"))
    return state_dict
end

function get_state_dict(m::HGFRobertaForQuestionAnswering, state_dict, prefix)
    get_state_dict(m.model, state_dict, joinname(prefix, "roberta"))
    get_state_dict(m, m.cls.dense, state_dict, joinname(prefix, "qa_outputs"))
    return state_dict
end
