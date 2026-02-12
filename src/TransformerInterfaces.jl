module TransformerInterfaces

using ..Transformers: Transformers

export AbstractTransformerModel, AbstractTransformerConfig, AbstractTransformerTokenizer
export load_model, load_config, load_tokenizer, save_model
export forward, get_encoder, get_embeddings
export get_vocab, get_eos_id, get_pad_id

"""
    AbstractTransformerModel

Abstract base type for all Transformer models.
"""
abstract type AbstractTransformerModel end

"""
    AbstractTransformerConfig

Abstract base type for all Transformer configurations.
"""
abstract type AbstractTransformerConfig <: AbstractDict{Symbol,Any} end

"""
    AbstractTransformerTokenizer

Abstract base type for all Transformer tokenizers.
"""
abstract type AbstractTransformerTokenizer end

"""
    load_model(model_type, model_name; kwargs...)

Load a model of type `model_type` from `model_name`.
"""
function load_model end

"""
    load_config(model_name; kwargs...)

Load a configuration from `model_name`.
"""
function load_config end

"""
    load_tokenizer(model_name; kwargs...)

Load a tokenizer from `model_name`.
"""
function load_tokenizer end

"""
    save_model(model, path; kwargs...)

Save a model to `path`.
"""
function save_model end

"""
    forward(model, input)

Forward pass of the model.
"""
function forward end

"""
    get_encoder(model)

Get the encoder component of the model.
"""
function get_encoder end

"""
    get_embeddings(model)

Get the embeddings component of the model.
"""
function get_embeddings end

"""
    get_vocab(tokenizer)

Get the vocabulary of the tokenizer.
"""
function get_vocab end

"""
    get_eos_id(tokenizer)

Get the end-of-sequence token ID.
"""
function get_eos_id end

"""
    get_pad_id(tokenizer)

Get the padding token ID.
"""
function get_pad_id end

end
