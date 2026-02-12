@hgfcfg :bart struct HGFBartConfig
    vocab_size::Int = 50265
    d_model::Int = 1024
    encoder_layers::Int = 12
    decoder_layers::Int = 12
    encoder_attention_heads::Int = 16
    decoder_attention_heads::Int = 16
    encoder_ffn_dim::Int = 4096
    decoder_ffn_dim::Int = 4096
    activation_function::String = "gelu"
    dropout::Float64 = 0.1
    attention_dropout::Float64 = 0.0
    activation_dropout::Float64 = 0.0
    classifier_dropout::Float64 = 0.0
    max_position_embeddings::Int = 1024
    init_std::Float64 = 0.02
    is_encoder_decoder::Bool = true
    pad_token_id::Int = 1
    bos_token_id::Int = 0
    eos_token_id::Int = 2
    decoder_start_token_id::Int = 2
    forced_eos_token_id::Int = 2
    scale_embedding::Bool = false
    use_cache::Bool = true
    num_labels::Int = 3
    id2label::Any = nothing
    label2id::Any = nothing
end
