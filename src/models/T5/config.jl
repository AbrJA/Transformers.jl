@hgfcfg :t5 struct HGFT5Config
    vocab_size::Int = 32128
    d_model::Int = 512
    d_kv::Int = 64
    d_ff::Int = 2048
    num_layers::Int = 6
    num_decoder_layers::Any = nothing
    num_heads::Int = 8
    relative_attention_num_buckets::Int = 32
    relative_attention_max_distance::Int = 128
    dropout_rate::Float64 = 0.1
    layer_norm_epsilon::Float64 = 1e-6
    initializer_factor::Float64 = 1.0
    feed_forward_proj::String = "relu"
    is_encoder_decoder::Bool = true
    use_cache::Bool = true
    pad_token_id::Int = 0
    eos_token_id::Int = 1
    dense_act_fn::String = "relu"
    is_gated_act::Bool = false
    tie_word_embeddings::Bool = true
end

function HGFConfig{:t5}(cfg, overwrite)
    if haskey(cfg, :feed_forward_proj)
        feed_forward_proj = cfg[:feed_forward_proj]
        if feed_forward_proj == "gated-gelu"
            overwrite[:dense_act_fn] = "gelu"
            overwrite[:is_gated_act] = true
        elseif feed_forward_proj == "gated-silu"
            overwrite[:dense_act_fn] = "silu"
            overwrite[:is_gated_act] = true
        end
    end
    return HGFConfig(:t5, cfg, overwrite)
end
