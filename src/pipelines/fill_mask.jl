using ..Transformers
using ..TransformerLayers
using ..TransformerTokenizers
using ..HuggingFaceModels

using FuncPipelines
using TextEncodeBase
using TextEncodeBase: nested2batch, nestedcall

struct FillMaskPipeline{M,T} <: AbstractPipeline
    model::M
    tokenizer::T
    top_k::Int
end

function FillMaskPipeline(
    model::Union{Nothing,HuggingFaceModels.HGFPreTrained}=nothing,
    config=nothing,
    tokenizer=nothing;
    top_k=5,
    kwargs...
)
    if isnothing(model) || isnothing(tokenizer)
        error("Model and tokenizer must be provided for now.")
    end

    # Handle tokenizer modification
    # We want to insert a mask extraction step into the tokenizer pipeline
    new_tkr = Transformers.TextEncoders.TransformerTextEncoder(tokenizer) do e
        # Helper to safely slice based on structure
        # Standard pipeline has nested2batch at index 6 usually.
        # We search for it.
        pipes = e.process.pipes
        idx = findfirst(p -> typeof(p).parameters[1] == :token && occursin("nested2batch", string(p)), pipes)

        # Fallback to user hardcoded indices if search fails (or just use 5 if we trust standard)
        split_idx = isnothing(idx) ? 5 : idx - 1

        # Get mask token from tokenizer or config
        # TODO: Generic mask token retrieval. For now assume [MASK] or <mask, try to guess
        mask_token = "[MASK]"
        if hasfield(typeof(e.tokenizer), :syms) # specialized tokenizer
            # ...
        end
        # RoBERTa uses <mask>, BERT uses [MASK]
        # We can check vocab
        if !("[MASK]" in e.vocab) && "<mask>" in e.vocab
            mask_token = "<mask>"
        end

        # Construct new pipeline
        # 1. Pre-mask steps
        pre = e.process[1:split_idx]

        # 2. Mask extraction step
        # We need to perform nested2batch on the mask boolean array
        mask_pipe = Pipeline{:masked_position}(nested2batch ∘ nestedcall(isequal(mask_token)), :token)

        # 3. Post-mask steps (excluding the final tuple construction usually)
        # The standard pipeline ends with PipeGet or implicit tuple?
        # Standard: ... -> nested2batch -> sequence_mask -> (token, attention_mask)
        # We want to replace the final return with our tuple.

        # e.process[split_idx+1:end-1] keeps steps after insertion but drops the last one
        post = e.process[split_idx+1:end-1]

        pre |> mask_pipe |> post |> PipeGet{(:token, :attention_mask, :masked_position)}()
    end

    return FillMaskPipeline(model, new_tkr, top_k)
end

function (p::FillMaskPipeline)(input::AbstractString)
    # 1. Tokenize (now returns masked_position too)
    encoded = encode(p.tokenizer, input)

    # 2. Forward pass
    output = p.model(encoded)
    logits = output.logit # (vocab, seq, batch)

    # 3. Decode
    # encoded.masked_position is a OneHotArray or Boolean Matrix?
    # nested2batch on booleans -> Matrix{Bool} usually
    mask_bool = encoded.masked_position

    # We need to extract logits where mask_bool is true
    # logits: [vocab, seq, batch]
    # mask_bool: [seq, batch]

    # Get indices where mask is true
    # Julia column major: indices are (seq_idx, batch_idx)
    # logits are (vocab_idx, seq_idx, batch_idx)

    # We can iterate batches
    results = []

    # Assuming batch size 1 for single string input
    # But encode handles list too?
    # For now support single string

    # To be generic:
    B = size(mask_bool, 2)
    for b in 1:B
        # mask_bool[:, b] is vector for this seq
        # find masks
        seq_mask = mask_bool[:, b]
        mask_indices = findall(seq_mask) # Indices in sequence

        batch_results = []
        for seq_idx in mask_indices
            # Get logits for this position
            # logits[:, seq_idx, b]
            token_logits = logits[:, seq_idx, b]

            # Top-k
            top_indices = partialsortperm(token_logits, 1:p.top_k, rev=true)
            top_scores = softmax(token_logits)[top_indices]
            top_tokens = decode(p.tokenizer, top_indices)

            push!(batch_results, collect(zip(top_tokens, top_scores)))
        end
        push!(results, batch_results)
    end

    # Flatten if single input
    return length(results) == 1 ? results[1] : results
end
