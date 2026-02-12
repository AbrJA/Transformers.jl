using Transformers
using Transformers.HuggingFace
using Transformers.TextEncoders

# Load tokenizer
tkr = load_tokenizer("distilbert-base-uncased")

# Inspect tokenizer for mask token
println("Type of tokenizer: ", typeof(tkr))
println("Field names: ", fieldnames(typeof(tkr)))

# Try to find mask token
mask_token = "[MASK]" # This is for BERT/DistilBERT. RoBERTa uses <mask>
println("Mask token: ", mask_token)

# Encode a sentence with mask
text = "The quick brown [MASK] jumps over the lazy dog."
encoded = encode(tkr, text)

println("Encoded tokens: ", encoded.token)
println("Encoded keys: ", keys(encoded))

# Find mask index
vocab = tkr.vocab
mask_id = vocab[mask_token] # lookup_index(vocab, mask_token) usually works for Vocab
println("Mask ID from vocab: ", mask_id)

mask_indices = findall(x -> x == mask_id, encoded.token)
println("Mask indices: ", mask_indices)


# Load model
# Use correct task name for DistilBert
model = load_model("distilbert-base-uncased"; task=:formaskedlm) # explicit task

# Move to CPU for testing
using Flux
model = cpu(model)

# Test pipeline logic
input = encoded
output = model(input)
logits = output.logit

println("Logits type: ", typeof(logits))
println("Logits size: ", size(logits))

# Extract logits for mask
# Julia is column-major: (vocab, seq, batch)
for idx in mask_indices
    # idx is CartesianIndex or LinearIndex?
    # encoded.token is a Matrix (seq_len, batch) usually? No, OneHotArray or indices.
    # TextEncoders usually return indices 1-based.

    # If encoded.token is (seq_len, batch), then:
    # logits is (vocab, seq_len, batch)

    # Let's check dimensions
    println("Logits size: ", size(logits))

    # Get logits for the first mask
    # We need the column index of the mask in the sequence
    seq_idx = idx[1]
    batch_idx = idx[2]

    mask_logits = logits[:, seq_idx, batch_idx]
    println("Mask logits size: ", size(mask_logits))

    # Top-k
    k = 5
    top_indices = partialsortperm(mask_logits, 1:k, rev=true)
    top_scores = softmax(mask_logits)[top_indices]
    top_tokens = [vocab[i] for i in top_indices]

    println("Top predictions:")
    for (t, s) in zip(top_tokens, top_scores)
        println("  $t: $s")
    end
end
