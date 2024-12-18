using Transformers
using FuncPipelines, TextEncodeBase
using TextEncodeBase: nested2batch, nestedcall
using Flux, StatsBase

tkr = Transformers.HuggingFace.load_tokenizer("distilbert/distilbert-base-cased")
tkr = Transformers.TextEncoders.BertTextEncoder(tkr) do e
    e.process[1:5] |> Pipeline{:masked_position}(nested2batch ∘ nestedcall(isequal("[MASK]")), :token) |> e.process[6:end-1] |> PipeGet{(:token, :attention_mask, :masked_position)}()
end

model = Transformers.HuggingFace.load_model("distilbert/distilbert-base-cased", :ForMaskedLM)

query = "[MASK] is the Capital of France"
input = Transformers.TextEncoders.encode(tkr, query)

input_ids = input.masked_position

model_output = model(input)

mask_logits = model_output.logit[:, :, 1]

mask_probabilities = softmax(mask_logits, dims=1)
predicted_token_id = map(argmax, eachcol(mask_probabilities))

predicted_token = Transformers.TextEncoders.decode(tkr, predicted_token_id)[input_ids]