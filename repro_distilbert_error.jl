using Transformers
using Transformers.HuggingFace
using Pickle

@info "Loading DistilBERT model..."
model_name = "hf-internal-testing/tiny-random-DistilBertModel"
cfg = HuggingFace.load_config(model_name; cache=false)
model = HuggingFace.load_model(:distilbert, model_name, :model; config=cfg, cache=false)

@info "Building input..."
input = (
    token=rand(1:100, 10, 2),
    attention_mask=ones(Int32, 10, 2),
)

@info "Running forward pass..."
try
    output = model(input)
    @info "Success!"
catch e
    @error "Forward pass failed" exception = (e, catch_backtrace())
end
