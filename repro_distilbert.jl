
import Pkg
Pkg.activate(; temp=true)
Pkg.develop(path=".")
Pkg.add("Pickle")
Pkg.add("Test")
Pkg.resolve()

using Transformers
using Transformers.HuggingFace
using Transformers.HuggingFace: HGFConfig
using Test
using Pickle

@testset "DistilBert Reproduction (Tiny Random)" begin
    println("Loading DistilBert config...")
    model_name = "hf-tiny-model-private/tiny-random-DistilBertModel"
    cfg = Transformers.HuggingFace.load_config(model_name)

    println("Creating DistilBert Model...")
    model = Transformers.HuggingFace.load_model(model_name, :model; config=cfg)

    @test model isa Transformers.HuggingFace.HGFDistilBertModel

    println("Model created. Running forward pass...")
    input = (
        token=rand(1:100, 10, 2),
        attention_mask=ones(Int32, 10, 2)  # Fixed: Use Int32 instead of Int
    )
    output = model(input)
    @test haskey(output, :hidden_state)
    println("Forward pass successful.")

    # Test ForMaskedLM
    println("Testing ForMaskedLM...")
    lm_model = Transformers.HuggingFace.load_model(model_name, :formaskedlm; config=cfg)
    @test lm_model isa Transformers.HuggingFace.HGFDistilBertForMaskedLM
    lm_out = lm_model(input)
    @test haskey(lm_out, :logit)

    println("All DistilBert variants verified successfully!")
end
