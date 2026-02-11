using Transformers
using Flux
using Test

@testset "Modules Loading" begin
    # Check submodule definitions
    @test isdefined(Transformers, :TransformerLayers)
    @test isdefined(Transformers, :TransformerTokenizers)
    @test isdefined(Transformers, :HuggingFaceModels)
    @test isdefined(Transformers, :TransformerDatasets)

    # Check backward compatibility aliases
    @test isdefined(Transformers, :Layers)
    @test isdefined(Transformers, :TextEncoders)
    @test isdefined(Transformers, :HuggingFace)
    @test isdefined(Transformers, :Datasets)

    @test Transformers.Layers === Transformers.TransformerLayers
    @test Transformers.TextEncoders === Transformers.TransformerTokenizers.TextEncoders
    @test Transformers.HuggingFace === Transformers.HuggingFaceModels
    @test Transformers.Datasets === Transformers.TransformerDatasets

    # Test basic functionality accessibility
    @test Transformers.Layers.TransformerBlock isa Type
    @test Transformers.TextEncoders.BertTextEncoder isa Type
    @test Transformers.HuggingFace.load_model isa Function

    # Test device functionality from TransformerLayers
    @test Transformers.enable_gpu isa Function
    @test Transformers.todevice isa Function

    # Test re-exported symbols
    @test isdefined(Transformers, :TransformerBlock) # Should be re-exported
    @test isdefined(Transformers, :enable_gpu) # Should be re-exported

    println("Modules loaded successfully!")
end
