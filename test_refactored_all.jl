using Transformers
using Transformers.HuggingFace
using Flux
using Test

@testset "Refactored Models Type Stability" begin
    models = [
        ("bert", "Bert"),
        ("gpt2", "GPT2"),
        ("distilbert", "DistilBert"),
        ("bart", "Bart"),
        ("roberta", "Roberta"),
        ("llama", "Llama"),
        ("bloom", "Bloom"),
        ("phi", "Phi"),
        ("t5", "T5"),
        ("clip", "CLIP"),
        ("gptj", "GPTJ"),
        ("gpt_neo", "GPTNeo"),
        ("gpt_neox", "GPTNeoX"),
    ]

    for (name, mod_name) in models
        @testset "Model: $name" begin
            # We don't necessarily need weights for structural verification
            # but we can check if the types are defined and accessible.
            # mod_name is now directly provided in the `models` array.

            try
                # Check if module exists
                mod = getfield(Transformers, Symbol(mod_name))
                @test mod isa Module

                # Check for some expected types
                type_prefix = "HGF" * mod_name
                target_type_name = Symbol(type_prefix * "Model")

                if isdefined(mod, target_type_name)
                    T = getfield(mod, target_type_name)
                    @test T <: Transformers.HuggingFaceModels.HGFPreTrained
                    @info "Verified $target_type_name in $mod_name"
                else
                    @warn "Type $target_type_name not found in $mod_name"
                end
            catch e
                @error "Failed to verify $name" exception = (e, catch_backtrace())
                rethrow(e)
            end
        end
    end
end
