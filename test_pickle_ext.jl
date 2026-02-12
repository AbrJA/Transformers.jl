
using Transformers
using Transformers.HuggingFaceModels
using Pickle
using Test

@testset "Pickle Extension" begin
    # Verify hooks are populated
    @test HuggingFaceModels.LOAD_PICKLE[] != nothing
    @test HuggingFaceModels.SAVE_PICKLE[] != nothing
    @test HuggingFaceModels.IS_PICKLE[] != nothing
    @test HuggingFaceModels.UNWRAP_PICKLE[] != nothing

    # Verify definition matches expected function
    @test HuggingFaceModels.LOAD_PICKLE[] == Pickle.Torch.THload
    @test HuggingFaceModels.IS_PICKLE[] == Pickle.Torch.islazy

    println("Pickle extension loaded successfully!")
end
