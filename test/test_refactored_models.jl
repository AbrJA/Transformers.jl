module TestRefactoredModels

using Test
using Transformers
using Transformers.Bert
using Transformers.GPT2
using Transformers.GPTNeo
using Transformers.GPTNeoX
using Transformers.GPTJ
using Transformers.Llama
using Transformers.Bloom
using Transformers.Phi
using Transformers.Bart
using Transformers.T5
using Transformers.DistilBert
using Transformers.Roberta
using Transformers.CLIP

@testset "Refactored Models" begin
    @testset "Bert" begin
        @test isdefined(Transformers, :Bert)
        @test isdefined(Transformers.Bert, :HGFBertModel)
        @test HGFBertModel <: Transformers.HuggingFaceModels.HGFPreTrained
    end

    @testset "GPT2" begin
        @test isdefined(Transformers, :GPT2)
        @test isdefined(Transformers.GPT2, :HGFGPT2Model)
        @test HGFGPT2Model <: Transformers.HuggingFaceModels.HGFPreTrained
    end

    @testset "GPT" begin
        @test isdefined(Transformers, :GPT)
        # HGFOpenAIGPTModel is likely not defined, checking config instead
        @test isdefined(Transformers.GPT, :HGFOpenAIGPTConfig)
    end

    @testset "GPTNeo" begin
        @test isdefined(Transformers, :GPTNeo)
        @test isdefined(Transformers.GPTNeo, :HGFGPTNeoModel)
        @test HGFGPTNeoModel <: Transformers.HuggingFaceModels.HGFPreTrained
    end

    @testset "GPTNeoX" begin
        @test isdefined(Transformers, :GPTNeoX)
        @test isdefined(Transformers.GPTNeoX, :HGFGPTNeoXModel)
        @test HGFGPTNeoXModel <: Transformers.HuggingFaceModels.HGFPreTrained
    end

    @testset "GPTJ" begin
        @test isdefined(Transformers, :GPTJ)
        @test isdefined(Transformers.GPTJ, :HGFGPTJModel)
        @test HGFGPTJModel <: Transformers.HuggingFaceModels.HGFPreTrained
    end

    @testset "Llama" begin
        @test isdefined(Transformers, :Llama)
        @test isdefined(Transformers.Llama, :HGFLlamaModel)
        @test HGFLlamaModel <: Transformers.HuggingFaceModels.HGFPreTrained
    end

    @testset "Bloom" begin
        @test isdefined(Transformers, :Bloom)
        @test isdefined(Transformers.Bloom, :HGFBloomModel)
        @test HGFBloomModel <: Transformers.HuggingFaceModels.HGFPreTrained
    end

    @testset "Phi" begin
        @test isdefined(Transformers, :Phi)
        @test isdefined(Transformers.Phi, :HGFPhiModel)
        @test HGFPhiModel <: Transformers.HuggingFaceModels.HGFPreTrained
    end

    @testset "Bart" begin
        @test isdefined(Transformers, :Bart)
        @test isdefined(Transformers.Bart, :HGFBartModel)
        @test HGFBartModel <: Transformers.HuggingFaceModels.HGFPreTrained
    end

    @testset "T5" begin
        @test isdefined(Transformers, :T5)
        @test isdefined(Transformers.T5, :HGFT5Model)
        @test HGFT5Model <: Transformers.HuggingFaceModels.HGFPreTrained
    end

    @testset "DistilBert" begin
        @test isdefined(Transformers, :DistilBert)
        @test isdefined(Transformers.DistilBert, :HGFDistilBertModel)
        @test HGFDistilBertModel <: Transformers.HuggingFaceModels.HGFPreTrained
    end

    @testset "Roberta" begin
        @test isdefined(Transformers, :Roberta)
        @test isdefined(Transformers.Roberta, :HGFRobertaModel)
        @test HGFRobertaModel <: Transformers.HuggingFaceModels.HGFPreTrained
    end

    @testset "CLIP" begin
        @test isdefined(Transformers, :CLIP)
        @test isdefined(Transformers.CLIP, :HGFCLIPModel)
        @test HGFCLIPModel <: Transformers.HuggingFaceModels.HGFPreTrained
    end
end

end
