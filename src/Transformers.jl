module Transformers

using Flux

# Core Interfaces
include("Interfaces.jl")
using .Interfaces
export Interfaces

# Re-export core modules
include("TransformerLayers.jl")
using .TransformerLayers
export TransformerLayers

# Re-export tokenizers
include("TransformerTokenizers.jl")
using .TransformerTokenizers
export TransformerTokenizers

# Re-export datasets
include("TransformerDatasets.jl")
using .TransformerDatasets
export TransformerDatasets

# Re-export HuggingFace models
include("HuggingFaceModels.jl")
using .HuggingFaceModels
export HuggingFaceModels

# Re-export Pipelines
include("pipelines/TransformerPipelines.jl")
using .TransformerPipelines
export TransformerPipelines

# Models (New Modular Structure)
include("models/Bert/bert.jl")
using .Bert
export Bert

include("models/GPT2/gpt2.jl")
using .GPT2
export GPT2

include("models/GPT/gpt.jl")
using .GPT
export GPT

include("models/GPTNeo/gpt_neo.jl")
using .GPTNeo
export GPTNeo

include("models/GPTNeoX/gpt_neox.jl")
using .GPTNeoX
export GPTNeoX

include("models/GPTJ/gptj.jl")
using .GPTJ
export GPTJ

include("models/Llama/llama.jl")
using .Llama
export Llama

include("models/Bloom/bloom.jl")
using .Bloom
export Bloom

include("models/Phi/phi.jl")
using .Phi
export Phi

include("models/Bart/bart.jl")
using .Bart
export Bart

include("models/T5/t5.jl")
using .T5
export T5

include("models/DistilBert/distilbert.jl")
using .DistilBert
export DistilBert

include("models/Roberta/roberta.jl")
using .Roberta
export Roberta

include("models/CLIP/clip.jl")
using .CLIP
export CLIP

# Backward compatibility aliases and exports
const Layers = TransformerLayers

const HuggingFace = HuggingFaceModels
const Datasets = TransformerDatasets

export Layers, TextEncoders, HuggingFace, Datasets

# Re-export commonly used symbols from submodules (to match old API)
# Layers
for n in names(Layers; all=false)
    if Base.isexported(Layers, n)
        @eval export $n
    end
end

# TextEncoders?
# Old Transformers exported TextEncoders module, but not its symbols directly?
# Transformers.jl (old) had `using .TextEncoders`.
# If `using .TextEncoders`, do symbols get exported? No. Only if `export` explicitly lists them.
# Checked Transformers.jl (old): `export Layers, TextEncoders, HuggingFace, Masks`
# It did NOT export `encode`, `decode` etc. Users used `TextEncoders.encode`.
# So exporting the module `TextEncoders` is enough.
# Which is done via `const TextEncoders = ...` and `export TextEncoders`.

# What about `enable_gpu`, `todevice`?
# They are in `TransformerLayers`.
# Verify if they are exported by `TransformerLayers`. Yes, I added them.
# So the loop above re-exports them.

# Masks?
# NeuralAttentionlib exports Masks.
# Old Tranformers exported Masks.
# `TransformerLayers` uses NeuralAttentionlib but doesn't explicitly export Masks?
# Let's check `TransformerLayers.jl` (created in step 405).
# It exports `Seq2Seq` etc. It does NOT export `Masks`.
# So usage `Transformers.Masks` might fail.
# I should export `Masks` from `TransformerLayers` or `Transformers.jl`.
# I'll add `using NeuralAttentionlib: Masks` and `export Masks` to `Transformers.jl`.

using NeuralAttentionlib: Masks
export Masks

end # module
