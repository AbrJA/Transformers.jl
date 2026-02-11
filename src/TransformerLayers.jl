module TransformerLayers

using StructWalk
using NeuralAttentionlib
using Flux
using Functors
using NNlib
using ChainRulesCore

export Seq2Seq, Transformer,
    TransformerBlock, TransformerDecoderBlock,
    PreNormTransformerBlock, PostNormTransformerBlock,
    PreNormTransformerDecoderBlock, PostNormTransformerDecoderBlock,
    Embed, EmbedDecoder, FixedLenPositionEmbed, SinCosPositionEmbed

export enable_gpu, todevice, togpudevice, tocpudevice

include("layers/utils.jl")
include("layers/architecture.jl")
include("layers/base.jl")
include("layers/embed.jl")
include("layers/layer.jl")
include("layers/attention_op.jl")
include("layers/structwalk.jl")
include("layers/testmode.jl")

include("layers/device.jl")
include("layers/loss.jl")

end
