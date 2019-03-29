module Basic

using Flux
using ..Transformers: Abstract3DTensor, Container, batchedmul

export PositionEmbedding, Embed, getmask, Vocabulary, gather
export OneHotArray, indices2onehot, onehot2indices, onehotarray
export Transformer, TransformerDecoder

export NNTopo, @nntopo_str, @nntopo
export Stack, show_stackfunc, stack

export @toNd

export logkldivergence, logcrossentropy, logsoftmax3d

include("./extend3d.jl")
include("./stack/topology.jl")
include("./stack/stack.jl")
include("./embed/Embed.jl")
include("./mh_atten.jl")
include("./transformer.jl")
include("./loss.jl")

end
