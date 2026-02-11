module TransformerDatasets

using DataDeps
using HTTP
using WordTokenizers
using Fetch

const Container{T} = Union{NTuple{N,T},Vector{T}} where N

export Dataset, Train, Dev, Test
export dataset, datafile, get_batch, get_vocab, get_labels, batched


include("datasets/dataset.jl")

include("datasets/translate/wmt.jl")
using .WMT

include("datasets/translate/iwslt.jl")
using .IWSLT

include("datasets/qa/clozetest.jl")
using .ClozeTest

include("datasets/glue/glue.jl")
using .GLUE

end
