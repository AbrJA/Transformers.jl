module IWSLT
using Fetch
using DataDeps

import ..Dataset
import ..testfile, ..devfile, ..trainfile
import ..token_freq, ..get_vocab

export IWSLT2016

function __init__()
    iwslt2016_init()
end

include("./iwslt2016.jl")


end
