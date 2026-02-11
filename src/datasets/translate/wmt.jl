module WMT
using Fetch
using DataDeps
using BytePairEncoding

import ..Dataset
import ..testfile, ..trainfile, ..get_vocab

export GoogleWMT

function __init__()
    googlewmt_init()
end

include("./google_wmt.jl")


end
