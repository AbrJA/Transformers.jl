module TransformersCUDAExt

using Transformers
using Transformers.Flux
using CUDA

# Lazy-load FluxCUDAExt to avoid precompilation failure (Issue #214)
# Base.get_extension returns nothing during precompilation, so we
# defer the lookup to runtime.
function Transformers._toxdevice(adaptor::Flux.FluxCUDAAdaptor, x, cache)
    FluxCUDAExt = Base.get_extension(Flux, :FluxCUDAExt)
    if isnothing(FluxCUDAExt)
        error("FluxCUDAExt not loaded. Make sure CUDA.jl is properly installed and loaded.")
    end
    Transformers.__toxdevice(adaptor, cache, x, Flux._isleaf, FluxCUDAExt.check_use_cuda)
end

end
