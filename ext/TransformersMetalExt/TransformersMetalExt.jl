module TransformersMetalExt

using Transformers
using Transformers.Flux
using Metal

# Lazy-load FluxMetalExt to avoid precompilation failure
function Transformers._toxdevice(adaptor::Flux.FluxMetalAdaptor, x, cache)
    FluxMetalExt = Base.get_extension(Flux, :FluxMetalExt)
    if isnothing(FluxMetalExt)
        error("FluxMetalExt not loaded. Make sure Metal.jl is properly installed and loaded.")
    end
    Transformers.__toxdevice(adaptor, cache, x, Flux._isleaf, FluxMetalExt.check_use_metal)
end

end
