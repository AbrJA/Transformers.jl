module TransformersAMDGPUExt

using Transformers
using Transformers.Flux
using AMDGPU

# Lazy-load FluxAMDGPUExt to avoid precompilation failure
function Transformers._toxdevice(adaptor::Flux.FluxAMDGPUAdaptor, x, cache)
    FluxAMDGPUExt = Base.get_extension(Flux, :FluxAMDGPUExt)
    if isnothing(FluxAMDGPUExt)
        error("FluxAMDGPUExt not loaded. Make sure AMDGPU.jl is properly installed and loaded.")
    end
    Transformers.__toxdevice(adaptor, cache, x, FluxAMDGPUExt._exclude, FluxAMDGPUExt.check_use_amdgpu)
end

end
