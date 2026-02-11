using Flux

is_precompiling() = ccall(:jl_generating_output, Cint, ()) == 1

"""
    enable_gpu(t=true)

Enable GPU for `todevice`. Disable with `enable_gpu(false)`.

The GPU backend is determined automatically by Flux based on which GPU package
is loaded (CUDA.jl, AMDGPU.jl, or Metal.jl). Use `Flux.gpu_backend!` to
override the default selection.

Should only be called in user scripts, not during precompilation.

# Examples
```julia
using Transformers, CUDA
enable_gpu()

model = todevice(model)  # CPU by default

enable_gpu()              # enable GPU
model = todevice(model)  # now moves to GPU
```
"""
function enable_gpu(t::Bool=true)
    if t
        backend = _gpu_backend()
        if backend == "CUDA"
            @info "GPU enabled (CUDA)"
        elseif backend == "AMDGPU"
            @info "GPU enabled (AMDGPU)"
        elseif backend == "Metal"
            @info "GPU enabled (Metal)"
        elseif backend == nothing
            @warn "No supported GPU backend loaded (CUDA, AMDGPU, Metal). GPU enablement ignored."
        else
            @info "GPU enabled ($backend)"
        end
    else
        @info "GPU disabled"
    end
    return nothing
end

"""
    gpu_device()

Return the current GPU device (from Flux).
"""
gpu_device() = Flux.gpu_device()

"""
    todevice(x)

Move `x` to the current device (CPU or GPU).
"""
todevice(args...; kws...) = Flux.gpu(args...; kws...)

"""
    togpudevice(x)

Move `x` to the GPU device.
"""
@inline function togpudevice(args...; kws...)
    backend = _gpu_backend()
    if backend == "CUDA"
        return Flux.gpu(args...; kws...)
    elseif backend == "AMDGPU"
        return Flux.gpu(args...; kws...)
    elseif backend == "Metal"
        return Flux.gpu(args...; kws...)
    else
        return Flux.gpu(args...; kws...) # Fallback
    end
end

"""
    tocpudevice(x)

Move `x` to the CPU device.
"""
@inline tocpudevice(args...; kws...) = Flux.cpu(args...; kws...)

_gpu_backend() = Flux.gpu_backend()
