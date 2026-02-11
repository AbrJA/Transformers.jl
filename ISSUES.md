# Transformers.jl — Open GitHub Issues Summary

> Source: [github.com/chengchingwen/Transformers.jl/issues](https://github.com/chengchingwen/Transformers.jl/issues)
> Date: 2026-02-11 | 12 open issues

## Priority Classification

### 🔴 Critical — Blocking Users

| # | Issue | Impact |
|---|-------|--------|
| [#201](https://github.com/chengchingwen/Transformers.jl/issues/201) | **Can't compile with Flux v0.14.23** — `GPU_BACKEND` removed | Package completely broken with latest Flux |
| [#214](https://github.com/chengchingwen/Transformers.jl/issues/214) | **TransformersCUDAExt precompilation failure** — `FluxCUDAExt = nothing` | CUDA support broken, workaround: `__precompile__(false)` |
| [#215](https://github.com/chengchingwen/Transformers.jl/issues/215) | **Zygote.withgradient gives wrong results** — padding/mask bug | Training produces incorrect gradients for shorter sequences in batch |

### 🟠 High — Significant Pain Points

| # | Issue | Impact |
|---|-------|--------|
| [#199](https://github.com/chengchingwen/Transformers.jl/issues/199) | **Separate TextEncoder/Tokenizer from Transformers.jl** | Users forced to load Flux just for tokenization |
| [#204](https://github.com/chengchingwen/Transformers.jl/issues/204) | **Unsupported pre-tokenization method: Digits** | Can't load models using `Digits` pre-tokenizer (e.g., some LLMs) |
| [#213](https://github.com/chengchingwen/Transformers.jl/issues/213) | **No progress bar during model loading** | Poor UX for large model downloads |
| [#177](https://github.com/chengchingwen/Transformers.jl/issues/177) | **Storage of Downloaded Models** | Users can't control where models are cached |

### 🟡 Medium — Improvements

| # | Issue | Impact |
|---|-------|--------|
| [#210](https://github.com/chengchingwen/Transformers.jl/issues/210) | **Tutorial uses old Zygote API** | Tutorial doesn't work with current Zygote |
| [#212](https://github.com/chengchingwen/Transformers.jl/issues/212) | **Support for Mamba2** | Feature request for SSM architecture |

### 🟢 Low — Nice to Have

| # | Issue | Impact |
|---|-------|--------|
| [#218](https://github.com/chengchingwen/Transformers.jl/issues/218) | **Is this project abandoned?** | Community concern about maintenance |
| [#174](https://github.com/chengchingwen/Transformers.jl/issues/174) | **Improve documentation** | Docs are minimal compared to Python `transformers` |
| [#172](https://github.com/chengchingwen/Transformers.jl/issues/172) | **Download model weights on external drive** | Relates to #177 |

---

## Issue Details

### #201 — Flux v0.14.23 Compatibility (Critical)
- **Root cause**: `device.jl` uses `Flux.GPU_BACKEND` which was removed
- **Location**: [device.jl](file:///home/abr/Documents/GitHub/Julia/Transformers.jl/src/device.jl) lines 2, 26, 33, 37, 41, 62-68
- **Fix**: Use `Flux.gpu_backend()` function instead of `GPU_BACKEND` constant, or use `Flux.get_device()` (new API)

### #214 — CUDA Extension Precompilation (Critical)
- **Root cause**: `Base.get_extension(Flux, :FluxCUDAExt)` returns `nothing` during precompilation
- **Location**: [TransformersCUDAExt.jl](file:///home/abr/Documents/GitHub/Julia/Transformers.jl/ext/TransformersCUDAExt/TransformersCUDAExt.jl) line 7
- **Fix**: Lazy-evaluate the extension lookup or use `@static if` with proper precompile guards

### #215 — Zygote Gradient Bug (Critical)
- **Root cause**: Mask/padding handling differs between forward pass and gradient pass
- **Symptom**: Only the longest sequence in a batch matches between direct call and `Zygote.withgradient`
- **Likely location**: Attention mask handling in `NeuralAttentionlib` or layer composition

### #199 — Separate Tokenizer (High)
- **User need**: Load tokenizer without Flux overhead
- **Current workaround**: Complex manual extraction from Transformers.jl API
- **Proposed**: Move tokenizer to `TextEncodeBase` or standalone package

### #204 — Digits Pre-tokenizer (High)
- **Root cause**: `fast_tkr.jl` doesn't implement `Digits` pre-tokenization method
- **Location**: [fast_tkr.jl](file:///home/abr/Documents/GitHub/Julia/Transformers.jl/src/huggingface/tokenizer/fast_tkr.jl)
- **Fix**: Add `extract_pre_tokenization(::Val{:Digits}, ...)` handler

---

## Additional Code Bugs Found

1. **`save_model` undefined `force`**: In [models.jl](file:///home/abr/Documents/GitHub/Julia/Transformers.jl/src/huggingface/models/models.jl) line 108, references `force` variable that is never defined as a parameter (should be `force = false` keyword arg)
2. **Weight format detection prefers Pickle over SafeTensors**: In [weight.jl](file:///home/abr/Documents/GitHub/Julia/Transformers.jl/src/huggingface/weight.jl) `detect_weight_format` checks for Pickle first. SafeTensors is safer and faster — should be preferred.
3. **HasIndexMap.files vs .indexmap**: In [weight.jl](file:///home/abr/Documents/GitHub/Julia/Transformers.jl/src/huggingface/weight.jl) line 140, `load_state_dict_from!` accesses `status.files` but the struct field is named `indexmap`.
