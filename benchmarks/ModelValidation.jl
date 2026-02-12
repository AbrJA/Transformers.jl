"""
    ModelValidation

A registry-based framework for validating Transformers.jl model outputs against
Python HuggingFace `transformers` and benchmarking performance.

## Quick Start

```julia
include("benchmarks/ModelValidation.jl")
using .ModelValidation

# Run parity check for a single model
result = run_parity_check(:bert, :model)

# Run parity checks for all registered models
results = run_all_parity_checks()

# Generate markdown report
generate_parity_report(results)
```

## Adding a New Model

```julia
register_model!(
    model_type = :new_model,
    task = :model,
    tiny_model = "hf-internal-testing/tiny-random-NewModel",
    python_class = "NewModel",
    output_keys = [:hidden_state],
)
```
"""
module ModelValidation

using Printf
using PythonCall
using NeuralAttentionlib
using Transformers
using Transformers.HuggingFace
const np = Ref{Py}()

export ModelSpec, ParityResult, BenchmarkResult
export register_model!, get_registry, list_models
export run_parity_check, run_all_parity_checks
export run_benchmark, run_all_benchmarks
export generate_parity_report, generate_benchmark_report

# ============================================================================
# Python setup
# ============================================================================

const torch = Ref{Py}()
const trf = Ref{Py}()
const time_mod = Ref{Py}()

function ensure_python!()
    if !isassigned(torch)
        torch[] = pyimport("torch")
        trf[] = pyimport("transformers")
        time_mod[] = pyimport("time")
        np[] = pyimport("numpy")
        @info "Python modules loaded: torch, transformers, numpy"
    end
end

# ============================================================================
# ModelSpec: describes one model+task combination
# ============================================================================

"""
    ModelSpec

Describes a model+task combination for validation testing.
"""
struct ModelSpec
    model_type::Symbol
    task::Symbol
    tiny_model_name::String
    python_class::String
    python_module::String
    input_builder::Function
    python_input_builder::Function
    output_extractor::Function
    python_output_extractor::Function
    output_keys::Vector{Symbol}
    atol::Float64
end

# ============================================================================
# Result types
# ============================================================================

struct ParityResult
    spec::ModelSpec
    passed::Bool
    max_diff::Float64
    julia_output_shapes::Dict{Symbol,Tuple}
    python_output_shapes::Dict{Symbol,Tuple}
    julia_time_s::Float64
    python_time_s::Float64
    error::Union{Nothing,String}
end

struct BenchmarkResult
    spec::ModelSpec
    julia_median_ms::Float64
    julia_mean_ms::Float64
    julia_min_ms::Float64
    python_median_ms::Float64
    python_mean_ms::Float64
    python_min_ms::Float64
    speedup::Float64
    n_samples::Int
    error::Union{Nothing,String}
end

# ============================================================================
# Global registry
# ============================================================================

const MODEL_REGISTRY = Dict{Tuple{Symbol,Symbol},ModelSpec}()

"""
    register_model!(; model_type, task, tiny_model, python_class, kwargs...)

Register a model+task for validation testing.
"""
function register_model!(;
    model_type::Symbol,
    task::Symbol,
    tiny_model::String,
    python_class::String,
    python_module::String="transformers",
    input_builder::Function=default_encoder_input,
    python_input_builder::Function=default_python_encoder_input,
    output_extractor::Function=default_output_extractor,
    python_output_extractor::Function=default_python_output_extractor,
    output_keys::Vector{Symbol}=[:hidden_state],
    atol::Float64=1e-4,
)
    spec = ModelSpec(
        model_type, task, tiny_model, python_class, python_module,
        input_builder, python_input_builder,
        output_extractor, python_output_extractor,
        output_keys, atol,
    )
    MODEL_REGISTRY[(model_type, task)] = spec
    return spec
end

get_registry() = MODEL_REGISTRY
list_models() = sort(collect(keys(MODEL_REGISTRY)))

# ============================================================================
# Default input builders
# ============================================================================

function default_encoder_input(; seq_len::Int=10, batch::Int=2, vocab_size::Int=100)
    return (
        token=rand(1:vocab_size, seq_len, batch),
        attention_mask=NeuralAttentionlib.GenericSeqMask(ones(Int32, seq_len, batch)),
    )
end

function default_decoder_input(; seq_len::Int=10, batch::Int=2, vocab_size::Int=100)
    return (
        token=rand(1:vocab_size, seq_len, batch),
    )
end

function default_seq2seq_input(; enc_len::Int=10, dec_len::Int=8, batch::Int=2, vocab_size::Int=100)
    return (
        encoder_input=(
            token=rand(1:vocab_size, enc_len, batch),
            attention_mask=NeuralAttentionlib.GenericSeqMask(ones(Int32, enc_len, batch)),
        ),
        decoder_input=(
            token=rand(1:vocab_size, dec_len, batch),
        ),
    )
end

# ============================================================================
# Default Python input builders
# ============================================================================

function default_python_encoder_input(jl_input)
    ensure_python!()
    # Julia tokens are 1-indexed, Python expects 0-indexed
    input_ids = jl_input.token .- Int32(1)
    # Extract matrix from GenericSeqMask
    mask_mat = jl_input.attention_mask.mask
    # Julia is column-major (seq_len, ..., batch), Python expects (batch, ..., seq_len)
    # Use collect to ensure densified matrix for PythonCall conversion
    py_input_ids = torch[].tensor(np[].array(collect(permutedims(input_ids, (2, 1))))).long()

    # Handle both 2D and 3D masks
    p_mask = ndims(mask_mat) == 2 ? (2, 1) : (ndims(mask_mat), (1:ndims(mask_mat)-1)...)
    py_attention_mask = torch[].tensor(np[].array(collect(permutedims(Int32.(mask_mat), p_mask)))).long()

    return Dict(:input_ids => py_input_ids, :attention_mask => py_attention_mask)
end

function default_python_decoder_input(jl_input)
    ensure_python!()
    input_ids = jl_input.token .- Int32(1)
    py_input_ids = torch[].tensor(np[].array(collect(permutedims(input_ids, (2, 1))))).long()
    return Dict(:input_ids => py_input_ids)
end

function default_python_seq2seq_input(jl_input)
    ensure_python!()
    enc_ids = jl_input.encoder_input.token .- Int32(1)
    dec_ids = jl_input.decoder_input.token .- Int32(1)
    enc_mask_mat = jl_input.encoder_input.attention_mask.mask

    py_enc_ids = torch[].tensor(np[].array(collect(permutedims(enc_ids, (2, 1))))).long()
    py_dec_ids = torch[].tensor(np[].array(collect(permutedims(dec_ids, (2, 1))))).long()

    p_mask = ndims(enc_mask_mat) == 2 ? (2, 1) : (ndims(enc_mask_mat), (1:ndims(enc_mask_mat)-1)...)
    py_enc_mask = torch[].tensor(np[].array(collect(permutedims(Int32.(enc_mask_mat), p_mask)))).long()

    return Dict(
        :input_ids => py_enc_ids,
        :attention_mask => py_enc_mask,
        :decoder_input_ids => py_dec_ids,
    )
end

# ============================================================================
# Default output extractors
# ============================================================================

function default_output_extractor(output, keys::Vector{Symbol})
    results = Dict{Symbol,Array}()
    for k in keys
        if haskey(output, k)
            val = output[k]
            if val isa AbstractArray
                results[k] = Array(val)
            end
        end
    end
    return results
end

"""
Extract tensors from Python model output, converting to Julia arrays.
Python output shapes (batch, seq_len, dims) → Julia (dims, seq_len, batch)
"""
function default_python_output_extractor(py_output, keys::Vector{Symbol})
    results = Dict{Symbol,Array}()
    # Python output mapping: Julia key => Python attribute
    py_key_map = Dict(
        :hidden_state => "last_hidden_state",
        :logit => "logits",
        :pooled => "pooler_output",
    )
    for k in keys
        py_key = get(py_key_map, k, String(k))
        try
            val = pygetattr(py_output, py_key)
            np_val = val.detach().numpy()
            jl_val = pyconvert(Array{Float32}, np_val)
            # Permute from Python (batch, ...) to Julia (..., batch)
            if ndims(jl_val) == 3
                results[k] = permutedims(jl_val, (3, 2, 1))
            elseif ndims(jl_val) == 2
                results[k] = permutedims(jl_val, (2, 1))
            else
                results[k] = jl_val
            end
        catch e
            @warn "Could not extract key $k (python attr: $py_key)" exception = e
        end
    end
    return results
end

# ============================================================================
# Parity check runner
# ============================================================================

function run_parity_check(model_type::Symbol, task::Symbol; kwargs...)
    key = (model_type, task)
    haskey(MODEL_REGISTRY, key) || error("Model ($model_type, $task) not registered. Available: $(list_models())")
    return run_parity_check(MODEL_REGISTRY[key]; kwargs...)
end

function run_parity_check(spec::ModelSpec; verbose::Bool=true)
    verbose && @info "═══ Parity check: $(spec.model_type)/$(spec.task) ═══"

    try
        ensure_python!()

        # --- Julia side ---
        verbose && @info "  Loading Julia model: $(spec.tiny_model_name)"
        cfg = HuggingFace.load_config(spec.tiny_model_name; cache=false)
        jl_model = HuggingFace.load_model(spec.model_type, spec.tiny_model_name, spec.task; config=cfg, cache=false)

        jl_input = spec.input_builder()

        verbose && @info "  Running Julia forward pass..."
        jl_t0 = time()
        jl_output = jl_model(jl_input)
        jl_t1 = time()

        jl_extracted = spec.output_extractor(jl_output, spec.output_keys)

        # --- Python side ---
        verbose && @info "  Loading Python model: $(spec.python_class)"
        py_model_cls = pygetattr(trf[], spec.python_class)
        py_model = py_model_cls.from_pretrained(spec.tiny_model_name)
        py_model.eval()

        py_input = spec.python_input_builder(jl_input)

        verbose && @info "  Running Python forward pass..."
        py_t0 = time()
        local py_output
        torch[].no_grad().__enter__()
        try
            py_output = py_model(; py_input...)
        finally
            torch[].no_grad().__exit__(pybuiltins.None, pybuiltins.None, pybuiltins.None)
        end
        py_t1 = time()

        py_extracted = spec.python_output_extractor(py_output, spec.output_keys)

        # --- Compare ---
        verbose && @info "  Comparing outputs..."
        max_diff = 0.0
        jl_shapes = Dict{Symbol,Tuple}()
        py_shapes = Dict{Symbol,Tuple}()

        for k in spec.output_keys
            if haskey(jl_extracted, k) && haskey(py_extracted, k)
                jl_val = jl_extracted[k]
                py_val = py_extracted[k]
                jl_shapes[k] = size(jl_val)
                py_shapes[k] = size(py_val)

                if size(jl_val) == size(py_val)
                    diff = maximum(abs.(Float64.(jl_val) .- Float64.(py_val)))
                    max_diff = max(max_diff, diff)
                    verbose && @info "    $k: max_diff=$(@sprintf("%.6e", diff)), shape=$(size(jl_val))"
                else
                    verbose && @warn "    $k: shape mismatch! Julia=$(size(jl_val)) Python=$(size(py_val))"
                    max_diff = Inf
                end
            else
                jl_has = haskey(jl_extracted, k)
                py_has = haskey(py_extracted, k)
                verbose && @warn "    $k: missing (Julia=$jl_has, Python=$py_has)"
                max_diff = Inf
            end
        end

        passed = max_diff <= spec.atol
        if verbose
            if passed
                @info "  ✅ PASSED (max_diff=$(@sprintf("%.6e", max_diff)), atol=$(spec.atol))"
            else
                @warn "  ❌ FAILED (max_diff=$(@sprintf("%.6e", max_diff)), atol=$(spec.atol))"
            end
        end

        return ParityResult(spec, passed, max_diff, jl_shapes, py_shapes, jl_t1 - jl_t0, py_t1 - py_t0, nothing)

    catch e
        msg = sprint(showerror, e, catch_backtrace())
        verbose && @error "  💥 ERROR" exception = (e, catch_backtrace())
        return ParityResult(spec, false, Inf, Dict{Symbol,Tuple}(), Dict{Symbol,Tuple}(), 0.0, 0.0, msg)
    end
end

function run_all_parity_checks(; verbose::Bool=true, filter_fn=Returns(true))
    results = ParityResult[]
    for key in sort(collect(keys(MODEL_REGISTRY)))
        spec = MODEL_REGISTRY[key]
        filter_fn(spec) || continue
        GC.gc(true)
        result = run_parity_check(spec; verbose)
        push!(results, result)
    end
    return results
end

# ============================================================================
# Performance benchmark runner
# ============================================================================

function run_benchmark(model_type::Symbol, task::Symbol; kwargs...)
    key = (model_type, task)
    haskey(MODEL_REGISTRY, key) || error("Model ($model_type, $task) not registered")
    return run_benchmark(MODEL_REGISTRY[key]; kwargs...)
end

function run_benchmark(spec::ModelSpec; n_samples::Int=20, n_warmup::Int=3, verbose::Bool=true)
    verbose && @info "═══ Benchmark: $(spec.model_type)/$(spec.task) ═══"

    try
        ensure_python!()

        # --- Julia setup ---
        verbose && @info "  Loading Julia model..."
        cfg = HuggingFace.load_config(spec.tiny_model_name; cache=false)
        jl_model = HuggingFace.load_model(spec.model_type, spec.tiny_model_name, spec.task; config=cfg, cache=false)
        jl_input = spec.input_builder()

        # Warmup
        for _ in 1:n_warmup
            jl_model(jl_input)
        end
        GC.gc(true)

        # Timed runs
        jl_times = Float64[]
        for _ in 1:n_samples
            t = @elapsed jl_model(jl_input)
            push!(jl_times, t * 1000)
        end

        # --- Python setup ---
        verbose && @info "  Loading Python model..."
        py_model_cls = pygetattr(trf[], spec.python_class)
        py_model = py_model_cls.from_pretrained(spec.tiny_model_name)
        py_model.eval()
        py_input_dict = spec.python_input_builder(jl_input)
        py_kwargs = pyconvert(Dict{String,Py}, py_input_dict)

        # Warmup
        for _ in 1:n_warmup
            torch[].no_grad().__enter__()
            try
                py_model(; py_kwargs...)
            finally
                torch[].no_grad().__exit__(pybuiltins.None, pybuiltins.None, pybuiltins.None)
            end
        end

        # Timed runs
        py_times = Float64[]
        for _ in 1:n_samples
            t0 = pyconvert(Float64, time_mod[].perf_counter())
            torch[].no_grad().__enter__()
            try
                py_model(; py_kwargs...)
            finally
                torch[].no_grad().__exit__(pybuiltins.None, pybuiltins.None, pybuiltins.None)
            end
            t1 = pyconvert(Float64, time_mod[].perf_counter())
            push!(py_times, (t1 - t0) * 1000)
        end

        jl_med = _median(jl_times)
        py_med = _median(py_times)
        speedup = py_med / jl_med

        verbose && @info @sprintf(
            "  Julia: %.2fms (median)  Python: %.2fms (median)  Speedup: %.2fx",
            jl_med, py_med, speedup
        )

        return BenchmarkResult(
            spec,
            jl_med, sum(jl_times) / length(jl_times), minimum(jl_times),
            py_med, sum(py_times) / length(py_times), minimum(py_times),
            speedup, n_samples, nothing,
        )

    catch e
        msg = sprint(showerror, e, catch_backtrace())
        verbose && @error "  💥 ERROR" exception = (e, catch_backtrace())
        return BenchmarkResult(spec, Inf, Inf, Inf, Inf, Inf, Inf, 0.0, 0, msg)
    end
end

function run_all_benchmarks(; n_samples::Int=20, verbose::Bool=true, filter_fn=Returns(true))
    results = BenchmarkResult[]
    for key in sort(collect(keys(MODEL_REGISTRY)))
        spec = MODEL_REGISTRY[key]
        filter_fn(spec) || continue
        GC.gc(true)
        result = run_benchmark(spec; n_samples, verbose)
        push!(results, result)
    end
    return results
end

# ============================================================================
# Helpers
# ============================================================================

function _median(v::Vector{Float64})
    s = sort(v)
    n = length(s)
    isodd(n) ? s[(n+1)÷2] : (s[n÷2] + s[n÷2+1]) / 2
end

# ============================================================================
# Report generation
# ============================================================================

function generate_parity_report(results::Vector{ParityResult}; io::IO=stdout)
    println(io, "# Model Parity Report\n")
    println(io, "| Model | Task | Status | Max Diff | Tolerance | Julia (ms) | Python (ms) |")
    println(io, "|-------|------|--------|----------|-----------|------------|-------------|")
    for r in results
        status = isnothing(r.error) ? (r.passed ? "✅ PASS" : "❌ FAIL") : "💥 ERROR"
        diff_str = isinf(r.max_diff) ? "N/A" : @sprintf("%.2e", r.max_diff)
        jl_ms = @sprintf("%.2f", r.julia_time_s * 1000)
        py_ms = @sprintf("%.2f", r.python_time_s * 1000)
        println(io, "| $(r.spec.model_type) | $(r.spec.task) | $status | $diff_str | $(@sprintf("%.0e", r.spec.atol)) | $jl_ms | $py_ms |")
    end
    n_pass = count(r -> r.passed, results)
    println(io, "\n**Summary:** $n_pass/$(length(results)) passed\n")
end

function generate_benchmark_report(results::Vector{BenchmarkResult}; io::IO=stdout)
    println(io, "# Performance Benchmark Report\n")
    println(io, "| Model | Task | Julia (ms) | Python (ms) | Speedup | Samples |")
    println(io, "|-------|------|------------|-------------|---------|---------|")
    for r in results
        if isnothing(r.error)
            emoji = r.speedup >= 1.0 ? "🟢" : "🔴"
            println(io, @sprintf("| %s | %s | %.2f | %.2f | %s %.2fx | %d |",
                r.spec.model_type, r.spec.task, r.julia_median_ms, r.python_median_ms,
                emoji, r.speedup, r.n_samples))
        else
            println(io, "| $(r.spec.model_type) | $(r.spec.task) | 💥 ERROR | - | - | 0 |")
        end
    end
    println(io)
end

# ============================================================================
# Default model registrations
# ============================================================================

function register_default_models!()
    # --- Encoder models (BERT-like) ---
    register_model!(model_type=:distilbert, task=:model,
        tiny_model="hf-internal-testing/tiny-random-DistilBertModel",
        python_class="DistilBertModel", output_keys=[:hidden_state])
    register_model!(model_type=:distilbert, task=:formaskedlm,
        tiny_model="hf-internal-testing/tiny-random-DistilBertForMaskedLM",
        python_class="DistilBertForMaskedLM", output_keys=[:logit])

    register_model!(model_type=:bert, task=:model,
        tiny_model="hf-internal-testing/tiny-random-BertModel",
        python_class="BertModel", output_keys=[:hidden_state])
    register_model!(model_type=:bert, task=:formaskedlm,
        tiny_model="hf-internal-testing/tiny-random-BertForMaskedLM",
        python_class="BertForMaskedLM", output_keys=[:logit])

    register_model!(model_type=:roberta, task=:model,
        tiny_model="hf-internal-testing/tiny-random-RobertaModel",
        python_class="RobertaModel", output_keys=[:hidden_state])

    # --- Decoder models (GPT-like) ---
    register_model!(model_type=:gpt2, task=:model,
        tiny_model="hf-internal-testing/tiny-random-GPT2Model",
        python_class="GPT2Model",
        input_builder=default_decoder_input, python_input_builder=default_python_decoder_input,
        output_keys=[:hidden_state])
    register_model!(model_type=:gpt2, task=:lmheadmodel,
        tiny_model="hf-internal-testing/tiny-random-GPT2LMHeadModel",
        python_class="GPT2LMHeadModel",
        input_builder=default_decoder_input, python_input_builder=default_python_decoder_input,
        output_keys=[:logit])

    register_model!(model_type=:gpt_neo, task=:model,
        tiny_model="hf-internal-testing/tiny-random-GPTNeoModel",
        python_class="GPTNeoModel",
        input_builder=default_decoder_input, python_input_builder=default_python_decoder_input,
        output_keys=[:hidden_state])

    register_model!(model_type=:gptj, task=:model,
        tiny_model="hf-internal-testing/tiny-random-GPTJModel",
        python_class="GPTJModel",
        input_builder=default_decoder_input, python_input_builder=default_python_decoder_input,
        output_keys=[:hidden_state])

    register_model!(model_type=:gpt_neox, task=:model,
        tiny_model="hf-internal-testing/tiny-random-GPTNeoXModel",
        python_class="GPTNeoXModel",
        input_builder=default_decoder_input, python_input_builder=default_python_decoder_input,
        output_keys=[:hidden_state])

    register_model!(model_type=:bloom, task=:model,
        tiny_model="hf-internal-testing/tiny-random-BloomModel",
        python_class="BloomModel",
        input_builder=default_decoder_input, python_input_builder=default_python_decoder_input,
        output_keys=[:hidden_state])

    register_model!(model_type=:phi, task=:model,
        tiny_model="hf-internal-testing/tiny-random-PhiModel",
        python_class="PhiModel",
        input_builder=default_decoder_input, python_input_builder=default_python_decoder_input,
        output_keys=[:hidden_state])

    # --- Seq2Seq models ---
    register_model!(model_type=:t5, task=:model,
        tiny_model="hf-internal-testing/tiny-random-T5Model",
        python_class="T5Model",
        input_builder=default_seq2seq_input, python_input_builder=default_python_seq2seq_input,
        output_keys=[:hidden_state], atol=1e-3)

    register_model!(model_type=:bart, task=:model,
        tiny_model="hf-internal-testing/tiny-random-BartModel",
        python_class="BartModel",
        input_builder=default_seq2seq_input, python_input_builder=default_python_seq2seq_input,
        output_keys=[:hidden_state], atol=1e-3)

    @info "ModelValidation: registered $(length(MODEL_REGISTRY)) models"
end

register_default_models!()

end # module
