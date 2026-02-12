#!/usr/bin/env julia
"""
Standalone performance benchmark — compares Julia vs Python forward pass timing.

Usage:
    julia --project=benchmarks benchmarks/run_benchmarks.jl                        # all models
    julia --project=benchmarks benchmarks/run_benchmarks.jl --model bert           # specific model
    julia --project=benchmarks benchmarks/run_benchmarks.jl --samples 50           # more samples
    julia --project=benchmarks benchmarks/run_benchmarks.jl --output benchmarks/results/perf.md
"""

if !haskey(Base.loaded_modules, Base.PkgId(Base.UUID("6099a3de-0909-46bc-b1f4-468b9a2dfc0d"), "PythonCall"))
    @info "Resolving benchmarks environment..."
    import Pkg
    Pkg.instantiate()
end

using CondaPkg
using PythonCall
using Transformers

# Parse CLI arguments
function parse_args()
    model_filter = nothing
    task_filter = nothing
    output_file = nothing
    n_samples = 20
    verbose = true

    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--model" && i < length(ARGS)
            model_filter = Symbol(ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--task" && i < length(ARGS)
            task_filter = Symbol(lowercase(ARGS[i+1]))
            i += 2
        elseif ARGS[i] == "--output" && i < length(ARGS)
            output_file = ARGS[i+1]
            i += 2
        elseif ARGS[i] == "--samples" && i < length(ARGS)
            n_samples = parse(Int, ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--quiet"
            verbose = false
            i += 1
        elseif ARGS[i] in ["--help", "-h"]
            println("""
            Usage: julia --project=benchmarks benchmarks/run_benchmarks.jl [OPTIONS]

            Options:
              --model MODEL    Filter by model type (e.g., bert, gpt2, distilbert)
              --task TASK       Filter by task (e.g., model, formaskedlm)
              --output FILE    Write markdown report to file
              --samples N      Number of timed samples per model (default: 20)
              --quiet          Suppress verbose output
              --help, -h       Show this help
            """)
            exit(0)
        else
            @warn "Unknown argument: $(ARGS[i])"
            i += 1
        end
    end

    return (; model_filter, task_filter, output_file, n_samples, verbose)
end

include(joinpath(@__DIR__, "ModelValidation.jl"))
using .ModelValidation

function main()
    args = parse_args()

    filter_fn = spec -> begin
        (!isnothing(args.model_filter) && spec.model_type != args.model_filter) && return false
        (!isnothing(args.task_filter) && spec.task != args.task_filter) && return false
        return true
    end

    models = filter(filter_fn, collect(values(get_registry())))
    if isempty(models)
        @error "No models match filters. Available: $(list_models())"
        exit(1)
    end
    @info "Running benchmarks for $(length(models)) model(s), $(args.n_samples) samples each..."
    println()

    results = run_all_benchmarks(; n_samples=args.n_samples, verbose=args.verbose, filter_fn)

    println("\n" * "="^70)
    generate_benchmark_report(results)

    if !isnothing(args.output_file)
        mkpath(dirname(abspath(args.output_file)))
        open(abspath(args.output_file), "w") do io
            generate_benchmark_report(results; io)
        end
        @info "Report written to $(args.output_file)"
    end
end

main()
