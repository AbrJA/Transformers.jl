using Flux

"""
    AbstractPipeline

Base type for all pipelines. A pipeline generally consists of:
- **Pre-processing**: Tokenization and input formatting.
- **Model Inference**: Running the model.
- **Post-processing**: Decoding outputs (logits to text/probabilities).

Pipelines are callable: `pipeline(input)`.
"""
abstract type AbstractPipeline end

(p::AbstractPipeline)(input) = p(input)

"""
    pipeline(task::String; model=nothing, config=nothing, tokenizer=nothing, kwargs...)
    pipeline(task::Symbol; kwargs...)

Factory function to create a pipeline for a specific task.

# Arguments
- `task`: The task name (e.g., "feature-extraction", "fill-mask", "question-answering", "text-generation").
- `model`: (Optional) A pre-loaded model or model name.
- `config`: (Optional) A configuration object.
- `tokenizer`: (Optional) A tokenizer.

# Supported Tasks
- `"feature-extraction"` / `:feature_extraction`
- `"fill-mask"` / `:fill_mask`
- `"question-answering"` / `:question_answering`
- `"text-generation"` / `:text_generation`
"""
function pipeline(task::AbstractString; kwargs...)
    return pipeline(Symbol(replace(task, "-" => "_")); kwargs...)
end

function pipeline(task::Symbol; model=nothing, config=nothing, tokenizer=nothing, kwargs...)
    if task == :feature_extraction
        # return FeatureExtractionPipeline(model, config, tokenizer; kwargs...)
        error("FeatureExtractionPipeline not implemented yet")
    elseif task == :fill_mask
        return FillMaskPipeline(model, config, tokenizer; kwargs...)
    elseif task == :question_answering
        return QuestionAnsweringPipeline(model, config, tokenizer; kwargs...)
    elseif task == :text_generation
        return TextGenerationPipeline(model, config, tokenizer; kwargs...)
    else
        error("Unknown task: $task. Supported tasks: :feature_extraction, :fill_mask, :question_answering, :text_generation")
    end
end

# Placeholder struct definitions to allow compilation before implementation
struct FeatureExtractionPipeline <: AbstractPipeline end
# FillMaskPipeline is implemented in fill_mask.jl
struct QuestionAnsweringPipeline <: AbstractPipeline end
struct TextGenerationPipeline <: AbstractPipeline end
