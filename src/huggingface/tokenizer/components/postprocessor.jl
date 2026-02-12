
# PostProcessor Extraction

function extract_trunc_pad(tokenizer_dict)
    # In huggingface's setting, padding and truncation can be applied independently.
    # Therefore, both of them would have a `max_length` fields.
    # But in our setting, they must be applied together (since we are going to convert to tensor).
    # We use the `max_length` of `truncation` as the possible final output length.
    process_config = Dict{Symbol,Any}()
    pad_dict = tokenizer_dict["padding"]
    trunc_dict = tokenizer_dict["truncation"]
    trunc = nothing
    if !isnothing(pad_dict)
        process_config[:fixedsize] = true
        process_config[:pad_end] = get(pad_dict, "direction", "Left") == "Left" ? :head : :tail
        padsym = get(pad_dict, "pad_token", nothing)
        isnothing(padsym) || (process_config[:padsym] = padsym)
        trunc = get(pad_dict, "max_length", trunc)
        isnothing(trunc) || (process_config[:trunc] = trunc)
    end
    if !isnothing(trunc_dict)
        strategy = get(trunc_dict, "strategy", nothing)
        !isnothing(strategy) && strategy != "LongestFirst" &&
            tokenizer_warn("truncation strategy $strategy not support, only LongestFirst")
        get(trunc_dict, "stride", 0) != 0 &&
            tokenizer_warn("truncation stride is not 0")
        process_config[:trunc_end] = get(trunc_dict, "direction", "Left") == "Left" ? :head : :tail
        trunc = get(trunc_dict, "max_length", trunc)
        isnothing(trunc) || (process_config[:trunc] = trunc)
    end
    return process_config
end

extract_post_processor(::Nothing, tokenizer_dict, process_config) = process_config
extract_post_processor(post_processor_dict, tokenizer_dict, process_config) =
    extract_post_processor(Symbol(post_processor_dict["type"]), post_processor_dict, tokenizer_dict, process_config)
@valsplit extract_post_processor(
    Val(post_processor_type::Symbol), post_processor_dict, tokenizer_dict, process_config
) =
    load_error("Unsupported post processor method: $post_processor_type")

function extract_term(term)
    if haskey(term, "SpecialToken")
        @assert length(term) == 1
        term = term["SpecialToken"]
        id = term["id"]
        type_id = term["type_id"] + 1
        return ConstTerm(id, type_id)
    elseif haskey(term, "Sequence")
        @assert length(term) == 1
        term = term["Sequence"]
        id = term["id"]
        type_id = term["type_id"] + 1
        if id == "A"
            id = 1
        elseif id == "B"
            id = 2
        else
            load_error("Unknown pattern in TemplateProcessing: $term")
        end
        return IndexInputTerm{String}(id, type_id)
    else
        load_error("Unknown pattern in TemplateProcessing: $term")
    end
end

function extract_post_processor(::Val{:TemplateProcessing}, post_processor_dict, tokenizer_dict, process_config)
    all(Base.splat(==), zip(post_processor_dict["single"], post_processor_dict["pair"])) ||
        load_error("Un-mergeable pattern for TemplateProcessing")
    special_tokens = post_processor_dict["special_tokens"]
    single_term = map(extract_term, post_processor_dict["single"])
    pair_term = map(extract_term, post_processor_dict["pair"][length(single_term)+1:end])
    process = Pipelines(
        Pipeline{:token}(grouping_sentence, :token),
        Pipeline{(:token, :segment)}(SequenceTemplate(single_term..., RepeatedTerm(pair_term...)), :token),
    )
    process_config[:process] = process
    return process_config
end

function extract_post_processor(::Val{:BertProcessing}, post_processor_dict, tokenizer_dict, process_config)
    sepsym, sepid = post_processor_dict["sep"]
    startsym, startid = post_processor_dict["cls"]
    trunc = get(process_config, :trunc, nothing)
    trunc_end = get(process_config, :trunc_end, :tail)
    padsym = get(process_config, :padsym, "[PAD]")
    pad_end = get(process_config, :pad_end, :tail)
    fixedsize = get(process_config, :fixedsize, false)
    process = TextEncoders.BertPipeline(; startsym, endsym=sepsym, padsym, trunc, trunc_end, pad_end, fixedsize)
    process_config[:process] = process
    return process_config
end

function extract_post_processor(::Val{:RobertaProcessing}, post_processor_dict, tokenizer_dict, process_config)
    @assert !post_processor_dict["add_prefix_space"] load_error_msg("add_prefix_space is unsupported")
    sepsym, sepid = post_processor_dict["sep"]
    startsym, startid = post_processor_dict["cls"]
    trunc = get(process_config, :trunc, nothing)
    trunc_end = get(process_config, :trunc_end, :tail)
    padsym = get(process_config, :padsym, "[PAD]")
    pad_end = get(process_config, :pad_end, :tail)
    fixedsize = get(process_config, :fixedsize, false)
    # Roberta uses same structure as Bert but handled slightly differently?
    # Actually RobertaProcessing in hgf is: <s> A </s> </s> B </s>
    # Wait, the current implementation in fast_tkr.jl was:
    # SequenceTemplate(
    #     ConstTerm(startsym), InputTerm{String}(), ConstTerm(sepsym),
    #     RepeatedTerm(ConstTerm(sepsym), InputTerm{String}(), ConstTerm(sepsym))),
    # :token),
    # This is DIFFERENT from StandardBertPipeline which assumes segment IDs.
    # So we can't reuse StandardBertPipeline blindly for Roberta unless we verify segment handling.
    # Let's keep Roberta dynamic for now to be safe, or implement StandardRobertaPipeline later.
    process = Pipelines(
        Pipeline{:token}(grouping_sentence, :token),
        Pipeline{(:token, :segment)}(
            SequenceTemplate(
                ConstTerm(startsym), InputTerm{String}(), ConstTerm(sepsym),
                RepeatedTerm(ConstTerm(sepsym), InputTerm{String}(), ConstTerm(sepsym))),
            :token),
    )
    process_config[:process] = process
    return process_config
end

function extract_post_processor(::Val{:ByteLevel}, post_processor_dict, tokenizer_dict, process_config)
    process = Pipelines(
        Pipeline{:token}(grouping_sentence, :token),
        Pipeline{:token}(SequenceTemplate(RepeatedTerm(InputTerm{String}()))(Val(1)), :token),
    )
    process_config[:process] = process
    return process_config
end

function extract_processor(tokenizer_json)
    process_config = extract_trunc_pad(tokenizer_json)
    process_config = extract_post_processor(tokenizer_json["post_processor"], tokenizer_json, process_config)
    return process_config
end
