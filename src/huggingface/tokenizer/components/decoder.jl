
# Decoder Extraction

function reduce_nestedcall(fs)
    return foldl(fs; init=[]) do init, f
        f isa typeof(identity) && return init
        isempty(init) && return push!(init, f)
        f0 = pop!(init)
        if f0 isa Base.Fix1{typeof(nestedcall)} && f isa Base.Fix1{typeof(nestedcall)}
            push!(init, nestedcall(f.x ∘ f0.x))
        else
            push!(init, f0, f)
        end
        return init
    end
end

function build_pipeline(fs)
    isempty(fs) && return identity
    length(fs) == 1 && return fs[]
    return foldl(Iterators.drop(fs, 1); init=Pipeline{:token}(first(fs), 1)) do pipe, f
        pipe |> Pipeline{:token}(f, :token)
    end |> PipeGet{:token}()
end

function extract_decoder(tokenizer_dict, config)
    decodes = Any[identity]
    textprocesses = Any[TextEncodeBase.join_text]
    decodes, textprocesses = extract_decoder(tokenizer_dict["decoder"], decodes, textprocesses)
    config[:clean_up_tokenization_spaces] && !(nestedcall(cleanup) in textprocesses) &&
        push!(textprocesses, nestedcall(cleanup))
    # https://github.com/huggingface/transformers/blob/25245ec26dc29bcf6102e1b4ddd0dfd02e720cf5/src/transformers/models/clip/tokenization_clip_fast.py#L95-L107
    if getconfigname(config) == :clip
        suffix = tokenizer_dict["model"]["end_of_word_suffix"]
        p = suffix => " "
        remove_suffix(s) = replace(s, p)
        remove_tail_space(s) = string_strip(' ', s; start=0, stop=1)
        push!(decodes, nestedcall(remove_suffix))
        push!(textprocesses, nestedcall(remove_tail_space))
    end
    decode = build_pipeline(reduce_nestedcall(decodes))
    textprocess = build_pipeline(reduce_nestedcall(textprocesses))
    return decode, textprocess
end

extract_decoder(::Nothing, decode, textprocess) = decode, textprocess
extract_decoder(decoder_dict, decode, textprocess) = extract_decoder(Symbol(decoder_dict["type"]), decoder_dict, decode, textprocess)

@valsplit extract_decoder(Val(decoder_type::Symbol), decoder_dict, decode, textprocess) = load_error("Unsupported decoder method: $decoder_type")

function extract_decoder(::Val{:Replace}, decoder_dict, decode, textprocess)
    @assert isone(length(decoder_dict["pattern"])) load_error_msg("Multiple pattern")
    if haskey(decoder_dict["pattern"], "Regex")
        pattern = RuRegex(normalizer_dict["pattern"]["Regex"])
    elseif haskey(decoder_dict["pattern"], "String")
        pattern = decoder_dict["pattern"]["String"]
    else
        load_error_msg("Only support regex or String pattern")
    end
    content = decoder_dict["content"]
    p = pattern => content
    Replace(s) = replace(s, p)
    push!(decode, nestedcall(Replace))
    return decode, textprocess
end

function extract_decoder(::Val{:ByteFallback}, decoder_dict, decode, textprocess)
    push!(decode, nestedcall(fallback2byte))
    return decode, textprocess
end

function extract_decoder(::Val{:Strip}, decoder_dict, decode, textprocess)
    content = decoder_dict["content"]
    @assert length(content) == 1 load_error_msg("Strip decoder with string content")
    char = content[1]
    start = decoder_dict["start"]
    stop = decoder_dict["stop"]
    Strip(s) = string_strip(char, s; start, stop)
    push!(textprocess, nestedcall(Strip))
    return decode, textprocess
end

function extract_decoder(::Val{:Fuse}, decoder_dict, decode, textprocess)
    !(TextEncodeBase.join_text in textprocess) && push!(textprocess, TextEncodeBase.join_text)
    return decode, textprocess
end

function extract_decoder(::Val{:Metaspace}, decoder_dict, decode, textprocess)
    @assert !haskey(decoder_dict, "str_rep") || decoder_dict["replacement"] == decoder_dict["str_rep"]
    replacement = collect(decoder_dict["replacement"])[]::Char
    p = replacement => ' '
    metaspace2space(s) = replace(s, p)
    push!(decode, nestedcall(metaspace2space))
    if decoder_dict["add_prefix_space"]
        remove_prefix_space(s) = string_strip(' ', s; start=1, stop=0)
        push!(textprocess, nestedcall(remove_prefix_space))
    end
    return decode, textprocess
end

function extract_decoder(::Val{:BPEDecoder}, decoder_dict, decode, textprocess)
    suffix = decoder_dict["suffix"]
    p = suffix => " "
    remove_suffix(s) = replace(s, p)
    remove_tail_space(s) = string_strip(' ', s; start=0, stop=1)
    push!(decode, nestedcall(remove_suffix))
    push!(textprocess, nestedcall(remove_tail_space))
    return decode, textprocess
end

function extract_decoder(::Val{:ByteLevel}, decoder_dict, decode, textprocess)
    push!(decode, nestedcall(TextEncodeBase.CodeUnMap(gpt2_codemap())))
    return decode, textprocess
end

function extract_decoder(::Val{:WordPiece}, decoder_dict, decode, textprocess)
    prefix = decoder_dict["prefix"]
    function remove_conti_prefix(s)
        if startswith(s, prefix)
            return String(SubString(s, 1 + ncodeunits(prefix)))
        else
            return " $s"
        end
    end
    push!(decode, nestedcall(remove_conti_prefix))
    remove_prefix_space(s) = string_strip(' ', s; start=1, stop=0)
    push!(textprocess, nestedcall(remove_prefix_space))
    return decode, textprocess
end

function extract_decoder(::Val{:Sequence}, decoder_dict, decode, textprocess)
    for sub_decoder_dict in decoder_dict["decoders"]
        decode, textprocess = extract_decoder(sub_decoder_dict, decode, textprocess)
    end
    return decode, textprocess
end
