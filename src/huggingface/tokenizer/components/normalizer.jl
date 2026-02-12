
# Normalizer Extraction
extract_normalizer(::Nothing, tokenization, tokenizer_dict) = tokenization
extract_normalizer(normalizer_dict, tokenization, tokenizer_dict) =
    extract_normalizer(Symbol(normalizer_dict["type"]), normalizer_dict, tokenization, tokenizer_dict)

@valsplit extract_normalizer(Val(normalizer_type::Symbol), normalizer_dict, tokenization, tokenizer_dict) =
    load_error("Unsupported normalizer method: $normalizer_type")

function extract_normalizer(::Val{:BertNormalizer}, normalizer_dict, tokenization, tokenizer_dict)
    # bert normalizer is done in bert pre tokenization
    @assert normalizer_dict["clean_text"] load_error_msg("bert normalize without clean_text")
    check = Ref{Bool}(false)
    scan(x -> (x isa BertUnCasedPreTokenization || x isa BertCasedPreTokenization) && (check[] = true),
        TokenizerStyle(), tokenization)
    check[] || load_error("BertNormalizer without BertPreTokenizer is unsupported")
    return tokenization
end

extract_normalizer(::Val{:Lowercase}, normalizer_dict, tokenization, tokenizer_dict) =
    TextEncodeBase.SentenceReplaceNormalizer(TextEncodeBase.LowercaseNormalizer(tokenization), "İ" => "İ")

extract_normalizer(::Val{:NFD}, normalizer_dict, tokenization, tokenizer_dict) =
    TextEncodeBase.UnicodeNormalizer(tokenization, :NFD)

extract_normalizer(::Val{:NFKD}, normalizer_dict, tokenization, tokenizer_dict) =
    TextEncodeBase.UnicodeNormalizer(tokenization, :NFKD)

extract_normalizer(::Val{:NFC}, normalizer_dict, tokenization, tokenizer_dict) =
    TextEncodeBase.UnicodeNormalizer(tokenization, :NFC)

extract_normalizer(::Val{:NFKC}, normalizer_dict, tokenization, tokenizer_dict) =
    TextEncodeBase.UnicodeNormalizer(tokenization, :NFKC)

function extract_normalizer(::Val{:Replace}, normalizer_dict, tokenization, tokenizer_dict)
    @assert isone(length(normalizer_dict["pattern"])) load_error_msg("Multiple pattern")
    if haskey(normalizer_dict["pattern"], "Regex")
        pattern = RuRegex(normalizer_dict["pattern"]["Regex"])
    elseif haskey(normalizer_dict["pattern"], "String")
        pattern = normalizer_dict["pattern"]["String"]
    else
        load_error_msg("Only support regex or String pattern")
    end
    content = normalizer_dict["content"]
    return ReplaceNormalizer(tokenization, pattern => content)
end

function extract_normalizer(::Val{:Precompiled}, normalizer_dict, tokenization, tokenizer_dict)
    precompiled = UnigramLanguageModel.PrecompiledNorm(normalizer_dict["precompiled_charsmap"])
    return PrecompiledNormalizer(tokenization, precompiled)
end

function extract_normalizer(::Val{:Prepend}, normalizer_dict, tokenization, tokenizer_dict)
    prepend = normalizer_dict["prepend"]
    return SentenceFuncNormalizer(tokenization, ensure_prefix(prepend))
end

function extract_normalizer(::Val{:Sequence}, normalizer_dict, tokenization, tokenizer_dict)
    for sub_normalizer_dict in Iterators.reverse(normalizer_dict["normalizers"])
        tokenization = extract_normalizer(sub_normalizer_dict, tokenization, tokenizer_dict)
    end
    return tokenization
end
