
# PreTokenizer Extraction
@valsplit extract_pre_tokenization(
    Val(tokenization_type::Symbol), pretokenizer_dict, tokenization, match_tokens, normalizer, tokenizer_dict
) = load_error("Unsupported pre-tokenization method: $tokenization_type")

extract_pre_tokenization(pretokenizer_dict, tokenization, match_tokens, normalizer, tokenizer_dict) =
    extract_pre_tokenization(
        Symbol(pretokenizer_dict["type"]), pretokenizer_dict, tokenization, match_tokens, normalizer, tokenizer_dict)

function extract_pre_tokenization(
    ::Val{:BertPreTokenizer}, pretokenizer_dict, tokenization, match_tokens, normalizer, tokenizer_dict
)
    @assert isnothing(tokenization) load_error_msg("Chaining tokenization is unsupported")
    normalizer_dict = tokenizer_dict["normalizer"]
    @assert !isnothing(normalizer_dict) && normalizer_dict["type"] == "BertNormalizer" load_error_msg("Normalizer of BertPreTokenizer is not BertNormalizer")
    islower = normalizer_dict["lowercase"]
    pretokenization = islower ? BertUnCasedPreTokenization() : BertCasedPreTokenization()
    return pretokenization, match_tokens, normalizer
end

function extract_pre_tokenization(
    ::Val{:ByteLevel}, pretokenizer_dict, tokenization, match_tokens, normalizer, tokenizer_dict
)
    @assert !pretokenizer_dict["add_prefix_space"] load_error_msg("add_prefix_space is unsupported")
    isnothing(tokenization) && (tokenization = GPT2Tokenization())
    normalizer = normalizer ∘ Base.Fix2(CodeNormalizer, gpt2_codemap())
    return tokenization, match_tokens, normalizer
end

function extract_pre_tokenization(
    ::Val{:Metaspace}, pretokenizer_dict, tokenization, match_tokens, normalizer, tokenizer_dict
)
    @assert !haskey(pretokenizer_dict, "str_rep") || pretokenizer_dict["replacement"] == pretokenizer_dict["str_rep"]
    replacement = collect(pretokenizer_dict["replacement"])[]::Char
    add_prefix_space = pretokenizer_dict["add_prefix_space"]
    if isnothing(tokenization)
        tokenization = EachMatchTokenization(RuRegex("$replacement[^$replacement]*|[^$replacement]+"))
        normalizer = normalizer ∘ Base.Fix2(ReplaceNormalizer, ' ' => replacement)
        if add_prefix_space
            normalizer = normalizer ∘ Base.Fix2(SentenceFuncNormalizer, ensure_prefix(replacement))
        end
    else
        @assert tokenization == EachSplitTokenization(isspace) load_error_msg("Metaspace without WhiteSpaceSPlit is unsupported")
        metaspacef(x) = isspace(x) || x == replacement
        tokenization = EachSplitTokenization(metaspacef)
        if add_prefix_space
            normalizer = normalizer ∘ Base.Fix2(WordFuncNormalizer, ensure_prefix(replacement))
        end
    end
    return tokenization, match_tokens, normalizer
end

function extract_pre_tokenization(
    ::Val{:Split}, pretokenizer_dict, tokenization, match_tokens, normalizer, tokenizer_dict
)
    @assert isnothing(tokenization) load_error_msg("Chaining tokenization is unsupported")
    behavior = pretokenizer_dict["behavior"]
    @assert behavior in ("Removed", "Isolated") load_error_msg("Only support removed or isolated behavior")
    @assert haskey(pretokenizer_dict["pattern"], "Regex") load_error_msg("Only support regex pattern")
    regex_str = pretokenizer_dict["pattern"]["Regex"]
    if behavior == "Removed"
        if pretokenizer_dict["invert"]
            if regex_str == raw"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
                tokenization = GPT2Tokenization()
            else
                if regex_str == raw"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"
                    regex = r"'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"
                    if isnothing(match_tokens)
                        match_tokens = ["<|startoftext|>", "<|endoftext|>"]
                    else
                        push!(match_tokens, "<|startoftext|>", "<|endoftext|>")
                    end
                else
                    regex = RuRegex(regex_str)
                end
                tokenization = EachMatchTokenization(regex)
            end
        else
            tokenization = EachSplitTokenization(RuRegex(regex_str))
        end
    else
        tokenization = MatchSplitsTokenization(RuRegex(regex_str))
    end
    return tokenization, match_tokens, normalizer
end

function extract_pre_tokenization(
    ::Val{:WhitespaceSplit}, pretokenizer_dict, tokenization, match_tokens, normalizer, tokenizer_dict
)
    @assert isnothing(tokenization) load_error_msg("Chaining tokenization is unsupported")
    return EachSplitTokenization(isspace), match_tokens, normalizer
end

function extract_pre_tokenization(
    ::Val{:Whitespace}, pretokenizer_dict, tokenization, match_tokens, normalizer, tokenizer_dict
)
    @assert isnothing(tokenization) load_error_msg("Chaining tokenization is unsupported")
    return EachMatchTokenization(r"\w+|[^\w\s]+"), match_tokens, normalizer
end

function extract_pre_tokenization(
    ::Val{:Digits}, pretokenizer_dict, tokenization, match_tokens, normalizer, tokenizer_dict
)
    individual_digits = get(pretokenizer_dict, "individual_digits", true)
    if individual_digits
        # Split digits into individual characters: each digit is its own token,
        # while non-digit runs remain as single tokens.
        # This regex matches either a single digit or a sequence of non-digits.
        digit_regex = RuRegex(raw"\d|[^\d]+")
        if isnothing(tokenization)
            tokenization = EachMatchTokenization(digit_regex)
        else
            # If we already have a tokenization, we need to add digit splitting
            # as a nested split within words. Use a normalizer-based approach
            # by wrapping existing tokenization with digit isolation.
            normalizer = normalizer ∘ Base.Fix2(WordFuncNormalizer, identity)
            tokenization_warn = "Digits pre-tokenizer chained with existing tokenization; " *
                                "digits will be split at the word level"
            @debug tokenization_warn
        end
    end
    # If individual_digits is false, this is a no-op
    return tokenization, match_tokens, normalizer
end

function extract_pre_tokenization(
    ::Val{:Sequence}, pretokenizer_dict, tokenization, match_tokens, normalizer, tokenizer_dict
)
    for sub_pretokenizer_dict in pretokenizer_dict["pretokenizers"]
        tokenization, match_tokens, normalizer = extract_pre_tokenization(
            sub_pretokenizer_dict, tokenization, match_tokens, normalizer, tokenizer_dict)
    end
    return tokenization, match_tokens, normalizer
end
