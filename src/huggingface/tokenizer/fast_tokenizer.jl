using StructWalk: scan
using FuncPipelines
using LRUCache
using TextEncodeBase
using TextEncodeBase: CodeNormalizer, ReplaceNormalizer, WordReplaceNormalizer,
    SentenceFuncNormalizer, WordFuncNormalizer,
    MatchTokenization, EachSplitTokenization, EachMatchTokenization, MatchSplitsTokenization,
    TokenizerStyle, nestedcall
using TextEncodeBase: SequenceTemplate, ConstTerm, InputTerm, RepeatedTerm, IndexInputTerm
using TextEncodeBase.RustRegex
using ..TextEncoders: BertUnCasedPreTokenization, BertCasedPreTokenization, TextTokenizer,
    grouping_sentence, string_strip
using ..WordPieceModel
using BytePairEncoding
using BytePairEncoding: CachedBPE, ByteFallbackBPE, GPT2Tokenization, gpt2_codemap, fallback2byte
using ..UnigramLanguageModel
using ..UnigramLanguageModel: PrecompiledNormalizer, CachedUnigram

struct NoTokenization <: TextEncodeBase.BaseTokenization end
TextEncodeBase.splitting(::NoTokenization, s::TextEncodeBase.SentenceStage) = Base.vect(TextEncodeBase.getvalue(s))

# https://github.com/huggingface/transformers/blob/235e5d4991e8a0984aa78db91087b49622c7740e/src/transformers/tokenization_utils_base.py#L3798
# NOT https://github.com/huggingface/tokenizers/blob/daf361676bdfd14088f7e0bc087effc6a9cfdf3e/tokenizers/src/decoders/wordpiece.rs#L31
cleanup(s) = replace(
    replace(replace(s, " ." => ".", " ?" => "?", " !" => "!", " ," => ","), " ' " => "'"),
    " n't" => "n't", " 'm" => "'m", #= " do not" => " don't", =#
    " 's" => "'s", " 've" => "'ve", " 're" => "'re")

add_prefix(prefix) = Base.Fix1(add_prefix, prefix)
add_prefix(prefix, str) = prefix * str
ensure_prefix(prefix) = Base.Fix1(ensure_prefix, prefix)
ensure_prefix(prefix, str) = String(startswith(str, prefix) ? str : add_prefix(prefix, str))

function extract_added_token(added_token)
    vidx = added_token["id"] + 1
    token = added_token["content"]
    isspecial = added_token["special"]

    added_token["rstrip"] || added_token["lstrip"] && tokenizer_warn(
        "match token `$token` require to match with space on either side but that is not implemented here"
    )
    added_token["single_word"] && tokenizer_warn(
        "match token `$token` does not match inside of a word but that is not implemented here"
    )
    return vidx, token, isspecial
end

extract_and_add_tokens!(::Nothing, _) = nothing
function extract_and_add_tokens!(added_token_list, vocab_list)
    iszero(length(added_token_list)) && return nothing
    added_token_list = sort(added_token_list; by=Base.Fix2(getindex, "id"))
    match_tokens = String[]
    for added_token in added_token_list
        vidx, token, isspecial = extract_added_token(added_token)
        if isspecial
            if vidx > length(vocab_list)
                # special tokens not in the vocab already
                @assert vidx == length(vocab_list) + 1
                push!(vocab_list, token)
            end
            @assert vocab_list[vidx] == token
        else
            n_vocab = length(vocab_list)
            if vidx == n_vocab + 1
                push!(vocab_list, token)
            elseif vidx <= n_vocab
                @assert vocab_list[vidx] == token "Two word has same index: $(token) and $(vocab_list[idx])"
            else
                error("There is a gap in the vocabulary")
            end
        end
        push!(match_tokens, token)
    end
    return match_tokens
end

function guessing_tokenization_method(model_dict)
    # method should be one of WordLevel, WordPiece, BPE, Unigram
    if haskey(model_dict, "type")
        # if serialization object has "type" field, use that value directly.
        return Symbol(model_dict["type"])
    elseif haskey(model_dict, "merges")
        # bpe method must have "merges"
        # https://github.com/huggingface/tokenizers/blob/06025e4ca151dcc7f6a4872e8857e8f175e8d3ac/tokenizers/src/models/bpe/serialization.rs#L59
        return :BPE
    elseif haskey(model_dict, "max_input_chars_per_word") || haskey(model_dict, "continuing_subword_prefix")
        # only bpe and wordpiece has "continuing_subword_prefix", so if it doesn't has "merges" but has
        # "continuing_subword_prefix", it should be wordpiece.
        # https://github.com/huggingface/tokenizers/blob/06025e4ca151dcc7f6a4872e8857e8f175e8d3ac/tokenizers/src/models/wordpiece/serialization.rs#L40-L41
        return :WordPiece
    elseif haskey(model_dict, "unk_id")
        # unigram only have "vocab" and "unk_id", but it seems only unigram use "unk_id" (others use "unk_token").
        # https://github.com/huggingface/tokenizers/blob/06025e4ca151dcc7f6a4872e8857e8f175e8d3ac/tokenizers/src/models/unigram/serialization.rs#L28
        return :Unigram
    elseif haskey(model_dict, "unk_token")
        # wordlevel only have "vocab" and "unk_token". At this point, it's probably wordlevel, but it should at least
        # have "unk_token".
        # https://github.com/huggingface/tokenizers/blob/06025e4ca151dcc7f6a4872e8857e8f175e8d3ac/tokenizers/src/models/wordlevel/serialization.rs#L30
        return :WordLevel
    else
        # Otherwise we raise an error.
        error("Failed to guess the tokenization method")
    end
end

@valsplit extract_tokenization_method(Val(method::Symbol), model_dict) =
    load_error("Unsupported tokenization method: $method")

extract_tokenization_method(model_dict) = extract_tokenization_method(
    guessing_tokenization_method(model_dict), model_dict)

function extract_tokenization_method(::Val{:WordPiece}, model_dict)
    @assert model_dict["continuing_subword_prefix"] == "##"
    unk_token = model_dict["unk_token"]
    max_char = model_dict["max_input_chars_per_word"]
    vocab_list = reverse_keymap_to_list(model_dict["vocab"])
    wordpiece = WordPiece(vocab_list, unk_token; max_char)
    return Base.Fix2(WordPieceTokenization, wordpiece), wordpiece, unk_token, vocab_list
end

empty2nothing(::Nothing) = nothing
empty2nothing(s) = isempty(s) ? nothing : s

function extract_tokenization_method(::Val{:BPE}, model_dict)
    @assert isnothing(model_dict["dropout"]) "BPE with dropout unsupported"
    !model_dict["fuse_unk"] && tokenizer_warn("fuse_unk is unsupported")
    byte_fallback = get(model_dict, "byte_fallback", false)
    unk_token = model_dict["unk_token"]
    sepsym = empty2nothing(model_dict["continuing_subword_prefix"])
    endsym = empty2nothing(model_dict["end_of_word_suffix"])
    merges = rank_from_lines(model_dict["merges"]; endsym)
    vocab_list = reverse_keymap_to_list(model_dict["vocab"])
    if byte_fallback
        bpe = ByteFallbackBPE(vocab_list, merges, sepsym, endsym)
    else
        cache = LRU{AbstractString,Vector{String}}(; maxsize=1000)
        bpe = CachedBPE(BPE(merges, sepsym, endsym), cache)
    end
    return Base.Fix2(BPETokenization, bpe), bpe, unk_token, vocab_list
end

function extract_tokenization_method(::Val{:Unigram}, model_dict)
    unki = model_dict["unk_id"] + 1
    score_list = model_dict["vocab"]
    vocab_list = Vector{String}(undef, length(score_list))
    scores = Vector{Float64}(undef, length(score_list))
    for (i, entry) in enumerate(score_list)
        @assert length(entry) == 2
        vocab_list[i] = entry[1]
        scores[i] = entry[2]
    end
    unk = vocab_list[unki]
    unigram = Unigram(vocab_list, scores, unki)
    cache = LRU{AbstractString,Vector{String}}(; maxsize=1000)
    unigram = CachedUnigram(unigram, cache)
    return Base.Fix2(UnigramTokenization, unigram), unigram, unk, vocab_list
end

# extract_tokenization_method(M::Val{:WordLevel}, model_dict)

function extract_tokenizer_model(model_dict)
    method, object, unk, vocab_list = extract_tokenization_method(model_dict)
    return method, object, unk, vocab_list
end

# Component Extraction
include("components/normalizer.jl")
include("components/pretokenizer.jl")
include("components/decoder.jl")
include("components/postprocessor.jl")

function load_fast_tokenizer_components(tokenizer_json, config)
    tokenizer_dict = json_load(tokenizer_json)
    method, tokenization_object, unk, vocab_list = extract_tokenizer_model(tokenizer_dict["model"])
    match_tokens = extract_and_add_tokens!(tokenizer_dict["added_tokens"], vocab_list)
    base_tokenization, match_tokens = extract_base_tokenization(method, match_tokens, tokenizer_dict)
    match_tokens = empty_then_nothing(match_tokens)
    process_config = extract_processor(tokenizer_dict)
    decode, textprocess = extract_decoder(tokenizer_dict, config)
    return base_tokenization, match_tokens, vocab_list, unk, tokenization_object, process_config, decode, textprocess
end

load_fast_tokenizer(type, tokenizer_json, config) = load_fast_tokenizer(tokenizer_json, config) # default ignoring type
function load_fast_tokenizer(tokenizer_json, config)
    base_tokenization, match_tokens, vocab_list, unk, tokenization_object, process_config, decode, textprocess =
        load_fast_tokenizer_components(tokenizer_json, config)
    isnothing(match_tokens) || (base_tokenization = MatchTokenization(base_tokenization, match_tokens))
    isnothing(unk) && (unk = "<unk>") # dummy unk token, wouldn't appear in vocabulary
    unk isa AbstractString || (unk = vocab_list[unk])
    vocab = Vocab(vocab_list, unk)
    tokenizer = TextTokenizer(base_tokenization)
    return tokenizer, vocab, process_config, decode, textprocess
end
