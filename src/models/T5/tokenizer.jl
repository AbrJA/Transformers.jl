using ..TransformerInterfaces

struct T5Tokenizer <: AbstractTransformerTokenizer
    sp
    extra::Dict{Symbol,Any}
end

TransformerInterfaces.get_vocab(t::T5Tokenizer) = t.sp
TransformerInterfaces.get_eos_id(t::T5Tokenizer) = 1
TransformerInterfaces.get_pad_id(t::T5Tokenizer) = 0
# T5 doesn't have a bos token, usually using pad as start token

function load_tokenizer(::Val{:t5}, model_name::AbstractString; possible_files=nothing, kws...)
    possible_files = ensure_possible_files(possible_files, model_name; kws...)
    sp_file = hgf_file(possible_files, model_name, "spiece.model"; kws...)
    sp = load_tokenizer_file(sp_file)
    extra = Dict{Symbol,Any}()
    return T5Tokenizer(sp, extra)
end
