using TextEncodeBase
using TextEncodeBase: nestedcall, nested2batch, getvalue, getmeta
using TextEncodeBase: SequenceTemplate, ConstTerm, InputTerm, RepeatedTerm

struct BertPipeline{F1,F2} <: Function
    startsym::String
    endsym::String
    padsym::String
    trunc::Union{Nothing,Int}
    trunc_end::Symbol
    pad_end::Symbol
    fixedsize::Bool
    maskf::F1
    truncf::F2
end

function BertPipeline(; startsym="[CLS]", endsym="[SEP]", padsym="[PAD]",
    trunc=nothing, trunc_end=:tail, pad_end=:tail, fixedsize=false)
    maskf = TextEncoders.get_mask_func(trunc, pad_end)
    truncf = TextEncoders.get_trunc_pad_func(fixedsize, trunc, trunc_end, pad_end)
    return BertPipeline(startsym, endsym, padsym, trunc, trunc_end, pad_end, fixedsize, maskf, truncf)
end

function (p::BertPipeline)(x)
    # 1. string_getvalue
    # This turns TokenStage into string (or whatever getvalue returns)
    # bert_default_preprocess effectively does:
    # x |> Pipeline{:token}(nestedcall(string_getvalue), 1)

    # TextEncoders.annotate_strings puts things into Batch{Sentence} or Sentence.
    # We assume 'x' is already annotated.

    # 1. String conversion and grouping
    # Pipeline{:token}(nestedcall(string_getvalue), 1) |> Pipeline{:token}(grouping_sentence, :token)
    # 'x' comes in as e.g. Batch{Sentence}
    # nestedcall(string_getvalue)(x)

    tokens = nestedcall(string_getvalue)(x)
    tokens = grouping_sentence(tokens) # This handles Batch structure

    # 2. Sequence Template (CLS, SEP)
    # This is the complex part.
    # The template is: [CLS] A [SEP] B [SEP] ...
    # We can use SequenceTemplate directly if it's efficient, or reimplement specific logic.
    # using SequenceTemplate is conceptually static if the template is constant.
    # But here we constructed it dynamically in bert_default_preprocess.
    # For StandardBertPipeline we can assume standard template:
    # CLS S1 SEP S2 SEP ...

    # Let's use the same SequenceTemplate but constructed once if possible?
    # Or just construct it here. SequenceTemplate construction is cheap.
    # Function execution is what matters.

    template = SequenceTemplate(
        ConstTerm(p.startsym, 1), InputTerm{String}(1), ConstTerm(p.endsym, 1),
        RepeatedTerm(InputTerm{String}(2), ConstTerm(p.endsym, 2); dynamic_type_id=true)
    )

    # Apply template
    # Pipeline{:token_segment}(SequenceTemplate(...), :token)
    # return (token, segment)
    token_segment = template(tokens)

    # Extract token and segment
    # Pipeline{:token}(nestedcall(first), :token_segment)
    # Pipeline{:segment}(nestedcall(last), :token_segment)
    toks = nestedcall(first)(token_segment)
    segs = nestedcall(last)(token_segment)

    # 3. Attention Mask
    # Pipeline{:attention_mask}(maskf, :token)
    # maskf depends on 'toks' length usually
    atten_mask = p.maskf(toks)

    # 4. Truncate and Pad Tokens
    # Pipeline{:token}(truncf(padsym), :token)
    toks = p.truncf(p.padsym)(toks)

    # 5. Batch Tokens
    # Pipeline{:token}(nested2batch, :token)
    toks = nested2batch(toks)

    # 6. Truncate and Pad Segments
    # Pipeline{:segment}(truncf(1), :segment)
    segs = p.truncf(1)(segs)

    # 7. Batch Segments
    # Pipeline{:segment}(nested2batch, :segment)
    segs = nested2batch(segs)

    # 8. Sequence Mask (usually identity for batch? No, it's for original length?)
    # Pipeline{:sequence_mask}(identity, :attention_mask)
    # Wait, 'identity' on 'attention_mask' input.
    # So sequence_mask = atten_mask
    seq_mask = atten_mask

    # 9. Return NamedTuple or Tuple matching PipeGet
    # PipeGet{(:token, :segment, :attention_mask, :sequence_mask)}()
    return (token=toks, segment=segs, attention_mask=atten_mask, sequence_mask=seq_mask)
end

# Implement Base.show for nice printing
Base.show(io::IO, p::BertPipeline) = print(io, "BertPipeline(trunc=$(p.trunc))")
