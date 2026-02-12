using Transformers
using Transformers.DistilBert
using Transformers.TransformerLayers

println("Checking HGFDistilBertModel constructor...")
try
    # Mock some components
    # DistilBert model fields are: embeddings, transformer
    # embeddings is usually a Chain(CompositeEmbedding, DropoutLayer(LayerNorm)) or similar
    # transformer is a Transformer

    e = randn(Float32, 10, 10)
    embed = TransformerLayers.CompositeEmbedding(token=TransformerLayers.Embed(e))
    trf = Transformer((), nothing)

    println("Types: ", typeof(embed), ", ", typeof(trf))

    m = HGFDistilBertModel(embed, trf)
    println("Success! Model created: ", typeof(m))
catch e
    println("Failed!")
    showerror(stdout, e, catch_backtrace())
    println()
end
