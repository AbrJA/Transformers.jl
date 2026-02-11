using Flux
using Functors

module TestMod
using Flux
using Functors
struct MyLayer
    x
end
Flux.@layer MyLayer
end
println("Module simulation passed")
