module NSOperators

using LinearAlgebra

using Fields
using PoissonSolver

include("cache.jl")
include("projector.jl")
include("residual.jl")

# TODO: implementation should be independent of fields package

end
