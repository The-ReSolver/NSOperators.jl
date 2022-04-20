module NSOperators

using LinearAlgebra

using Fields
using PoissonSolver

export Cache, update_v!, update_p!, update_r!
export localresidual!, ℜ, dℜ!

include("cache.jl")
include("residual.jl")

# TODO: implementation should be independent of fields package

end
