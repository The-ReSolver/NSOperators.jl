module NSOperators

using Fields
using PoissonSolver

export Cache, update_v!, update_p!, update_r!
export localresidual!, ā, dā!

include("cache.jl")
include("residual.jl")

# TODO: implementation should be independent of fields package

end
