# This script will benchmark the time taken to run the caching updating
# functions.

using BenchmarkTools

using NSOperators
using Fields
using ChebUtils
using FDGrids

# initialise cache object
Ny = 33; Nz = 33; Nt = 33
y = chebpts(Ny)
# NOTE: the Chebyshev differentiation is faster???
# Dy = DiffMatrix(y, 3, 1); Dy2 = DiffMatrix(y, 3, 2)
Dy = chebdiff(Ny); Dy2 = chebddiff(Ny)
grid = Grid(y, Nz, Nt, Dy, Dy2, zeros(Ny), 1.0, 1.0)
U = VectorField(grid)
cache = Cache(grid, zeros(Ny), zeros(Ny), zeros(Ny), 1.0, 1.0)

# NOTE: this shouldn't be creating any allocations
@btime update_v!($U, $cache)
@btime update_p!($cache)
@btime update_r!($cache)
