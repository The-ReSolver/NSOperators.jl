using NSOperators
using Fields
using ChebUtils
using PoissonSolver
using FFTW
using LinearAlgebra
using Random
using Test

include("residualgradfd.jl")

include("test_cache.jl")
include("test_residual.jl")
