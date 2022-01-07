@testset "Cache construction            " begin
    # initialise variables and arrays
    Ny = rand(3:50)
    Nz = rand(3:50)
    Nt = rand(3:50)
    y = rand(Float64, Ny)
    Dy = rand(Float64, (Ny, Ny))
    Dy2 = rand(Float64, (Ny, Ny))
    ws = rand(Float64, Ny)
    ω = abs(randn())
    β = abs(randn())
    ū = rand(Float64, Ny)
    dūdy = rand(Float64, Ny)
    d2ūdy2 = rand(Float64, Ny)
    dp̄dy = rand(Float64, Ny)
    Re = abs(randn())
    Ro = randn()

    # construct fields
    grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)
    U = SpectralField(grid)
    u = PhysicalField(grid)

    # initialise useful types
    spec_type = SpectralField{Ny, Nz, Nt, typeof(grid), Float64, Array{ComplexF64, 3}}
    phys_type = PhysicalField{Ny, Nz, Nt, typeof(grid), Float64, Array{Float64, 3}}
    vec_type = VectorField{3, spec_type}
    plan_type = Tuple{FFTPlan!{Ny, Nz, Nt, FFTW.rFFTWPlan{Float64, -1, false, 3, Vector{Int}}}, IFFTPlan!{Ny, Nz, Nt, FFTW.rFFTWPlan{ComplexF64, 1, false, 3, Vector{Int}}}}

    # construct local residual
    @test typeof(Cache(U, u, ū, dūdy, d2ūdy2, dp̄dy, Re, Ro)) == Cache{Float64, spec_type, phys_type, vec_type, Matrix{Complex{Float64}}, plan_type}

    # catch errors
    @test_throws ArgumentError Cache(SpectralField(Grid(rand(Ny - 1), Nz, Nt, Dy, Dy2, ws, ω, β)), u, ū, dūdy, d2ūdy2, dp̄dy, Re, Ro)
    @test_throws ArgumentError Cache(U, u, rand(Ny - 1), dūdy, d2ūdy2, dp̄dy, Re, Ro)
    @test_throws MethodError Cache(U, u, rand(Int, Ny), dūdy, d2ūdy2, dp̄dy, Re, Ro)
    @test_throws MethodError Cache(U, u, ū, dūdy, d2ūdy2, dp̄dy, 1+1im, Ro)
end
