@testset "Leray projection constructor  " begin
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

    # initialise grid and field
    grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)
    U = SpectralField(grid)
    u = PhysicalField(grid)

    # construct projection type
    @test typeof(Project!(U, u)) == Project!{SpectralField{Ny, Nz, Nt, typeof(grid), Float64, Array{ComplexF64, 3}}}

    # catch errors
    @test_throws ArgumentError Project!(SpectralField(Grid(rand(Float64, Ny - 1), Nz, Nt, Dy, Dy2, ws, ω, β)), u)
    @test_throws MethodError Project!(U, rand(Float64, (Ny, Nz)))
end

@testset "Leray projection calculation  " begin
    # construct a random field
    Ny = 64
    Nz = 64
    Nt = 64
    y = chebpts(Ny)
    Dy = chebdiff(Ny)
    Dy2 = chebddiff(Ny)
    ws = rand(Float64, Ny)
    ω = 1.0
    β = 1.0
    grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)
    U = VectorField(grid)
    for i in 1:3
        U[i] .= rand(ComplexF64, (Ny, (Nz >> 1) + 1, Nt))
    end
    u = PhysicalField(grid)

    # construct projection object
    project! = Project!(U[1], u)

    # perform projection
    project!(U)

    # initialise derivative fields
    dUdy = VectorField(grid)
    dUdz = VectorField(grid)
    div = SpectralField(grid)

    # calculate divergence of projected vector field
    ddy!(U, dUdy)
    ddz!(U, dUdz)
    div .= dUdy[2] .+ dUdz[3]

    @test norm(div) ≈ 1e-8
end
