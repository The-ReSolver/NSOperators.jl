@testset "Projector constructor  " begin
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
    @test typeof(Projector!(U, u)) == Projector!{SpectralField{Ny, Nz, Nt, typeof(grid), Float64, Array{ComplexF64, 3}}}

    # catch errors
    @test_throws ArgumentError Projector!(SpectralField(Grid(rand(Float64, Ny - 1), Nz, Nt, Dy, Dy2, ws, ω, β)), u)
    @test_throws MethodError Projector!(U, rand(Float64, (Ny, Nz)))
end

@testset "Projector identity test" begin
    # construct incompressible vector field
    Ny = 64; Nz = 64; Nt = 64
    y = chebpts(Ny)
    Dy = chebdiff(Ny)
    Dy2 = chebddiff(Ny)
    ws = rand(Float64, Ny)
    ω = 1.0
    β = 1.0
    u_fun(y, z, t) = sin(π*y)*exp(cos(z))*sin(t)
    v_fun(y, z, t) = (cos(π*y) + 1)*cos(z)*sin(t)
    w_fun(y, z, t) = π*sin(π*y)*sin(z)*sin(t)
    grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)
    u = VectorField(PhysicalField(grid, u_fun),
                    PhysicalField(grid, v_fun),
                    PhysicalField(grid, w_fun))
    U = VectorField(grid)
    FFT! = FFTPlan!(grid; flags=FFTW.ESTIMATE)
    IFFT! = IFFTPlan!(grid; flags=FFTW.ESTIMATE)
    FFT!(U, u)
    U_aux = VectorField(copy(U[1]), copy(U[2]), copy(U[3]))

    # check divergence is zero
    dVdy = SpectralField(grid)
    dWdz = SpectralField(grid)
    div = SpectralField(grid)
    ddy!(U[2], dVdy)
    ddz!(U[3], dWdz)
    div .= dVdy .+ dWdz
    @test norm(div) < 1e-12

    # initialise projector
    projector! = Projector!(U[1], PhysicalField(grid))

    # perform projection
    projector!(U)

    # check vector field didn't change
    @test U ≈ U_aux
end

@testset "Projector calculation  " begin
    # construct a random field
    Ny = 64; Nz = 64; Nt = 64
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

    # construct projection object
    projector! = Projector!(U[1], PhysicalField(grid))

    # perform projection
    projector!(U)

    # initialise derivative fields
    dVdy = SpectralField(grid)
    dWdz = SpectralField(grid)
    div = SpectralField(grid)

    # calculate divergence of projected vector field
    ddy!(U[2], dVdy)
    ddz!(U[3], dWdz)
    div .= dVdy .+ dWdz

    @test norm(div) ≈ 1e-8
end
