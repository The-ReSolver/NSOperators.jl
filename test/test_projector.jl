@testset "Projector constructor         " begin
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

@testset "Projector incompressible field" begin
    # construct incompressible vector field
    Ny = 64; Nz = 64; Nt = 64
    y = chebpts(Ny)
    Dy = chebdiff(Ny)
    Dy2 = chebddiff(Ny)
    ws = quadweights(y, 2)
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

@testset "Projector compressible field  " begin
    # construct compressible vector field
    Ny = 64; Nz = 64; Nt = 64
    y = chebpts(Ny)
    Dy = chebdiff(Ny)
    Dy2 = chebddiff(Ny)
    ws = quadweights(y, 2)
    ω = 1.0
    β = 1.0
    u_fun(y, z, t) = (1 - y^2)*exp(cos(z))*atan(sin(t))
    v_fun(y, z, t) = sin(π*y)*exp(sin(z))*atan(cos(t))
    w_fun(y, z, t) = (cos(π*y) + 1)*cos(z)*exp(cos(t))
    grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)
    u = VectorField(PhysicalField(grid, u_fun),
                    PhysicalField(grid, v_fun),
                    PhysicalField(grid, w_fun))
    U = VectorField(grid)
    FFT! = FFTPlan!(grid; flags=FFTW.ESTIMATE)
    FFT!(U, u)

    # check divergence is non-zero
    dVdy = SpectralField(grid)
    dWdz = SpectralField(grid)
    div = SpectralField(grid)
    ddy!(U[2], dVdy)
    ddz!(U[3], dWdz)
    div .= dVdy .+ dWdz
    @test norm(div) > 1e-3

    # initialise projector
    projector! = Projector!(U[1], PhysicalField(grid))

    # perform projection
    projector!(U)

    # check divergence of new vector field
    ddy!(U[2], dVdy)
    ddz!(U[3], dWdz)
    div .= dVdy .+ dWdz
    @test norm(div) < 1e-11
end
