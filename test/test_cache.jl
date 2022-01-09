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

@testset "Cache velocity update         " begin
    # initialise incompressible velocity field
    Ny = 32; Nz = 32; Nt = 32
    y = chebpts(Ny)
    Dy = chebdiff(Ny)
    Dy2 = chebddiff(Ny)
    ws = rand(Float64, Ny)
    ω = 1.0
    β = 1.0
    grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)
    u_fun(y, z, t) = sin(π*y)*exp(cos(z))*sin(t)
    v_fun(y, z, t) = (cos(π*y) + 1)*cos(z)*sin(t)
    w_fun(y, z, t) = π*sin(π*y)*sin(z)*sin(t)
    u = VectorField(PhysicalField(grid, u_fun),
                    PhysicalField(grid, v_fun),
                    PhysicalField(grid, w_fun))
    U = VectorField(grid)
    FFT! = FFTPlan!(grid; flags=FFTW.ESTIMATE)
    FFT!(U, u)
    Re = abs(rand())
    Ro = abs(rand())

    # initialise cache and update velocity field
    Re = abs(rand())
    Ro = abs(rand())
    cache = Cache(U[1], u[1], rand(Ny), rand(Ny), rand(Ny), rand(Ny), Re, Ro)
    NSOperators._update_vel!(U, cache)

    # initialise fields for results
    dudt_fun(y, z, t) = sin(π*y)*exp(cos(z))*cos(t)
    dvdt_fun(y, z, t) = (cos(π*y) + 1)*cos(z)*cos(t)
    dwdt_fun(y, z, t) = π*sin(π*y)*sin(z)*cos(t)
    dudz_fun(y, z, t) = -sin(π*y)*sin(z)*exp(cos(z))*sin(t)
    dvdz_fun(y, z, t) = -(cos(π*y) + 1)*sin(z)*sin(t)
    dwdz_fun(y, z, t) = π*sin(π*y)*cos(z)*sin(t)
    d2udz2_fun(y, z, t) = sin(π*y)*((sin(z)^2) - cos(z))*exp(cos(z))*sin(t)
    d2vdz2_fun(y, z, t) = -(cos(π*y) + 1)*cos(z)*sin(t)
    d2wdz2_fun(y, z, t) = -π*sin(π*y)*sin(z)*sin(t)
    dudy_fun(y, z, t) = π*cos(π*y)*exp(cos(z))*sin(t)
    dvdy_fun(y, z, t) = -π*sin(π*y)*cos(z)*sin(t)
    dwdy_fun(y, z, t) = (π^2)*cos(π*y)*sin(z)*sin(t)
    d2udy2_fun(y, z, t) = -(π^2)*sin(π*y)*exp(cos(z))*sin(t)
    d2vdy2_fun(y, z, t) = -(π^2)*cos(π*y)*cos(z)*sin(t)
    d2wdy2_fun(y, z, t) = -(π^3)*sin(π*y)*sin(z)*sin(t)
    v_dudy_fun(y, z, t) = v_fun(y, z, t)*dudy_fun(y, z, t)
    w_dudz_fun(y, z, t) = w_fun(y, z, t)*dudz_fun(y, z, t)
    v_dvdy_fun(y, z, t) = v_fun(y, z, t)*dvdy_fun(y, z, t)
    w_dvdz_fun(y, z, t) = w_fun(y, z, t)*dvdz_fun(y, z, t)
    v_dwdy_fun(y, z, t) = v_fun(y, z, t)*dwdy_fun(y, z, t)
    w_dwdz_fun(y, z, t) = w_fun(y, z, t)*dwdz_fun(y, z, t)
    # dvdz_dwdy_fun(y, z, t) = dvdz_fun(y, z, t)*dwdy_fun(y, z, t)
    # dvdy_dwdz_fun(y, z, t) = dvdy_fun(y, z, t)*dwdz_fun(y, z, t)
    dvdz_dwdy_fun(y, z, t) = -(π^2)*cos(π*y)*(cos(π*y) + 1)*(sin(z)^2)*(sin(t)^2)
    dvdy_dwdz_fun(y, z, t) = -(π^2)*(sin(π*y)^2)*(cos(z)^2)*(sin(t)^2)
    dudt = PhysicalField(grid, dudt_fun)
    dvdt = PhysicalField(grid, dvdt_fun)
    dwdt = PhysicalField(grid, dwdt_fun)
    dudz = PhysicalField(grid, dudz_fun)
    dvdz = PhysicalField(grid, dvdz_fun)
    dwdz = PhysicalField(grid, dwdz_fun)
    d2udz2 = PhysicalField(grid, d2udz2_fun)
    d2vdz2 = PhysicalField(grid, d2vdz2_fun)
    d2wdz2 = PhysicalField(grid, d2wdz2_fun)
    dudy = PhysicalField(grid, dudy_fun)
    dvdy = PhysicalField(grid, dvdy_fun)
    dwdy = PhysicalField(grid, dwdy_fun)
    d2udy2 = PhysicalField(grid, d2udy2_fun)
    d2vdy2 = PhysicalField(grid, d2vdy2_fun)
    d2wdy2 = PhysicalField(grid, d2wdy2_fun)
    v_dudy = PhysicalField(grid, v_dudy_fun)
    w_dudz = PhysicalField(grid, w_dudz_fun)
    v_dvdy = PhysicalField(grid, v_dvdy_fun)
    w_dvdz = PhysicalField(grid, w_dvdz_fun)
    v_dwdy = PhysicalField(grid, v_dwdy_fun)
    w_dwdz = PhysicalField(grid, w_dwdz_fun)
    dvdz_dwdy = PhysicalField(grid, dvdz_dwdy_fun)
    dvdy_dwdz = PhysicalField(grid, dvdy_dwdz_fun)
    dUdt = SpectralField(grid)
    dVdt = SpectralField(grid)
    dWdt = SpectralField(grid)
    dUdz = SpectralField(grid)
    dVdz = SpectralField(grid)
    dWdz = SpectralField(grid)
    d2Udz2 = SpectralField(grid)
    d2Vdz2 = SpectralField(grid)
    d2Wdz2 = SpectralField(grid)
    dUdy = SpectralField(grid)
    dVdy = SpectralField(grid)
    dWdy = SpectralField(grid)
    d2Udy2 = SpectralField(grid)
    d2Vdy2 = SpectralField(grid)
    d2Wdy2 = SpectralField(grid)
    V_dUdy = SpectralField(grid)
    W_dUdz = SpectralField(grid)
    V_dVdy = SpectralField(grid)
    W_dVdz = SpectralField(grid)
    V_dWdy = SpectralField(grid)
    W_dWdz = SpectralField(grid)
    dVdz_dWdy = SpectralField(grid)
    dVdy_dWdz = SpectralField(grid)
    FFT!(dUdt, dudt)
    FFT!(dVdt, dvdt)
    FFT!(dWdt, dwdt)
    FFT!(dUdz, dudz)
    FFT!(dVdz, dvdz)
    FFT!(dWdz, dwdz)
    FFT!(d2Udz2, d2udz2)
    FFT!(d2Vdz2, d2vdz2)
    FFT!(d2Wdz2, d2wdz2)
    FFT!(dUdy, dudy)
    FFT!(dVdy, dvdy)
    FFT!(dWdy, dwdy)
    FFT!(d2Udy2, d2udy2)
    FFT!(d2Vdy2, d2vdy2)
    FFT!(d2Wdy2, d2wdy2)
    FFT!(V_dUdy, v_dudy)
    FFT!(W_dUdz, w_dudz)
    FFT!(V_dVdy, v_dvdy)
    FFT!(W_dVdz, w_dvdz)
    FFT!(V_dWdy, v_dwdy)
    FFT!(W_dWdz, w_dwdz)
    FFT!(dVdz_dWdy, dvdz_dwdy)
    FFT!(dVdy_dWdz, dvdy_dwdz)

    # check if they agree
    @test dUdt ≈ cache.spec_cache[1]
    @test dVdt ≈ cache.spec_cache[2]
    @test dWdt ≈ cache.spec_cache[3]
    @test dUdz ≈ cache.spec_cache[4]
    @test dVdz ≈ cache.spec_cache[5]
    @test dWdz ≈ cache.spec_cache[6]
    @test d2Udz2 ≈ cache.spec_cache[7]
    @test d2Vdz2 ≈ cache.spec_cache[8]
    @test d2Wdz2 ≈ cache.spec_cache[9]
    @test dUdy ≈ cache.spec_cache[10]
    @test dVdy ≈ cache.spec_cache[11]
    @test dWdy ≈ cache.spec_cache[12]
    @test d2Udy2 ≈ cache.spec_cache[13]
    @test d2Vdy2 ≈ cache.spec_cache[14]
    @test d2Wdy2 ≈ cache.spec_cache[15]
    @test V_dUdy ≈ cache.spec_cache[20]
    @test W_dUdz ≈ cache.spec_cache[21]
    @test V_dVdy ≈ cache.spec_cache[22]
    @test W_dVdz ≈ cache.spec_cache[23]
    @test V_dWdy ≈ cache.spec_cache[24]
    @test W_dWdz ≈ cache.spec_cache[25]
    @test dVdz_dWdy ≈ cache.spec_cache[26]
    @test dVdy_dWdz ≈ cache.spec_cache[27]
end

@testset "Cache pressure update         " begin
    # initialise incompressible velocity field
    Ny = 32; Nz = 32; Nt = 32
    y = chebpts(Ny)
    Dy = chebdiff(Ny)
    Dy2 = chebddiff(Ny)
    ws = rand(Float64, Ny)
    ω = 1.0
    β = 1.0
    grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)
    u_fun(y, z, t) = sin(π*y)*exp(cos(z))*sin(t)
    v_fun(y, z, t) = (cos(π*y) + 1)*cos(z)*sin(t)
    w_fun(y, z, t) = π*sin(π*y)*sin(z)*sin(t)
    u = VectorField(PhysicalField(grid, u_fun),
                    PhysicalField(grid, v_fun),
                    PhysicalField(grid, w_fun))
    U = VectorField(grid)
    FFT! = FFTPlan!(grid; flags=FFTW.ESTIMATE)
    FFT!(U, u)

    # initialise cache and update pressure field
    Re = abs(rand()); Ro = abs(rand())
    cache = Cache(U[1], u[1], rand(Ny), rand(Ny), rand(Ny), rand(Ny), Re, Ro)
    NSOperators._update_vel!(U, cache)
    NSOperators._update_p!(U, cache)

    # compute laplacian of pressure field
    P = cache.spec_cache[16]
    d2Pdy2 = SpectralField(grid)
    d2Pdz2 = SpectralField(grid)
    ΔP = SpectralField(grid)
    d2dy2!(P, d2Pdy2)
    d2dz2!(P, d2Pdz2)
    ΔP .= d2Pdy2 .+ d2Pdz2

    # initialise rhs field
    # NOTE: the laplacian of pressure and rhs nonlinear terms do not individually match (at mean spanwise mode), but collectively (their difference) they do. why??? 
    rhs_fun(y, z, t) = 2*(π^2)*(sin(t)^2)*(cos(π*y)*(cos(π*y) + 1)*(sin(z)^2) - (sin(π*y)^2)*(cos(z)^2)) - Ro*π*cos(π*y)*exp(cos(z))*sin(t)
    rhs_phys = PhysicalField(grid, rhs_fun)
    rhs = SpectralField(grid)
    FFT!(rhs, rhs_phys)

    @test ΔP ≈ rhs
end
