@testset "Cache construction            " begin
    # initialise variables and arrays
    Ny = rand(3:50); Nz = rand(3:50); Nt = rand(3:50)
    y = rand(Float64, Ny)
    Dy = rand(Float64, (Ny, Ny))
    Dy2 = rand(Float64, (Ny, Ny))
    ws = rand(Float64, Ny)
    ω = abs(randn())
    β = abs(randn())
    ū = rand(Float64, Ny)
    dūdy = rand(Float64, Ny)
    d2ūdy2 = rand(Float64, Ny)
    Re = abs(randn()); Ro = randn()

    # construct fields
    grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)
    U = SpectralField(grid)
    u = PhysicalField(grid)

    # initialise useful types
    spec_type = SpectralField{Ny, Nz, Nt, typeof(grid), Float64, Array{ComplexF64, 3}}
    phys_type = PhysicalField{Ny, Nz, Nt, typeof(grid), Float64, Array{Float64, 3}}
    plan_type = Tuple{FFTPlan!{Ny, Nz, Nt, FFTW.rFFTWPlan{Float64, -1, false, 3, Vector{Int}}}, IFFTPlan!{Ny, Nz, Nt, FFTW.rFFTWPlan{ComplexF64, 1, false, 3, Vector{Int}}}}

    # construct local residual
    @test typeof(Cache(U, u, ū, dūdy, d2ūdy2, Re, Ro)) == Cache{Float64, spec_type, phys_type, Matrix{Complex{Float64}}, plan_type}

    # catch errors
    @test_throws ArgumentError Cache(SpectralField(Grid(rand(Ny - 1), Nz, Nt, Dy, Dy2, ws, ω, β)), u, ū, dūdy, d2ūdy2, Re, Ro)
    @test_throws ArgumentError Cache(U, u, rand(Ny - 1), dūdy, d2ūdy2, Re, Ro)
    @test_throws MethodError Cache(U, u, rand(Int, Ny), dūdy, d2ūdy2, Re, Ro)
    @test_throws MethodError Cache(U, u, ū, dūdy, d2ūdy2, 1+1im, Ro)
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
    Re = abs(rand()); Ro = abs(rand())
    cache = Cache(U[1], u[1], rand(Ny), rand(Ny), rand(Ny), Re, Ro)
    update_v!(U, cache)

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
    dvdz_dwdy_fun(y, z, t) = dvdz_fun(y, z, t)*dwdy_fun(y, z, t)
    dvdy_dwdz_fun(y, z, t) = dvdy_fun(y, z, t)*dwdz_fun(y, z, t)
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
    ws = chebws(Dy)
    ω = 1.0
    β = 1.0
    grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)
    dūdy_fun(y) = 1.0
    dūdy = [dūdy_fun(y[i]) for i in 1:Ny]
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
    cache = Cache(U[1], u[1], rand(Ny), dūdy, rand(Ny), Re, Ro)
    update_v!(U, cache)
    update_p!(cache)

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
    rhs_fun(y, z, t) = 2*(π^2)*(sin(t)^2)*(cos(π*y)*(cos(π*y) + 1)*(sin(z)^2) - (sin(π*y)^2)*(cos(z)^2)) - Ro*(π*cos(π*y)*exp(cos(z))*sin(t) + dūdy_fun(y))
    rhs_phys = PhysicalField(grid, rhs_fun)
    rhs = SpectralField(grid)
    FFT!(rhs, rhs_phys)

    @test ΔP ≈ rhs

    # initialise bc fields
    top_wall_fun(y, z, t) = cache.Re_recip*(π^2)*cos(z)*sin(t) - cache.Ro
    low_wall_fun(y, z, t) = cache.Re_recip*(π^2)*cos(z)*sin(t) + cache.Ro
    top_wall = SpectralField(grid)
    low_wall = SpectralField(grid)
    FFT!(top_wall, PhysicalField(grid, top_wall_fun))
    FFT!(low_wall, PhysicalField(grid, low_wall_fun))

    @test cache.spec_cache[17][1, :, :] ≈ top_wall[1, :, :]
    @test cache.spec_cache[17][end, :, :] ≈ low_wall[end, :, :]
end

@testset "Cache residual update         " begin
    # # initialise incompressible velocity field
    # Ny = 32; Nz = 32; Nt = 32
    # y = chebpts(Ny)
    # Dy = chebdiff(Ny)
    # Dy2 = chebddiff(Ny)
    # ws = rand(Float64, Ny)
    # ω = 1.0
    # β = 1.0
    # grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)
    # u_fun(y, z, t) = sin(π*y)*exp(cos(z))*sin(t)
    # v_fun(y, z, t) = (cos(π*y) + 1)*cos(z)*sin(t)
    # w_fun(y, z, t) = π*sin(π*y)*sin(z)*sin(t)
    # u = VectorField(PhysicalField(grid, u_fun),
    #                 PhysicalField(grid, v_fun),
    #                 PhysicalField(grid, w_fun))
    # U = VectorField(grid)
    # FFT! = FFTPlan!(grid; flags=FFTW.ESTIMATE)
    # FFT!(U, u)
    # Re = abs(rand())
    # Ro = abs(rand())

    # # initialise cache and update velocity field
    # Re = abs(rand()); Ro = abs(rand())
    # cache = Cache(U[1], u[1], rand(Ny), rand(Ny), rand(Ny), Re, Ro)
    # update_v!(U, cache)
    # update_p!(cache)
    # localresidual!(U, cache)
    # update_r!(cache)

    # # initialise fields for results
    # drxdt_fun(y, z, t) = 1.0
    # drydt_fun(y, z, t) = 1.0
    # drzdt_fun(y, z, t) = 1.0
    # drxdz_fun(y, z, t) = 1.0
    # drydz_fun(y, z, t) = 1.0
    # drzdz_fun(y, z, t) = 1.0
    # d2rxdz2_fun(y, z, t) = 1.0
    # d2rydz2_fun(y, z, t) = 1.0
    # d2rzdz2_fun(y, z, t) = 1.0
    # drxdy_fun(y, z, t) = 1.0
    # drydy_fun(y, z, t) = 1.0
    # drzdy_fun(y, z, t) = 1.0
    # d2rxdy2_fun(y, z, t) = 1.0
    # d2rydy2_fun(y, z, t) = 1.0
    # d2rzdy2_fun(y, z, t) = 1.0
    # dudz_fun(y, z, t) = -sin(π*y)*sin(z)*exp(cos(z))*sin(t)
    # dvdz_fun(y, z, t) = -(cos(π*y) + 1)*sin(z)*sin(t)
    # dwdz_fun(y, z, t) = π*sin(π*y)*cos(z)*sin(t)
    # dudy_fun(y, z, t) = π*cos(π*y)*exp(cos(z))*sin(t)
    # dvdy_fun(y, z, t) = -π*sin(π*y)*cos(z)*sin(t)
    # dwdy_fun(y, z, t) = (π^2)*cos(π*y)*sin(z)*sin(t)
    # v_drxdy_fun(y, z, t) = v_fun(y, z, t)*drxdy_fun(y, z, t)
    # w_drxdz_fun(y, z, t) = w_fun(y, z, t)*drxdz_fun(y, z, t)
    # v_drydy_fun(y, z, t) = v_fun(y, z, t)*drydy_fun(y, z, t)
    # w_drydz_fun(y, z, t) = w_fun(y, z, t)*drydz_fun(y, z, t)
    # v_drzdy_fun(y, z, t) = v_fun(y, z, t)*drzdy_fun(y, z, t)
    # w_drzdz_fun(y, z, t) = w_fun(y, z, t)*drzdz_fun(y, z, t)
    # rx_dudy_fun(y, z, t) = rx_fun(y, z, t)*dudy_fun(y, z, t)
    # ry_dvdy_fun(y, z, t) = ry_fun(y, z, t)*dvdy_fun(y, z, t)
    # rz_dwdy_fun(y, z, t) = rz_fun(y, z, t)*dwdy_fun(y, z, t)
    # rx_dudz_fun(y, z, t) = rx_fun(y, z, t)*dudz_fun(y, z, t)
    # ry_dvdz_fun(y, z, t) = ry_fun(y, z, t)*dvdz_fun(y, z, t)
    # rz_dwdz_fun(y, z, t) = rz_fun(y, z, t)*dwdz_fun(y, z, t)
    # drxdt_phys = PhysicalField(grid, drxdt_fun)
    # drydt_phys = PhysicalField(grid, drydt_fun)
    # drzdt_phys = PhysicalField(grid, drzdt_fun)
    # drxdz_phys = PhysicalField(grid, drxdz_fun)
    # drydz_phys = PhysicalField(grid, drydz_fun)
    # drzdz_phys = PhysicalField(grid, drzdz_fun)
    # d2rxdz2_phys = PhysicalField(grid, d2rxdz2_fun)
    # d2rydz2_phys = PhysicalField(grid, d2rydz2_fun)
    # d2rzdz2_phys = PhysicalField(grid, d2rzdz2_fun)
    # drxdy_phys = PhysicalField(grid, drxdy_fun)
    # drydy_phys = PhysicalField(grid, drydy_fun)
    # drzdy_phys = PhysicalField(grid, drzdy_fun)
    # d2rxdy2_phys = PhysicalField(grid, d2rxdy2_fun)
    # d2rydy2_phys = PhysicalField(grid, d2rydy2_fun)
    # d2rzdy2_phys = PhysicalField(grid, d2rzdy2_fun)
    # v_drxdy = PhysicalField(grid, v_drxdy_fun)
    # w_drxdz = PhysicalField(grid, w_drxdz_fun)
    # v_drydy = PhysicalField(grid, v_drydy_fun)
    # w_drydz = PhysicalField(grid, w_drydz_fun)
    # v_drzdy = PhysicalField(grid, v_drzdy_fun)
    # w_drzdz = PhysicalField(grid, w_drzdz_fun)
    # rx_dudy = PhysicalField(grid, rx_dudy_fun)
    # ry_dvdy = PhysicalField(grid, ry_dvdy_fun)
    # rz_dwdy = PhysicalField(grid, rz_dwdy_fun)
    # rx_dudz = PhysicalField(grid, rx_dudz_fun)
    # ry_dvdz = PhysicalField(grid, ry_dvdz_fun)
    # rz_dwdz = PhysicalField(grid, rz_dwdz_fun)
    # drxdt = SpectralField(grid)
    # drydt = SpectralField(grid)
    # drzdt = SpectralField(grid)
    # drxdz = SpectralField(grid)
    # drydz = SpectralField(grid)
    # drzdz = SpectralField(grid)
    # d2rxdz2 = SpectralField(grid)
    # d2rydz2 = SpectralField(grid)
    # d2rzdz2 = SpectralField(grid)
    # drxdy = SpectralField(grid)
    # drydy = SpectralField(grid)
    # drzdy = SpectralField(grid)
    # d2rxdy2 = SpectralField(grid)
    # d2rydy2 = SpectralField(grid)
    # d2rzdy2 = SpectralField(grid)
    # V_drxdy = SpectralField(grid)
    # W_drxdz = SpectralField(grid)
    # V_drydy = SpectralField(grid)
    # W_drydz = SpectralField(grid)
    # V_drzdy = SpectralField(grid)
    # W_drzdz = SpectralField(grid)
    # rx_dUdy = SpectralField(grid)
    # ry_dVdy = SpectralField(grid)
    # rz_dWdy = SpectralField(grid)
    # rx_dUdz = SpectralField(grid)
    # ry_dVdz = SpectralField(grid)
    # rz_dWdz = SpectralField(grid)
    # FFT!(drxdt, drxdt_phys)
    # FFT!(drydt, drydt_phys)
    # FFT!(drzdt, drzdt_phys)
    # FFT!(drxdz, drxdz_phys)
    # FFT!(drydz, drydz_phys)
    # FFT!(drzdz, drzdz_phys)
    # FFT!(d2rxdz2, d2rxdz2_phys)
    # FFT!(d2rydz2, d2rydz2_phys)
    # FFT!(d2rzdz2, d2rzdz2_phys)
    # FFT!(drxdy, drxdy_phys)
    # FFT!(drydy, drydy_phys)
    # FFT!(drzdy, drzdy_phys)
    # FFT!(d2rxdy2, d2rxdy2_phys)
    # FFT!(d2rydy2, d2rydy2_phys)
    # FFT!(d2rzdy2, d2rzdy2_phys)
    # FFT!(V_drxdy, v_drxdy)
    # FFT!(W_drxdz, w_drxdz)
    # FFT!(V_drydy, v_drydy)
    # FFT!(W_drydz, w_drydz)
    # FFT!(V_drzdy, v_drzdy)
    # FFT!(W_drzdz, w_drzdz)
    # FFT!(rx_dUdy, rx_dudy)
    # FFT!(ry_dVdy, ry_dvdy)
    # FFT!(rz_dWdy, rz_dwdy)
    # FFT!(rx_dUdz, rx_dudz)
    # FFT!(ry_dVdz, ry_dvdz)
    # FFT!(rz_dWdz, rz_dwdz)

    # # check if they agree
    # @test drxdt ≈ cache.spec_cache[42]
    # @test drydt ≈ cache.spec_cache[43]
    # @test drzdt ≈ cache.spec_cache[44]
    # @test drxdz ≈ cache.spec_cache[45]
    # @test drydz ≈ cache.spec_cache[46]
    # @test drzdz ≈ cache.spec_cache[47]
    # @test d2rxdz2 ≈ cache.spec_cache[48]
    # @test d2rydz2 ≈ cache.spec_cache[49]
    # @test d2rzdz2 ≈ cache.spec_cache[50]
    # @test drxdy ≈ cache.spec_cache[51]
    # @test drydy ≈ cache.spec_cache[52]
    # @test drzdy ≈ cache.spec_cache[53]
    # @test d2rxdy2 ≈ cache.spec_cache[54]
    # @test d2rydy2 ≈ cache.spec_cache[55]
    # @test d2rzdy2 ≈ cache.spec_cache[56]
    # @test V_drxdy ≈ cache.spec_cache[57]
    # @test W_drxdz ≈ cache.spec_cache[58]
    # @test V_drydy ≈ cache.spec_cache[59]
    # @test W_drydz ≈ cache.spec_cache[60]
    # @test V_drzdy ≈ cache.spec_cache[61]
    # @test W_drzdz ≈ cache.spec_cache[62]
    # @test rx_dUdy ≈ cache.spec_cache[63]
    # @test ry_dVdy ≈ cache.spec_cache[64]
    # @test rz_dWdy ≈ cache.spec_cache[65]
    # @test rx_dUdz ≈ cache.spec_cache[66]
    # @test ry_dVdz ≈ cache.spec_cache[67]
    # @test rz_dWdz ≈ cache.spec_cache[68]
end
