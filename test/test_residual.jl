@testset "Local residual constructor    " begin
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
    plan_type = Tuple{FFTPlan!{Ny, Nz, Nt, FFTW.rFFTWPlan{Float64, -1, false, 3, Vector{Int}}}, IFFTPlan!{Ny, Nz, Nt, FFTW.rFFTWPlan{ComplexF64, 1, false, 3, Vector{Int}}}}

    # construct local residual
    @test typeof(NSOperators._LocalResidual!(U, u, ū, dūdy, d2ūdy2, dp̄dy, Re, Ro)) == NSOperators._LocalResidual!{Float64, spec_type, phys_type, Matrix{Complex{Float64}}, plan_type}

    # catch errors
    @test_throws ArgumentError NSOperators._LocalResidual!(SpectralField(Grid(rand(Ny - 1), Nz, Nt, Dy, Dy2, ws, ω, β)), u, ū, dūdy, d2ūdy2, dp̄dy, Re, Ro)
    @test_throws ArgumentError NSOperators._LocalResidual!(U, u, rand(Ny - 1), dūdy, d2ūdy2, dp̄dy, Re, Ro)
    @test_throws MethodError NSOperators._LocalResidual!(U, u, rand(Int, Ny), dūdy, d2ūdy2, dp̄dy, Re, Ro)
    @test_throws MethodError NSOperators._LocalResidual!(U, u, ū, dūdy, d2ūdy2, dp̄dy, 1+1im, Ro)
end

@testset "Local residual calculation    " begin
    # initialise variables and arrays
    Ny = 16
    Nz = 16
    Nt = 16
    y = chebpts(Ny)
    Dy = chebdiff(Ny)
    Dy2 = chebddiff(Ny)
    ws = quadweights(y, 2)
    ω = 1.0
    β = 1.0
    Re = abs(randn())
    Ro = abs(randn())

    # initialise functions
    ū_fun(y) = y
    dūdy_fun(y) = 1.0
    d2ūdy2_fun(y) = 0.0
    dp̄dy_fun(y) = 2*y
    u_fun(y, z, t) = sin(π*y)*exp(cos(z))*sin(t)
    v_fun(y, z, t) = (cos(π*y) + 1)*cos(z)*sin(t)
    w_fun(y, z, t) = π*sin(π*y)*sin(z)*sin(t)
    d2vdy2_fun(y, z, t) = -(π^2)*cos(π*y)*cos(z)*sin(t)
    rhs_fun(y, z, t) = -(-2*π*(sin(z)^2)*sin(t)*(cos(π*y) + 1) - Ro*cos(π*y)exp(cos(z)))*sin(t)
    rx_fun_no_p(y, z, t) = sin(π*y)*exp(cos(z))*cos(t) + (1 - Ro)*(cos(π*y) + 1)*cos(z)*sin(t) - (1/Re)*sin(π*y)*exp(cos(z))*sin(t)*(sin(z)^2 - cos(z) - π^2) + π*exp(cos(z))*(sin(t)^2)*(cos(π*y)*(cos(π*y) + 1)*cos(z) - (sin(π*y)^2)*(sin(z)^2))
    ry_fun_no_p(y, z, t) = (cos(π*y) + 1)*cos(z)*cos(t) + (1/Re)*cos(z)*sin(t)*((π^2 + 1)*cos(π*y) + 1) + Ro*sin(π*y)*exp(cos(z))*sin(t) - π*(cos(π*y) + 1)*sin(π*y)*(sin(t)^2)
    rz_fun_no_p(y, z, t) = π*sin(π*y)*sin(z)*cos(t) + ((π*(π^2 + 1))/Re)*sin(π*y)*sin(z)*sin(t) + ((π^2)/2)*sin(2*z)*(sin(t)^2)*(cos(π*y) + 1)

    # initialise laplacian
    lapl = Laplace(Ny, Nz, β, Dy2, Dy)

    # initialise grid
    grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)

    # initialise transforms
    FFT! = FFTPlan!(grid)
    IFFT! = IFFTPlan!(grid)

    # initialise mean vectors
    ū = [ū_fun(y[i]) for i in 1:Ny]
    dūdy = [dūdy_fun(y[i]) for i in 1:Ny]
    d2ūdy2 = [d2ūdy2_fun(y[i]) for i in 1:Ny]
    dp̄dy = [dp̄dy_fun(y[i]) for i in 1:Ny]

    # initialise fields
    P = SpectralField(grid)
    dPdy = SpectralField(grid)
    dPdz = SpectralField(grid)
    d2vdy2 = PhysicalField(grid, d2vdy2_fun)
    d2Vdy2 = SpectralField(grid)
    rhs_phys = PhysicalField(grid, rhs_fun)
    rhs_spec = SpectralField(grid)
    u = VectorField(PhysicalField(grid, u_fun),
                    PhysicalField(grid, v_fun),
                    PhysicalField(grid, w_fun))
    U = VectorField(grid)
    res_phys = VectorField(PhysicalField(grid, rx_fun_no_p),
                            PhysicalField(grid, ry_fun_no_p),
                            PhysicalField(grid, rz_fun_no_p))
    res_spec = VectorField(grid)

    # transform physical fields to spectral fields
    FFT!(d2Vdy2, d2vdy2)
    FFT!(rhs_spec, rhs_phys)
    FFT!(U, u)
    FFT!(res_spec, res_phys)

    # initialise boundary data for pressure equation
    bc_data = ((1/Re).*d2Vdy2[1, :, :], (1/Re).*d2Vdy2[end, :, :])

    # solve pressure equation
    solve!(P, lapl, rhs_spec, bc_data)

    # compute pressure gradients
    ddy!(P, dPdy)
    ddz!(P, dPdz)

    # include pressure gradients in residual field
    res_spec[2] .+= dPdy
    res_spec[3] .+= dPdz

    # calculate local residual type
    local_residual! = NSOperators._LocalResidual!(U[1], u[1], ū, dūdy, d2ūdy2, dp̄dy, Re, Ro)
    res_calc = VectorField(grid)
    local_residual!(res_calc, U)
    display(round.(parent(res_calc[1]), digits=5)[5, :, :])
    println()
    display(round.(parent(res_spec[1]), digits=5)[5, :, :])
    # @test local_residual!(VectorField(grid), U)[1][:, 2, 2] ≈ res_spec[1][:, 2, 2] rtol=1e-2
end
