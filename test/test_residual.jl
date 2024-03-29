@testset "Local residual calculation    " begin
    # initialise variables and arrays
    Ny = 32; Nz = 32; Nt = 32
    y = chebpts(Ny)
    Dy = chebdiff(Ny)
    Dy2 = chebddiff(Ny)
    ws = chebws(Dy)
    ω = 1.0
    β = 1.0
    Re = abs(randn()); Ro = abs(randn())

    # initialise functions
    ū_fun(y) = y
    dūdy_fun(y) = 1.0
    d2ūdy2_fun(y) = 0.0
    r_mean_x_fun(y) = (π/2)*(cos(2*π*y) + cos(π*y))*0.56515910399 # modified bessel function of the first kind, first order evaluated at 1!
    r_mean_y_fun(y) = -(π/2)*sin(π*y)*(cos(π*y) + 1) + Ro*y
    r_mean_z_fun(y) = 0.0
    u_fun(y, z, t) = sin(π*y)*exp(cos(z))*sin(t)
    v_fun(y, z, t) = (cos(π*y) + 1)*cos(z)*sin(t)
    w_fun(y, z, t) = π*sin(π*y)*sin(z)*sin(t)
    d2vdy2_fun(y, z, t) = -(π^2)*cos(π*y)*cos(z)*sin(t)
    rhs_fun(y, z, t) = 2*(π^2)*(sin(t)^2)*(cos(π*y)*(cos(π*y) + 1)*(sin(z)^2) - (sin(π*y)^2)*(cos(z)^2)) - Ro*(π*cos(π*y)*exp(cos(z))*sin(t) + dūdy_fun(y))
    rx_fun_no_p(y, z, t) = sin(π*y)*exp(cos(z))*cos(t) + (1 - Ro)*(cos(π*y) + 1)*cos(z)*sin(t) - (1/Re)*sin(π*y)*exp(cos(z))*sin(t)*(sin(z)^2 - cos(z) - π^2) + π*exp(cos(z))*(sin(t)^2)*(cos(π*y)*(cos(π*y) + 1)*cos(z) - (sin(π*y)^2)*(sin(z)^2))
    ry_fun_no_p(y, z, t) = (cos(π*y) + 1)*cos(z)*cos(t) + (1/Re)*cos(z)*sin(t)*((π^2 + 1)*cos(π*y) + 1) + Ro*sin(π*y)*exp(cos(z))*sin(t) - π*(cos(π*y) + 1)*sin(π*y)*(sin(t)^2)
    rz_fun_no_p(y, z, t) = π*sin(π*y)*sin(z)*cos(t) + ((π*(π^2 + 1))/Re)*sin(π*y)*sin(z)*sin(t) + ((π^2)/2)*sin(2*z)*(sin(t)^2)*(cos(π*y) + 1)

    # initialise laplacian
    lapl = Laplace(Ny, Nz, β, Dy2, Dy)

    # initialise grid
    grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)

    # initialise transforms
    FFT! = FFTPlan!(grid; flags=ESTIMATE)

    # initialise mean vectors
    ū = [ū_fun(y[i]) for i in 1:Ny]
    dūdy = [dūdy_fun(y[i]) for i in 1:Ny]
    d2ūdy2 = [d2ūdy2_fun(y[i]) for i in 1:Ny]

    # initialise fields
    P = SpectralField(grid)
    dPdy = SpectralField(grid)
    dPdz = SpectralField(grid)
    d2vdy2 = PhysicalField(grid, d2vdy2_fun)
    d2Vdy2 = SpectralField(grid)
    rhs_phys = PhysicalField(grid, rhs_fun)
    rhs_spec = SpectralField(grid)
    u = VectorField(grid, u_fun, v_fun, w_fun)
    U = VectorField(grid)
    res_phys = VectorField(grid, rx_fun_no_p, ry_fun_no_p, rz_fun_no_p)
    res_spec = VectorField(grid)

    # transform physical fields to spectral fields
    FFT!(d2Vdy2, d2vdy2)
    FFT!(rhs_spec, rhs_phys)
    FFT!(U, u)
    FFT!(res_spec, res_phys)

    # initialise boundary data for pressure equation
    bc_data = ((1/Re).*d2Vdy2[1, :, :], (1/Re).*d2Vdy2[end, :, :])
    bc_data[1][1, 1] = -Ro
    bc_data[2][1, 1] = Ro

    # solve pressure equation
    solve!(P, lapl, rhs_spec, bc_data)

    # compute pressure gradients
    ddy!(P, dPdy)
    ddz!(P, dPdz)

    # include pressure gradients in residual field
    res_spec[2] .+= dPdy
    res_spec[3] .+= dPdz

    # compute the mean constraint vectors
    r_mean_x = [r_mean_x_fun(y[i]) for i in 1:Ny]
    r_mean_y = [r_mean_y_fun(y[i]) for i in 1:Ny]
    r_mean_z = [r_mean_z_fun(y[i]) for i in 1:Ny]

    # include mean constraint in residual fields
    @views begin
        res_spec[1][:, 1, 1] = r_mean_x
        res_spec[2][:, 1, 1] = r_mean_y + dPdy[:, 1, 1]
        res_spec[3][:, 1, 1] = r_mean_z
    end

    # calculate local residual type
    cache = Cache(U[1], u[1], ū, dūdy, d2ūdy2, Re, Ro)
    update_v!(U, cache)
    update_p!(cache)
    res_calc = localresidual!(U, cache)
    # @test res_calc ≈ res_spec

    i = 3
    Ny = 2
    Nz = 8
    Nt = 8
    display(round.(res_calc[i][Ny, 1:Nz, 1:Nt]; digits=5))
    println()
    display(round.(res_spec[i][Ny, 1:Nz, 1:Nt]; digits=5))
    println()
    println(res_calc[i] ≈ res_spec[i])

    # divergence
    drydy = SpectralField(grid)
    drzdz = SpectralField(grid)
    div_r = SpectralField(grid)
    ddy!(res_calc[2], drydy)
    ddz!(res_calc[3], drzdz)
    div_r .= drydy .+ drzdz
    # @test norm(div_r) < 5e-8

    # boundary values
    d2udy2_fun(y, z, t) = -(π^2)*sin(π*y)*exp(cos(z))*sin(t)
    d2wdy2_fun(y, z, t) = -(π^3)*sin(π*y)*sin(z)*sin(t)
    d2Udy2 = SpectralField(grid)
    d2Wdy2 = SpectralField(grid)
    FFT!(d2Udy2, PhysicalField(grid, d2udy2_fun))
    FFT!(d2Wdy2, PhysicalField(grid, d2wdy2_fun))
    # @test res_calc[1][1, :, :] ≈ (-1/Re)*d2Udy2[1, :, :] atol=1e-6 # NOTE: tolerances is here to deal with zero arrays
    # @test res_calc[1][end, :, :] ≈ (-1/Re)*d2Udy2[end, :, :] atol=1e-6
    # @test res_calc[2][1, :, :] ≈ SpectralField(grid)[1, :, :] atol=1e-6
    # @test res_calc[2][end, :, :] ≈ SpectralField(grid)[end, :, :] atol=1e-6
    # @test res_calc[3][1, :, :] ≈ (-1/Re)*d2Wdy2[1, :, :] + dPdz[1, :, :]
    # @test res_calc[3][end, :, :] ≈ (-1/Re)*d2Wdy2[end, :, :] + dPdz[end, :, :]
end

@testset "Global residual   " begin
    # # initialise variables and arrays
    # Ny = 32; Nz = 32; Nt = 32
    # y = chebpts(Ny)
    # Dy = chebdiff(Ny)
    # Dy2 = chebddiff(Ny)
    # ws = chebws(Dy)
    # ω = 1.0
    # β = 1.0
    # Re = 1.0; Ro = 1.0

    # # initialise functions
    # ū_fun(y) = y
    # dūdy_fun(y) = 1.0
    # d2ūdy2_fun(y) = 0.0
    # u_fun(y, z, t) = (1 - y^2)*exp(cos(z))*sin(t)
    # v_fun(y, z, t) = (cos(π*y) + 1)*cos(z)*sin(t)
    # w_fun(y, z, t) = π*sin(π*y)*sin(z)*cos(sin(t))

    # # initialise laplacian
    # lapl = Laplace(Ny, Nz, β, Dy2, Dy)

    # # initialise grid
    # grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)

    # # initialise transforms
    # FFT! = FFTPlan!(grid; flags=ESTIMATE)

    # # initialise mean vectors
    # ū = [ū_fun(y[i]) for i in 1:Ny]
    # dūdy = [dūdy_fun(y[i]) for i in 1:Ny]
    # d2ūdy2 = [d2ūdy2_fun(y[i]) for i in 1:Ny]

    # # initialise fields
    # u = VectorField(grid, u_fun, v_fun, w_fun)
    # U = VectorField(grid)
    # FFT!(U, u)

    # # calculate reisudal
    # cache = Cache(U[1], u[1], ū, dūdy, d2ūdy2, Re, Ro)
    # update_v!(U, cache)
    # update_p!(cache)
    # localresidual!(U, cache)
    # res = ℜ(cache)

    # # pressure norm components
    # dPdy = copy(cache.spec_cache[17])
    # dPdz = copy(cache.spec_cache[18])
    # lry_no_pr = copy(cache.spec_cache[37])
    # lrz_no_pr = copy(cache.spec_cache[38])
    # lry_no_pr .= lry_no_pr .- dPdy
    # lrz_no_pr .= lrz_no_pr .- dPdz
    # ∇P_norm = 0.5*(norm(VectorField(dPdy, dPdz))^2)
    # NS_pressure_dot = Fields.dot(VectorField(lry_no_pr, lrz_no_pr), VectorField(dPdy, dPdz))

    # # NOTE: this is done since the mathematica script this answer is derived
    # #       from doesn't include pressure
    # @test res - NS_pressure_dot - ∇P_norm ≈ 8253.897168444093
end

@testset "Residual gradient calculation " begin
    # # initialise grid
    # Ny = 32; Nz = 32; Nt = 32
    # y = chebpts(Ny)
    # # Dy = chebdiff(Ny); Dy2 = chebddiff(Ny)
    # Dy = DiffMatrix(y, 5, 1); Dy2 = DiffMatrix(y, 5, 2)
    # ws = chebws(Ny)
    # ω = 1.0
    # β = 1.0
    # grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)

    # # initialise transform plans
    # FFT! = FFTPlan!(grid; flags=ESTIMATE)

    # # initialise fluctuation velocity field
    # u_fun(y, z, t) = sin(π*y)*exp(cos(z))*sin(t)
    # v_fun(y, z, t) = (cos(π*y) + 1)*cos(z)*sin(t)
    # w_fun(y, z, t) = π*sin(π*y)*sin(z)*sin(t)
    # u = VectorField(grid, u_fun, v_fun, w_fun)
    # U = VectorField(grid)
    # FFT!(U, u)

    # # initialise mean fields
    # ū_fun(y) = y
    # dūdy_fun(y) = 1.0
    # d2ūdy2_fun(y) = 0.0
    # dp̄dy_fun(y) = 2*y
    # ū = [ū_fun(y[i]) for i in 1:Ny]
    # dūdy = [dūdy_fun(y[i]) for i in 1:Ny]
    # d2ūdy2 = [d2ūdy2_fun(y[i]) for i in 1:Ny]
    # dp̄dy = [dp̄dy_fun(y[i]) for i in 1:Ny]

    # # initialise cache
    # # Re = abs(rand()); Ro = abs(rand())
    # Re = 1.0; Ro = 1.0
    # cache = Cache(U[1], u[1], ū, dūdy, d2ūdy2, Re, Ro)

    # # calculate residual gradient
    # update_v!(U, cache)
    # update_p!(cache)
    # localresidual!(U, cache)
    # update_r!(cache)
    # grad_calc = dℜ!(cache)

    # # * other useful functions
    # dudy_fun(y, z, t) = π*cos(π*y)*exp(cos(z))*sin(t)
    # dudz_fun(y, z, t) = -sin(π*y)*sin(z)*exp(cos(z))*sin(t)
    # d2udy2_fun(y, z, t) = -(π^2)*sin(π*y)*exp(cos(z))*sin(t)
    # d2udz2_fun(y, z, t) = sin(π*y)*((sin(z)^2) - cos(z))*exp(cos(z))*sin(t)
    # d2udydz_fun(y, z, t) = -π*cos(π*y)*sin(z)*exp(cos(z))*sin(t)
    # dvdt_fun(y, z, t) = (cos(π*y) + 1)*cos(z)*cos(t)
    # dvdy_fun(y, z, t) = -π*sin(π*y)*cos(z)*sin(t)
    # dvdz_fun(y, z, t) = -(cos(π*y) + 1)*sin(z)*sin(t)
    # dwdt_fun(y, z, t) = π*sin(π*y)*sin(z)*cos(t)
    # dwdy_fun(y, z, t) = (π^2)*cos(π*y)*sin(z)*sin(t)
    # dwdz_fun(y, z, t) = π*sin(π*y)*cos(z)*sin(t)

    # # TODO: setup test that uses exact functions for residual gradient

    # # NOTE: V_dVdy
    # # d2vdy2_fun(y, z, t) = -(π^2)*cos(π*y)*cos(z)*sin(t)
    # # fun2(y, z, t) = -(v_fun(y, z, t)^2)*d2vdy2_fun(y, z, t)
    # # field2 = SpectralField(grid)
    # # FFT!(field2, PhysicalField(grid, fun2))

    # # NOTE: W_dVdz
    # # dvdz_fun(y, z, t) = -(cos(π*y) + 1)*sin(z)*sin(t)
    # # dwdz_fun(y, z, t) = π*sin(π*y)*cos(z)*sin(t)
    # # d2vdz2_fun(y, z, t) = -(cos(π*y) + 1)*cos(z)*sin(t)
    # # w_dvdz_sqrd_fun(y, z, t) = w_fun(y, z, t)*(dvdz_fun(y, z, t)^2)
    # # fun2(y, z, t) = -w_fun(y, z, t)*dwdz_fun(y, z, t)*dvdz_fun(y, z, t) - (w_fun(y, z, t)^2)*d2vdz2_fun(y, z, t)
    # # W_dVdz_sqrd = SpectralField(grid)
    # # field2 = SpectralField(grid)
    # # FFT!(W_dVdz_sqrd, PhysicalField(grid, w_dvdz_sqrd_fun))
    # # FFT!(field2, PhysicalField(grid, fun2))

    # # NOTE: V_dWdy
    # # dwdy_fun(y, z, t) = (π^2)*cos(π*y)*sin(z)*sin(t)
    # # dvdy_fun(y, z, t) = -π*sin(π*y)*cos(z)*sin(t)
    # # d2wdy2_fun(y, z, t) = -(π^3)*sin(π*y)*sin(z)*sin(t)
    # # v_dwdy_sqrd_fun(y, z, t) = v_fun(y, z, t)*(dwdy_fun(y, z, t)^2)
    # # fun2(y, z, t) = -v_fun(y, z, t)*dvdy_fun(y, z, t)*dwdy_fun(y, z, t) - (v_fun(y, z, t)^2)*d2wdy2_fun(y, z, t)
    # # V_dWdy_sqrd = SpectralField(grid)
    # # field2 = SpectralField(grid)
    # # FFT!(V_dWdy_sqrd, PhysicalField(grid, v_dwdy_sqrd_fun))
    # # FFT!(field2, PhysicalField(grid, fun2))

    # # NOTE: W_dWdz
    # # d2wdz2_fun(y, z, t) = -π*sin(π*y)*sin(z)*sin(t)
    # # fun2(y, z, t) = -(w_fun(y, z, t)^2)*d2wdz2_fun(y, z, t)
    # # field2 = SpectralField(grid)
    # # FFT!(field2, PhysicalField(grid, fun2))

    # # esimate residual gradient using differencing
    # grad_fd = VectorField(grid)
    # dℜ_fd!(U, cache, grad_fd; step=1e-3)

    # # * r = V_dUdy + W_dUdz GOOD
    # # dℜx_fun(y, z, t) = -v_fun(y, z, t)*dvdy_fun(y, z, t)*dudy_fun(y, z, t) - (v_fun(y, z, t)^2)*d2udy2_fun(y, z, t) - v_fun(y, z, t)*dwdy_fun(y, z, t)*dudz_fun(y, z, t) -
    # #                     2*v_fun(y, z, t)*w_fun(y, z, t)*d2udydz_fun(y, z, t) - w_fun(y, z, t)*dvdz_fun(y, z, t)*dudy_fun(y, z, t) -
    # #                     w_fun(y, z, t)*dwdz_fun(y, z, t)*dudz_fun(y, z, t) - (w_fun(y, z, t)^2)*d2udz2_fun(y, z, t)
    # # dℜy_fun(y, z, t) = v_fun(y, z, t)*(dudy_fun(y, z, t)^2) + w_fun(y, z, t)*dudy_fun(y, z, t)*dudz_fun(y, z, t)
    # # dℜz_fun(y, z, t) = v_fun(y, z, t)*dudy_fun(y, z, t)*dudz_fun(y, z, t) + w_fun(y, z, t)*(dudz_fun(y, z, t)^2)
    # # dℜ_p = VectorField(grid, dℜx_fun, dℜy_fun, dℜz_fun)
    # # dℜ_s = VectorField(grid)
    # # FFT!(dℜ_s, dℜ_p)

    # # * r = V_dUdy GOOD
    # dℜx_fun(y, z, t) = -v_fun(y, z, t)*dvdy_fun(y, z, t)*dudy_fun(y, z, t) - (v_fun(y, z, t)^2)*d2udy2_fun(y, z, t)
    # dℜy_fun(y, z, t) = v_fun(y, z, t)*(dudy_fun(y, z, t)^2)
    # dℜz_fun(y, z, t) = 0.0
    # dℜ_p = VectorField(grid, dℜx_fun, dℜy_fun, dℜz_fun)
    # dℜ_s = VectorField(grid)
    # FFT!(dℜ_s, dℜ_p)

    # # * r = W_dUdz GOOD
    # # dℜx_fun(y, z, t) = -w_fun(y, z, t)*dwdz_fun(y, z, t)*dudz_fun(y, z, t) - (w_fun(y, z, t)^2)*d2udz2_fun(y, z, t)
    # # dℜy_fun(y, z, t) = 0.0
    # # dℜz_fun(y, z, t) = w_fun(y, z, t)*(dudz_fun(y, z, t)^2)
    # # dℜ_p = VectorField(grid, dℜx_fun, dℜy_fun, dℜz_fun)
    # # dℜ_s = VectorField(grid)
    # # FFT!(dℜ_s, dℜ_p)

    # # * r = v*dūdy GOOD
    # # dℜx_fun(y, z, t) = 0.0
    # # dℜy_fun(y, z, t) = v_fun(y, z, t)*dūdy_fun(y)
    # # dℜz_fun(y, z, t) = 0.0
    # # dℜ_p = VectorField(grid, dℜx_fun, dℜy_fun, dℜz_fun)
    # # dℜ_s = VectorField(grid)
    # # FFT!(dℜ_s, dℜ_p)

    # n = 1
    # a = 8
    # b = 8
    # c = b
    # display(round.(dℜ_s[n][a, 1:b, 1:c]; digits=10))
    # println()
    # display(round.(grad_calc[n][a, 1:b, 1:c]; digits=10))
    # println()
    # display(round.(grad_fd[n][a, 1:b, 1:c]; digits=10))
    # println()
    # # display(round.(dℜ_s[n][a, 1:b, 1:c]./grad_calc[n][a, 1:b, 1:c]; digits=10))
    # # println()

    # # println(≈(dℜ_s[n][a, 1:b, 1:c], grad_calc[n][a, 1:b, 1:c]; rtol=1e-3))

    # println(≈(dℜ_s[1], grad_calc[1]; rtol=1e-3))
    # println(≈(dℜ_s[2], grad_calc[2]; rtol=1e-4))
    # println(≈(dℜ_s[3], grad_calc[3]; rtol=1e-4))

    # println(≈(grad_fd[1], grad_calc[1]; rtol=1e-3))
    # println(≈(grad_fd[2], grad_calc[2]; rtol=1e-4))
    # println(≈(grad_fd[3], grad_calc[3]; rtol=1e-4))

    # # a = 2
    # # display(round.(grad_fd[comp][a, 1:8, 1:10]; digits=5))
    # # println()
    # # display(round.(grad_calc[comp][a, 1:8, 1:10]; digits=5))
    # # println()
    # # display(round.(field2[a, 1:8, 1:10]; digits=5))
    # # @test VectorField(grad_calc...) ≈ grad_fd

    # # IFFT! = IFFTPlan!(grid; flags=ESTIMATE)
    # # grad_fd_phys = VectorField(grid; field_type=:physical)
    # # grad_calc_phys = VectorField(grid; field_type=:physical)
    # # IFFT!(grad_fd_phys, grad_fd, VectorField(grid))
    # # IFFT!(grad_calc_phys, grad_calc, VectorField(grid))

    # # using Plots
    # # # ENV["GKSwstype"] = "nul"
    # # for nt in 1:Nt
    # #     @views begin
    # #         p1 = contour(points(grid)[2], sort(points(grid)[1]), grad_fd_phys[1][:, :, nt])
    # #         p2 = contour(points(grid)[2], sort(points(grid)[1]), grad_calc_phys[1][:, :, nt])
    # #     end
    # #     plot(p1, p2, layout=(1, 2), legend=false)
    # #     savefig("./plots/dRx_Nt=$(string(nt)).pdf")
    # # end
    # # for nt in 1:Nt
    # #     @views begin
    # #         p1 = contour(points(grid)[2], sort(points(grid)[1]), grad_fd_phys[2][:, :, nt])
    # #         p2 = contour(points(grid)[2], sort(points(grid)[1]), grad_calc_phys[2][:, :, nt])
    # #     end
    # #     plot(p1, p2, layour=(1, 2), legend=false)
    # #     savefig("./plots/dRy_Nt=$(string(nt)).pdf")
    # # end
    # # for nt in 1:Nt
    # #     @views begin
    # #         p1 = contour(points(grid)[2], sort(points(grid)[1]), grad_fd_phys[3][:, :, nt])
    # #         p2 = contour(points(grid)[2], sort(points(grid)[1]), grad_calc_phys[3][:, :, nt])
    # #     end
    # #     plot(p1, p2, layour=(1, 2), legend=false)
    # #     savefig("./plots/dRz_Nt=$(string(nt)).pdf")
    # # end
end

@testset "Residual gradient " begin
    # # initialise grid
    # Ny = 32; Nz = 32; Nt = 32
    # y = chebpts(Ny)
    # Dy = DiffMatrix(y, 5, 1); Dy2 = DiffMatrix(y, 5, 2)
    # ws = chebws(Ny)
    # ω = 1.0
    # β = 1.0
    # grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)
    
    # # initialise transform plans
    # FFT! = FFTPlan!(grid; flags=ESTIMATE)

    # # initialise fluctuation velocity field
    # u_fun(y, z, t) = sin(π*y)*exp(cos(z))*sin(t)
    # v_fun(y, z, t) = (cos(π*y) + 1)*cos(z)*sin(t)
    # w_fun(y, z, t) = π*sin(π*y)*sin(z)*sin(t)
    # u = VectorField(grid, u_fun, v_fun, w_fun)
    # U = VectorField(grid)
    # FFT!(U, u)

    # # initialise mean fields
    # ū_fun(y) = y
    # dūdy_fun(y) = 1.0
    # d2ūdy2_fun(y) = 0.0
    # dp̄dy_fun(y) = 2*y
    # ū = [ū_fun(y[i]) for i in 1:Ny]
    # dūdy = [dūdy_fun(y[i]) for i in 1:Ny]
    # d2ūdy2 = [d2ūdy2_fun(y[i]) for i in 1:Ny]
    # dp̄dy = [dp̄dy_fun(y[i]) for i in 1:Ny] # ? is this required ?

    # # compute exact residual gradient field


    # # initialise cache
    # # Re = abs(rand()); Ro = abs(rand())
    # Re = 1.0; Ro = 1.0
    # cache = Cache(U[1], u[1], ū, dūdy, d2ūdy2, Re, Ro)

    # # calculate residual gradient
    # update_v!(U, cache)
    # update_p!(cache)
    # localresidual!(U, cache)
    # update_r!(cache)
    # grad_calc = dℜ!(cache)
end
