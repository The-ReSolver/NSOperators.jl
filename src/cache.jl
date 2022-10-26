# The file contains the definition for the global cache of the optimisation.

# TODO: can the typing of this struct be simplified to only include the information that is absolutely necessary
# TODO: add extra constructor that only takes ū and computes the gradients automatically
struct Cache{T, S, P, BC, PLANS}
    spec_cache::Vector{S}
    phys_cache::Vector{P}
    bc_cache::NTuple{2, BC}
    mean_data::NTuple{3, Vector{T}}
    plans::PLANS
    lapl::Laplace
    Re_recip::T
    Ro::T

    # constructor for a mean fixed field
    function Cache(U::S, u::P, ū::Vector{T}, dūdy::Vector{T}, d2ūdy2::Vector{T}, Re::T, Ro::T) where {T<:Real, S<:AbstractArray{Complex{T}, 3}, P<:AbstractArray{T, 3}}
        # check compatible sizes
        U.grid == u.grid || throw(ArgumentError("Fields are on different grids!"))
        (size(U.grid.y) == size(ū) && size(U.grid.y) == size(dūdy) && size(U.grid.y) == size(d2ūdy2)) || throw(ArgumentError("Mean data on different grids!"))

        # initialise cached arrays
        spec_cache = [similar(U) for _ in 1:69]
        phys_cache = [similar(u) for _ in 1:37]

        # initialise transforms
        FFT! = FFTPlan!(u)
        IFFT! = IFFTPlan!(U)

        # initialise laplace operator
        lapl = Laplace(size(u)[1], size(u)[2], get_β(u), get_Dy(u), get_Dy2(u))

        # initialise boundary data cache
        bc_cache = (Matrix{Complex{T}}(undef, size(U)[2], size(U)[3]),
                    Matrix{Complex{T}}(undef, size(U)[2], size(U)[3]))

        args = (spec_cache, phys_cache, bc_cache,
                (ū, dūdy, d2ūdy2), (FFT!, IFFT!), lapl, 1/Re, Ro)
        new{T, S, P, typeof(bc_cache[1]), typeof((FFT!, IFFT!))}(args...)
    end
end

Cache(grid::G, ū::Vector{T}, dūdy::Vector{T}, d2ūdy2::Vector{T}, Re::T, Ro::T) where {G, T<:Real} = Cache(SpectralField(grid), PhysicalField(grid), ū, dūdy, d2ūdy2, Re, Ro)

# methods to access type fields safely
ū(cache::Cache) = cache.mean_data[1]
dūdy(cache::Cache) = cache.mean_data[2]
d2ūdy2(cache::Cache) = cache.mean_data[3]
FFT!(cache::Cache) = cache.plans[1]
IFFT!(cache::Cache) = cache.plans[2]
spec_methods = [:dUdt, :dVdt, :dWdt, :dUdz, :dVdz, :dWdz, :d2Udz2, :d2Vdz2,
                :d2Wdz2, :dUdy, :dVdy, :dWdy, :d2Udy2, :d2Vdy2, :d2Wdy2, :P,
                :dPdy, :dPdz, :poiss_rhs, :V_dUdy, :W_dUdz, :V_dVdy, :W_dVdz,
                :V_dWdy, :W_dWdz, :dVdz_dWdy, :dVdy_dWdz, :ifft_tmp1,
                :ifft_tmp2, :ifft_tmp3, :ifft_tmp4, :ifft_tmp5, :ifft_tmp6,
                :ifft_tmp7, :ifft_tmp8, :rx, :ry, :rz, :drxdt, :drydt, :drzdt,
                :drxdz, :drydz, :drzdz, :d2rxdz2, :d2rydz2, :d2rzdz2, :drxdy,
                :drydy, :drzdy, :d2rxdy2, :d2rydy2, :d2rzdy2, :V_drxdy,
                :W_drxdz, :V_drydy, :W_drydz, :V_drzdy, :W_drzdz, :rx_dUdy,
                :ry_dVdy, :rz_dWdy, :rx_dUdz, :ry_dVdz, :rz_dWdz, :ifft_tmp9,
                :dℜx, :dℜy, :dℜz]
phys_methods = [:v, :w, :dudz, :dvdz, :dwdz, :dudy, :dvdy, :dwdy, :v_dudy,
                :w_dudz, :v_dvdy, :w_dvdz, :v_dwdy, :w_dwdz, :dvdz_dwdy,
                :dvdy_dwdz, :drxdz_p, :drydz_p, :drzdz_p, :drxdy_p, :drydy_p,
                :drzdy_p, :v_drxdy, :w_drxdz, :v_drydy, :w_drydz, :v_drzdy,
                :w_drzdz, :rx_dudy, :ry_dvdy, :rz_dwdy, :rx_dudz, :ry_dvdz,
                :rz_dwdz, :rx_p, :ry_p, :rz_p]
for (i, method) in enumerate(spec_methods)
    @eval begin
        ($method)(cache::Cache) = cache.spec_cache[$i]
    end
end
for (i, method) in enumerate(phys_methods)
    @eval begin
        ($method)(cache::Cache) = cache.phys_cache[$i]
    end
end

function update_v!(U::AbstractVector{S}, cache::Cache{T, S}) where {T, S}
    # assign spectral aliases
    dUdt = NSOperators.dUdt(cache)
    dVdt = NSOperators.dVdt(cache)
    dWdt = NSOperators.dWdt(cache)
    dUdz = NSOperators.dUdz(cache)
    dVdz = NSOperators.dVdz(cache)
    dWdz = NSOperators.dWdz(cache)
    d2Udz2 = NSOperators.d2Udz2(cache)
    d2Vdz2 = NSOperators.d2Vdz2(cache)
    d2Wdz2 = NSOperators.d2Wdz2(cache)
    dUdy = NSOperators.dUdy(cache)
    dVdy = NSOperators.dVdy(cache)
    dWdy = NSOperators.dWdy(cache)
    d2Udy2 = NSOperators.d2Udy2(cache)
    d2Vdy2 = NSOperators.d2Vdy2(cache)
    d2Wdy2 = NSOperators.d2Wdy2(cache)
    V_dUdy = NSOperators.V_dUdy(cache)
    W_dUdz = NSOperators.W_dUdz(cache)
    V_dVdy = NSOperators.V_dVdy(cache)
    W_dVdz = NSOperators.W_dVdz(cache)
    V_dWdy = NSOperators.V_dWdy(cache)
    W_dWdz = NSOperators.W_dWdz(cache)
    dVdz_dWdy = NSOperators.dVdz_dWdy(cache)
    dVdy_dWdz = NSOperators.dVdy_dWdz(cache)
    ifft_tmp1 = NSOperators.ifft_tmp1(cache)
    ifft_tmp2 = NSOperators.ifft_tmp2(cache)
    ifft_tmp3 = NSOperators.ifft_tmp3(cache)
    ifft_tmp4 = NSOperators.ifft_tmp4(cache)
    ifft_tmp5 = NSOperators.ifft_tmp5(cache)
    ifft_tmp6 = NSOperators.ifft_tmp6(cache)
    ifft_tmp7 = NSOperators.ifft_tmp7(cache)
    ifft_tmp8 = NSOperators.ifft_tmp8(cache)

    # assign physical aliases
    v = NSOperators.v(cache)
    w = NSOperators.w(cache)
    dudz = NSOperators.dudz(cache)
    dvdz = NSOperators.dvdz(cache)
    dwdz = NSOperators.dwdz(cache)
    dudy = NSOperators.dudy(cache)
    dvdy = NSOperators.dvdy(cache)
    dwdy = NSOperators.dwdy(cache)
    v_dudy = NSOperators.v_dudy(cache)
    w_dudz = NSOperators.w_dudz(cache)
    v_dvdy = NSOperators.v_dvdy(cache)
    w_dvdz = NSOperators.w_dvdz(cache)
    v_dwdy = NSOperators.v_dwdy(cache)
    w_dwdz = NSOperators.w_dwdz(cache)
    dvdz_dwdy = NSOperators.dvdz_dwdy(cache)
    dvdy_dwdz = NSOperators.dvdy_dwdz(cache)

    # assign transform aliases
    FFT! = NSOperators.FFT!(cache)
    IFFT! = NSOperators.IFFT!(cache)

    # compute derivatives
    @sync begin
        Base.Threads.@spawn ddt!(U[1], dUdt)
        Base.Threads.@spawn ddt!(U[2], dVdt)
        Base.Threads.@spawn ddt!(U[3], dWdt)
        Base.Threads.@spawn ddz!(U[1], dUdz)
        Base.Threads.@spawn ddz!(U[2], dVdz)
        Base.Threads.@spawn ddz!(U[3], dWdz)
        Base.Threads.@spawn d2dz2!(U[1], d2Udz2)
        Base.Threads.@spawn d2dz2!(U[2], d2Vdz2)
        Base.Threads.@spawn d2dz2!(U[3], d2Wdz2)
        Base.Threads.@spawn ddy!(U[1], dUdy)
        Base.Threads.@spawn ddy!(U[2], dVdy)
        Base.Threads.@spawn ddy!(U[3], dWdy)
        Base.Threads.@spawn d2dy2!(U[1], d2Udy2)
        Base.Threads.@spawn d2dy2!(U[2], d2Vdy2)
        Base.Threads.@spawn d2dy2!(U[3], d2Wdy2)
    end

    # compute nonlinear components
    @sync begin
        Base.Threads.@spawn IFFT!(v, U[2], ifft_tmp1)
        Base.Threads.@spawn IFFT!(w, U[3], ifft_tmp2)
        Base.Threads.@spawn IFFT!(dudz, dUdz, ifft_tmp3)
        Base.Threads.@spawn IFFT!(dvdz, dVdz, ifft_tmp4)
        Base.Threads.@spawn IFFT!(dwdz, dWdz, ifft_tmp5)
        Base.Threads.@spawn IFFT!(dudy, dUdy, ifft_tmp6)
        Base.Threads.@spawn IFFT!(dvdy, dVdy, ifft_tmp7)
        Base.Threads.@spawn IFFT!(dwdy, dWdy, ifft_tmp8)
    end

    @sync begin
        Base.Threads.@spawn v_dudy .= v.*dudy
        Base.Threads.@spawn w_dudz .= w.*dudz
        Base.Threads.@spawn v_dvdy .= v.*dvdy
        Base.Threads.@spawn w_dvdz .= w.*dvdz
        Base.Threads.@spawn v_dwdy .= v.*dwdy
        Base.Threads.@spawn w_dwdz .= w.*dwdz
        Base.Threads.@spawn dvdz_dwdy .= dvdz.*dwdy
        Base.Threads.@spawn dvdy_dwdz .= dvdy.*dwdz
    end

    @sync begin
        Base.Threads.@spawn FFT!(V_dUdy, v_dudy)
        Base.Threads.@spawn FFT!(W_dUdz, w_dudz)
        Base.Threads.@spawn FFT!(V_dVdy, v_dvdy)
        Base.Threads.@spawn FFT!(W_dVdz, w_dvdz)
        Base.Threads.@spawn FFT!(V_dWdy, v_dwdy)
        Base.Threads.@spawn FFT!(W_dWdz, w_dwdz)
        Base.Threads.@spawn FFT!(dVdz_dWdy, dvdz_dwdy)
        Base.Threads.@spawn FFT!(dVdy_dWdz, dvdy_dwdz)
    end

    return
end

function update_p!(cache::Cache{T, S}) where {T, S}
    # assign aliases
    dUdy = NSOperators.dUdy(cache)
    d2Vdy2 = NSOperators.d2Vdy2(cache)
    P = NSOperators.P(cache)
    dPdy = NSOperators.dPdy(cache)
    dPdz = NSOperators.dPdz(cache)
    poiss_rhs = NSOperators.poiss_rhs(cache)
    dVdz_dWdy = NSOperators.dVdz_dWdy(cache)
    dVdy_dWdz = NSOperators.dVdy_dWdz(cache)
    dūdy = NSOperators.dūdy(cache)

    # compute rhs of pressure equation
    # @. poiss_rhs = -2.0*(dVdz_dWdy - dVdy_dWdz) - cache.Ro*dUdy
    @. poiss_rhs = -cache.Ro*dUdy
    @views begin
        @. poiss_rhs[:, 1, 1] -= cache.Ro*dūdy
    end

    # extract boundary condition data
    # @views begin
    #     @. cache.bc_cache[1] = cache.Re_recip*d2Vdy2[1, :, :]
    #     @. cache.bc_cache[2] = cache.Re_recip*d2Vdy2[end, :, :]
    # end
    # cache.bc_cache[1][1, 1] = -cache.Ro
    # cache.bc_cache[2][1, 1] = cache.Ro
    cache.bc_cache[1] .= 0.0
    cache.bc_cache[2] .= 0.0

    # solve the pressure equation
    # solve!(P, cache.lapl, poiss_rhs, cache.bc_cache)
    solve!(P, cache.lapl, poiss_rhs)

    # compute pressure gradients
    ddy!(P, dPdy)
    ddz!(P, dPdz)

    return
end

function update_r!(cache::Cache{T, S}) where {T, S, V<:AbstractVector{S}}
    # assign spectral aliases
    ifft_tmp1 = NSOperators.ifft_tmp1(cache)
    ifft_tmp2 = NSOperators.ifft_tmp2(cache)
    ifft_tmp3 = NSOperators.ifft_tmp3(cache)
    ifft_tmp4 = NSOperators.ifft_tmp4(cache)
    ifft_tmp5 = NSOperators.ifft_tmp5(cache)
    ifft_tmp6 = NSOperators.ifft_tmp6(cache)
    ifft_tmp7 = NSOperators.ifft_tmp7(cache)
    ifft_tmp8 = NSOperators.ifft_tmp8(cache)
    rx = NSOperators.rx(cache)
    ry = NSOperators.ry(cache)
    rz = NSOperators.rz(cache)
    drxdt = NSOperators.drxdt(cache)
    drydt = NSOperators.drydt(cache)
    drzdt = NSOperators.drzdt(cache)
    drxdz = NSOperators.drxdz(cache)
    drydz = NSOperators.drydz(cache)
    drzdz = NSOperators.drzdz(cache)
    d2rxdz2 = NSOperators.d2rxdz2(cache)
    d2rydz2 = NSOperators.d2rydz2(cache)
    d2rzdz2 = NSOperators.d2rzdz2(cache)
    drxdy = NSOperators.drxdy(cache)
    drydy = NSOperators.drydy(cache)
    drzdy = NSOperators.drzdy(cache)
    d2rxdy2 = NSOperators.d2rxdy2(cache)
    d2rydy2 = NSOperators.d2rydy2(cache)
    d2rzdy2 = NSOperators.d2rzdy2(cache)
    V_drxdy = NSOperators.V_drxdy(cache)
    W_drxdz = NSOperators.W_drxdz(cache)
    V_drydy = NSOperators.V_drydy(cache)
    W_drydz = NSOperators.W_drydz(cache)
    V_drzdy = NSOperators.V_drzdy(cache)
    W_drzdz = NSOperators.W_drzdz(cache)
    rx_dUdy = NSOperators.rx_dUdy(cache)
    ry_dVdy = NSOperators.ry_dVdy(cache)
    rz_dWdy = NSOperators.rz_dWdy(cache)
    rx_dUdz = NSOperators.rx_dUdz(cache)
    ry_dVdz = NSOperators.ry_dVdz(cache)
    rz_dWdz = NSOperators.rz_dWdz(cache)
    ifft_tmp9 = NSOperators.ifft_tmp9(cache)


    # assign physical aliases
    v = NSOperators.v(cache)
    w = NSOperators.w(cache)
    dudz = NSOperators.dudz(cache)
    dvdz = NSOperators.dvdz(cache)
    dwdz = NSOperators.dwdz(cache)
    dudy = NSOperators.dudy(cache)
    dvdy = NSOperators.dvdy(cache)
    dwdy = NSOperators.dwdy(cache)
    drxdz_p = NSOperators.drxdz_p(cache)
    drydz_p = NSOperators.drydz_p(cache)
    drzdz_p = NSOperators.drzdz_p(cache)
    drxdy_p = NSOperators.drxdy_p(cache)
    drydy_p = NSOperators.drydy_p(cache)
    drzdy_p = NSOperators.drzdy_p(cache)
    v_drxdy = NSOperators.v_drxdy(cache)
    w_drxdz = NSOperators.w_drxdz(cache)
    v_drydy = NSOperators.v_drydy(cache)
    w_drydz = NSOperators.w_drydz(cache)
    v_drzdy = NSOperators.v_drzdy(cache)
    w_drzdz = NSOperators.w_drzdz(cache)
    rx_dudy = NSOperators.rx_dudy(cache)
    ry_dvdy = NSOperators.ry_dvdy(cache)
    rz_dwdy = NSOperators.rz_dwdy(cache)
    rx_dudz = NSOperators.rx_dudz(cache)
    ry_dvdz = NSOperators.ry_dvdz(cache)
    rz_dwdz = NSOperators.rz_dwdz(cache)
    rx_p = NSOperators.rx_p(cache)
    ry_p = NSOperators.ry_p(cache)
    rz_p = NSOperators.rz_p(cache)

    # assign transform aliases
    FFT! = NSOperators.FFT!(cache)
    IFFT! = NSOperators.IFFT!(cache)

    # compute derivatives
    @sync begin
        Base.Threads.@spawn ddt!(rx, drxdt)
        Base.Threads.@spawn ddt!(ry, drydt)
        Base.Threads.@spawn ddt!(rz, drzdt)
        Base.Threads.@spawn ddz!(rx, drxdz)
        Base.Threads.@spawn ddz!(ry, drydz)
        Base.Threads.@spawn ddz!(rz, drzdz)
        Base.Threads.@spawn d2dz2!(rx, d2rxdz2)
        Base.Threads.@spawn d2dz2!(ry, d2rydz2)
        Base.Threads.@spawn d2dz2!(rz, d2rzdz2)
        Base.Threads.@spawn ddy!(rx, drxdy)
        Base.Threads.@spawn ddy!(ry, drydy)
        Base.Threads.@spawn ddy!(rz, drzdy)
        Base.Threads.@spawn d2dy2!(rx, d2rxdy2)
        Base.Threads.@spawn d2dy2!(ry, d2rydy2)
        Base.Threads.@spawn d2dy2!(rz, d2rzdy2)
    end

    # compute nonlinear components
    @sync begin
        Base.Threads.@spawn IFFT!(rx_p, rx, ifft_tmp1)
        Base.Threads.@spawn IFFT!(ry_p, ry, ifft_tmp2)
        Base.Threads.@spawn IFFT!(rz_p, rz, ifft_tmp3)
        Base.Threads.@spawn IFFT!(drxdz_p, drxdz, ifft_tmp4)
        Base.Threads.@spawn IFFT!(drydz_p, drydz, ifft_tmp5)
        Base.Threads.@spawn IFFT!(drzdz_p, drzdz, ifft_tmp6)
        Base.Threads.@spawn IFFT!(drxdy_p, drxdy, ifft_tmp7)
        Base.Threads.@spawn IFFT!(drydy_p, drydy, ifft_tmp8)
        Base.Threads.@spawn IFFT!(drzdy_p, drzdy, ifft_tmp9)
    end

    @sync begin
        Base.Threads.@spawn v_drxdy .= v.*drxdy_p
        Base.Threads.@spawn w_drxdz .= w.*drxdz_p
        Base.Threads.@spawn v_drydy .= v.*drydy_p
        Base.Threads.@spawn w_drydz .= w.*drydz_p
        Base.Threads.@spawn v_drzdy .= v.*drzdy_p
        Base.Threads.@spawn w_drzdz .= w.*drzdz_p
        Base.Threads.@spawn rx_dudy .= rx_p.*dudy
        Base.Threads.@spawn ry_dvdy .= ry_p.*dvdy
        Base.Threads.@spawn rz_dwdy .= rz_p.*dwdy
        Base.Threads.@spawn rx_dudz .= rx_p.*dudz
        Base.Threads.@spawn ry_dvdz .= ry_p.*dvdz
        Base.Threads.@spawn rz_dwdz .= rz_p.*dwdz
    end

    @sync begin
        Base.Threads.@spawn FFT!(V_drxdy, v_drxdy)
        Base.Threads.@spawn FFT!(W_drxdz, w_drxdz)
        Base.Threads.@spawn FFT!(V_drydy, v_drydy)
        Base.Threads.@spawn FFT!(W_drydz, w_drydz)
        Base.Threads.@spawn FFT!(V_drzdy, v_drzdy)
        Base.Threads.@spawn FFT!(W_drzdz, w_drzdz)
        Base.Threads.@spawn FFT!(rx_dUdy, rx_dudy)
        Base.Threads.@spawn FFT!(ry_dVdy, ry_dvdy)
        Base.Threads.@spawn FFT!(rz_dWdy, rz_dwdy)
        Base.Threads.@spawn FFT!(rx_dUdz, rx_dudz)
        Base.Threads.@spawn FFT!(ry_dVdz, ry_dvdz)
        Base.Threads.@spawn FFT!(rz_dWdz, rz_dwdz)
    end

    return
end
