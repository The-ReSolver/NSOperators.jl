# The file contains the definition for the global cache of the optimisation.

# TODO: can the typing of this struct be simplified to only include the information that is absolutely necessary
#       i.e. simplify the type information to reduce redundancy as much as possible
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
        lapl = Laplace(size(u)[1], size(u)[2], get_β(get_grid(u)), u.grid.Dy[2], u.grid.Dy[1])

        # initialise boundary data cache
        bc_cache = (Matrix{Complex{T}}(undef, size(U)[2], size(U)[3]),
                    Matrix{Complex{T}}(undef, size(U)[2], size(U)[3]))

        args = (spec_cache, phys_cache, bc_cache,
                (ū, dūdy, d2ūdy2), (FFT!, IFFT!), lapl, 1/Re, Ro)
        new{T, S, P, typeof(bc_cache[1]), typeof((FFT!, IFFT!))}(args...)
    end
end

# TODO: can I create methods for grid the fields without having to call them explicitely?
Cache(grid::G, ū::Vector{T}, dūdy::Vector{T}, d2ūdy2::Vector{T}, Re::T, Ro::T) where {G, T<:Real} = Cache(SpectralField(grid), PhysicalField(grid), ū, dūdy, d2ūdy2, Re, Ro)

# methods to access type fields safely
ū(cache::Cache) = cache.mean_data[1]
dūdy(cache::Cache) = cache.mean_data[2]
d2ūdy2(cache::Cache) = cache.mean_data[3]
dUdt(cache::Cache) = cache.spec_cache[1]
dVdt(cache::Cache) = cache.spec_cache[2]
dWdt(cache::Cache) = cache.spec_cache[3]
dUdz(cache::Cache) = cache.spec_cache[4]
dVdz(cache::Cache) = cache.spec_cache[5]
dWdz(cache::Cache) = cache.spec_cache[6]
d2Udz2(cache::Cache) = cache.spec_cache[7]
d2Vdz2(cache::Cache) = cache.spec_cache[8]
d2Wdz2(cache::Cache) = cache.spec_cache[9]
dUdy(cache::Cache) = cache.spec_cache[10]
dVdy(cache::Cache) = cache.spec_cache[11]
dWdy(cache::Cache) = cache.spec_cache[12]
d2Udy2(cache::Cache) = cache.spec_cache[13]
d2Vdy2(cache::Cache) = cache.spec_cache[14]
d2Wdy2(cache::Cache) = cache.spec_cache[15]
P(cache::Cache) = cache.spec_cache[16]
dPdy(cache::Cache) = cache.spec_cache[17]
dPdz(cache::Cache) = cache.spec_cache[18]
poiss_rhs(cache::Cache) = cache.spec_cache[19]
V_dUdy(cache::Cache) = cache.spec_cache[20]
W_dUdz(cache::Cache) = cache.spec_cache[21]
V_dVdy(cache::Cache) = cache.spec_cache[22]
W_dVdz(cache::Cache) = cache.spec_cache[23]
V_dWdy(cache::Cache) = cache.spec_cache[24]
W_dWdz(cache::Cache) = cache.spec_cache[25]
dVdz_dWdy(cache::Cache) = cache.spec_cache[26]
dVdy_dWdz(cache::Cache) = cache.spec_cache[27]
ifft_tmp1(cache::Cache) = cache.spec_cache[28]
ifft_tmp2(cache::Cache) = cache.spec_cache[29]
ifft_tmp3(cache::Cache) = cache.spec_cache[30]
ifft_tmp4(cache::Cache) = cache.spec_cache[31]
ifft_tmp5(cache::Cache) = cache.spec_cache[32]
ifft_tmp6(cache::Cache) = cache.spec_cache[33]
ifft_tmp7(cache::Cache) = cache.spec_cache[34]
ifft_tmp8(cache::Cache) = cache.spec_cache[35]
rx(cache::Cache) = cache.spec_cache[36]
ry(cache::Cache) = cache.spec_cache[37]
rz(cache::Cache) = cache.spec_cache[38]
drxdt(cache::Cache) = cache.spec_cache[42]
drydt(cache::Cache) = cache.spec_cache[43]
drzdt(cache::Cache) = cache.spec_cache[44]
drxdz(cache::Cache) = cache.spec_cache[45]
drydz(cache::Cache) = cache.spec_cache[46]
drzdz(cache::Cache) = cache.spec_cache[47]
d2rxdz2(cache::Cache) = cache.spec_cache[48]
d2rydz2(cache::Cache) = cache.spec_cache[49]
d2rzdz2(cache::Cache) = cache.spec_cache[50]
drxdy(cache::Cache) = cache.spec_cache[51]
drydy(cache::Cache) = cache.spec_cache[52]
drzdy(cache::Cache) = cache.spec_cache[53]
d2rxdy2(cache::Cache) = cache.spec_cache[54]
d2rydy2(cache::Cache) = cache.spec_cache[55]
d2rzdy2(cache::Cache) = cache.spec_cache[56]
V_drxdy(cache::Cache) = cache.spec_cache[57]
W_drxdz(cache::Cache) = cache.spec_cache[58]
V_drydy(cache::Cache) = cache.spec_cache[59]
W_drydz(cache::Cache) = cache.spec_cache[60]
V_drzdy(cache::Cache) = cache.spec_cache[61]
W_drzdz(cache::Cache) = cache.spec_cache[62]
rx_dUdy(cache::Cache) = cache.spec_cache[63]
ry_dVdy(cache::Cache) = cache.spec_cache[64]
rz_dWdy(cache::Cache) = cache.spec_cache[65]
rx_dUdz(cache::Cache) = cache.spec_cache[66]
ry_dVdz(cache::Cache) = cache.spec_cache[67]
rz_dWdz(cache::Cache) = cache.spec_cache[68]
ifft_tmp9(cache::Cache) = cache.spec_cache[69]
v(cache::Cache) = cache.phys_cache[1]
w(cache::Cache) = cache.phys_cache[2]
dudz(cache::Cache) = cache.phys_cache[3]
dvdz(cache::Cache) = cache.phys_cache[4]
dwdz(cache::Cache) = cache.phys_cache[5]
dudy(cache::Cache) = cache.phys_cache[6]
dvdy(cache::Cache) = cache.phys_cache[7]
dwdy(cache::Cache) = cache.phys_cache[8]
v_dudy(cache::Cache) = cache.phys_cache[9]
w_dudz(cache::Cache) = cache.phys_cache[10]
v_dvdy(cache::Cache) = cache.phys_cache[11]
w_dvdz(cache::Cache) = cache.phys_cache[12]
v_dwdy(cache::Cache) = cache.phys_cache[13]
w_dwdz(cache::Cache) = cache.phys_cache[14]
dvdz_dwdy(cache::Cache) = cache.phys_cache[15]
dvdy_dwdz(cache::Cache) = cache.phys_cache[16]
drxdz_p(cache::Cache) = cache.phys_cache[17]
drydz_p(cache::Cache) = cache.phys_cache[18]
drzdz_p(cache::Cache) = cache.phys_cache[19]
drxdy_p(cache::Cache) = cache.phys_cache[20]
drydy_p(cache::Cache) = cache.phys_cache[21]
drzdy_p(cache::Cache) = cache.phys_cache[22]
v_drxdy(cache::Cache) = cache.phys_cache[23]
w_drxdz(cache::Cache) = cache.phys_cache[24]
v_drydy(cache::Cache) = cache.phys_cache[25]
w_drydz(cache::Cache) = cache.phys_cache[26]
v_drzdy(cache::Cache) = cache.phys_cache[27]
w_drzdz(cache::Cache) = cache.phys_cache[28]
rx_dudy(cache::Cache) = cache.phys_cache[29]
ry_dvdy(cache::Cache) = cache.phys_cache[30]
rz_dwdy(cache::Cache) = cache.phys_cache[31]
rx_dudz(cache::Cache) = cache.phys_cache[32]
ry_dvdz(cache::Cache) = cache.phys_cache[33]
rz_dwdz(cache::Cache) = cache.phys_cache[34]
rx_p(cache::Cache) = cache.phys_cache[35]
ry_p(cache::Cache) = cache.phys_cache[36]
rz_p(cache::Cache) = cache.phys_cache[37]
FFT!(cache::Cache) = cache.plans[1]
IFFT!(cache::Cache) = cache.plans[2]

function update_v!(U::V, cache::Cache{T, S}) where {T, S, V<:AbstractVector{S}}
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

function update_p!(cache::Cache{T, S}) where {T, S, V<:AbstractVector{S}}
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
    @. poiss_rhs = -2.0*(dVdz_dWdy - dVdy_dWdz) - cache.Ro*dUdy
    @views begin
        @. poiss_rhs[:, 1, 1] -= cache.Ro*dūdy
    end

    # extract boundary condition data
    @views begin
        @. cache.bc_cache[1] = cache.Re_recip*d2Vdy2[1, :, :]
        @. cache.bc_cache[2] = cache.Re_recip*d2Vdy2[end, :, :]
    end
    cache.bc_cache[1][1, 1] = -cache.Ro
    cache.bc_cache[2][1, 1] = cache.Ro

    # solve the pressure equation
    solve!(P, cache.lapl, poiss_rhs, cache.bc_cache)

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
