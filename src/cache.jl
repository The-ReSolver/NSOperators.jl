# The file contains the definition for the global cache of the optimisation.

export Cache, update_v!, update_p!, update_r!

# TODO: can the typing of this struct be simplified to only include the information that is absolutely necessary
struct Cache{T, S, P, BC, PLANS}
    spec_cache::Vector{S}
    phys_cache::Vector{P}
    bc_cache::NTuple{2, BC}
    mean_data::NTuple{3, Vector{T}}
    plans::PLANS
    lapl::Laplace
    Re_recip::T
    Ro::T

    function Cache(U::S, u::P, ū::Vector{T}, dūdy::Vector{T}, d2ūdy2::Vector{T}, Re::T, Ro::T) where {T<:Real, S<:AbstractArray{Complex{T}, 3}, P<:AbstractArray{T, 3}}
        # check compatible sizes
        U.grid == u.grid || throw(ArgumentError("Fields are on different grids!"))
        (size(U.grid.y) == size(ū) && size(U.grid.y) == size(dūdy) && size(U.grid.y) == size(d2ūdy2)) || throw(ArgumentError("Mean data on different grids!"))

        # initialise cached arrays
        spec_cache = [similar(U) for i in 1:69]
        phys_cache = [similar(u) for i in 1:29]

        # initialise transforms
        FFT! = FFTPlan!(u)
        IFFT! = IFFTPlan!(U)

        # initialise laplace operator
        lapl = Laplace(size(u)[1], size(u)[2], u.grid.dom[2], u.grid.Dy[2], u.grid.Dy[1])

        # initialise boundary data cache
        bc_cache = (Matrix{Complex{T}}(undef, size(U)[2], size(U)[3]),
                    Matrix{Complex{T}}(undef, size(U)[2], size(U)[3]))

        args = (spec_cache, phys_cache, bc_cache,
                (ū, dūdy, d2ūdy2), (FFT!, IFFT!), lapl, 1/Re, Ro)
        new{T, S, P, typeof(bc_cache[1]), typeof((FFT!, IFFT!))}(args...)
    end
end

# TODO: can I create methods for grid the fields without having to call them explicitely?
Cache(grid::G, ū::Vector{T}, dūdy::Vector{T}, d2ūdy2::Vector{T}, Re::T, Ro::T) where {G, T<:Real} = Cache(SpectralField(grid), PhysicalField(grid), ū::Vector{T}, dūdy::Vector{T}, d2ūdy2::Vector{T}, Re::T, Ro::T)

function update_v!(U::V, cache::Cache{T, S}) where {T, S, V<:AbstractVector{S}}
    # assign spectral aliases
    dUdt = cache.spec_cache[1]
    dVdt = cache.spec_cache[2]
    dWdt = cache.spec_cache[3]
    dUdz = cache.spec_cache[4]
    dVdz = cache.spec_cache[5]
    dWdz = cache.spec_cache[6]
    d2Udz2 = cache.spec_cache[7]
    d2Vdz2 = cache.spec_cache[8]
    d2Wdz2 = cache.spec_cache[9]
    dUdy = cache.spec_cache[10]
    dVdy = cache.spec_cache[11]
    dWdy = cache.spec_cache[12]
    d2Udy2 = cache.spec_cache[13]
    d2Vdy2 = cache.spec_cache[14]
    d2Wdy2 = cache.spec_cache[15]
    V_dUdy = cache.spec_cache[20]
    W_dUdz = cache.spec_cache[21]
    V_dVdy = cache.spec_cache[22]
    W_dVdz = cache.spec_cache[23]
    V_dWdy = cache.spec_cache[24]
    W_dWdz = cache.spec_cache[25]
    dVdz_dWdy = cache.spec_cache[26]
    dVdy_dWdz = cache.spec_cache[27]
    ifft_tmp1 = cache.spec_cache[28]
    ifft_tmp2 = cache.spec_cache[29]
    ifft_tmp3 = cache.spec_cache[30]
    ifft_tmp4 = cache.spec_cache[31]
    ifft_tmp5 = cache.spec_cache[32]
    ifft_tmp6 = cache.spec_cache[33]
    ifft_tmp7 = cache.spec_cache[34]
    ifft_tmp8 = cache.spec_cache[35]

    # assign physical aliases
    v = cache.phys_cache[1]
    w = cache.phys_cache[2]
    dudz = cache.phys_cache[3]
    dvdz = cache.phys_cache[4]
    dwdz = cache.phys_cache[5]
    dudy = cache.phys_cache[6]
    dvdy = cache.phys_cache[7]
    dwdy = cache.phys_cache[8]
    v_dudy = cache.phys_cache[9]
    w_dudz = cache.phys_cache[10]
    v_dvdy = cache.phys_cache[11]
    w_dvdz = cache.phys_cache[12]
    v_dwdy = cache.phys_cache[13]
    w_dwdz = cache.phys_cache[14]
    dvdz_dwdy = cache.phys_cache[15]
    dvdy_dwdz = cache.phys_cache[16]

    # assign transform aliases
    FFT! = cache.plans[1]
    IFFT! = cache.plans[2]

    # compute derivatives
    ddt!(U[1], dUdt)
    ddt!(U[2], dVdt)
    ddt!(U[3], dWdt)
    ddz!(U[1], dUdz)
    ddz!(U[2], dVdz)
    ddz!(U[3], dWdz)
    d2dz2!(U[1], d2Udz2)
    d2dz2!(U[2], d2Vdz2)
    d2dz2!(U[3], d2Wdz2)
    ddy!(U[1], dUdy)
    ddy!(U[2], dVdy)
    ddy!(U[3], dWdy)
    d2dy2!(U[1], d2Udy2)
    d2dy2!(U[2], d2Vdy2)
    d2dy2!(U[3], d2Wdy2)

    # compute nonlinear components
    IFFT!(v, U[2], ifft_tmp1)
    IFFT!(w, U[3], ifft_tmp2)
    IFFT!(dudz, dUdz, ifft_tmp3)
    IFFT!(dvdz, dVdz, ifft_tmp4)
    IFFT!(dwdz, dWdz, ifft_tmp5)
    IFFT!(dudy, dUdy, ifft_tmp6)
    IFFT!(dvdy, dVdy, ifft_tmp7)
    IFFT!(dwdy, dWdy, ifft_tmp8)
    v_dudy .= v.*dudy
    w_dudz .= w.*dudz
    v_dvdy .= v.*dvdy
    w_dvdz .= w.*dvdz
    v_dwdy .= v.*dwdy
    w_dwdz .= w.*dwdz
    dvdz_dwdy .= dvdz.*dwdy
    dvdy_dwdz .= dvdy.*dwdz
    FFT!(V_dUdy, v_dudy)
    FFT!(W_dUdz, w_dudz)
    FFT!(V_dVdy, v_dvdy)
    FFT!(W_dVdz, w_dvdz)
    FFT!(V_dWdy, v_dwdy)
    FFT!(W_dWdz, w_dwdz)
    FFT!(dVdz_dWdy, dvdz_dwdy)
    FFT!(dVdy_dWdz, dvdy_dwdz)

    return
end

function update_p!(cache::Cache{T, S}) where {T, S, V<:AbstractVector{S}}
    # assign aliases
    dUdy = cache.spec_cache[10]
    d2Vdy2 = cache.spec_cache[14]
    P = cache.spec_cache[16]
    dPdy = cache.spec_cache[17]
    dPdz = cache.spec_cache[18]
    poiss_rhs = cache.spec_cache[19]
    dVdz_dWdy = cache.spec_cache[26]
    dVdy_dWdz = cache.spec_cache[27]
    dūdy = cache.mean_data[2]

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
    ifft_tmp1 = cache.spec_cache[28]
    ifft_tmp2 = cache.spec_cache[29]
    ifft_tmp3 = cache.spec_cache[30]
    ifft_tmp4 = cache.spec_cache[31]
    ifft_tmp5 = cache.spec_cache[32]
    ifft_tmp6 = cache.spec_cache[33]
    ifft_tmp7 = cache.spec_cache[34]
    ifft_tmp8 = cache.spec_cache[35]
    rx = cache.spec_cache[36]
    ry = cache.spec_cache[37]
    rz = cache.spec_cache[38]
    drxdt = cache.spec_cache[42]
    drydt = cache.spec_cache[43]
    drzdt = cache.spec_cache[44]
    drxdz = cache.spec_cache[45]
    drydz = cache.spec_cache[46]
    drzdz = cache.spec_cache[47]
    d2rxdz2 = cache.spec_cache[48]
    d2rydz2 = cache.spec_cache[49]
    d2rzdz2 = cache.spec_cache[50]
    drxdy = cache.spec_cache[51]
    drydy = cache.spec_cache[52]
    drzdy = cache.spec_cache[53]
    d2rxdy2 = cache.spec_cache[54]
    d2rydy2 = cache.spec_cache[55]
    d2rzdy2 = cache.spec_cache[56]
    V_drxdy = cache.spec_cache[57]
    W_drxdz = cache.spec_cache[58]
    V_drydy = cache.spec_cache[59]
    W_drydz = cache.spec_cache[60]
    V_drzdy = cache.spec_cache[61]
    W_drzdz = cache.spec_cache[62]
    rx_dUdy = cache.spec_cache[63]
    ry_dVdy = cache.spec_cache[64]
    rz_dWdy = cache.spec_cache[65]
    rx_dUdz = cache.spec_cache[66]
    ry_dVdz = cache.spec_cache[67]
    rz_dWdz = cache.spec_cache[68]
    ifft_tmp9 = cache.spec_cache[69]


    # assign physical aliases
    v = cache.phys_cache[1]
    w = cache.phys_cache[2]
    dudz = cache.phys_cache[3]
    dvdz = cache.phys_cache[4]
    dwdz = cache.phys_cache[5]
    dudy = cache.phys_cache[6]
    dvdy = cache.phys_cache[7]
    dwdy = cache.phys_cache[8]
    drxdz_p = cache.phys_cache[9]
    drydz_p = cache.phys_cache[10]
    drzdz_p = cache.phys_cache[11]
    drxdy_p = cache.phys_cache[12]
    drydy_p = cache.phys_cache[13]
    drzdy_p = cache.phys_cache[14]
    v_drxdy = cache.phys_cache[15]
    w_drxdz = cache.phys_cache[16]
    v_drydy = cache.phys_cache[17]
    w_drydz = cache.phys_cache[18]
    v_drzdy = cache.phys_cache[19]
    w_drzdz = cache.phys_cache[20]
    rx_dudy = cache.phys_cache[21]
    ry_dvdy = cache.phys_cache[22]
    rz_dwdy = cache.phys_cache[23]
    rx_dudz = cache.phys_cache[24]
    ry_dvdz = cache.phys_cache[25]
    rz_dwdz = cache.phys_cache[26]
    rx_p = cache.phys_cache[27]
    ry_p = cache.phys_cache[28]
    rz_p = cache.phys_cache[29]

    # assign transform aliases
    FFT! = cache.plans[1]
    IFFT! = cache.plans[2]

    # compute derivatives
    ddt!(rx, drxdt)
    ddt!(ry, drydt)
    ddt!(rz, drzdt)
    ddz!(rx, drxdz)
    ddz!(ry, drydz)
    ddz!(rz, drzdz)
    d2dz2!(rx, d2rxdz2)
    d2dz2!(ry, d2rydz2)
    d2dz2!(rz, d2rzdz2)
    ddy!(rx, drxdy)
    ddy!(ry, drydy)
    ddy!(rz, drzdy)
    d2dy2!(rx, d2rxdy2)
    d2dy2!(ry, d2rydy2)
    d2dy2!(rz, d2rzdy2)

    # compute nonlinear components
    IFFT!(rx_p, rx, ifft_tmp1)
    IFFT!(ry_p, ry, ifft_tmp2)
    IFFT!(rz_p, rz, ifft_tmp3)
    IFFT!(drxdz_p, drxdz, ifft_tmp4)
    IFFT!(drydz_p, drydz, ifft_tmp5)
    IFFT!(drzdz_p, drzdz, ifft_tmp6)
    IFFT!(drxdy_p, drxdy, ifft_tmp7)
    IFFT!(drydy_p, drydy, ifft_tmp8)
    IFFT!(drzdy_p, drzdy, ifft_tmp9)
    v_drxdy .= v.*drxdy_p
    w_drxdz .= w.*drxdz_p
    v_drydy .= v.*drydy_p
    w_drydz .= w.*drydz_p
    v_drzdy .= v.*drzdy_p
    w_drzdz .= w.*drzdz_p
    rx_dudy .= rx_p.*dudy
    ry_dvdy .= ry_p.*dvdy
    rz_dwdy .= rz_p.*dwdy
    rx_dudz .= rx_p.*dudz
    ry_dvdz .= ry_p.*dvdz
    rz_dwdz .= rz_p.*dwdz
    FFT!(V_drxdy, v_drxdy)
    FFT!(W_drxdz, w_drxdz)
    FFT!(V_drydy, v_drydy)
    FFT!(W_drydz, w_drydz)
    FFT!(V_drzdy, v_drzdy)
    FFT!(W_drzdz, w_drzdz)
    FFT!(rx_dUdy, rx_dudy)
    FFT!(ry_dVdy, ry_dvdy)
    FFT!(rz_dWdy, rz_dwdy)
    FFT!(rx_dUdz, rx_dudz)
    FFT!(ry_dVdz, ry_dvdz)
    FFT!(rz_dWdz, rz_dwdz)

    return
end
