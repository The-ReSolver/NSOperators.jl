# This file contains the definitions required to compute the local residual
# of an incompressible fluctuation velocity field.

export localresidual!

struct _LocalResidual!{T, S, P, BC, PLANS}
    spec_cache::Vector{S}
    phys_cache::Vector{P}
    bc_cache::NTuple{2, BC}
    ū_data::NTuple{3, Vector{T}}
    dp̄dy::Vector{T}
    plans::PLANS
    lapl::Laplace
    Re_recip::T
    Ro::T

    function _LocalResidual!(U::S, u::P, ū::Vector{T}, dūdy::Vector{T}, d2ūdy2::Vector{T}, dp̄dy::Vector{T}, Re::T, Ro::T) where {T<:Real, S<:AbstractArray{Complex{T}, 3}, P<:AbstractArray{T, 3}}
        # check sizes of arguments are compatible
        (size(u)[1], (size(u)[2] >> 1) + 1, size(u)[3]) == size(U) || throw(ArgumentError("Arrays are not compatible sizes!"))
        (size(ū) == size(dūdy) && size(ū) == size(d2ūdy2)) || throw(ArgumentError("Vectors are not compatible sizes!"))

        # initialise cached arrays
        spec_cache = [similar(U) for i in 1:35]
        phys_cache = [similar(u) for i in 1:16]

        # initialise plans
        FFT = FFTPlan!(u)
        IFFT = IFFTPlan!(U)

        # initialise laplace operator
        lapl = Laplace(size(u)[1], size(u)[2], u.grid.dom[2], u.grid.Dy[2], u.grid.Dy[1])

        # initialise boundary data cache
        bc_cache = (Matrix{Complex{T}}(undef, size(U)[2], size(U)[3]),
                    Matrix{Complex{T}}(undef, size(U)[2], size(U)[3]))

        args = (spec_cache, phys_cache, bc_cache,
                (ū, dūdy, d2ūdy2), dp̄dy, (FFT, IFFT), lapl, 1/Re, Ro)
        new{T, S, P, typeof(bc_cache[1]), typeof((FFT, IFFT))}(args...)
    end
end

function (f::_LocalResidual!{T, S, P})(res::V, U::V) where {T, S, P, V<:AbstractVector{S}}
    # assign spectral array aliases
    dUdt = f.spec_cache[1]
    dVdt = f.spec_cache[2]
    dWdt = f.spec_cache[3]
    dUdz = f.spec_cache[4]
    dVdz = f.spec_cache[5]
    dWdz = f.spec_cache[6]
    d2Udz2 = f.spec_cache[7]
    d2Vdz2 = f.spec_cache[8]
    d2Wdz2 = f.spec_cache[9]
    dUdy = f.spec_cache[10]
    dVdy = f.spec_cache[11]
    dWdy = f.spec_cache[12]
    d2Udy2 = f.spec_cache[13]
    d2Vdy2 = f.spec_cache[14]
    d2Wdy2 = f.spec_cache[15]
    Pr = f.spec_cache[16]
    dPdy = f.spec_cache[17]
    dPdz = f.spec_cache[18]
    poiss_rhs = f.spec_cache[19]
    V_dUdy = f.spec_cache[20]
    W_dUdz = f.spec_cache[21]
    V_dVdy = f.spec_cache[22]
    W_dVdz = f.spec_cache[23]
    V_dWdy = f.spec_cache[24]
    W_dWdz = f.spec_cache[25]
    dVdz_dWdy = f.spec_cache[26]
    dVdy_dWdz = f.spec_cache[27]
    ifft_tmp1 = f.spec_cache[28]
    ifft_tmp2 = f.spec_cache[29]
    ifft_tmp3 = f.spec_cache[30]
    ifft_tmp4 = f.spec_cache[31]
    ifft_tmp5 = f.spec_cache[32]
    ifft_tmp6 = f.spec_cache[33]
    ifft_tmp7 = f.spec_cache[34]
    ifft_tmp8 = f.spec_cache[35]

    # assign physical array aliases
    v = f.phys_cache[1]
    w = f.phys_cache[2]
    dudz = f.phys_cache[3]
    dvdz = f.phys_cache[4]
    dwdz = f.phys_cache[5]
    dudy = f.phys_cache[6]
    dvdy = f.phys_cache[7]
    dwdy = f.phys_cache[8]
    v_dudy = f.phys_cache[9]
    w_dudz = f.phys_cache[10]
    v_dvdy = f.phys_cache[11]
    w_dvdz = f.phys_cache[12]
    v_dwdy = f.phys_cache[13]
    w_dwdz = f.phys_cache[14]
    dvdz_dwdy = f.phys_cache[15]
    dvdy_dwdz = f.phys_cache[16]

    # fft transform aliases
    FFT = f.plans[1]
    IFFT = f.plans[2]

    # compute all relevent derivatives
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
    IFFT(v, U[2], ifft_tmp1)
    IFFT(w, U[3], ifft_tmp2)
    IFFT(dudz, dUdz, ifft_tmp3)
    IFFT(dvdz, dVdz, ifft_tmp4)
    IFFT(dwdz, dWdz, ifft_tmp5)
    IFFT(dudy, dUdy, ifft_tmp6)
    IFFT(dvdy, dVdy, ifft_tmp7)
    IFFT(dwdy, dWdy, ifft_tmp8)
    v_dudy .= v.*dudy
    w_dudz .= w.*dudz
    v_dvdy .= v.*dvdy
    w_dvdz .= w.*dvdz
    v_dwdy .= v.*dwdy
    w_dwdz .= w.*dwdz
    dvdz_dwdy .= dvdz.*dwdy
    dvdy_dwdz .= dvdy.*dwdz
    FFT(V_dUdy, v_dudy)
    FFT(W_dUdz, w_dudz)
    FFT(V_dVdy, v_dvdy)
    FFT(W_dVdz, w_dvdz)
    FFT(V_dWdy, v_dwdy)
    FFT(W_dWdz, w_dwdz)
    FFT(dVdz_dWdy, dvdz_dwdy)
    FFT(dVdy_dWdz, dvdy_dwdz)

    # calculate rhs of pressure equation
    poiss_rhs .= -2.0.*(dVdz_dWdy .- dVdy_dWdz) .- f.Ro.*dUdy

    # calculate boundary condition data of pressure equation
    @views begin
        f.bc_cache[1] .= f.Re_recip.*d2Vdy2[1, :, :]
        f.bc_cache[2] .= f.Re_recip.*d2Vdy2[end, :, :]
    end

    # solve pressure equation
    solve!(Pr, f.lapl, poiss_rhs, f.bc_cache)

    # compute pressure gradient
    ddy!(Pr, dPdy)
    ddz!(Pr, dPdz)

    # calculate residual
    @views @inbounds begin
        for ny in 1:size(U[1])[1]
            res[1][ny, :, :] .= dUdt[ny, :, :] .+ U[2][ny, :, :].*f.ū_data[2][ny] .- f.Re_recip.*(d2Udy2[ny, :, :] .+ d2Udz2[ny, :, :]) .- f.Ro.*U[2][ny, :, :] .+ V_dUdy[ny, :, :] .+ W_dUdz[ny, :, :]
            res[2][ny, :, :] .= dVdt[ny, :, :] .- f.Re_recip.*(d2Vdy2[ny, :, :] .+ d2Vdz2[ny, :, :]) .+ f.Ro.*U[1][ny, :, :] .+ V_dVdy[ny, :, :] .+ W_dVdz[ny, :, :] .+ dPdy[ny, :, :]
            res[3][ny, :, :] .= dWdt[ny, :, :] .- f.Re_recip.*(d2Wdy2[ny, :, :] .+ d2Wdz2[ny, :, :]) .+ V_dWdy[ny, :, :] .+ W_dWdz[ny, :, :] .+ dPdz[ny, :, :]
        end
    end

    # calculate mean constraint
    @views begin
        res[1][:, 1, 1] .= f.Re_recip.*f.ū_data[3] .- V_dUdy[:, 1, 1] .- W_dUdz[:, 1, 1]
        res[2][:, 1, 1] .= (-f.Ro).*f.ū_data[1] .- f.dp̄dy .- V_dVdy[:, 1, 1] .- W_dVdz[:, 1, 1]
        res[3][:, 1, 1] .= .-V_dWdy[:, 1, 1] .- W_dWdz[:, 1, 1]
    end

    return res
end

# TODO: update cache for the new IFFT method
function localresidual!(res::V, U::V, cache::Cache{T, S}) where {T, S, V<:AbstractVector{S}}
    # assign spectral aliases
    dUdt = cache.spec_cache[1]
    dVdt = cache.spec_cache[2]
    dWdt = cache.spec_cache[3]
    d2Udz2 = cache.spec_cache[7]
    d2Vdz2 = cache.spec_cache[8]
    d2Wdz2 = cache.spec_cache[9]
    d2Udy2 = cache.spec_cache[13]
    d2Vdy2 = cache.spec_cache[14]
    d2Wdy2 = cache.spec_cache[15]
    dPdy = cache.spec_cache[17]
    dPdz = cache.spec_cache[18]
    V_dUdy = cache.spec_cache[20]
    W_dUdz = cache.spec_cache[21]
    V_dVdy = cache.spec_cache[22]
    W_dVdz = cache.spec_cache[23]
    V_dWdy = cache.spec_cache[24]
    W_dWdz = cache.spec_cache[25]

    # assign mean aliases
    ū = cache.mean_data[1]
    dūdy = cache.mean_data[2]
    d2ūdy2 = cache.mean_data[3]

    # calculate local residual
    @views @inbounds begin
        for ny in 1:size(U[1])[1]
            res[1][ny, :, :] .= dUdt[ny, :, :] .+ U[2][ny, :, :].*dūdy[ny] .- cache.Re_recip.*(d2Udy2[ny, :, :] .+ d2Udz2[ny, :, :]) .- cache.Ro.*U[2][ny, :, :] .+ V_dUdy[ny, :, :] .+ W_dUdz[ny, :, :]
            res[2][ny, :, :] .= dVdt[ny, :, :] .- cache.Re_recip.*(d2Vdy2[ny, :, :] .+ d2Vdz2[ny, :, :]) .+ cache.Ro.*U[1][ny, :, :] .+ V_dVdy[ny, :, :] .+ W_dVdz[ny, :, :] .+ dPdy[ny, :, :]
            res[3][ny, :, :] .= dWdt[ny, :, :] .- cache.Re_recip.*(d2Wdy2[ny, :, :] .+ d2Wdz2[ny, :, :]) .+ V_dWdy[ny, :, :] .+ W_dWdz[ny, :, :] .+ dPdz[ny, :, :]
        end
    end

    # calculate mean constraint
    @views begin
        res[1][:, 1, 1] .= d2ūdy2 .- V_dUdy[:, 1, 1] .- W_dUdz[:, 1, 1]
        res[2][:, 1, 1] .= cache.Ro.*ū .- f.dp̄dy .- V_dVdy[:, 1, 1] .- W_dVdz[:, 1, 1]
        res[3][:, 1, 1] .= .-V_dWdy[:, 1, 1] .- W_dWdz[:, 1, 1]
    end

    return res
end
