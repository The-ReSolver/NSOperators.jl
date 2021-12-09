# This file contains the definitions required to compute the local residual
# of an incompressible fluctuation velocity field.

export residual

# TODO: test this
# TODO: zero mode (mean constraint)
struct _LocalResidual!{T, S, P, BC, PLANS}
    spec_cache::Vector{S}
    phys_cache::Vector{P}
    bc_cache::NTuple{2, BC}
    ū::Vector{T}
    dūdy::Vector{T}
    plans::NTuple{2, PLANS}
    lapl::Laplace
    Re_recip::T
    Ro::T

    function LocalResidual!(û::S, u::P, ū::Vector{T}, dūdy::Vector{T}, Re::T, Ro::T) where {T<:Real, S<:AbstractArray{Complex{T}, 3}, P<:AbstractArray{T, 3}}
        # size of array
        size = size(u)

        # initialise cached arrays
        spec_cache = [similar(û) for i in 1:27]
        phys_cache = [similar(u) for i in 1:16]

        # define plans
        FFT = FFTPlan!(u)
        IFFT = IFFTPlan!(û)

        # define laplace operator
        lapl = Laplace(size[1], size[2], u.grid.dom[2], u.grid.Dy[2], u.grid.Dy[1])

        # define boundary data cache
        bc_cache = (Matrix{Complex{T}}(undef, size[2], size[3]),
                    Matrix{Complex{T}}(undef, size[2], size[3]))

        args = (spec_cache, phys_cache, bc_cache,
                ū, dūdy, (FFT, IFFT), lapl, 1/Re, Ro)
        new{T, S, P, typeof(bc_cache[1]), typeof(FFT)}(args...)
    end
end

function (f::_LocalResidual!{T, S, P})(res::Vector{S}, u::Vector{S}) where {T, S, P}
    # assign spectral array aliases
    dudt = f.spec_cache[1]
    dvdt = f.spec_cache[2]
    dwdt = f.spec_cache[3]
    dudz = f.spec_cache[4]
    dvdz = f.spec_cache[5]
    dwdz = f.spec_cache[6]
    d2udz2 = f.spec_cache[7]
    d2vdz2 = f.spec_cache[8]
    d2wdz2 = f.spec_cache[9]
    dudy = f.spec_cache[10]
    dvdy = f.spec_cache[11]
    dwdy = f.spec_cache[12]
    d2udy2 = f.spec_cache[13]
    d2vdy2 = f.spec_cache[14]
    d2wdy2 = f.spec_cache[15]
    p = f.spec_cache[16]
    dpdy = f.spec_cache[17]
    dpdz = f.spec_cache[18]
    poiss_rhs = f.spec_cache[19]
    v_dudy = f.spec_cache[20]
    w_dudz = f.spec_cache[21]
    v_dvdy = f.spec_cache[22]
    w_dvdz = f.spec_cache[23]
    v_dwdy = f.spec_cache[24]
    w_dwdz = f.spec_cache[25]
    dvdz_dwdy = f.spec_cache[26]
    dvdy_dwdz = f.spec_cache[27]

    # assign physical array aliases
    v_t = f.phys_cache[1]
    w_t = f.phys_cache[2]
    dudz_t = f.phys_cache[3]
    dvdz_t = f.phys_cache[4]
    dwdz_t = f.phys_cache[5]
    dudy_t = f.phys_cache[6]
    dvdy_t = f.phys_cache[7]
    dwdy_t = f.phys_cache[8]
    v_dudy_t = f.phys_cache[9]
    w_dudz_t = f.phys_cache[10]
    v_dvdy_t = f.phys_cache[11]
    w_dvdz_t = f.phys_cache[12]
    v_dwdy_t = f.phys_cache[13]
    w_dwdz_t = f.phys_cache[14]
    dvdz_dwdy_t = f.phys_cache[15]
    dvdy_dwdz_t = f.phys_cache[16]

    # compute all relevent derivatives
    ddt!(u[1], dudt)
    ddt!(u[2], dvdt)
    ddt!(u[3], dwdt)
    ddz!(u[1], dudz)
    ddz!(u[2], dvdz)
    ddz!(u[3], dwdz)
    d2dz2!(u[1], d2udz2)
    d2dz2!(u[2], d2vdz2)
    d2dz2!(u[3], d2wdz2)
    ddy!(u[1], dudy)
    ddy!(u[2], dvdy)
    ddy!(u[3], dwdy)
    d2dy2!(u[1], d2udy2)
    d2dy2!(u[2], d2vdy2)
    d2dy2!(u[3], d2wdy2)

    # compute nonlinear components
    f.IFFT(v_t, u[2])
    f.IFFT(w_t, u[3])
    f.IFFT(dudz_t, dudz)
    f.IFFT(dvdz_t, dvdz)
    f.IFFT(dwdz_t, dwdz)
    f.IFFT(dudy_t, dudy)
    f.IFFT(dvdy_t, dvdy)
    f.IFFT(dwdy_t, dwdy)
    v_dudy_t .= v_t.*dudy_t
    w_dudz_t .= w_t.*dudz_t
    v_dvdy_t .= v_t.*dvdy_t
    w_dvdz_t .= w_t.*dvdz_t
    v_dwdy_t .= v_t.*dwdy_t
    w_dwdz_t .= w_t.*dwdz_t
    dvdz_dwdy_t .= dvdz_t.*dwdy_t
    dvdy_dwdz_t .= dvdy_t.*dwdz_t
    f.FFT(v_dudy, v_dudy_t)
    f.FFT(w_dudz, w_dudz_t)
    f.FFT(v_dvdy, v_dvdy_t)
    f.FFT(w_dvdz, w_dvdz_t)
    f.FFT(v_dwdy, v_dwdy_t)
    f.FFT(w_dwdz, w_dwdz_t)
    f.FFT(dvdz_dwdy, dvdz_dwdy_t)
    f.FFT(dvdy_dwdz, dvdy_dwdz_t)

    # calculate rhs of pressure equation
    poiss_rhs .= 2.0.*(dvdz_dwdy .- dvdy_dwdz) .- f.Ro.*dudy

    # calculate boundary condition data of pressure equation
    f.bc_cache[1] .= f.Re_recip.*@view d2vdy2[1, :, :]
    f.bc_cache[2] .= f.Re_recip.*@view d2vdy2[end, :, :]

    # solve pressure equation
    solve!(p, f.lapl, poiss_rhs, f.bc_cache)

    # compute pressure gradient
    dpdy = ddy!(p, dpdy)
    dpdz = ddz!(p, dpdz)

    # calculate residual
    res[1] .= dudt .+ u[2].*f.dūdy .- f.Re_recip.*(d2udy2 .+ d2udz2) .- f.Ro.*u[2] .+ v_dudy .+ w_dudz
    res[2] .= dvdt .- f.Re_recip.*(d2vdy2 .+ d2vdz2) .+ f.Ro.*u[1] .+ v_dvdy .+ w_dvdz .+ dpdy
    res[3] .= dwdt .- f.Re_recip.*(d2wdy2 .+ d2wdz2) .+ v_dwdy .+ w_dwdz .+ dpdz
end
