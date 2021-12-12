# The file contains the definition for the global cache of the optimisation.

export Cache

struct Cache{T, S, P, BC, PLANS}
    spec_cache::Vector{S}
    phys_cache::Vector{P}
    bc_cache::NTuple{2, BC}
    mean_data::NTuple{4, Vector{T}}
    plans::PLANS
    lapl::Laplace
    Re_recip::T
    Ro::T

    function Cache(U::S, u::P, ū::Vector{T}, dūdy::Vector{T}, d2ūdy2::Vector{T}, dp̄dy::Vector{T}, Re::T, Ro::T) where {T<:Real, S<:AbstractArray{Complex{T}, 3}, P<:AbstractArray{T, 3}}
        # check compatible sizes
        (size(u)[1], (size(u)[2] >> 1) + 1, size(u)[3]) == size(U) || throw(ArgumentError("Arrays are not compatible sizes!"))
        (size(ū) == size(dūdy) && size(ū) == size(d2ūdy2)) || throw(ArgumentError("Vectors are not compatible sizes!"))

        # initialise cached arrays
        spec_cache = [similar(U) for i in 1:27]
        phys_cache = [similar(u) for i in 1:16]

        # initialise transforms
        FFT! = FFTPlan!(u)
        IFFT! = IFFTPlan!(U)

        # initialise laplace operator
        lapl = Laplace(size(u)[1], size(u)[2], u.grid.dom[2], u.grid.Dy[2], u.grid.Dy[1])

        # initialise boundary data cache
        bc_cache = (Matrix{Complex{T}}(undef, size(U)[2], size(U)[3]),
                    Matrix{Complex{T}}(undef, size(U)[2], size(U)[3]))

        args = (spec_cache, phys_cache, bc_cache,
                (ū, dūdy, d2ūdy2), dp̄dy, (FFT!, IFFT!), lapl, 1/Re, Ro)
        new{T, S, P, typeof(bc_cache[1]), typeof((FFT!, IFFT!))}(args...)
    end
end


"""
    Given a velocity field, update all variables stored in the cache such that
    they can be used to compute the required optimisation variables.
"""
function (f::Cache{T, S})(U::V) where {T, S, V<:AbstractVector{S}}
    # assign spectral aliases
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

    # assign physical aliases
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

    # assign transform aliases
    FFT = f.plans[1]
    IFFT = f.plans[2]

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
    IFFT(v, U[2])
    IFFT(w, U[3])
    IFFT(dudz, dUdz)
    IFFT(dvdz, dVdz)
    IFFT(dwdz, dWdz)
    IFFT(dudy, dUdy)
    IFFT(dvdy, dVdy)
    IFFT(dwdy, dWdy)
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

    # compute rhs of pressure equation
    poiss_rhs .= 2.0.*(dVdz_dWdy .- dVdy_dWdz) .- f.Ro.*dUdy

    # extract boundary condition data
    # TODO: make sure this doesn't assign any memory
    @views begin
        f.bc_cache[1] .= f.Re_recip.*d2Vdy2[1, :, :]
        f.bc_cache[2] .= f.Re_recip.*d2Vdy2[end, :, :]
    end

    # solve the pressure equation
    solve!(Pr, f.lapl, poiss_rhs, f.bc_cache)

    # compute pressure gradients
    ddy!(Pr, dPdy)
    ddz!(Pr, dPdz)

    return U
end