# This file contains a handful of useful vector calculus operations, using an
# assumed interface available on the input types that defines the required
# spatial and temporal derivatives.

function grad!(∇U::V, U::S) where {S, V<:AbstractVector{S}}
    # ddx!(U, ΔU[1]) # TODO: this should be defined for the interface to be more general
    ddy!(U, ∇U[2])
    ddz!(U, ∇U[3])

    return ∇U
end

function div!(∇U::S, U::V, tmp::S) where {S, V<:AbstractVector{S}}
    ddy!(U, ∇U)
    ddz!(U, tmp)
    ∇U .= tmp

    return ∇U
end

function lap!(ΔU::S, U::S, tmp::S) where {S}
    d2dy2!(U, ΔU)
    d2dz2!(U, tmp)
    ΔU .+= tmp

    return ∇U
end

function lap!(ΔU::V, U::V) where {V}
    for i in 1:size(V)[1]
        lap!(U[i], ΔU[i])
    end

    return ∇U
end

function convect!(U1∇U2::VS, U1::VS, U2::VS, stmp::VS, ptmp::VP) where {S, P, VS<:AbstractVector{S}, VP<:AbstractVector{P}}
    # velocity derivatives
    # ddx!(U2, stmp[1])
    ddy!(U2, stmp[2])
    ddz!(U2, stmp[3])

    # x convective component
    IFFT!(ptmp[1], U1, stmp[2])
    IFFT!(ptmp[2], stmp[1], stmp[2])
    ptmp[3] .= ptmp[1].*ptmp[2]
    FFT!(U1ΔU2, ptmp[3])

    # y convective component

    # z convective component
end

function left_convect!(U1::Vector{T}, U2::V); end

function right_convect!(U1::V, U2::Vector{T}); end

function semi_adj_convect!(U1::V, U2::V); end

function k̂_cross!(k̂_cross_U::V, U::V) where {V}
    k̂_cross_U[1] .= -U[2]
    k̂_cross_U[2] .= U[1]
    return k̂_croos_U
end
