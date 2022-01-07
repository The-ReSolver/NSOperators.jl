# This file contains the definitions required to compute the residual and
# associated gradients of an incompressible velocity field.

export ℜ, dℜ

function _localresidual!(U::V, cache::Cache{T, S}) where {T, S, V<:AbstractVector{S}}
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
    dp̄dy = cache.mean_data[4]

    # local residual cache
    res = cache.res_cache

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
        res[1][:, 1, 1] .= cache.Re_recip.*d2ūdy2 .- V_dUdy[:, 1, 1] .- W_dUdz[:, 1, 1]
        res[2][:, 1, 1] .= (-cache.Ro).*ū .- dp̄dy .- V_dVdy[:, 1, 1] .- W_dVdz[:, 1, 1]
        res[3][:, 1, 1] .= .-V_dWdy[:, 1, 1] .- W_dWdz[:, 1, 1]
    end

    return res
end

function ℜ(); end

function dℜ(); end
