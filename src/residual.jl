# This file contains the definitions required to compute the residual and
# associated gradients of an incompressible velocity field.

function localresidual!(U::V, cache::Cache{T, S}) where {T, S, V<:AbstractVector{S}}
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
    rx = cache.spec_cache[36]
    ry = cache.spec_cache[37]
    rz = cache.spec_cache[38]

    # assign mean aliases
    ū = cache.mean_data[1]
    dūdy = cache.mean_data[2]
    d2ūdy2 = cache.mean_data[3]

    # calculate local residual
    @. rx = dUdt + U[2]*dūdy - cache.Re_recip*(d2Udy2 + d2Udz2) - cache.Ro*U[2] + V_dUdy + W_dUdz
    @. ry = dVdt - cache.Re_recip*(d2Vdy2 + d2Vdz2) + cache.Ro*U[1] + V_dVdy + W_dVdz + dPdy
    @. rz = dWdt - cache.Re_recip*(d2Wdy2 + d2Wdz2) + V_dWdy + W_dWdz + dPdz

    # calculate mean constraint
    @views begin
        @. rx[:, 1, 1] = cache.Re_recip*d2ūdy2 - V_dUdy[:, 1, 1] - W_dUdz[:, 1, 1]
        @. ry[:, 1, 1] = (-cache.Ro)*ū - dPdy[:, 1, 1] - V_dVdy[:, 1, 1] - W_dVdz[:, 1, 1]
        # @. ry[:, 1, 1] = (-cache.Ro)*ū - V_dVdy[:, 1, 1] - W_dVdz[:, 1, 1] # NOTE: this produces similar results to mathematica notebook
        @. rz[:, 1, 1] = -V_dWdy[:, 1, 1] - W_dWdz[:, 1, 1]
    end

    return VectorField(rx, ry, rz)
end

ℜ(cache::Cache) = 0.5*(norm(VectorField(cache.spec_cache[36], cache.spec_cache[37], cache.spec_cache[38]))^2)

function dℜ!(cache::Cache)
    # assign spectral aliases
    rx = cache.spec_cache[36]
    ry = cache.spec_cache[37]
    dℜx = cache.spec_cache[39]
    dℜy = cache.spec_cache[40]
    dℜz = cache.spec_cache[41]
    drxdt = cache.spec_cache[42]
    drydt = cache.spec_cache[43]
    drzdt = cache.spec_cache[44]
    d2rxdz2 = cache.spec_cache[48]
    d2rydz2 = cache.spec_cache[49]
    d2rzdz2 = cache.spec_cache[50]
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

    # assign mean aliases
    dūdy = cache.mean_data[2]

    # calculate residual gradient
    @. dℜx = -drxdt - V_drxdy - W_drxdz - cache.Re_recip*(d2rxdy2 + d2rxdz2) + cache.Ro*ry                                         # NOTE: verified term-by-term
    @. dℜy = -drydt - V_drydy - W_drydz - cache.Re_recip*(d2rydy2 + d2rydz2) + rx*dūdy - cache.Ro*rx + rx_dUdy + ry_dVdy + rz_dWdy # NOTE: verified term-by-term
    @. dℜz = -drzdt - V_drzdy - W_drzdz - cache.Re_recip*(d2rzdy2 + d2rzdz2) + rx_dUdz + ry_dVdz + rz_dWdz                         # NOTE: verified term-by-term

    return VectorField(dℜx, dℜy, dℜz)
end
