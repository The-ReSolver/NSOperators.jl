# This file contains the definitions required to compute the residual and
# associated gradients of an incompressible velocity field.

function localresidual!(U::V, cache::Cache{T, S}) where {T, S, V<:AbstractVector{S}}
    # assign spectral aliases
    dUdt = NSOperators.dUdt(cache)
    dVdt = NSOperators.dVdt(cache)
    dWdt = NSOperators.dWdt(cache)
    d2Udz2 = NSOperators.d2Udz2(cache)
    d2Vdz2 = NSOperators.d2Vdz2(cache)
    d2Wdz2 = NSOperators.d2Wdz2(cache)
    d2Udy2 = NSOperators.d2Udy2(cache)
    d2Vdy2 = NSOperators.d2Vdy2(cache)
    d2Wdy2 = NSOperators.d2Wdy2(cache)
    dPdy = NSOperators.dPdy(cache)
    dPdz = NSOperators.dPdz(cache)
    V_dUdy = NSOperators.V_dUdy(cache)
    W_dUdz = NSOperators.W_dUdz(cache)
    V_dVdy = NSOperators.V_dVdy(cache)
    W_dVdz = NSOperators.W_dVdz(cache)
    V_dWdy = NSOperators.V_dWdy(cache)
    W_dWdz = NSOperators.W_dWdz(cache)
    rx = NSOperators.rx(cache)
    ry = NSOperators.ry(cache)
    rz = NSOperators.rz(cache)

    # assign mean aliases
    ū = NSOperators.ū(cache)
    dūdy = NSOperators.dūdy(cache)
    d2ūdy2 = NSOperators.d2ūdy2(cache)

    # calculate local residual
    @. rx = dUdt + U[2]*dūdy - cache.Re_recip*(d2Udy2 + d2Udz2) - cache.Ro*U[2] + V_dUdy + W_dUdz
    @. ry = dVdt - cache.Re_recip*(d2Vdy2 + d2Vdz2) + cache.Ro*U[1] + V_dVdy + W_dVdz + dPdy
    @. rz = dWdt - cache.Re_recip*(d2Wdy2 + d2Wdz2) + V_dWdy + W_dWdz + dPdz
    # @. rx = V_dUdy
    # @. ry = 0.0
    # @. rz = 0.0

    # calculate mean constraint
    @views begin
        @. rx[:, 1, 1] = -cache.Re_recip*d2ūdy2 + V_dUdy[:, 1, 1] + W_dUdz[:, 1, 1]
        @. ry[:, 1, 1] = cache.Ro*ū + dPdy[:, 1, 1] + V_dVdy[:, 1, 1] + W_dVdz[:, 1, 1]
        @. rz[:, 1, 1] = V_dWdy[:, 1, 1] + W_dWdz[:, 1, 1]
        # @. rx[:, 1, 1] = 0.0
        # @. ry[:, 1, 1] = 0.0
        # @. rz[:, 1, 1] = 0.0
    end

    return VectorField(rx, ry, rz)
end

ℜ(cache::Cache) = 0.5*norm(VectorField(rx(cache), ry(cache), rz(cache)))^2

function dℜ!(cache::Cache)
    # assign spectral aliases
    rx = NSOperators.rx(cache)
    ry = NSOperators.ry(cache)
    dℜx = NSOperators.dℜx(cache)
    dℜy = NSOperators.dℜy(cache)
    dℜz = NSOperators.dℜz(cache)
    drxdt = NSOperators.drxdt(cache)
    drydt = NSOperators.drydt(cache)
    drzdt = NSOperators.drzdt(cache)
    d2rxdz2 = NSOperators.d2rxdz2(cache)
    d2rydz2 = NSOperators.d2rydz2(cache)
    d2rzdz2 = NSOperators.d2rzdz2(cache)
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

    # assign mean aliases
    dūdy = NSOperators.dūdy(cache)

    # calculate residual gradient
    @. dℜx = -drxdt - V_drxdy - W_drxdz - cache.Re_recip*(d2rxdy2 + d2rxdz2) + cache.Ro*ry
    @. dℜy = -drydt - V_drydy - W_drydz - cache.Re_recip*(d2rydy2 + d2rydz2) + rx*dūdy - cache.Ro*rx + rx_dUdy + ry_dVdy + rz_dWdy
    @. dℜz = -drzdt - V_drzdy - W_drzdz - cache.Re_recip*(d2rzdy2 + d2rzdz2) + rx_dUdz + ry_dVdz + rz_dWdz
    # @. dℜx = -V_drxdy
    # @. dℜy = rx_dUdy
    # @. dℜz = 0.0

    return VectorField(dℜx, dℜy, dℜz)
end

unsafe_ℜdℜ!(cache::Cache{T, S}) where {T, S} = (ℜ(cache), dℜ!(cache))

function ℜdℜ!(U::AbstractVector{S}, cache::Cache{T, S}) where {T, S}
    update_v!(U, cache)
    update_p!(cache)
    localresidual!(U, cache)
    update_r!(cache)
    return ℜ(cache), dℜ!(cache)
end
