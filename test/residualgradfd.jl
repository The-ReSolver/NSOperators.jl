# This file contains the function definition to approximate the gradient of the
# global residual for a given incompressible velocity field. A second-order
# central differencing scheme will be used,

# TODO: try by perturbing in physical space

function dℜ_fd!(U::V, cache::Cache{T, S}, grad_fd::V; step::Float64=1e-6) where {T, S, V<:AbstractVector{S}}
    # loop over velocity field
    # for i in 1:size(U)[1], nt in 1:size(U[1])[3], nz in 1:size(U[1])[2], ny in 1:size(U[1])[1]
    for i in 1:3, nt in 1:size(U[1])[3], nz in 1:size(U[1])[2], ny in 1:size(U[1])[1]
        println("Ny: $ny, Nz: $nz, Nt: $nt")
        # initialise auxilery arrays
        Ur_for = VectorField(deepcopy(U[1]), deepcopy(U[2]), deepcopy(U[3]))
        Ur_back = VectorField(deepcopy(U[1]), deepcopy(U[2]), deepcopy(U[3]))
        Ui_for = VectorField(deepcopy(U[1]), deepcopy(U[3]), deepcopy(U[3]))
        Ui_back = VectorField(deepcopy(U[1]), deepcopy(U[3]), deepcopy(U[3]))

        # perturb real component
        Ur_for[i][ny, nz, nt] = U[i][ny, nz, nt] + step
        update_v!(Ur_for, cache)
        update_p!(cache)
        localresidual!(Ur_for, cache)
        gr_for_r = ℜ(cache)
        Ur_back[i][ny, nz, nt] = U[i][ny, nz, nt] - step
        update_v!(Ur_back, cache)
        update_p!(cache)
        localresidual!(Ur_back, cache)
        gr_back_r = ℜ(cache)

        # perturb imaginary component
        Ui_for[i][ny, nz, nt] = U[i][ny, nz, nt] + 1im*step
        update_v!(Ui_for, cache)
        update_p!(cache)
        localresidual!(Ui_for, cache)
        gr_for_i = ℜ(cache)
        Ui_back[i][ny, nz, nt] = U[i][ny, nz, nt] - 1im*step
        update_v!(Ui_back, cache)
        update_p!(cache)
        localresidual!(Ui_back, cache)
        gr_back_i = ℜ(cache)

        # calculate central difference and assign to gradient field
        grad_r = (gr_for_r - gr_back_r)/(2*step)
        grad_i = (gr_for_i - gr_back_i)/(2*step)
        grad_fd[i][ny, nz, nt] = grad_r + 1im*grad_i
    end

    return grad_fd
end
