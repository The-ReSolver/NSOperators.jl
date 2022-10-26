# This file contains the function definition to approximate the gradient of the
# global residual for a given incompressible velocity field. A second-order
# central differencing scheme will be used,

# TODO: try by perturbing in physical space
# TODO: chack if perturbing at the wall or not makes a difference
# TODO: perform FD on single point to get faster comparison

function dℜ_fd!(U::VectorField{N, S}, cache::Cache{T, S}, grad_fd::VectorField{N, S}; step::Float64=1e-6) where {T, N, Ny, Nz, Nt, S<:SpectralField{Ny, Nz, Nt}}
    # loop over velocity field
    # for i in 1:N, nt in 1:Nt, nz in 1:((Nz >> 1) + 1), ny in 2:(Ny - 1)
    for n in 1:N, nt in 1:Nt, nz in 1:((Nz >> 1) + 1), ny in 1:Ny
    # for n in 1:1, nt in 1:Nt, nz in 1:((Nz >> 1) + 1), ny in 1:Ny
        @printf "N: %2i, Ny: %2i, Nz: %2i, Nt: %2i\r" n ny nz nt
        # initialise auxilery arrays
        Ur_for = copy(U)
        Ur_back = copy(U)
        Ui_for = copy(U)
        Ui_back = copy(U)

        # convert to relative step size
        # TODO: add check to avoid letting step size be zero
        rel_step_real = step*abs(real(U[n][ny, nz, nt]))
        rel_step_imag = step*abs(imag(U[n][ny, nz, nt]))

        # perturb real component
        Ur_for[n][ny, nz, nt] = U[n][ny, nz, nt] + rel_step_real
        update_v!(Ur_for, cache)
        update_p!(cache)
        localresidual!(Ur_for, cache)
        gr_for_r = ℜ(cache)
        Ur_back[n][ny, nz, nt] = U[n][ny, nz, nt] - rel_step_real
        update_v!(Ur_back, cache)
        update_p!(cache)
        localresidual!(Ur_back, cache)
        gr_back_r = ℜ(cache)

        # perturb imaginary component
        Ui_for[n][ny, nz, nt] = U[n][ny, nz, nt] + 1im*rel_step_imag
        update_v!(Ui_for, cache)
        update_p!(cache)
        localresidual!(Ui_for, cache)
        gr_for_i = ℜ(cache)
        Ui_back[n][ny, nz, nt] = U[n][ny, nz, nt] - 1im*rel_step_imag
        update_v!(Ui_back, cache)
        update_p!(cache)
        localresidual!(Ui_back, cache)
        gr_back_i = ℜ(cache)

        # calculate central difference and assign to gradient field
        grad_r = (gr_for_r - gr_back_r)/(2*rel_step_real)
        grad_i = (gr_for_i - gr_back_i)/(2*rel_step_imag)
        grad_fd[n][ny, nz, nt] = grad_r + 1im*grad_i
    end
    print("\u1b[0K")

    return grad_fd
end
