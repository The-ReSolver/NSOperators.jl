# This file contains the definitions required to project a given vector into
# a divergence free sub-space.

export Projector!

struct Projector!{S}
    spec_cache::Vector{S}
    lapl::Laplace

    function Projector!(U::S, u::P) where {T, S<:AbstractArray{Complex{T}, 3}, P<:AbstractArray{T, 3}}
        # check sizes of arguments are compatible
        (size(u)[1], (size(u)[2] >> 1) + 1, size(u)[3]) == size(U) || throw(ArgumentError("Arrays are not compatible sizes!"))

        # initialise cached arrays
        spec_cache = [similar(U) for i in 1:6]

        # initialise laplacian
        lapl = Laplace(size(u)[1], size(u)[2], U.grid.dom[2], U.grid.Dy[2])

        new{S}(spec_cache, lapl)
    end
end

function (f::Projector!{S})(U::V) where {T, S, V<:AbstractVector{S}}
    # assign aliases
    ϕ = f.spec_cache[1]
    dϕdy = f.spec_cache[2]
    dϕdz = f.spec_cache[3]
    dUdy = f.spec_cache[4]
    dUdz = f.spec_cache[5]
    rhs = f.spec_cache[6]

    # compute rhs of poisson equation
    ddy!(U[2], dUdy)
    ddz!(U[3], dUdz)
    rhs .= .-dUdy .- dUdz

    # solve poisson equation
    solve!(ϕ, f.lapl, rhs)

    # compute gradient of scalar field
    ddy!(ϕ, dϕdy)
    ddz!(ϕ, dϕdz)

    # project original field
    U[2] .-= dϕdy
    U[3] .-= dϕdz
end
