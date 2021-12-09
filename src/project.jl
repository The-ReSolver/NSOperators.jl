# This file contains the definitions required to project a given vector into
# a divergence free sub-space.

struct Project!{S}
    spec_cache::Vector{S}
    lapl::Laplace

    function Project!(û::S) where {T, S<:AbstractArray{Complex{T}, 3}}
        # size of array
        size = size(û)

        # initialise cached arrays
        spec_cache = [similar(û) for i in 1:6]

        # initialise laplacian
        lapl = Laplace(size[1], size[2], û.grid.dom[2], û.grid.Dy[2], û.grid.Dy[1])

        new{S}(spec_cache, lapl)
    end
end

function (f::Project!{S})(u::Vector{S}) where {T, S<:AbstractArray{Complex{T}, 3}}
    # assign aliases
    ϕ = f.spec_cache[1]
    dϕdy = f.spec_cache[2]
    dϕdz = f.spec_cache[3]
    dudy = f.spec_cache[4]
    dudz = f.spec_cache[5]
    rhs = f.spec_cache[6]

    # compute rhs of poisson equation
    ddy!(u[2], dudy)
    ddz!(u[3], dudz)
    rhs .= .-dudy .- dudz

    # solve poisson equation
    solve!(ϕ, f.lapl, rhs)

    # compute gradient of scalar field
    ddy!(ϕ, dϕdy)
    ddz!(ϕ, dϕdz)

    # project original field
    u[2] .-= dϕdy
    u[3] .-= dϕdz
end
