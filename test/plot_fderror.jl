# THis script file plots the difference betweent he FD approximation of the
# residual gradient and the properly computed version.

# !----------------------------------------------------------------------------
# ! make sure U is unmutated during the computation of resdual gradient
# !----------------------------------------------------------------------------

# TODO: add option to take finite difference only at specific points

function gen_velocity_field(S, funcs::Vararg{Function})
    if length(funcs) == 0
        funcs = ((y, z, t)->sin(π*y)*exp(cos(z))*sin(t),
                (y, z, t)->(1 + cos(π*y))*cos(z)*sin(t),
                (y, z, t)->π*sin(π*y)*sin(z)*sin(t))
    end
    y = chebpts(S[1])
    Dy = DiffMatrix(y, 5, 1); Dy2 = DiffMatrix(y, 5, 2)
    ws = chebws(S[1])
    ω = 1.0; β = 1.0
    grid = Grid(y, S[2], S[3], Dy, Dy2, ws, ω, β)
    u = VectorField(grid, funcs...)
    U = VectorField(grid)
    FFT! = FFTPlan!(grid)
    FFT!(U, u)

    return U
end

function gen_mean_field(S, funcs::Vararg{Function})
    if length(funcs) == 0
        funcs = (y->y, y->1.0, y->0.0)
    end
    y = chebpts(S[1])
    ū = (funcs[1]).(y)
    dūdy = (funcs[2]).(y)
    d2ūdy2 = (funcs[3]).(y)

    return ū, dūdy, d2ūdy2
end

function norm_difference(step_range, U, ū, dūdy, d2ūdy2, Re, Ro)
    # construct cache
    cache = Cache(U[1].grid, ū, dūdy, d2ūdy2, Re, Ro)

    # initialise results
    diffnorm = zeros(length(step_range))

    for (i, step) in enumerate(step_range)
        # compute gradient
        update_v!(U, cache)
        update_p!(cache)
        localresidual!(U, cache)
        update_r!(cache)
        grad_comp = dℜ!(cache)

        # compute finite difference
        grad_fd = dℜ_fd!(U, cache, similar(U); step=step)

        # assign their normed difference to the result
        diffnorm[i] = norm(grad_comp - grad_fd)
    end

    return diffnorm
end

function loc_difference(step_range, U, ū, dūdy, d2ūdy2, Re, Ro, loc)
    # construct cache
    cache = Cache(U.grid, ū, dūdy, d2ūdy2, Re, Ro)

    # initialise results
    diffnorm = zeros(length(step_range))

    for (i, step) in enumerate(step_range)
        # compute gradient
        update_v!(U, cache)
        update_p!(cache)
        localresidual(U, cache)
        update_r!(cache)
        grad_comp = dℜ!(cache)

        # compute finite difference
        grad_fd = dℜ_fd!(U, cache, similar(U); step=step)

        # assign their normed difference to the result
        diffnorm[i] = norm(grad_comp[loc...] - grad_fd[loc...])
    end

    return diffnorm
end

function save_data(step_range, diffnorm, filetail::String="")
    open("./step_range_$filetail", "w") do f1
        write(f1, step_range)
    end
    open("./norm_difference_$filetail", "w") do f2
        write(f2, diffnorm)
    end
end

function plot_data(filetail)
    # extract the data from the file
    step_range = Vector{Float64}(undef, Int(filesize("./src/ste_range_$filetail")/sizeof(Float64)))
    diffnorm = Vector{Float64}(undef, length(step_range))
    open("./step_range_$filetail", "r") do f1
        read!(f1, step_range)
    end
    open("./diffnorm_$filetail", "r") do f2
        read!(f2, diffnorm)
    end

    # plot the data
    fig = figure(1)
    ax = fig.gca()
    ax.plot(step_range, diffnorm)

    # label axes
    ax.set_xlabel("h")
    ax.set_ylabel("Normed Difference")

    # save the figure
    savefig(ax, "FDerror_$filetail")
end

# generate velocity field
S = (64, 64, 64)
U = gen_velocity_field(S)
U_mean = gen_mean_field(S)
Re = 10.0; Ro = 0.5

# compute the norm differences
step_range = 10 .^ range(-9; stop=-1, length=20)
diffnorm = norm_difference(step_range, U, U_mean[1], U_mean[2], U_mean[3], Re, Ro)

# save the data generated for later use
save_data(step_range, diffnorm, "64box_Re10_Ro0.5_defaultvel")

# plot the result and save figure
# plot_data(step_range, diffnorm)
# plot_data("test")
