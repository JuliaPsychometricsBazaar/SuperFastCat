"""
Given quadrature points `x` and weights `w` and some extra weight term for the
pdf `f(x)`, compute the mean of the distribution with pdf `g(x)f(x)` where
`g(x)` is the whatever factors are included in `x` and `w` (if weighted
quadrature has been used -- otherwise just `f(x)`. This function
returns `(mean, normalising_constant)` where `normalising_constant`
is the integral of `g(x)f(x)`.
"""
function mean_and_c(x, w, fx)
    mean = zero(eltype(x))
    norm = zero(eltype(x))
    @fastmath @turbo for i in eachindex(x)
        common_factor = w[i] * fx[i] # f(x[i])
        mean += common_factor * x[i]
        norm += common_factor
    end
    (mean / norm, norm)
end

function mean_and_c(x, w)
    mean = zero(eltype(x))
    norm = zero(eltype(x))
    @fastmath @turbo for i in eachindex(x)
        mean += w[i] * x[i]
        norm += w[i]
    end
    (mean / norm, norm)
end

"""
As with `mean_and_c` but also returns the variance of the distribution:
`(variance, mean, normalising_constant)`.
"""
function var_mean_and_c(x, w, fx)
    mean, norm = mean_and_c(x, w, fx)
    raw_var = 0f0
    @fastmath @turbo for i in eachindex(x)
        raw_var += w[i] * fx[i] * (x[i] - mean) ^ 2
    end
    (
        raw_var / norm,
        mean,
        norm
    )
end