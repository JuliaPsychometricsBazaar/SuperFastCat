using LinearAlgebra: eigen!, SymTridiagonal

# Constructing Gaussian quadrature weights for an
# arbitrary weight function and integration bounds.

# for numerical stability, we apply the usual Lanczos
# Gram–Schmidt procedure to the basis {T₀,T₁,T₂,…} of
# Chebyshev polynomials on [-1,1] rather than to the
# textbook monomial basis {1,x,x²,…}.

# evaluate Chebyshev polynomial p(x) with coefficients a[i]
# by a Clenshaw recurrence.
function chebeval(x, a)
    if length(a) ≤ 2
        length(a) == 1 && return a[1] + x * zero(a[1])
        return a[1]+x*a[2]
    end
    bₖ = a[end-1] + 2x*a[end]
    bₖ₊₁ = oftype(bₖ, a[end])
    for j = lastindex(a)-2:-1:2
        bⱼ = a[j] + 2x*bₖ - bₖ₊₁
        bₖ, bₖ₊₁ = bⱼ, bₖ
    end
    return a[1] + x*bₖ - bₖ₊₁
end

# if a[i] are coefficients of Chebyshev series, compute the coefficients xa (in-place)
# of the series multiplied by x, using recurrence xTₙ = 0.5 (Tₙ₊₁+Tₙ₋₁) for n > 0
function chebx!(xa, a)
    resize!(xa, length(a)+1)
    xa .= 0
    for n = 2:lastindex(a)
        c = 0.5*a[n]
        xa[n-1] += c
        xa[n+1] += c
    end
    if !isempty(a)
        xa[2] += a[1]
    end
    return xa
end

struct WeightedGauss{T}
    N::Int
    alpha::Vector{T}
    beta::Vector{T}
    q_0::Vector{T}
    q_1::Vector{T}
    v::Vector{T}
    segbuf::Vector{QuadGK.Segment{T, T, T}}
    points_buf::Array{Float32, 2}
end

function WeightedGauss{T}(N) where {T}
    q_0, q_1, v = ntuple(3) do _
        sizehint!(Vector{T}(undef, 1), N + 1)
    end
    WeightedGauss{T}(
        N,
        Vector{T}(undef, N),
        Vector{T}(undef, N),
        q_0,
        q_1,
        v,
        alloc_segbuf(Float32, Float32, Float32; size=3),
        Array{Float32, 2}(undef, N, 2)
    )
end

"""
    gauss(W, N, a, b; rtol=sqrt(eps), quad=quadgk)

Return a pair `(x, w)` of `N` quadrature points `x[i]` and weights `w[i]` to
integrate functions on the interval `(a, b)` multiplied by the weight function
`W(x)`.  That is, `sum(w .* f.(x))` approximates the integral `∫ W(x)f(x)dx`
from `a` to `b`.

This function performs `2N` numerical integrals of polynomials against `W(x)`
using the integration function `quad` (defaults to `quadgk`) with relative tolerance `rtol`
(which defaults to half of the precision `eps` of the endpoints).
This is followed by an O(N²) calculations. So, using a large order `N` is expensive.

If `W` has lots of singularities that make it hard to integrate numerically,
you may need to decrease `rtol`.   You can also pass in a specialized quadrature routine
via the `quad` keyword argument, which should accept arguments `quad(f,a,b,rtol=_,atol=_)`
similar to `quadgk`.  (This is useful if your weight function has discontinuities, in which
case you might want to break up the integration interval at the discontinuities.)

The type used for calculations and the return value is the same as the type the
endpoints `a` and `b`.
"""
function (wg::WeightedGauss)(W, a, b, rtol, quad=quadgk)
    # Uses the Lanczos recurrence described in Trefethen & Bau,
    # Numerical Linear Algebra, to find the `N`-point Gaussian quadrature
    # using O(N) integrals and O(N²) operations, applied to Chebyshev basis:
    q_0 = wg.q_0
    q_1 = wg.q_1
    v = wg.v
    resize!(q_0, 1)
    resize!(q_1, 1)
    resize!(v, 1)
    xscale = 2/(b-a) # scaling from (a,b) to (-1,1)
    T = typeof(xscale)
    wg.beta[1] = 0
    v[1] = q_0[1] = 0 # 0 polynomial
    wint = first(quad(W, a, b, rtol=rtol, segbuf=wg.segbuf))
    (wint isa Real && wint > 0) ||
        throw(ArgumentError("weight W must be real and positive"))
    atol = rtol*wint
    q_1[1] = 1/sqrt(wint)
    for n = 1:wg.N
        chebx!(v, q_1) # v = x * q_1
        wg.alpha[n] = let v=v, q_1=q_1
            first(quad(a, b, rtol=rtol, atol=atol, segbuf=wg.segbuf) do x
                t = (x-a)*xscale - 1
                W(x) * chebeval(t, q_1) * chebeval(t, v)
            end)
        end
        n == wg.N && break
        for j = 1:length(q_0); v[j] -= wg.beta[n]*q_0[j]; end
        for j = 1:length(q_1); v[j] -= wg.alpha[n]*q_1[j]; end
        wg.beta[n+1] = let v=v
            sqrt(first(quad(a, b, rtol=rtol, atol=atol, segbuf=wg.segbuf) do x
                W(x) * chebeval((x-a)*xscale - 1, v)^2
            end))
        end
        v .*= inv(wg.beta[n+1])
        q_0,q_1,v = q_1,v,q_0
    end

    E = eigen!(SymTridiagonal((@view wg.alpha[begin:end]), (@view wg.beta[2:wg.N])))

    xs = @view wg.points_buf[:, 1]
    ws = @view wg.points_buf[:, 2]
    ws .= wint .* abs2.(@view E.vectors[1,:])
    xs .= (E.values .+ 1) ./ xscale .+ a
    return (xs, ws)
end