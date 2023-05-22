"""
This file is based on QuadGK.jl but modified to reduce the number of
allocations.
"""

module Quad

using LinearAlgebra: eigen!, SymTridiagonal, LAPACK
using RecursiveArrayTools
using QuadGK

export WeightedGauss, AvgWeightedGauss, abscissae, weights

abstract type WeightedGaussBase end

abscissae(wgb::WeightedGaussBase) = @view wgb.points_buf[:, 1]
weights(wgb::WeightedGaussBase) = @view wgb.points_buf[:, 2]

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

struct WeightedGauss{T} <: WeightedGaussBase 
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

function lanczos_jacobi_coeffs!(W, N, a, b, xscale, wint, rtol, q_0, q_1, v, alpha, beta, segbuf, quad=quadgk)
    # Uses the Lanczos recurrence described in Trefethen & Bau,
    # Numerical Linear Algebra, to find the `N`-point Gaussian quadrature
    # using O(N) integrals and O(N²) operations, applied to Chebyshev basis:
    
    # TODO: How does this compare with Stieltjes method? Is it definitely different?
    resize!(q_0, 1)
    resize!(q_1, 1)
    resize!(v, 1)
    beta[1] = 0
    v[1] = q_0[1] = 0 # 0 polynomial
    atol = rtol*wint
    q_1[1] = 1/sqrt(wint)
    for n = 1:N
        chebx!(v, q_1) # v = x * q_1
        alpha[n] = let v=v, q_1=q_1
            first(quad(a, b, rtol=rtol, atol=atol, segbuf=segbuf) do x
                t = (x-a)*xscale - 1
                W(x) * chebeval(t, q_1) * chebeval(t, v)
            end)
        end
        if n == N
            break
        end
        for j = 1:length(q_0)
            v[j] -= beta[n]*q_0[j]
        end
        for j = 1:length(q_1)
            v[j] -= alpha[n]*q_1[j]
        end
        beta[n+1] = let v=v
            sqrt(first(quad(a, b, rtol=rtol, atol=atol, segbuf=segbuf) do x
                W(x) * chebeval((x-a)*xscale - 1, v)^2
            end))
        end
        v .*= inv(beta[n+1])
        q_0,q_1,v = q_1,v,q_0
    end
end

function (wg::WeightedGauss)(W, a, b, rtol, quad=quadgk)
    xscale = 2/(b-a) # scaling from (a,b) to (-1,1)
    wint = first(quad(W, a, b, rtol=rtol, segbuf=wg.segbuf))
    if !(wint isa Real && wint > 0)
        throw(ArgumentError("weight W must be real and positive"))
    end
    
    lanczos_jacobi_coeffs!(W, wg.N, a, b, xscale, wint, rtol, wg.q_0, wg.q_1, wg.v, wg.alpha, wg.beta, wg.segbuf, quad)

    E = eigen!(SymTridiagonal((@view wg.alpha[begin:end]), (@view wg.beta[2:wg.N])))

    xs = @view wg.points_buf[:, 1]
    ws = @view wg.points_buf[:, 2]
    ws .= wint .* abs2.(@view E.vectors[1,:])
    xs .= (E.values .+ 1) ./ xscale .+ a
    return (xs, ws)
end

struct AvgWeightedGauss{T} <: WeightedGaussBase 
    N::Int
    alpha::Vector{T}
    beta::Vector{T}
    q_0::Vector{T}
    q_1::Vector{T}
    v::Vector{T}
    segbuf::Vector{QuadGK.Segment{T, T, T}}
    points_buf::Array{Float32, 2}
end

function AvgWeightedGauss{T}(N) where {T}
    num_alpha_betas = N + 2
    npts = 2 * N + 1
    q_0, q_1, v = ntuple(3) do _
        sizehint!(Vector{T}(undef, 1), num_alpha_betas + 1)
    end
    AvgWeightedGauss{T}(
        N,
        Vector{T}(undef, num_alpha_betas),
        Vector{T}(undef, num_alpha_betas),
        q_0,
        q_1,
        v,
        alloc_segbuf(Float32, Float32, Float32; size=3),
        Array{Float32, 2}(undef, npts, 2)
    )
end

function avg_gauss(wg)
    # Optimal averaged Jacobi matrix example
    # For l=3 => matrix size = 2l + 1 = 7x7
    # [ α_0  β_1  0    0    0    0    0
    #   β_1  α_1  β_2  0    0    0    0
    #   0    β_2  α_2  β_3  0    0    0
    #   0    0    β_3  α_3  β_3  0    0
    #   0    0    0    β_3  α_2  β_2  0
    #   0    0    0    0    β_2  α_1  β_1
    #   0    0    0    0    0    β_1  α_0 ]
    # Diagonal
    dv = vcat(
        (@view wg.alpha[begin: end - 1]), # α_0, ... α_l;  e.g. α_0, α_1, α_2, α_3
        (@view wg.alpha[end - 2: -1: begin]) # α_{l-1}, ... α_0;  e.g. α_2, α_1, α_0
    )
    # Off-diagonal
    ev = vcat(
        (@view wg.beta[begin + 1:end - 1]), # β_1, ... β_{l};  e.g. β_1, β_2, β_3
        (@view wg.beta[end - 1: -1:begin + 1]) # β_{l}, ... β_1;  e.g. β_3, β_2, β_1
    )
    @info "dv, ev"
    @show dv
    @show ev

    # Uses stemr == RRR rther than QR internally
    return LAPACK.stegr!('V', dv, ev)
end

function optimal_avg_gauss(wg)
    # Optimal averaged Jacobi matrix example
    # For l=3 => matrix size = 2l + 1 = 7x7
    # [ α_0  β_1  0    0    0    0    0
    #   β_1  α_1  β_2  0    0    0    0
    #   0    β_2  α_2  β_3  0    0    0
    #   0    0    β_3  α_3  β_4  0    0
    #   0    0    0    β_4  α_2  β_2  0
    #   0    0    0    0    β_2  α_1  β_1
    #   0    0    0    0    0    β_1  α_0 ]
    # Diagonal
    dv = vcat(
        (@view wg.alpha[begin: end - 1]), # α_0, ... α_l;  e.g. α_0, α_1, α_2, α_3
        (@view wg.alpha[end - 2: -1: begin]) # α_{l-1}, ... α_0;  e.g. α_2, α_1, α_0
    )
    # Off-diagonal
    ev = vcat(
        (@view wg.beta[begin + 1:end]), # β_1, ... β_{l+1};  e.g. β_1, β_2, β_3, β_4
        (@view wg.beta[end - 2: -1:begin + 1]) # β_{l-1}, ... β_1;  e.g. β_2, β_1
    )
    @info "dv, ev"
    @show dv
    @show ev

    return LAPACK.stegr!('V', dv, ev)
end

function (wg::AvgWeightedGauss)(W, a, b, rtol, quad=quadgk)
    xscale = 2/(b-a) # scaling from (a,b) to (-1,1)
    wint = first(quad(W, a, b, rtol=rtol, segbuf=wg.segbuf))
    (wint isa Real && wint > 0) ||
        throw(ArgumentError("weight W must be real and positive"))
    
    # TODO: Do not need to calculate α_{l+1}, just β_{l+1}
    # Should unroll loop and specialise last iteration
    lanczos_jacobi_coeffs!(W, wg.N + 2, a, b, xscale, wint, rtol, wg.q_0, wg.q_1, wg.v, wg.alpha, wg.beta, wg.segbuf, quad)
    # wg.alpha contains α_0, α_1, α_2, ..., α_{l+1}
    # wg.beta contain β_0, β_1, β_2, ..., β_{l+1}
    @show "alpha, beta"
    @show wg.alpha
    @show wg.beta
    
    values, vectors = optimal_avg_gauss(wg)

    @info "vals, vecs"
    @show values
    @show vectors

    xs = @view wg.points_buf[:, 1]
    ws = @view wg.points_buf[:, 2]
    ws .= wint .* abs2.(@view vectors[1,:])
    # Is some normalisation here not being done?
    xs .= (values .+ 1) ./ xscale .+ a
    return (xs, ws)
end

end