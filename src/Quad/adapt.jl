"""
This file contains support for progressive adaptive quadrature. In contrast to
adaptive quadrature QuadGK.jl, this is lower level and the user is given
control over when to stop. The idea is that it can be used to partially GPU
accelerate the process, as well as evalute multiple integrals
simultaneously and choose which to refine based on finding some aggregate
e.g. min/max.
"""

function evalrule(fx, s, w, gw)
    # Ik and Ig are integrals via Kronrod and Gauss rules, respectively
    n1 = 1 - (length(w) & 1) # 0 if even order, 1 if odd order
    if n1 == 0 # even: Gauss rule does not include x == 0
        Ik = fx[end] * w[end]
        Ig = zero(Ik)
    else # odd: don't count x==0 twice in Gauss rule
        f0 = fx[end]
        Ig = f0 * gw[end]
        Ik = f0 * w[end] +
            (fx[end - 1] + fx[end - 2]) * w[end-1]
    end
    for i = 1:length(gw)-n1
        fg = fx[4i] + fx[4i - 1]
        fk = fx[4i - 2] + fx[4i - 3]
        Ig += fg * gw[i]
        Ik += fg * w[2i] + fk * w[2i-1]
    end
    Ik_s, Ig_s = Ik * s, Ig * s # new variable since this may change the type
    E = norm(Ik_s - Ig_s)
    return (Ik_s, E)
end

struct ProgQuadGK{T <: AbstractFloat}
    order::Int
    max_depth::Int
    a::T
    b::T
    x::Vector{T}
    w::Vector{T}
    gw::Vector{T}
    segpoints::Matrix{T}
end

function abscissa_to_segpoints!(out, a, b, x)
    s = (b - a) / 2
    for i in 1:(length(x) - 1)
        out[2i - 1] = a + (1 + x[i]) * s
        out[2i] = a + (1 - x[i]) * s
    end
    out[2 * length(x) - 1] = a + s
end

function precompute!(pqgk::ProgQuadGK)
    t = kronrod(Float32, pqgk.order)
    pqgk.x .= t[1]
    pqgk.w .= t[2]
    pqgk.gw .= t[3]
    for depth in 1:pqgk.max_depth
        denom = 2 ^ (depth - 1)
        for num in 0:(denom - 1)
            a = pqgk.a + (pqgk.b - pqgk.a) * (num / denom)
            b = pqgk.a + (pqgk.b - pqgk.a) * ((num + 1) / denom)
            abscissa_to_segpoints!((@view pqgk.segpoints[:, denom + num]), a, b, pqgk.x)
        end
    end
end

"""
Construct a progressive adaptive quadrature object. The order is the order of the
Gauss-Kronrod rule to use. The max_depth is the maximum depth of the binary tree
of subintervals to use. The a and b are the endpoints of the interval to integrate
over.
"""
function ProgQuadGK(order, max_depth, a::T, b::T) where {T <: AbstractFloat}
    pqgk = ProgQuadGK{T}(
        order,
        max_depth,
        a,
        b,
        Array{T}(undef, order + 1),
        Array{T}(undef, order + 1),
        Array{T}(undef, (order + 1) ÷ 2),
        Array{T}(undef, 2 * order + 1, 2 ^ max_depth - 1)
    )
    precompute!(pqgk)
    pqgk
end

bintree_depth(idx) = 8sizeof(typeof(idx)) - leading_zeros(idx)
seg_width_at_depth(total_width, depth) = total_width * 2f0 ^ (-depth)

function eval_at_seg!(f, out, pqgk::ProgQuadGK, seg_idx)
    out .= f.(@view pqgk.segpoints[:, seg_idx])
end

"""
This is the main entrypoint
"""
function evalrule(fx::AbstractVector, pqgk::ProgQuadGK{<: AbstractFloat}, seg_idx::Integer)
    total_width = pqgk.b - pqgk.a
    evalrule(fx, seg_width_at_depth(total_width, bintree_depth(seg_idx)), pqgk.w, pqgk.gw)
end

function evalrule(fx::AbstractVector, pqgk::ProgQuadGK{T}, total_width::T) where {T <: AbstractFloat}
    evalrule(fx, total_width, pqgk.w, pqgk.gw)
end

function evalrule(f::Function, pqgk::ProgQuadGK, seg_idx)
    evalrule(f.(@view pqgk.segpoints[:, seg_idx]), pqgk, seg_idx)
end

function seg_child_idxs(seg_idx)
    (2 * seg_idx, 2 * seg_idx + 1)
end