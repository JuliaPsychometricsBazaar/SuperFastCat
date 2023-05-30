module SuperFastCat

using Distributions
using Random: Xoshiro
using QuadGK
using LogExpFunctions: logistic
using LoopVectorization
using Base.Order: Ordering, ForwardOrdering, ReverseOrdering, lt, ord
using QuickHeaps: FastForwardOrdering
using TimerOutputs
using Base: Threads
using FillArrays
using StaticArrays

include("./pushvectors.jl")

using .PushVectors: PushVector

const theta_lo = -10.0f0
const theta_hi = 10.0f0
const theta_width = theta_hi - theta_lo

bintree_depth(idx) = 8sizeof(typeof(idx)) - leading_zeros(idx)

include("./types.jl")
include("./stats.jl")
include("./dummydata.jl")
include("./Quad/Quad.jl")
include("./decisiontree.jl")

using .Quad

export FixedWDecisionTreeGenerationState, ResponsesLikelihood, ItemBank
export zero_subnormals_all, generate_dt_cat_exhaustive_point_ability
export random_responses, clumpy_4pl_item_bank, push_question_response!
export iteration_precompute!

Base.@kwdef struct ItemBank
    params::ItemBankT
    affines::Array{Float32, 2}
end

function ItemBank(params)
    ItemBank(
        params=params,
        affines=Array{Float32, 2}(undef, size(params, 1), 6)
    )
end

Base.length(ib::ItemBank) = size(ib.params, 1)

Base.@kwdef mutable struct ResponsesLikelihood
    questions::Vector{UInt32}
    responses::Vector{Bool}
    affines::PrecomputedLikelihoodT
    length::Int
end

function ResponsesLikelihood(max_depth)
    ResponsesLikelihood(
        questions=Vector{UInt32}(undef, max_depth),
        responses=Vector{Bool}(undef, max_depth),
        affines=Array{Float32, 2}(undef, max_depth, 4),
        length=0
    )
end

function questions(rl::ResponsesLikelihood)
    @view rl.questions[1:rl.length]
end

function responses(rl::ResponsesLikelihood)
    @view rl.responses[1:rl.length]
end

function showqa(io::IO, rl::ResponsesLikelihood)
    print(io, "*$(length(rl))*  ")
    for i in 1:length(rl)
        sym = rl.responses[i] ? "✔" : "✘"
        print(io, "q$(rl.questions[i]): $sym, ")
    end
    println()
end

function push_question_response!(lh::ResponsesLikelihood, item_bank::ItemBank, question, response)
    lh.length += 1
    replace_question_response!(lh, item_bank, question, response)
end

function replace_question_response!(lh::ResponsesLikelihood, item_bank::ItemBank, question, response)
    lh.questions[lh.length] = question
    lh.responses[lh.length] = response
    question_affs = @view item_bank.affines[question, :]
    if response
        yc = idxr_ir_pos_y_c
        ym = idxr_ir_pos_y_m
    else
        yc = idxr_ir_neg_y_c
        ym = idxr_ir_neg_y_m
    end
    lh.affines[lh.length, :] .= (
        question_affs[idxr_x_c],
        question_affs[idxr_x_m],
        question_affs[yc],
        question_affs[ym],
    )
end

function Base.length(lh::ResponsesLikelihood)
    lh.length
end

function Base.resize!(lh::ResponsesLikelihood, size)
    lh.length = size
end

function logistic_affine(x, xc, xm, yc, ym)
    z = muladd(x, xm, xc)
    y = logistic(z)
    muladd(y, ym, yc)
end

function (rl::ResponsesLikelihood)(x::Float32)::Float32
    res::Float32 = 1.0
    # @fastmath @turbo 
    for i in 1:length(rl)
        aff = logistic_affine(
            x,
            rl.affines[i, idxr_x_c],
            rl.affines[i, idxr_x_m],
            rl.affines[i, idxr_lh_y_c],
            rl.affines[i, idxr_lh_y_m]
        )
        res *= aff
    end
    res
end

struct ItemResponse
    affines::Matrix{Float32}
    question::Int
    response::Bool
end

LoopVectorization.can_turbo(::ItemResponse, ::Val{1}) = true

function (ir::ItemResponse)(x)
    affines = @view ir.affines[ir.question, :]
    xc = affines[idxr_x_c]
    xm = affines[idxr_x_m]
    if ir.response
        yc = affines[idxr_ir_pos_y_c]
        ym = affines[idxr_ir_pos_y_m]
    else
        yc = affines[idxr_ir_neg_y_c]
        ym = affines[idxr_ir_neg_y_m]
    end
    logistic_affine(x, xc, xm, yc, ym)
end

const logistic_normal_scaler = 1.702

function precompute_x_affine_normal_as_logistic(params::ItemBankT, question)
    scaled_discrimination = logistic_normal_scaler * params[question, idxr_discrimination]
    c = -(params[question, idxr_difficulty] * scaled_discrimination)
    m = scaled_discrimination
    (c, m)
end

function precompute_y_affine(params::ItemBankT, question, response)
    guess = params[question, idxr_guess]
    slip = params[question, idxr_slip]
    y_guess_slip = guess + slip
    if response > 0
        y_c = guess
        y_m = 1.0 - y_guess_slip 
    else
        y_c = 1.0 - guess
        y_m = y_guess_slip - 1.0
    end
    return (y_c, y_m)
end

function precompute!(item_bank::ItemBank)
    # @fastmath @turbo 
    for item_idx in 1:length(item_bank)
        param_prec = @view item_bank.affines[item_idx, :]
        (param_prec[idxr_x_c], param_prec[idxr_x_m]) = precompute_x_affine_normal_as_logistic(item_bank.params, item_idx)
        (param_prec[idxr_ir_neg_y_c], param_prec[idxr_ir_neg_y_m]) = precompute_y_affine(item_bank.params, item_idx, 0)
        (param_prec[idxr_ir_pos_y_c], param_prec[idxr_ir_pos_y_m]) = precompute_y_affine(item_bank.params, item_idx, 1)
    end
end

function tree_size(max_depth)
    2^(max_depth + 1) - 1
end

include("./roughbest.jl")

function zero_subnormals_all()
    Threads.@threads :static for i in 1:Threads.nthreads()
        set_zero_subnormals(true)
    end
end

using SnoopPrecompile

include("./fixedw.jl")
include("./Slow.jl")
include("./slow_trees.jl")

using IntervalArithmetic

include("./intervalbest.jl")
include("./iterqwk.jl")

end
