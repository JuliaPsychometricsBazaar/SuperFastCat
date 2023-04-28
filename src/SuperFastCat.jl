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

include("./types.jl")
include("./stats.jl")
include("./dummydata.jl")
include("./wgauss.jl")
include("./decisiontree.jl")

export DecisionTreeGenerationState, ResponsesLikelihood, ItemBank
export zero_subnormals_all, generate_dt_cat_exhaustive_point_ability
export random_responses, clumpy_4pl_item_bank, push_question_response!

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

function parameter_prune(dist_buf, sort_idxs_buf, current_ability, params)
    #Kung's
    # Calculate distance from current ability
    dist_buf .= abs.(params[:, idxr_difficulty] .- current_ability)
    sort_idxs_buf .= sortperm(dist_buf)
    bests = Array
    # 
end

Base.@kwdef struct ParameterBasedPruningBuffers
    dists::Vector{Float32}
    sort_idxs::Vector{Int}
end

function ParameterBasedPruningBuffers(num_items)
    ParameterBasedPruningBuffers(
        dists=Vector{Float32}(undef, num_items),
        sort_idxs=Vector{Int}(undef, num_items),
    )
end

"""
Rough best has property that it is always sorted according to `ordering`. The
best measure is at the beginning `bests[1]`. The rest of the list is within `margin` of the best.
"""
@Base.kwdef struct RoughBest{OrderingT <: Ordering}
    ordering::OrderingT
    best_idxs::Vector{Int}
    best_measures::Vector{Float32}
    margin::Float32
end

function RoughBest(ordering::Ordering, capacity::Int, margin)
    best_idxs = Vector{Int}()
    sizehint!(best_idxs, capacity)
    best_measures = Vector{Float32}()
    sizehint!(best_measures, capacity)
    RoughBest(
        ordering=ordering,
        best_idxs=best_idxs,
        best_measures=best_measures,
        margin=margin
    )
end

KnownForwardOrderings = Union{FastForwardOrdering, ForwardOrdering}
KnownReverseOrderings = ReverseOrdering{KnownForwardOrderings}

apply_margin(::KnownForwardOrderings, measure, margin) = measure + margin
apply_margin(::KnownReverseOrderings, measure, margin) = meaure - margin

function add_to_rough_best!(bests::RoughBest, idx, measure::Float32)
    if length(bests.best_idxs) == 0
        # First item
        push!(bests.best_idxs, idx)
        push!(bests.best_measures, measure)
    else
        cur_best_measure = bests.best_measures[1]
        if lt(bests.ordering, measure, cur_best_measure)
            # Added item is new best -- insert at front and evict everything > best_measure + margin
            insert!(bests.best_idxs, 1, idx)
            insert!(bests.best_measures, 1, measure)
            margin_measure = apply_margin(bests.ordering, measure, bests.margin)
            insert_idx = searchsortedfirst(bests.best_measures, margin_measure, bests.ordering)
            resize!(bests.best_idxs, insert_idx - 1)
            resize!(bests.best_measures, insert_idx - 1)
        else
            # Try to see if we can insert it somewhere
            margin_measure = apply_margin(bests.ordering, cur_best_measure, bests.margin)
            if lt(bests.ordering, measure, margin_measure)
                # We want to keep it since it is within `margin` of best
                insert_idx = searchsortedfirst(bests.best_measures, measure, bests.ordering)
                insert!(bests.best_idxs, insert_idx, idx)
                insert!(bests.best_measures, insert_idx, measure)
            end
        end
    end
end

function Base.empty!(bests::RoughBest)
    empty!(bests.best_idxs)
    empty!(bests.best_measures)
end

function Base.resize!(bests::RoughBest, size)
    resize!(bests.best_idxs, size)
    resize!(bests.best_measures, size)
end

function Base.length(bests::RoughBest)
    length(bests.best_idxs)
end

"""
This is all the state needed for decision tree
`generate_dt_cat_exhaustive_point_ability`. The idea is that nothing (much)
is calculated here, but initialisation allocates all the buffers, so that
we can make sure that the next step does not allocate anything unexpected.
"""
Base.@kwdef struct DecisionTreeGenerationState
    # Input
    # \- Input / Item bank
    item_bank::ItemBank
    # State
    # \- State / Likelihood
    likelihood::ResponsesLikelihood
    # \- State / Tree position
    state_tree::TreePosition
    # Outputs
    # \- Outputs / Decision tree
    decision_tree_result::DecisionTree
    # Buffers
    # \- Buffers / Parameter based item pruning
    parameter_pruning::ParameterBasedPruningBuffers
    # \- Buffers / Integration points
    weighted_gauss::WeightedGauss
    # \- Buffers / f(x) values at integration points
    ir_fx::Vector{Float32}
    # \- Buffers / Item-response at integration points
    item_response_quadrature_buf::Vector{Float32}
    # \- Buffers / Rough best list
    rough_best::RoughBest
end

function DecisionTreeGenerationState(item_bank::ItemBankT, max_depth, weighted_quadpts=5)
    # XXX: This is hardcoded for now, but should be found based on error estimate
    margin::Float32 = 0.1
    num_items = size(item_bank, 1)
    rough_best = RoughBest(FastForwardOrdering(), ceil(Int, sqrt(num_items) + 3), margin)
    DecisionTreeGenerationState(
        item_bank=ItemBank(item_bank),
        likelihood=ResponsesLikelihood(max_depth + 1), # +1 for final ability estimates
        state_tree=TreePosition(max_depth),
        decision_tree_result=DecisionTree(max_depth),
        parameter_pruning=ParameterBasedPruningBuffers(num_items),
        weighted_gauss=WeightedGauss{Float32}(weighted_quadpts),
        ir_fx=Vector{Float32}(undef, weighted_quadpts),
        item_response_quadrature_buf=Vector{Float32}(undef, num_items),
        rough_best=rough_best 
    )
end

function calc_ability(state::DecisionTreeGenerationState, x, w)::Float32
    if state.state_tree.cur_depth == 0
        return 0.0f0
    else
        #llf = ResponsesLikelihood(precomputed, cur_depth)
        #return optimize(llf, state.parent_ability, LBFGS(); autodiff = :forward)
        return mean_and_c(x, w)[1]
    end
end

function expected_var(item_bank, ir_fx_buf, lh_quad_xs, lh_quad_ws, ability, item_idx)::Float32
    res = 0f0
    @inbounds @fastmath for resp in false:true
        ir = ItemResponse(item_bank.affines, item_idx, resp)
        @turbo ir_fx_buf .= ir.(lh_quad_xs)
        prob = ir(ability)
        # XXX: Could be faster to get all outcomes from the ItemResponse at the same time
        (var, _, _) = var_mean_and_c(lh_quad_xs, lh_quad_ws, ir_fx_buf)
        res += prob * var
    end
    res 
end

#=function expected_var_turbo(state::DecisionTreeGenerationState, lh_quad_xs, lh_quad_ws, ability, item_idx)::Float32
    res = 0f0
    @fastmath @turbo for resp in false:true
        ir = ItemResponse(state.item_bank.affines, item_idx, resp)
        ir_fx = ir.(lh_quad_xs)
        prob = ir(ability)
        # XXX: Could be faster to get all outcomes from the ItemResponse at the same time
        (var, _, _) = var_mean_and_c(lh_quad_xs, lh_quad_ws, ir_fx)
        res += prob * var
    end
    res 
end=#

function generate_dt_cat_exhaustive_point_ability(state::DecisionTreeGenerationState)
    ## Step 0. Precompute item bank
    @timeit "precompute item bank" begin
        precompute!(state.item_bank)
    end

    while true
        #showqa(stdout, state.likelihood)
        #println(stdout, responses_idx(state.likelihood.responses))
        ## Step 1. Get specialised gaussian quadrature points
        @timeit "calculate gaussian quadrature points" begin
            # @fastmath @turbo 
            #lh_quad_xs, lh_quad_ws = gauss(state.likelihood, 5, -10.0f0, 10.0f0, rtol=1e-3)
            lh_quad_xs, lh_quad_ws = state.weighted_gauss(state.likelihood, -10.0f0, 10.0f0, 1f-3)
        end

        ## Step 2. Compute a point estimate of ability
        @timeit "calculate ability point estimate" begin
            ability = calc_ability(state, lh_quad_xs, lh_quad_ws)
        end
        
        ## Step 3. Find quickly the nearby ones
        @timeit "rough calculation" begin
            
            empty!(state.rough_best)
            # @fastmath @turbo 
            for item_idx in 1:length(state.item_bank)
                if item_idx in questions(state.likelihood)
                    continue
                end
                #@info "expected_var" expected_var typeof(expected_var) isbits(expected_var)
                #@timeit "add to rough best outer" begin
                ev = expected_var(state.item_bank, state.ir_fx, lh_quad_xs, lh_quad_ws, ability, item_idx)
                add_to_rough_best!(state.rough_best, item_idx, ev)
                #end
            end
        end
        
        #@info "Filtered" (length(state.item_bank) - length(state.likelihood)) length(state.rough_best)
        
        ## Step 4. Exact calculation with QuadGK TODO
        #for (item_idx, expected_var) in rough_best.bests
            #ir = ItemResponse(state.item_bank, item_idx, resp)
            #prob = ir(x)``
            #add_to_rough_best!(rough_best, item_idx, expected_var)
        #end
        @timeit "select next item and add to dt" begin
            next_item = state.rough_best.best_idxs[1]
            insert!(state.decision_tree_result, responses(state.likelihood), ability, next_item)
            if state.state_tree.cur_depth == state.state_tree.max_depth
                @timeit "final state ability calculation" begin
                    for resp in (false, true)
                        resize!(state.likelihood, state.state_tree.cur_depth)
                        push_question_response!(state.likelihood, state.item_bank, next_item, resp)
                        lh_quad_xs, lh_quad_ws = state.weighted_gauss(state.likelihood, -10.0f0, 10.0f0, 1f-3)
                        ability = calc_ability(state, lh_quad_xs, lh_quad_ws)
                        insert!(state.decision_tree_result, responses(state.likelihood), ability)
                    end
                end
            end
        end

        @timeit "move to next node" begin
            ## Step ___. Move to next position within the tree
            if next!(state.state_tree, state.likelihood, state.item_bank, next_item, ability)
                break
            end
        end
    end
    state.decision_tree_result
end

function zero_subnormals_all()
    Threads.@threads :static for i in 1:Threads.nthreads()
        set_zero_subnormals(true)
    end
end

using SnoopPrecompile
@precompile_setup begin
    rng = Xoshiro(42)
    max_depth = 2
    @precompile_all_calls begin
        params = clumpy_4pl_item_bank(rng, 2, 3)
        state = DecisionTreeGenerationState(params, max_depth)
        dt_cat = generate_dt_cat_exhaustive_point_ability(state)
    end
end

include("./Slow.jl")

using IntervalArithmetic

include("./intervalbest.jl")
include("./iterqwk.jl")

end
