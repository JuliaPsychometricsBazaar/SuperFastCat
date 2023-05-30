"""
This is all the state needed for decision tree
`generate_dt_cat_exhaustive_point_ability`. The idea is that nothing (much)
is calculated here, but initialisation allocates all the buffers, so that
we can make sure that the next step does not allocate anything unexpected.
"""
Base.@kwdef struct FixedWDecisionTreeGenerationState
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
    # \- Buffers / Integration points
    weighted_gauss::WeightedGauss
    # \- Buffers / f(x) values at integration points
    ir_fx::Vector{Float32}
    # \- Buffers / Item-response at integration points
    item_response_quadrature_buf::Vector{Float32}
    # \- Buffers / Rough best list
    rough_best::RoughBest
end

function FixedWDecisionTreeGenerationState(item_bank::ItemBankT, max_depth; weighted_quadpts=5)
    # XXX: This is hardcoded for now, but should be found based on error estimate
    margin::Float32 = 0.1
    num_items = size(item_bank, 1)
    rough_best = RoughBest(FastForwardOrdering(), ceil(Int, sqrt(num_items) + 3), margin)
    FixedWDecisionTreeGenerationState(
        item_bank=ItemBank(item_bank),
        likelihood=ResponsesLikelihood(max_depth + 1), # +1 for final ability estimates
        state_tree=TreePosition(max_depth),
        decision_tree_result=DecisionTree(max_depth),
        weighted_gauss=WeightedGauss{Float32}(weighted_quadpts),
        ir_fx=Vector{Float32}(undef, weighted_quadpts),
        item_response_quadrature_buf=Vector{Float32}(undef, num_items),
        rough_best=rough_best 
    )
end

function calc_ability(state::FixedWDecisionTreeGenerationState)::Float32
    if state.state_tree.cur_depth == 0
        return 0.0f0
    else
        return mean_and_c(abscissae(state.weighted_gauss), weights(state.weighted_gauss))[1]
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

#=function expected_var_turbo(state::FixedWDecisionTreeGenerationState, lh_quad_xs, lh_quad_ws, ability, item_idx)::Float32
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

function best_item(state::FixedWDecisionTreeGenerationState, ability)
    best_ev = Inf
    best_idx = -1
    for item_idx in 1:length(state.item_bank)
        if item_idx in questions(state.likelihood)
            continue
        end
        ev = expected_var(
            state.item_bank,
            state.ir_fx,
            abscissae(state.weighted_gauss),
            weights(state.weighted_gauss),
            ability,
            item_idx
        )
        if ev < best_ev
            best_ev = ev
            best_idx = item_idx
        end
    end
    return best_idx
end

function precompute!(state::FixedWDecisionTreeGenerationState)
    precompute!(state.item_bank)
end

function iteration_precompute!(state::FixedWDecisionTreeGenerationState)
    state.weighted_gauss(state.likelihood, theta_lo, theta_hi, 1f-3)
end

function generate_dt_cat_exhaustive_point_ability(state::FixedWDecisionTreeGenerationState)
    ## Step 0. Precompute item bank
    @timeit "precompute item bank" begin
        precompute!(state)
    end

    while true
        ## Step 1. Get specialised gaussian quadrature points
        @timeit "calculate gaussian quadrature points" begin
            iteration_precompute!(state)
        end

        ## Step 2. Compute a point estimate of ability
        @timeit "calculate ability point estimate" begin
            ability = calc_ability(state)
        end
        
        ## Step 3. Find quickly the nearby ones
        @timeit "rough calculation" begin
            next_item = best_item(state, ability)
        end
        
        @timeit "select next item and add to dt" begin
            insert!(state.decision_tree_result, responses(state.likelihood), ability, next_item)
            if state.state_tree.cur_depth == state.state_tree.max_depth
                @timeit "final state ability calculation" begin
                    for resp in (false, true)
                        resize!(state.likelihood, state.state_tree.cur_depth)
                        push_question_response!(state.likelihood, state.item_bank, next_item, resp)
                        state.weighted_gauss(state.likelihood, theta_lo, theta_hi, 1f-3)
                        ability = calc_ability(state)
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

@precompile_setup begin
    rng = Xoshiro(42)
    max_depth = 2
    @precompile_all_calls begin
        params = clumpy_4pl_item_bank(rng, 2, 3)
        state = FixedWDecisionTreeGenerationState(params, max_depth)
        dt_cat = generate_dt_cat_exhaustive_point_ability(state)
    end
end