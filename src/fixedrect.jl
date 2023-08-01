Base.@kwdef struct FixedRectDecisionTreeGenerationState
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
    # \- Buffers / integration points
    quadpts::Vector{Float32}
    # \- Buffers / lh(x) values at integration points
    lh_x::Vector{Float32}
    # \- Buffers / f(x) values at integration points
    ir_fx::Vector{Float32}
end

function FixedRectDecisionTreeGenerationState(item_bank::ItemBankT, max_depth; quadpts=39)
    FixedRectDecisionTreeGenerationState(
        item_bank=ItemBank(item_bank),
        likelihood=ResponsesLikelihood(max_depth + 1), # +1 for final ability estimates
        state_tree=TreePosition(max_depth),
        decision_tree_result=DecisionTree(max_depth),
        quadpts=range(theta_lo, theta_hi, quadpts),
        lh_x=Vector{Float32}(undef, quadpts),
        ir_fx=Vector{Float32}(undef, quadpts)
    )
end

function calc_ability(state::FixedRectDecisionTreeGenerationState)::Float32
    if state.state_tree.cur_depth == 0
        return 0.0f0
    else
        return mean_and_c(state.quadpts, state.lh_x)[1]
    end
end

function rect_expected_var(item_bank, lh, ir_fx_buf, pts, ability, item_idx)::Float32
    res = 0f0
    @inbounds @fastmath for resp in false:true
        ir = ItemResponse(item_bank.affines, item_idx, resp)
        @turbo ir_fx_buf .= ir.(pts) .* lh
        prob = ir(ability)
        # XXX: Could be faster to get all outcomes from the ItemResponse at the same time
        (var, _, _) = var_mean_and_c(pts, ir_fx_buf)
        res += prob * var
    end
    res 
end

function best_item(state::FixedRectDecisionTreeGenerationState, ability)
    best_ev = Inf
    best_idx = -1
    for item_idx in 1:length(state.item_bank)
        if item_idx in questions(state.likelihood)
            continue
        end
        ev = rect_expected_var(
            state.item_bank,
            state.lh_x,
            state.ir_fx,
            state.quadpts,
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

function precompute!(state::FixedRectDecisionTreeGenerationState)
    precompute!(state.item_bank)
end

function iteration_precompute!(state::FixedRectDecisionTreeGenerationState)
    state.lh_x .= state.likelihood.(state.quadpts)
end

function generate_dt_cat_exhaustive_point_ability(state::FixedRectDecisionTreeGenerationState)
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
                        state.lh_x .= state.likelihood.(state.quadpts)
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
        state = FixedRectDecisionTreeGenerationState(params, max_depth)
        dt_cat = generate_dt_cat_exhaustive_point_ability(state)
    end
end