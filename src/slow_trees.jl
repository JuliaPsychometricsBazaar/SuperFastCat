export SlowDecisionTreeGenerationState, generate_dt_cat_exhaustive_point_ability

Base.@kwdef struct SlowDecisionTreeGenerationState
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
end

function SlowDecisionTreeGenerationState(item_bank::ItemBankT, max_depth)
    SlowDecisionTreeGenerationState(
        item_bank=ItemBank(item_bank),
        # +1 for final ability estimates
        likelihood=ResponsesLikelihood(max_depth + 1),
        state_tree=TreePosition(max_depth),
        decision_tree_result=DecisionTree(max_depth)
    )
end

function best_item(state::SlowDecisionTreeGenerationState, ability)
    best_ev = Inf
    best_idx = -1
    for item_idx in 1:length(state.item_bank)
        if item_idx in questions(state.likelihood)
            continue
        end
        #  expected_var(state.item_bank, state.ir_fx, lh_quad_xs, lh_quad_ws, ability, item_idx)
        ev = Slow.slow_expected_var(state.item_bank.params, state.likelihood, ability, item_idx, theta_lo, theta_hi)
        if ev < best_ev
            best_ev = ev
            best_idx = item_idx
        end
    end
    return best_idx
end

function calc_ability(state::SlowDecisionTreeGenerationState)
    if state.state_tree.cur_depth == 0
        return 0.0
    else
        Slow.slow_mean_and_c((x -> Slow.slow_likelihood(state.item_bank.params, state.likelihood, x)), theta_lo, theta_hi)[1]
    end
end

function generate_dt_cat_exhaustive_point_ability(state::SlowDecisionTreeGenerationState)
    while true
        ## Step 1. Compute a point estimate of ability
        @timeit "calculate ability point estimate" begin
            ability = calc_ability(state)
        end
        
        ## Step 2. Get best next item
        @timeit "next item rule" begin
            next_item = best_item(state, ability)
        end
        
        if next_item == -1
            error("No best item found")
        end
        
        @timeit "add to dt" begin
            insert!(state.decision_tree_result, responses(state.likelihood), ability, next_item)
            if state.state_tree.cur_depth == state.state_tree.max_depth
                @timeit "final state ability calculation" begin
                    for resp in (false, true)
                        resize!(state.likelihood, state.state_tree.cur_depth)
                        push_question_response!(state.likelihood, state.item_bank, next_item, resp)
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