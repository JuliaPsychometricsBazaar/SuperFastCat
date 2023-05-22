"""
The idea of this script is to compare the decision trees -- mainly finding out
where in the ``top-k'' each system comes relative to the more accurate quadgk
based one
"""

using Random: Xoshiro
using SuperFastCat
using SuperFastCat: ProgQuadGKDecisionTreeGenerationState, responses_idx, bintree_depth, questions, theta_lo, theta_hi, calc_ability, responses, next!, precompute!
using SuperFastCat.Slow
using TimerOutputs

include("./utils.jl")

function walk(rw, slow_state; other_states...)
    for (_, state) in other_states
        precompute!(state)
    end
    while true
        @show "Responses"
        @show responses(slow_state.likelihood)
        ridx = responses_idx(responses(slow_state.likelihood))
        depth = bintree_depth(ridx)
        slow_ability = calc_ability(slow_state)
        items = []
        for item_idx in 1:length(slow_state.item_bank)
            if item_idx in questions(slow_state.likelihood)
                continue
            end
            ev = Slow.slow_expected_var(slow_state.item_bank.params, slow_state.likelihood, slow_ability, item_idx, theta_lo, theta_hi)
            push!(items, (ev, item_idx))
        end
        sort!(items, by=x->x[1])
        @show "items" items
        for (name, state) in other_states
            @show name
            @show responses(state.likelihood)
            iteration_precompute!(state)
            best_ev = Inf
            best_idx = -1
            state_ability = calc_ability(state)
            @show "ability" slow_ability state_ability
            for item_idx in 1:length(state.item_bank)
                if item_idx in questions(state.likelihood)
                    continue
                end
                ev = Slow.slow_expected_var(state.item_bank.params, state.likelihood, state_ability, item_idx, theta_lo, theta_hi)
                if ev < best_ev
                    best_ev = ev
                    best_idx = item_idx
                end
            end
            @info "best_idx" name best_idx
            for (k, (_, gold_idx)) in enumerate(items)
                if best_idx == gold_idx
                    write_rec(
                        rw;
                        ridx=ridx,
                        depth=depth ,
                        type="item_params",
                        system=name,
                        k=k
                    )
                    break
                end
            end
        end
        next_item = items[1][2]
        insert!(slow_state.decision_tree_result, responses(slow_state.likelihood), slow_ability, next_item)
        for (_, state) in other_states
            next!(state.state_tree, state.likelihood, state.item_bank, next_item, slow_ability)
        end
        if next!(slow_state.state_tree, slow_state.likelihood, slow_state.item_bank, next_item, slow_ability)
            break
        end
    end
end

function main(outfn)
    reset_timer!()
    zero_subnormals_all()
    
    rng = Xoshiro(42)
    params = clumpy_4pl_item_bank(rng, 3, 1000) # 1000
    max_depth = 5
    state = SlowDecisionTreeGenerationState(params, max_depth)
    open_rec_writer(outfn) do rw
        walk(
            rw,
            state;
            fixedw=DecisionTreeGenerationState(params, max_depth; weighted_quadpts=5),
            iterqwk=ProgQuadGKDecisionTreeGenerationState(params, max_depth; quad_order=5, quad_max_depth=4),
      )
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS[1])
end
