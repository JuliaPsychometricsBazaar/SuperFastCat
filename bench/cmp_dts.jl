"""
The idea of this script is to compare the decision trees -- mainly finding out
where in the ``top-k'' each system comes relative to the more accurate quadgk
based one
"""

using Random: Xoshiro
using SuperFastCat
using SuperFastCat: ProgQuadGKDecisionTreeGenerationState, responses_idx, bintree_depth, questions, theta_lo, theta_hi, calc_ability, responses, next!, precompute!, best_item, summarize_items
using SuperFastCat.Slow
using TimerOutputs

include("./utils.jl")

function walk(rw, slow_state, systems)
    for system in systems
        precompute!(system[:state])
    end
    while true
        #@show "Responses"
        #@show responses(slow_state.likelihood)
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
        #@show items
        for system in systems
            name = system[:name]
            state = system[:state]
            timing = @timed begin
                #@show name
                #@show responses(state.likelihood)
                iteration_precompute!(state)
                state_ability = calc_ability(state)
                #@show "ability" slow_ability state_ability
                best_idx = best_item(state, state_ability)
                #@show "best_idx" name best_idx
            end
            for (k, (_, gold_idx)) in enumerate(items)
                if best_idx == gold_idx
                    write_rec(
                        rw;
                        ridx=ridx,
                        depth=depth ,
                        type="next_item",
                        system=name,
                        k=k,
                        time=timing[:time]
                    )
                    break
                end
            end
        end
        next_item = items[1][2]
        insert!(slow_state.decision_tree_result, responses(slow_state.likelihood), slow_ability, next_item)
        for system in systems
            state = system[:state]
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
    params = clumpy_4pl_item_bank(rng, 3, 1000)
    max_depth = 5
    state = SlowDecisionTreeGenerationState(params, max_depth)
    other_states = [
        Dict(:name => :fixedw_5, :weighted_quadpts => 5, :state => FixedWDecisionTreeGenerationState(params, max_depth; weighted_quadpts=5)),
        Dict(:name => :fixedw_8, :weighted_quadpts => 8, :state => FixedWDecisionTreeGenerationState(params, max_depth; weighted_quadpts=8)),
        Dict(:name => :fixedw_11, :weighted_quadpts => 11, :state => FixedWDecisionTreeGenerationState(params, max_depth; weighted_quadpts=11)),
        Dict(:name => :iterqwk, :state => ProgQuadGKDecisionTreeGenerationState(params, max_depth; quad_order=5, quad_max_depth=4)),
    ]
    open_rec_writer(outfn) do rw
        walk(rw, state, other_states)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS[1])
end
