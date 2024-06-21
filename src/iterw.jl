struct ProgQuadW
    order::Int
    max_depth::Int
    a::Float32
    b::Float32
    x::Vector{Float32}
    w::Vector{Float32}
    gw::Vector{Float32}
    segpoints::Matrix{Float32}
end

function ProgQuadW(order, max_depth, a, b)
    ProgQuadW(
        order,
        max_depth,
        a,
        b,
        Array{Float32}(undef, order + 1),
        Array{Float32}(undef, order + 1),
        Array{Float32}(undef, (order + 1) ÷ 2),
        Array{Float32}(undef, 2 * order + 1, 2 ^ max_depth - 1)
    )
end

Base.@kwdef struct ProgQuadWDecisionTreeGenerationState
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
    # \- Buffers / Progressive quadrature
    prog_quadw::ProgQuadW
    # \- Buffers / likelihood values at integration points
    lh_vals::Vector{Float32}
    # \- Buffers / likelihood values * some f(x) at integration points
    lh_fx::Vector{Float32}
    # \- Buffers / likelihood values * some interval f(x) at integration points
    lh_fx_interval::Vector{Interval{Float32}}
    # \- Buffers / Item-response at integration points
    item_response_quadrature_buf::Vector{Float32}
    # \- Buffers / Interval best list
    interval_best::IntervalBest
end

function ProgQuadWDecisionTreeGenerationState(item_bank::ItemBankT, max_depth; quad_order=5, quad_max_depth=4)
    num_items = size(item_bank, 1)
    num_segpts = quad_order * 2 + 1
    interval_best = IntervalBest(FastForwardOrdering(), ceil(Int, sqrt(num_items) + 3))
    ProgQuadWDecisionTreeGenerationState(
        item_bank=ItemBank(item_bank),
        likelihood=ResponsesLikelihood(max_depth + 1), # +1 for final ability estimates
        state_tree=TreePosition(max_depth),
        decision_tree_result=DecisionTree(max_depth),
        prog_quadgk=ProgQuadGK(quad_order, quad_max_depth, -10.0f0, 10.0f0),
        lh_vals=Vector{Float32}(undef, num_segpts),
        lh_fx=Vector{Float32}(undef, num_segpts),
        lh_fx_interval=Vector{Interval{Float32}}(undef, num_segpts),
        item_response_quadrature_buf=Vector{Float32}(undef, num_items),
        interval_best=interval_best
    )
end

function precompute!(state::ProgQuadWDecisionTreeGenerationState)
    precompute!(state.item_bank)
    precompute!(state.prog_quadgk)
end

function best_item(state::ProgQuadWDecisionTreeGenerationState, ability)
    @timeit "first pass" begin
        for item_idx in 1:length(state.item_bank)
            if item_idx in questions(state.likelihood)
                continue
            end
            res = 0f0 ± 0f0
            for resp in false:true
                ir = ItemResponse(state.item_bank.affines, item_idx, resp)
                prob = ir(ability)
                res += prob * initial_var_estimate(state, 10.0f0, ir)
            end
            res
            add_to_interval_best!(state.interval_best, item_idx, res)
        end
    end

    #@info "Filtered" (length(state.item_bank) - length(state.likelihood)) length(state.interval_best)

    @timeit "adaptive passes" begin
        cur_depth = 1
        # Rule is to
        #  * Process each item round-robin (fairness)
        #  * Prioritise the current earliest item (best-first heuristic -- this can knock out later items from consideration more quickly)
        bests_contents_idx = 1
        while true
            if length(state.interval_best) == 1
                # Success!
                break
            end
            while bests_contents_idx <= length(state.interval_best)
                if state.interval_best.contents[bests_contents_idx].depth == cur_depth
                    break
                end
                bests_contents_idx += 1
            end
            if bests_contents_idx > length(state.interval_best)
                # Next depth
                cur_depth += 1
                bests_contents_idx = 1
                if cur_depth > state.prog_quadgk.max_depth
                    # Unsuccessful, but giving up early
                    break
                end
            end
            cur_item = state.interval_best.contents[bests_contents_idx]
            @assert cur_depth == cur_item.depth "cur_depth $cur_depth != cur_item.depth $(cur_item.depth)"
            item_idx = cur_item.item_bank_idx
            acc = cur_item.acc
            segs = cur_item.segs
            if segs === nothing
                segs = []
                s1_idx = 2
                s2_idx = 3
            else
                # pick the minimum seg & split it
                seg = heappop!(segs, Reverse)
                s1_idx = 2 * seg.seg_idx
                s2_idx = 2 * seg.seg_idx + 1
                acc = acc - seg.vals
            end
            seg_width = seg_width_at_depth(20.0f0, cur_depth)
            res = 0f0 ± 0f0
            for resp in false:true
                ir = ItemResponse(state.item_bank.affines, item_idx, resp)
                # TODO: Make prob an interval
                prob = ir(ability)
                for seg_idx in (s1_idx, s2_idx)
                    vals = refine_mean_and_c(state, seg_width, ir, seg_idx)
                    acc += vals
                    push!(segs, Segment(seg_idx, vals))
                end
                c = acc[c_pt_idx] ± acc[c_err_idx]
                unnorm_mean = acc[unnorm_mean_pt_idx] ± acc[unnorm_mean_err_idx]
                mean = unnorm_mean / c
                var = refined_var_estimate(state, segs, ir, mean, c)
                res += prob * var
            end
            update_measure!(state.interval_best, bests_contents_idx, res, acc, segs)
            bests_contents_idx = 1
        end
    end
    next_item = state.interval_best.contents[1].item_bank_idx
    empty!(state.interval_best)
    return next_item
end

function generate_dt_cat_prog_quadgk_point_ability(state::ProgQuadWDecisionTreeGenerationState)
    ## Step 0. Precompute item bank
    @timeit "precompute item bank" begin
        precompute!(state)
    end

    while true
        @timeit "calculate ability point estimate" begin
            ability = calc_ability(state)
        end
        
        @timeit "apply next item rule" begin
            next_item = best_item(state, ability)
        end
        
        @assert length(state.interval_best) > 0 "Got no best item"
        
        if length(state.interval_best) > 1
            @warn "Got more than one best item" length(state.interval_best) state.interval_best
        end
        
        @timeit "select next item and add to dt" begin
            insert!(state.decision_tree_result, responses(state.likelihood), ability, next_item)
            if state.state_tree.cur_depth == state.state_tree.max_depth
                @timeit "final state ability calculation" begin
                    for resp in (false, true)
                        resize!(state.likelihood, state.state_tree.cur_depth)
                        push_question_response!(state.likelihood, state.item_bank, next_item, resp)
                        ability = Slow.slow_mean_and_c(state.likelihood, -10.0f0, 10.0f0)[1]
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