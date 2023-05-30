"""
This file is partially based on QuadGK.jl, The idea is to iteratively refine
estimates of the postposterior variance of the likelihood. In order to do so,
estimates are calculated as intervals, based on errors from the difference
between the Gauss and Kronrod quadrature rules. Note that these errors are not
strict bounds, so this is not a rigorous interval method. Perhaps with
calibration of the errors we could get a probability of the true value being in
the interval.

The idea is to use the interval estimates to prune the search space. After the
first pass, adaptive quadrature is used to filter down the list of items with a
best value better than the worst value of the current best item.

Because of the lack of vectorisation and use of intervals, this is currently a
lot slower than the weighted quadrature method. Maybe they can be combined, and
perhaps the use of per-item intervals could be replaced with a method which
creates a worst-case error bound for the whole item bank.
"""

using QuadGK: kronrod
using LinearAlgebra: norm
using DataStructures: heappop!
import Base.Order.Reverse

Base.@kwdef struct ProgQuadGKDecisionTreeGenerationState
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
    prog_quadgk::ProgQuadGK
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

function ProgQuadGKDecisionTreeGenerationState(item_bank::ItemBankT, max_depth; quad_order=5, quad_max_depth=4)
    num_items = size(item_bank, 1)
    num_segpts = quad_order * 2 + 1
    interval_best = IntervalBest(FastForwardOrdering(), ceil(Int, sqrt(num_items) + 3))
    ProgQuadGKDecisionTreeGenerationState(
        item_bank=ItemBank(item_bank),
        likelihood=ResponsesLikelihood(max_depth + 1), # +1 for final ability estimates
        state_tree=TreePosition(max_depth),
        decision_tree_result=DecisionTree(max_depth),
        prog_quadgk=ProgQuadGK(quad_order, quad_max_depth, theta_lo, theta_hi),
        lh_vals=Vector{Float32}(undef, num_segpts),
        lh_fx=Vector{Float32}(undef, num_segpts),
        lh_fx_interval=Vector{Interval{Float32}}(undef, num_segpts),
        item_response_quadrature_buf=Vector{Float32}(undef, num_items),
        interval_best=interval_best
    )
end

function calc_ability(state::ProgQuadGKDecisionTreeGenerationState)::Float32
    if state.state_tree.cur_depth == 0
        return 0.0f0
    else
        # TODO: use IterativeQuadGK instead of QuadGK here
        return Slow.slow_mean_and_c(state.likelihood, theta_lo, theta_hi)[1]
    end
end

function update_lh_values!(state::ProgQuadGKDecisionTreeGenerationState, ir::ItemResponse, seg_idx)
    state.lh_vals .= state.likelihood.(@view state.prog_quadgk.segpoints[:, seg_idx]) .* ir.(@view state.prog_quadgk.segpoints[:, seg_idx])
end

function update_lh_x_values!(state::ProgQuadGKDecisionTreeGenerationState, seg_idx)
    state.lh_fx .= state.lh_vals .* (@view state.prog_quadgk.segpoints[:, seg_idx])
end

function update_lh_sqdist_vals!(state::ProgQuadGKDecisionTreeGenerationState, seg_idx, mean)
    state.lh_fx_interval .= state.lh_vals .* ((@view state.prog_quadgk.segpoints[:, seg_idx]) .- mean) .^ 2
end

function initial_var_estimate(state::ProgQuadGKDecisionTreeGenerationState, seg_width, ir)
    seg_idx = 1
    update_lh_values!(state, ir, seg_idx)
    c_pt, c_err = evalrule(state.lh_vals, state.prog_quadgk, seg_width)
    c = c_pt ± c_err
    update_lh_x_values!(state, seg_idx)
    unnorm_mean_pt, unnorm_mean_err = evalrule(state.lh_fx, state.prog_quadgk, seg_width)
    unnorm_mean = unnorm_mean_pt ± unnorm_mean_err
    mean = unnorm_mean / c
    update_lh_sqdist_vals!(state, seg_idx, mean)
    unnorm_var_pt, unnorm_var_err = evalrule(state.lh_fx_interval, state.prog_quadgk, seg_width)
    unnorm_var = unnorm_var_pt ± sup(unnorm_var_err)
    unnorm_var / c
end

function seg_unnorm_var(state::ProgQuadGKDecisionTreeGenerationState, seg_width, ir, mean, seg_idx)
    update_lh_values!(state, ir, seg_idx)
    update_lh_sqdist_vals!(state, seg_idx, mean)
    evalrule(state.lh_fx_interval, state.prog_quadgk, seg_width)
end

function refine_mean_and_c(state::ProgQuadGKDecisionTreeGenerationState, seg_width, ir, seg_idx)
    update_lh_values!(state, ir, seg_idx)
    seg_c_pt, seg_c_err = evalrule(state.lh_vals, state.prog_quadgk, seg_width)
    update_lh_x_values!(state, seg_idx)
    seg_unnorm_mean_pt, seg_unnorm_mean_err = evalrule(state.lh_fx, state.prog_quadgk, seg_width)
    SVector(seg_c_pt, seg_c_err, seg_unnorm_mean_pt, seg_unnorm_mean_err)
end

function refined_var_estimate(state::ProgQuadGKDecisionTreeGenerationState, segs, ir, mean, c)
    # Since we have a new mean at this point, we have to go back through all
    # the old segments and recompute the variance for these too
    unnorm_var_pt = 0f0
    unnorm_var_err = 0f0
    for seg in segs
        seg_idx = seg.seg_idx
        depth = bintree_depth(seg_idx)
        seg_width = seg_width_at_depth(theta_width, depth)
        seg_unnorm_var_pt, seg_unnorm_var_err = seg_unnorm_var(state, seg_width, ir, mean, seg_idx)
        unnorm_var_pt += seg_unnorm_var_pt
        unnorm_var_err += seg_unnorm_var_err
    end
    unnorm_var = unnorm_var_pt ± sup(unnorm_var_err)
    unnorm_var / c
end

function precompute!(state::ProgQuadGKDecisionTreeGenerationState)
    precompute!(state.item_bank)
    #precompute!(state.prog_quadgk)
    # Should maybe add this back
end

iteration_precompute!(state::ProgQuadGKDecisionTreeGenerationState) = nothing

@inline function refine_all_mean_and_c(state, irs, seg_width, s1_idx, s2_idx)
    # binary divided segs  * resp true/false
    StaticArrays.sacollect(
        SMatrix{2, 2},
        SVector(refine_mean_and_c(state, seg_width, irs[resp + 1], seg_idx))
        for seg_idx in (s1_idx, s2_idx),
            resp in false:true
    )
end

function best_item(state::ProgQuadGKDecisionTreeGenerationState, ability)
    @timeit "first pass" begin
        for item_idx in 1:length(state.item_bank)
            if item_idx in questions(state.likelihood)
                continue
            end
            res = 0f0 ± 0f0
            for resp in false:true
                ir = ItemResponse(state.item_bank.affines, item_idx, resp)
                # TODO: Make prob an interval
                # TODO: Add turbos
                prob = ir(ability)
                res += prob * initial_var_estimate(state, theta_width / 2, ir)
            end
            res
            add_to_interval_best!(state.interval_best, item_idx, res)
        end
    end

    #@show "after first pass"
    #print(summarize_items(state.interval_best))

    @timeit "adaptive passes" begin
        cur_expansions = 1
        # Rule is to
        #  * Process each item round-robin (fairness)
        #  * Prioritise the current earliest item (best-first heuristic -- this can knock out later items from consideration more quickly)
        while true
            if length(state.interval_best) == 1
                # Success!
                break
            end
            bests_contents_idx = 1
            while bests_contents_idx <= length(state.interval_best)
                if state.interval_best.contents[bests_contents_idx].num_expansions == cur_expansions
                    break
                end
                bests_contents_idx += 1
            end
            if bests_contents_idx > length(state.interval_best)
                # Next depth
                cur_expansions += 1
                bests_contents_idx = 1
            end
            cur_item = state.interval_best.contents[bests_contents_idx]
            #@assert cur_depth == cur_item.depth "cur_depth $cur_depth != cur_item.depth $(cur_item.depth)"
            item_idx = cur_item.item_bank_idx
            acc = cur_item.acc
            segs = cur_item.segs
            if segs === nothing
                segs = []
                (s1_idx, s2_idx) = seg_child_idxs(1)
                cur_depth = 1
            else
                # pick the minimum seg & split it
                seg = heappop!(segs, Reverse)
                (s1_idx, s2_idx) = seg_child_idxs(seg.seg_idx)
                acc = acc - seg.vals
                cur_depth = bintree_depth(seg.seg_idx)
            end
            next_depth = cur_depth + 1
            if next_depth > state.prog_quadgk.max_depth
                # Unsuccessful, but giving up early
                break
            end
            next_seg_width = seg_width_at_depth(theta_width, cur_depth + 1)
            res = 0f0 ± 0f0
            irs = (
                ItemResponse(state.item_bank.affines, item_idx, false),
                ItemResponse(state.item_bank.affines, item_idx, true)
            )
            mean_and_cs = refine_all_mean_and_c(state, irs, next_seg_width, s1_idx, s2_idx)
            for (seg_idx, sii) in ((s1_idx, 1), (s2_idx, 2))
                seg_mean_and_cs = reinterpret(reshape, Float32, mean_and_cs[sii, :])
                push!(segs, Segment(seg_idx, seg_mean_and_cs))
                acc += seg_mean_and_cs
            end
            #=
            @show acc
            @show sum([seg.vals for seg in segs])
            for seg in segs
                @show seg
                seg_idx = seg.seg_idx
                depth = bintree_depth(seg_idx)
                pos = seg_idx - (2 ^ (depth - 1))
                seg_width = seg_width_at_depth(theta_width, depth) * 2f0
                @show depth pos seg_width
                sir = Slow.SlowItemResponse(state.item_bank.params, item_idx, false)
                seg_slow_var_mean_c = Slow.slow_var_mean_and_c(x -> Slow.slow_likelihood(state.item_bank.params, state.likelihood, x) * sir(x), theta_lo + seg_width * pos, theta_lo + seg_width * (pos + 1))
                @show seg_slow_var_mean_c
            end
            =#
            for resp in false:true
                respi = resp + 1
                prob = irs[respi](ability)
                c = acc[c_pt_idx, respi] ± acc[c_err_idx, respi]
                unnorm_mean = acc[unnorm_mean_pt_idx, respi] ± acc[unnorm_mean_err_idx, respi]
                mean = unnorm_mean / c
                var = refined_var_estimate(state, segs, irs[respi], mean, c)
                #=
                @show "res" resp prob c unnorm_mean mean var
                sir = Slow.SlowItemResponse(state.item_bank.params, item_idx, resp); slow_var_mean_c = Slow.slow_var_mean_and_c(x -> Slow.slow_likelihood(state.item_bank.params, state.likelihood, x) * sir(x), theta_lo, theta_hi)
                @show slow_var_mean_c
                if slow_var_mean_c[3] ∉ c
                    error("c out of range")
                end
                if slow_var_mean_c[1] ∉ var
                    error("var out of range")
                end
                =#
                res += prob * var
            end
            update_measure!(state.interval_best, bests_contents_idx, res, acc, segs)
            #print(summarize_items(state.interval_best))
        end
    end
    next_item = state.interval_best.contents[1].item_bank_idx
    empty!(state.interval_best)
    return next_item
end

function generate_dt_cat_prog_quadgk_point_ability(state::ProgQuadGKDecisionTreeGenerationState)
    ## Step 0. Precompute item bank
    @timeit "precompute item bank" begin
        precompute!(state)
    end

    while true
        @timeit "calculate ability point estimate" begin
            ability = calc_ability(state)
        end
        
        #scores = Array{Interval{Float32}}(undef, length(state.item_bank))
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
                        ability = Slow.slow_mean_and_c(state.likelihood, theta_lo, theta_hi)[1]
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