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

struct ProgQuadGK
    order::Int
    max_depth::Int
    a::Float32
    b::Float32
    x::Vector{Float32}
    w::Vector{Float32}
    gw::Vector{Float32}
    segpoints::Matrix{Float32}
end

function ProgQuadGK(order, max_depth, a, b)
    ProgQuadGK(
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

function abscissa_to_segpoints!(out, a, b, x)
    s = (b - a) / 2
    for i in 1:(length(x) - 1)
        out[2i - 1] = a + (1 + x[i]) * s
        out[2i] = a + (1 - x[i]) * s
    end
    out[2 * length(x) - 1] = a + s
end

function precompute!(iqgk::ProgQuadGK)
    t = kronrod(Float32, iqgk.order)
    @info "kronrod" iqgk.order iqgk.max_depth t
    iqgk.x .= t[1]
    iqgk.w .= t[2]
    iqgk.gw .= t[3]
    for depth in 1:iqgk.max_depth
        denom = 2 ^ (depth - 1)
        for num in 0:(denom - 1)
            a = iqgk.a + (iqgk.b - iqgk.a) * (num / denom)
            b = iqgk.a + (iqgk.b - iqgk.a) * ((num + 1) / denom)
            abscissa_to_segpoints!((@view iqgk.segpoints[:, denom + num]), a, b, iqgk.x)
        end
    end
    @info "precompute done"
end


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
    c_pt, c_err = evalrule(state.lh_vals, seg_width, state.prog_quadgk.w, state.prog_quadgk.gw)
    c = c_pt ± c_err
    update_lh_x_values!(state, seg_idx)
    unnorm_mean_pt, unnorm_mean_err = evalrule(state.lh_fx, seg_width, state.prog_quadgk.w, state.prog_quadgk.gw)
    unnorm_mean = unnorm_mean_pt ± unnorm_mean_err
    mean = unnorm_mean / c
    update_lh_sqdist_vals!(state, seg_idx, mean)
    unnorm_var_pt, unnorm_var_err = evalrule(state.lh_fx_interval, seg_width, state.prog_quadgk.w, state.prog_quadgk.gw)
    unnorm_var = unnorm_var_pt ± sup(unnorm_var_err)
    unnorm_var / c
end

function seg_unnorm_var(state::ProgQuadGKDecisionTreeGenerationState, seg_width, ir, mean, seg_idx)
    update_lh_values!(state, ir, seg_idx)
    update_lh_sqdist_vals!(state, seg_idx, mean)
    evalrule(state.lh_fx_interval, seg_width, state.prog_quadgk.w, state.prog_quadgk.gw)
end

depth_at_seg_idx(seg_idx) = 8sizeof(typeof(seg_idx)) * leading_zeros(seg_idx)
seg_width_at_depth(total_width, depth) = total_width * 2f0 ^ (-depth)

function refine_mean_and_c(state::ProgQuadGKDecisionTreeGenerationState, seg_width, ir, seg_idx)
    update_lh_values!(state, ir, seg_idx)
    seg_c_pt, seg_c_err = evalrule(state.lh_vals, seg_width, state.prog_quadgk.w, state.prog_quadgk.gw)
    update_lh_x_values!(state, seg_idx)
    seg_unnorm_mean_pt, seg_unnorm_mean_err = evalrule(state.lh_fx, seg_width, state.prog_quadgk.w, state.prog_quadgk.gw)
    SVector(seg_c_pt, seg_c_err, seg_unnorm_mean_pt, seg_unnorm_mean_err)
end

function refined_var_estimate(state::ProgQuadGKDecisionTreeGenerationState, segs, ir, mean, c)
    # Since we have a new mean at this point, we have to go back through all
    # the old segments and recompute the variance for these too
    unnorm_var_pt = 0f0
    unnorm_var_err = 0f0
    for seg in segs
        depth = depth_at_seg_idx(seg.seg_idx)
        seg_width = seg_width_at_depth(theta_width, depth)
        seg_idx = seg.seg_idx
        seg_unnorm_var_pt, seg_unnorm_var_err = seg_unnorm_var(state, seg_width, ir, mean, seg_idx)
        unnorm_var_pt += seg_unnorm_var_pt
        unnorm_var_err += seg_unnorm_var_err
    end
    unnorm_var = unnorm_var_pt ± sup(unnorm_var_err)
    unnorm_var / c
end

function generate_dt_cat_prog_quadgk_point_ability(state::ProgQuadGKDecisionTreeGenerationState)
    ## Step 0. Precompute item bank
    @timeit "precompute item bank" begin
        precompute!(state.item_bank)
        precompute!(state.prog_quadgk)
    end

    while true
        @timeit "calculate ability point estimate" begin
            ability = calc_ability(state)
        end
        
        #scores = Array{Interval{Float32}}(undef, length(state.item_bank))
        @timeit "apply next item rule" begin
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
                    depths() = repr([i.depth for i in state.interval_best.contents])
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
                    seg_width = seg_width_at_depth(theta_width, cur_depth)
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
        end
        
        @assert length(state.interval_best) > 0 "Got no best item"
        
        if length(state.interval_best) > 1
            @warn "Got more than one best item" length(state.interval_best) state.interval_best
        end
        
        @timeit "select next item and add to dt" begin
            next_item = state.interval_best.contents[1].item_bank_idx
            empty!(state.interval_best)
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