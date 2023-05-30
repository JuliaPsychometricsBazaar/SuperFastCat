using Makie
using GLMakie
using Random
using SuperFastCat
using SuperFastCat: ProgQuadGKDecisionTreeGenerationState, responses_idx, bintree_depth, questions, theta_lo, theta_hi, calc_ability, responses, next!, precompute!
using SuperFastCat.Slow: slow_likelihood

function main()
    rng = Xoshiro(42)
    params = clumpy_4pl_item_bank(rng, 3, 100000)

    xs = range(-10.0, 10.0, 100)

    max_depth = 20
    state = SlowDecisionTreeGenerationState(params, max_depth)
    fixedw = FixedWDecisionTreeGenerationState(params, max_depth; weighted_quadpts=5)
    iterqwk = ProgQuadGKDecisionTreeGenerationState(params, max_depth; quad_order=5, quad_max_depth=4)
    states = [state, fixedw, iterqwk]
    
    for s in (fixedw, iterqwk)
        precompute!(s)
    end

    qrs = random_responses(rng, length(state.item_bank), 2)
    for s in states
        s.state_tree.cur_depth = length(qrs)
        for (q, r) in qrs
            push_question_response!(s.likelihood, s.item_bank, q, r)
        end
    end

    for (name, s) in ((:fixedw, fixedw), (:iterqwk, iterqwk))
        @show name
        @show s.likelihood.(convert(AbstractArray{Float32}, xs))
        iteration_precompute!(s)
    end

    f = Figure()
    sc = display(f);
    ax = Axis(f[1, 1])

    lines!(xs, slow_likelihood.(Ref(params), Ref(state.likelihood), xs))
    @show slow_likelihood.(Ref(params), Ref(state.likelihood), xs)
    for s in states
        ability = calc_ability(s)
        @show ability
        vlines(ability)
        lines!(xs, s.likelihood.(convert(AbstractArray{Float32}, xs)))
        @show s.likelihood.(convert(AbstractArray{Float32}, xs))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end