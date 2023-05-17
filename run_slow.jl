"""
This is the main entry point for the slow version of the algorithm. The slow
version simply uses QuadGK for all integrals.
"""

using Random: Xoshiro
using JLD2
using SuperFastCat
using TimerOutputs

function main()
    reset_timer!()
    zero_subnormals_all()
    @timeit "Whole script" begin
        @timeit "Generate item bank" begin
            rng = Xoshiro(42)
            params = clumpy_4pl_item_bank(rng, 3, 1000)
            max_depth = 6
        end
        @timeit "Allocate" begin
            state = SlowDecisionTreeGenerationState(params, max_depth)
        end
        @timeit "generate_dt_cat_exhaustive_point_ability" begin
            dt_cat = generate_dt_cat_exhaustive_point_ability(state)
        end
    end
    print_timer()
    jldsave("dt.jld2"; questions=dt_cat.questions, abilities=dt_cat.ability_estimates)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end