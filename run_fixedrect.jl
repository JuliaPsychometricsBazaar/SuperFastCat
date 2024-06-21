using Random: Xoshiro
using JLD2
using SuperFastCat
using TimerOutputs


function main()
    zero_subnormals_all()
    @timeit "Whole script" begin
        @timeit "Generate item bank" begin
            rng = Xoshiro(42)
            params = clumpy_4pl_item_bank(rng, 3, 100000)
            #params = clumpy_4pl_item_bank(rng, 3, 10)
            #max_depth = size(params, 1)
            max_depth = 8
        end
        @timeit "Allocate" begin
            state = FixedRectDecisionTreeGenerationState(params, max_depth)
        end
        @timeit "generate_dt_cat_exhaustive_point_ability" begin
            dt_cat = generate_dt_cat_exhaustive_point_ability(state)
        end
    end
    print_timer()
    @info "dt_cat" dt_cat
    jldsave("dt.jld2"; questions=dt_cat.questions, abilities=dt_cat.ability_estimates)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
