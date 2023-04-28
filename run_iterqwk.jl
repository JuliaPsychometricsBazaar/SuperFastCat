using Random: Xoshiro
using JLD2
using SuperFastCat
using SuperFastCat: ProgQuadGKDecisionTreeGenerationState, generate_dt_cat_prog_quadgk_point_ability
using TimerOutputs


function main()
    reset_timer!()
    zero_subnormals_all()
    @timeit "Whole script" begin
        @timeit "Generate item bank" begin
            rng = Xoshiro(42)
            params = clumpy_4pl_item_bank(rng, 3, 100000)
            max_depth = 3
        end
        @timeit "Allocate" begin
            state = ProgQuadGKDecisionTreeGenerationState(params, max_depth)
        end
        @timeit "generate_dt_cat_prog_quadgk_point_ability" begin
            dt_cat = generate_dt_cat_prog_quadgk_point_ability(state)
        end
    end
    print_timer()
    @info "dt_cat" dt_cat
    jldsave("dt.jld2"; questions=dt_cat.questions, abilities=dt_cat.ability_estimates)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end