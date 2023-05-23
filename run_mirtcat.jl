include("src/FixRCall.jl")

using Random: Xoshiro
using JLD2
using SuperFastCat
using SuperFastCat: ProgQuadGKDecisionTreeGenerationState, generate_dt_cat_prog_quadgk_point_ability, DecisionTree
using TimerOutputs
using CondaPkg
using RCall
using FittedItemBanks


function params_to_mirt(params)
    @info "params_to_mirt" params
    @info "robject" robject(params)
    params[:, 1] = -params[:, 1]
    params[:, 4] .= 1.0 .- params[:, 4]
    rcopy(R"""
    library(mirtCAT)
    mat <- $params
    colnames(mat) <- c("d", "a1", "g", "u")
    print(mat)
    print("\n")
    generate.mirt_object(mat, "4PL")
    """)
end

function main()
    reset_timer!()
    @timeit "Whole script" begin
        @timeit "Generate item bank" begin
            rng = Xoshiro(42)
            params = clumpy_4pl_item_bank(rng, 3, 100)
            mirt_params = params_to_mirt(params)
            @info "mirt_params" mirt_params
        end
        @timeit "Create MIRT design" begin
            mirt_design = rcopy(R"""
                mirtCAT(df=NULL, mo=$mirt_params, design_elements=TRUE, criteria="MEPV", next_item="MEPV")
            """)
        end
        
        cur_depth = 0
        max_depth = 4
        todo = []
        responses = []
        decision_tree_result = DecisionTree(max_depth)

        while true
            #@info "iteration!" cur_depth max_depth length(todo) R"$mirt_design$person$items_answered"  R"$mirt_design$person$thetas_history"
            @timeit "find next item" begin
                R"""
                new_item <- findNextItem($mirt_design)
                """
            end
            ability = R"extract.mirtCAT(mirt_design$person, 'thetas')"
            @timeit "update design" begin
                if cur_depth < max_depth
                    @rput mirt_design
                    mirt_design = R"""
                    old_mirt_design <- c(
                        person = mirt_design$person$copy(),
                        test = mirt_design$test,
                        design = mirt_design$design
                    )
                    updateDesign(mirt_design, new_item = new_item, new_response = 0)
                    """
                    push!(responses, false)
                    @rget new_item
                    insert!(decision_tree_result, responses, ability, new_item)
                    @rget old_mirt_design
                    cur_depth += 1
                    push!(todo, (cur_depth, old_mirt_design))
                elseif !isempty(todo)
                    cur_depth, mirt_design = pop!(todo)
                    mirt_design = R"""
                    updateDesign($mirt_design, new_item = new_item, new_response = 1)
                    """
                    responses = responses[1:cur_depth]
                    responses[cur_depth] = true
                else
                    break
                end
            end
        end
    end
    print_timer()
    decision_tree_result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end