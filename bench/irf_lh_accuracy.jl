"""
This script writes out a record file that can be used to plot the accuracy of
IRFs from the affines approach versus the slow approach.
"""

using SuperFastCat
using SuperFastCat: precompute!, idxr_difficulty, idxr_discrimination, idxr_guess, idxr_slip
using SuperFastCat: idxr_x_c, idxr_x_m, idxr_lh_y_c, idxr_lh_y_m
using SuperFastCat: idxr_ir_neg_y_c, idxr_ir_neg_y_m, idxr_ir_pos_y_c, idxr_ir_pos_y_m
using SuperFastCat.Slow
using Random: Xoshiro
using LogExpFunctions: logistic

include("./utils.jl")

function main(outfn)
    rng = Xoshiro(42)
    params = clumpy_4pl_item_bank(rng, 3, 100)
    item_bank = ItemBank(params)
    precompute!(item_bank)
    lh = ResponsesLikelihood(101)
    resize!(lh, 0)
    open_rec_writer(outfn) do rw
        for q in 1:100
            write_rec(
                rw;
                type="item_params",
                difficulty=item_bank.params[q, idxr_difficulty],
                discrimination=item_bank.params[q, idxr_discrimination],
                guess=item_bank.params[q, idxr_guess],
                slip=item_bank.params[q, idxr_slip],
                x_c=item_bank.affines[q, idxr_x_c],
                x_m=item_bank.affines[q, idxr_x_m],
                y_neg_c=item_bank.affines[q, idxr_ir_neg_y_c],
                y_neg_m=item_bank.affines[q, idxr_ir_neg_y_m],
                y_pos_c=item_bank.affines[q, idxr_ir_pos_y_c],
                y_pos_m=item_bank.affines[q, idxr_ir_pos_y_m],
            )
            for r in (false, true)
                resize!(lh, 0)
                push_question_response!(lh, item_bank, q, r)
                xs = range(-5.0, 5.0, 100)
                fast_ys = lh.(convert(Vector{Float32}, xs))
                slow_ys = slow_likelihood.(Ref(params), Ref(lh), xs)
                for (x, fast_y, slow_y) in zip(xs, fast_ys, slow_ys)
                    write_rec(
                        rw;
                        type="irf_point",
                        q=q,
                        r=r,
                        x=x,
                        fast_lh=fast_y,
                        slow_lh=slow_y,
                    )
                end
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS[1])
end