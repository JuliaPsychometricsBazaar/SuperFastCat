"""
This gathers data by running the ability estimation and next item rules for a
single random response pattern with increasing numbers of responses. It
compares methods:
 * QuadGK
 * fixed gauss
 * weighted gauss

The fixed/weighted gauss are tried with different numbers of points and quadgk
target err values.

Both values and timings are also taken, but not much optimisation is done, so a
different setup may be needed for accurate timings.
"""

using QuadGK
using SuperFastCat
using SuperFastCat: precompute!, WeightedGauss, questions, responses
using SuperFastCat: idxr_discrimination, idxr_difficulty, idxr_guess, idxr_slip, logistic_normal_scaler
using SuperFastCat: mean_and_c, expected_var
using SuperFastCat.Slow
using Random: Xoshiro
using LogExpFunctions: logistic
using IterTools

const pts_choices = (3, 5, 7, 9, 11)
const err_choices = (1f-3, 1f-5, 1f-7, 1f-9)

include("./utils.jl")

function main(outfn, num_responses)
    rng = Xoshiro(42)
    params = clumpy_4pl_item_bank(rng, 3, 100000)
    item_bank = ItemBank(params)
    precompute!(item_bank)
    lh = ResponsesLikelihood(num_responses + 1)
    weighted_gausses = Dict()
    normal_gausses = Dict()
    ir_fx = Dict()
    for pts in pts_choices
        weighted_gausses[pts] = WeightedGauss{Float32}(pts)
        normal_gausses[pts] = gauss(pts, -10.0f0, 10.0f0)
        ir_fx[pts] = Vector{Float32}(undef, pts)
    end
    resize!(lh, 0)
    qrs = random_responses(rng, length(item_bank), num_responses)
    open_rec_writer(outfn) do rw
        for ((q, r), (next_q, _)) in partition(qrs, 2, 1)
            push_question_response!(lh, item_bank, q, r)
            rec_ctx(rw, lh_len=length(lh), questions=questions(lh), responses=responses(lh)) do
                #(var, mean, c) = slow_var_mean_and_c(lh, -10.0, 10.0)
                ability_acc, c_acc, num_err, denom_err = slow_mean_and_c(x -> slow_likelihood(params, lh, x), -10.0, 10.0)
                write_rec(
                    rw;
                    type="quadgk",
                    quantity="ability",
                    val=ability_acc,
                    c=c_acc,
                    num_err=num_err,
                    denom_err=denom_err,
                )
                slow_expected_variance, timed_stats = proc_timed(
                    @timed slow_expected_var(params, lh, ability_acc, next_q, -10.0, 10.0)
                )
                write_rec(
                    rw;
                    type="quadgk",
                    quantity="expected_variance",
                    val=slow_expected_variance,
                    timed_stats=timed_stats
                )
                for pts in pts_choices
                    fixed_quad_xs, fixed_quad_ws = normal_gausses[pts]
                    (ability_fixed, ability_fixed_c), timed_stats = proc_timed(@timed mean_and_c(fixed_quad_xs, fixed_quad_ws, slow_likelihood.(Ref(params), Ref(lh), fixed_quad_xs)))
                    write_rec(
                        rw;
                        type="fixed_gauss",
                        quantity="ability",
                        pts=pts,
                        val=ability_fixed,
                        c=ability_fixed_c,
                        timed_stats=timed_stats,
                    )
                    orig_lh_quad_slow_xs, orig_lh_quad_slow_ws = gauss(x -> slow_likelihood(params, lh, x), pts, -10.0f0, 10.0f0)
                    (ability_orig_wgauss_slow, ability_orig_wgauss_slow_c), timed_stats = proc_timed(@timed mean_and_c(orig_lh_quad_slow_xs, orig_lh_quad_slow_ws))
                    write_rec(
                        rw;
                        type="orig_wgauss_slow_lh",
                        quantity="ability",
                        pts=pts,
                        val=ability_orig_wgauss_slow,
                        c=ability_orig_wgauss_slow_c,
                        timed_stats=timed_stats,
                    )
                    orig_lh_quad_xs, orig_lh_quad_ws = gauss(lh, pts, -10.0f0, 10.0f0)
                    (ability_orig_wgauss, ability_orig_wgauss_c), timed_stats = proc_timed(@timed mean_and_c(orig_lh_quad_xs, orig_lh_quad_ws))
                    write_rec(
                        rw;
                        type="orig_wgauss",
                        quantity="ability",
                        pts=pts,
                        val=ability_orig_wgauss,
                        c=ability_orig_wgauss_c,
                        timed_stats=timed_stats,
                    )
                    # TODO: Add fixed_gauss expected_var
                    for quadgk_err in err_choices
                        (lh_quad_xs, lh_quad_ws), timed_stats = proc_timed(@timed weighted_gausses[pts](lh, -10.0f0, 10.0f0, quadgk_err))
                        write_rec(
                            rw;
                            type="weighted_gauss",
                            quantity="weights",
                            pts=pts,
                            quadgk_err=quadgk_err,
                            timed_stats=timed_stats,
                        )
                        (ability_wgauss, ability_wgauss_c), timed_stats = proc_timed(@timed mean_and_c(lh_quad_xs, lh_quad_ws))
                        write_rec(
                            rw;
                            type="weighted_gauss",
                            quantity="ability",
                            pts=pts,
                            val=ability_wgauss,
                            c=ability_wgauss_c,
                            quadgk_err=quadgk_err,
                            timed_stats=timed_stats,
                        )
                        expected_variance_2wgauss, timed_stats = proc_timed(@timed expected_var(item_bank, ir_fx[pts], lh_quad_xs, lh_quad_ws, ability_wgauss, next_q))
                        write_rec(
                            rw;
                            type="weighted_gauss",
                            quantity="expected_variance",
                            pts=pts,
                            val=expected_variance_2wgauss,
                            quadgk_err=quadgk_err,
                            timed_stats=timed_stats,
                        )
                        expected_variance_wgauss_qgk, timed_stats = @timed expected_var(item_bank, ir_fx[pts], lh_quad_xs, lh_quad_ws, ability_acc, next_q)
                        write_rec(
                            rw;
                            type="weighted_gauss+quadgk_ability",
                            quantity="expected_variance",
                            pts=pts,
                            val=expected_variance_wgauss_qgk,
                            quadgk_err=quadgk_err,
                            timed_stats=timed_stats,
                        )
                    end
                end
            end
        end
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS[1], parse(Int, ARGS[2]))
end
