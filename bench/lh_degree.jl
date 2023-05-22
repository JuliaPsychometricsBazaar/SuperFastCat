"""
The aim of this function is to see how much of the likelihood function is
captured by low-degree polynomial approximations.

Conclusion: Although wiggly likelihood functions may need more higher number of
coefficients in the Chebychev basis expansion, the main factor seems to be that
the number of coefficients increases as the main area of mass of the likelihood
function gets smaller, which tends to happen as the number of question asked in
the test increases.
"""

using ApproxFun
using Random: Xoshiro
using SuperFastCat
using SuperFastCat.Slow
using Base.Iterators: take
using GLMakie

include("./utils.jl")


function coef_coverage(f, lo, hi)
    approx_f = Fun(f, Segment(lo, hi))
    coefs = coefficients(approx_f)
    abs_coefs = abs.(coefs)
    sum_coefs = sum(abs_coefs)
    cumsum(abs_coefs) ./ sum_coefs
end

function setup()
    rng = Xoshiro(42)
    params = clumpy_4pl_item_bank(rng, 3, 100000)
    ib = ItemBank(params)
    lh = ResponsesLikelihood(200)

    (
        () -> begin
            resize!(lh, 0);
            random_responses(rng, length(ib), 100)
        end,
        (q, r) -> push_question_response!(lh, ib, q, r),
        x -> slow_likelihood(params, lh, x)
    )
end

function main(outfn)
    reset_get_qr!, push_qr!, lh_func = setup()

    open_rec_writer(outfn) do rw
        for lh_idx in 1:100
            qrs = reset_get_qr!()
            for (lh_len, (q, r)) in enumerate(qrs)
                push_qr!(q, r)
                covs = coef_coverage(lh_func, -5.0, 5.0)
                xs = range(-5.0, 5.0, 100)
                write_rec(
                    rw;
                    lh_idx=lh_idx,
                    lh_len=lh_len,
                    covs=covs,
                    xs=xs,
                    ys=lh_func.(xs)
                )
            end
        end
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS[1])
end