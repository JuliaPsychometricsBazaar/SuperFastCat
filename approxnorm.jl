"""
This script looks at the magnitude of the normalising constant used for
calcuating the mean/variance of the ability likelihood functions in for a CAT.
It compares them with a few heurisitcs, which could be used to scale the
factors of the likelihood to avoid underflow. It also looks at when subnormals
begin to appear at sampled quadrature points.

TODO:
 * Use realistic fixed quadrature points rather (or probably in addition to) than adaptive ones
   * Are some of the problems mitigated by precomputed quadrature
 * Consider also more realisitic response patterns
 * Make some charts
"""
using QuadGK
using Base.Iterators: take
using Random: Xoshiro
using SuperFastCat
using SuperFastCat: precompute, idxr_slip, idxr_guess, idxr_difficulty, idxr_discrimination, push_question_response!, showqa
using StatsBase


const lo = -10.0f0
const hi = 10.0f0

function rough_norm_single(params, item_idx, resp, a, b)
    slip = params[item_idx, idxr_slip]
    guess = params[item_idx, idxr_guess]
    diff_height = 1.0f0 - slip - guess
    diff = params[item_idx, idxr_difficulty]
    diff_weight = resp ? b - diff : diff - a
    guess + (diff_weight * diff_height) / (b - a) 
end

function rough_norm(params, item_idx, resp, a, b)
     rough_norm_single(params, item_idx, resp, a, b) * (b - a)
end

function rough_norm(params, rl::ResponsesLikelihood, a, b)
    prod(rough_norm_single(params, item_idx, resp, a, b) for (item_idx, resp) in first(zip(rl.questions, rl.responses), length(rl))) * (b - a)
end

function compare_norms(params, lh)
    mem = []
    function mem_lh(x)
        l = lh(x)
        push!(mem, l)
        l
    end
    int, err = quadgk(mem_lh, lo, hi)
    int /= (hi - lo)
    err /= (hi - lo)
    norm = rough_norm(params, lh, lo, hi)
    maxent = 0.5f0 ^ length(lh)
    showqa(stdout, lh)
    println("int: $int, err: $err, norm: $norm (ratio: $(norm / int)) maxent: $maxent (ratio: $(maxent / int))")
    println("subnormals: $(count(issubnormal, mem))/$(length(mem))")
end

function amain()
    rng = Xoshiro(42)
    params = clumpy_4pl_item_bank(rng, 3, 100000)
    ib = ItemBank(params)
    precompute(ib)
    lh  = ResponsesLikelihood(200)
    println("## Single question")
    for question in [74063, 15595, 95108, 95355, 85876, 79905, 90598, 78013]
        for resp in [true, false]
            slip = params[question, idxr_slip]
            guess = params[question, idxr_guess]
            diff = params[question, idxr_difficulty]
            disc = params[question, idxr_discrimination]
            println("slip: $slip, guess: $guess, diff: $diff, disc: $disc")
            resize!(lh, 0)
            push_question_response!(lh, ib, question, resp)
            int, err = quadgk(lh, lo, hi)
            norm = rough_norm(params, question, resp, lo, hi)
            println("int: $int, err: $err, norm: $norm, diff: $(int - norm)")
        end
    end
    println()
    println()
    println("## Multi question")
    for resps in 0:15
        resize!(lh, 0)
        for (question, resp) in zip([74063, 15595, 95108, 95355], digits(resps, base=2, pad=4))
            push_question_response!(lh, ib, question, resp > 0)
        end
        compare_norms(params, lh)
    end
    println()
    println()
    println("## Multi question pt. 2")
    for _ in 1:20
        println()
        resize!(lh, 0)
        question_idxs = sample(rng, 1:size(params, 1), 100; replace=false)
        resps = rand(rng, Bool, 100)
        qrs = zip(question_idxs, resps)
        for _ in 1:20
            for (q, r) in take(qrs, 5)
                push_question_response!(lh, ib, q, r)
            end
            compare_norms(params, lh)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    amain()
end
