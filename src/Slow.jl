module Slow

using LogExpFunctions: logistic as liblogistic
using QuadGK

import ..ResponsesLikelihood, ..logistic_normal_scaler
import ..idxr_discrimination, ..idxr_difficulty, ..idxr_guess , ..idxr_slip 

export SlowItemResponse, slow_likelihood, slow_mean_and_c, slow_var_mean_and_c, slow_expected_var


logistic(x::Real) = liblogistic(x)
logistic(x) = inv(exp(-x) + one(x))

struct SlowItemResponse
    params::Matrix{Float64}
    question::Int
    response::Bool
end

function (ir::SlowItemResponse)(x)
    a = ir.params[ir.question, idxr_discrimination]
    b = ir.params[ir.question, idxr_difficulty]
    c = ir.params[ir.question, idxr_guess]
    d = ir.params[ir.question, idxr_slip]
    range = 1.0 - c - d
    if ir.response
        muladd(range, logistic(a * logistic_normal_scaler * (x - b)), c)
    else
        muladd(-range, logistic(a * logistic_normal_scaler * (x - b)), 1.0 - c)
    end
end

function slow_likelihood(params, rl::ResponsesLikelihood, x)
    prod(SlowItemResponse(params, question, response)(x) for (question, response) in first(zip(rl.questions, rl.responses), length(rl)); init=1.0)
end

function slow_mean_and_c(lh, lo, hi; atol=nothing, rtol=nothing)
    mean, num_err = quadgk(x -> lh(x) * x, lo, hi; atol=atol, rtol=rtol)
    c, denom_err = quadgk(lh, lo, hi; atol=atol, rtol=rtol)
    (mean / c, c, num_err, denom_err)
end

function slow_var_mean_and_c(lh, lo, hi; atol=nothing, rtol=nothing)
    (mean, c, num_err, denom_err) = slow_mean_and_c(lh, lo, hi; atol=atol, rtol=rtol)
    var, var_err = quadgk(x -> lh(x) * (x - mean) ^ 2, lo, hi; atol=atol, rtol=rtol)
    (var / c, mean, c, num_err, denom_err, var_err)
end

function slow_expected_var(params, rl, ability, item_idx, lo, hi; atol=nothing, rtol=nothing)
    res = 0.0
    for resp in (false, true)
        ir = SlowItemResponse(params, item_idx, resp)
        prob = ir(ability)
        (var, _, _, _, _, _) = slow_var_mean_and_c(x -> slow_likelihood(params, rl, x) * ir(x), lo, hi; atol=atol, rtol=rtol)
        res += prob * var
    end
    res 
end

end
