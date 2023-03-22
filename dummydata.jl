pclamp(x) = clamp.(x, 0.0, 1.0)
abs_rand(rng, dist, dims...) = abs.(rand(rng, dist, dims...))
clamp_rand(rng, dist, dims...) = pclamp.(rand(rng, dist, dims...))

function clumpy_4pl_item_bank(rng, num_clumps, num_questions)
    clump_dist_mat = hcat(
        Normal.(rand(rng, Normal(), num_clumps), 0.1),  # Difficulty
        Normal.(abs_rand(rng, Normal(1.0, 0.2), num_clumps), 0.1),  # Discrimination
        Normal.(clamp_rand(rng, Normal(0.0, 0.2), num_clumps), 0.02),  # Guess
        Normal.(clamp_rand(rng, Normal(0.0, 0.2), num_clumps))  # Slip
    )
    params_clumps = mapslices(Product, clump_dist_mat; dims=[2])[:, 1]
    # TODO: Resample the clumps to create a correlated distribution
    params = Array{Float32, 2}(undef, num_questions, 4)
    for (question_idx, clump) in enumerate(sample(rng, params_clumps, num_questions; replace=true))
        (difficulty, discrimination, guess, slip) = rand(rng, clump)
        params[question_idx, :] = [difficulty, abs(discrimination), pclamp(guess), pclamp(slip)]
    end
    params
end
