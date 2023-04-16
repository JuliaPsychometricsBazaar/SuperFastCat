"""
The purpose script is useful for looking at allocations.

You can use:
    $ pprof -http localhost:8080 allocs.jl alloc-profile.pb.gz
    $ julia --project=. allocs.jl
"""
using Profile
using Random: Xoshiro
using SuperFastCat
using PProf


function main()
    zero_subnormals_all()
    rng = Xoshiro(42)
    params = clumpy_4pl_item_bank(rng, 3, 100)
    max_depth = 3
    state = DecisionTreeGenerationState(params, max_depth)
    generate_dt_cat_exhaustive_point_ability(state)
    state = DecisionTreeGenerationState(params, max_depth)
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate=1 generate_dt_cat_exhaustive_point_ability(state)
    PProf.Allocs.pprof(from_c = false)
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
