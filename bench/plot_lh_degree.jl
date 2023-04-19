using DuckDB
using Makie
using GLMakie
using DataFrames
using DBInterface
using Parquet
using JSON3


function main(inf)
    lh_lens = Set()
    lh_idxs = Set()
    lh_covs = Dict()
    open(inf, "r") do io
        for obj in JSON3.read(io; jsonlines=true)
            lh_idx = obj[:lh_idx]
            push!(lh_idxs, lh_idx)
            lh_len = obj[:lh_len]
            push!(lh_lens, lh_len)
            lh_covs[(lh_idx, lh_len)] = obj
        end
    end

    f = Figure()
    sg = SliderGrid(
        f[1, 3],
        (label = "LH", range = 1:length(lh_idxs)),
        (label = "Qs", range = 1:length(lh_lens)),
        width = 300,
        tellheight = false
    )

    lh_idx = sg.sliders[1].value
    lh_len = sg.sliders[2].value

    data = @lift lh_covs[($lh_idx, $lh_len)]

    lh_ax = Axis(f[1, 1])
    lines!(
        lh_ax,
        (@lift $data[:xs]),
        (@lift $data[:ys])
    )
    cov_ax = Axis(f[1, 2])
    barplot!(
        cov_ax,
        (@lift $data[:covs])
    )
    xlims!(cov_ax, 1, 300)
    on(data) do _
        autolimits!(lh_ax)
    end
    f
end

if abspath(PROGRAM_FILE) == @__FILE__
    display(main(ARGS[1]))
end