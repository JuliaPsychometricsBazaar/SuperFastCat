using Parquet2
using Makie
using GLMakie
using DataFrames
using Parquet2: Dataset


function load(inf)
    ds = Dataset(inf)
    DataFrame(ds; copycols=false)
end

function main(pointsf, paramsf)
    GLMakie.activate!()
    points = load(pointsf)
    params = load(paramsf)
    points[!, :q] = Int.(points[!, :q])
    f = Figure()
    ax = Axis(f[1, 1])
    function inspector_label(self, i, p)
        par = params[points[i, :q], :]
        repr("text/plain", Dict(pairs(merge(points[i, :], par))))
    end
    pick_scatter = scatter!(
        ax,
        points[!, :x],
        points[!, :fast_lh] - points[!, :slow_lh],
        inspector_label = inspector_label,
        markersize = 1
    )

    picked_df = Observable{Union{Nothing, DataFrame}}(nothing)
    picked_title = Observable{String}("")

    on(events(f).mousebutton, priority = 2) do event
        if event.button == Mouse.left && event.action == Mouse.press
            plt, i = pick(f)
            if plt != pick_scatter
                return Consume(false)
            end
            @info "pick" plt i
            selected = points[(points.q .== points[i, :q]) .& (points.r .== points[i, :r]), :]
            @info "pick2" selected
            picked_df[] = selected
            picked_title[] = "Question #$(points[i, :q])"
            Consume(true)
        end
        Consume(false)
    end

    DataInspector(f)
    
    function picked_df_col(xcol, ycol)
        function inner(picked_df)
            if picked_df === nothing
                return (Point2{Float64})[]
            end
            return Point2{Float64}[Point2f(x, y) for (x, y) in zip(picked_df[!, xcol], picked_df[!, ycol])]
        end
        return lift(inner, picked_df) 
    end
    
    ax_irf = Axis(f[1, 2], title=picked_title)
    limits!(ax_irf, -5.0, 5.0, 0.0, 1.0)
    lines!(
        ax_irf,
        picked_df_col(:x, :fast_lh),
        markersize = 1,
        label = "fast"
    )
    lines!(
        ax_irf,
        picked_df_col(:x, :slow_lh),
        markersize = 1,
        label = "slow"
    )
    axislegend(ax_irf)
    f
end


if abspath(PROGRAM_FILE) == @__FILE__
    display(main(ARGS[1], ARGS[2]))
    readline()
end