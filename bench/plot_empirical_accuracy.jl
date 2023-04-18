using Parquet2
using Makie
using GLMakie
using DataFrames
using Parquet2: Dataset


function load(inf)
    ds = Dataset(inf)
    DataFrame(ds; copycols=false)
end

function main(pointsf)
    GLMakie.activate!()
    points = load(pointsf)
    points[!, :q] = Int.(points[!, :q])
    f = Figure()
    ax = Axis(f[1, 1])
    function inspector_label(self, i, p)
        par = params[points[i, :q], :]
        repr("text/plain", Dict(pairs(merge(points[i, :], par))))
    end
    scatter = scatter!(
        ax,
        points[!, :x],
        points[!, :err],
        inspector_label = inspector_label,
        markersize = 1
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    display(main(ARGS[1]))
    readline()
end