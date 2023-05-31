using DuckDB
using DBInterface
using Makie
using DataFrames
using GLMakie
using Gadfly
using PrettyTables


function draw_time_density(df)
    f = Figure()
    Axis(f[1, 1], xscale = Makie.pseudolog10)
    for grp in groupby(df, :system)
        vals = collect(skipmissing(grp[!, :time]))
        @show vals
        density!(vals, strokewidth = 1, strokecolor = :black, fillcolor = :blue, alpha = 0.5)
    end
    f
end

function draw_time_boxplot(df)
    set_default_plot_size(14cm, 20cm)
    plot_all = Gadfly.plot(df, x=:system, y=:time, Geom.boxplot)
    filter_cond = df[!, :system] .∉ Ref(Set((
        "fixedw_20_tm3",
        "fixedw_11_tm3",
        "fixedw_20_tm5",
        "fixedw_11_tm5",
        "iterqwk"
    )))
    df_filtered = df[filter_cond, :]
    @show df df_filtered
    plot_filter = Gadfly.plot(df_filtered, x=:system, y=:time, Geom.boxplot)
    vstack(plot_all, plot_filter)
end

const kvals = (1, 3, 10, 30, 100)

function report_top_ks(df)
    systems = []
    cols = [[] for kval_ in kvals]
    for grp in groupby(df, :system)
        push!(systems, grp[1, :system])
        for (idx, kval) in enumerate(kvals)
            push!(cols[idx], sum(grp[!, :k] .<= kval) / nrow(grp) * 100)
        end
    end
    @show cols
    dt = DataFrame(
        "system" => systems,
        ["P@$kval" => cols[idx] for (idx, kval) in enumerate(kvals)]...
    )
    println(dt)
    pretty_table(dt)
end

function main(inf)
    conn = DBInterface.connect(DuckDB.DB, inf)
    df = DataFrame(DBInterface.execute(conn, "SELECT * FROM next_item"))
    #draw_time_density(df)
    #draw_time_boxplot(df)
    report_top_ks(df)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS[1])
end