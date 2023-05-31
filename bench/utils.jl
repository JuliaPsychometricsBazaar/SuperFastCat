using JSON3


struct RecWriter
    io::IO
    ctx::Vector{Dict}
end

function open_rec_writer(f, filename)
    open(filename, "w") do io
        f(RecWriter(io, []))
    end
end

function rec_ctx(f, rw::RecWriter; kwargs...)
    push!(rw.ctx, Dict(kwargs...))
    f()
    pop!(rw.ctx)
end

function write_rec(rw::RecWriter; kwargs...)
    out_rec = merge(reverse([rw.ctx..., kwargs])...)
    #@info "write_rec" out_rec
    JSON3.write(rw.io, out_rec)
    println(rw.io)
    flush(rw.io)
end

function proc_timed(timed)
    (
        timed.value,
        Dict(
            :time => timed[:time],
            :bytes => timed[:bytes],
            :gctime => timed[:gctime],
            :allocd => timed[:gcstats].allocd,
            :total_time => timed[:gcstats].total_time,
        )
    )
end