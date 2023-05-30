"""
Rough best has property that it is always sorted according to `ordering`. The
best measure is at the beginning `bests[1]`. The rest of the list is within `margin` of the best.
"""
@Base.kwdef struct RoughBest{OrderingT <: Ordering}
    ordering::OrderingT
    best_idxs::Vector{Int}
    best_measures::Vector{Float32}
    margin::Float32
end

function RoughBest(ordering::Ordering, capacity::Int, margin)
    best_idxs = Vector{Int}()
    sizehint!(best_idxs, capacity)
    best_measures = Vector{Float32}()
    sizehint!(best_measures, capacity)
    RoughBest(
        ordering=ordering,
        best_idxs=best_idxs,
        best_measures=best_measures,
        margin=margin
    )
end

KnownForwardOrderings = Union{FastForwardOrdering, ForwardOrdering}
KnownReverseOrderings = ReverseOrdering{KnownForwardOrderings}

apply_margin(::KnownForwardOrderings, measure, margin) = measure + margin
apply_margin(::KnownReverseOrderings, measure, margin) = meaure - margin

function add_to_rough_best!(bests::RoughBest, idx, measure::Float32)
    if length(bests.best_idxs) == 0
        # First item
        push!(bests.best_idxs, idx)
        push!(bests.best_measures, measure)
    else
        cur_best_measure = bests.best_measures[1]
        if lt(bests.ordering, measure, cur_best_measure)
            # Added item is new best -- insert at front and evict everything > best_measure + margin
            insert!(bests.best_idxs, 1, idx)
            insert!(bests.best_measures, 1, measure)
            margin_measure = apply_margin(bests.ordering, measure, bests.margin)
            insert_idx = searchsortedfirst(bests.best_measures, margin_measure, bests.ordering)
            resize!(bests.best_idxs, insert_idx - 1)
            resize!(bests.best_measures, insert_idx - 1)
        else
            # Try to see if we can insert it somewhere
            margin_measure = apply_margin(bests.ordering, cur_best_measure, bests.margin)
            if lt(bests.ordering, measure, margin_measure)
                # We want to keep it since it is within `margin` of best
                insert_idx = searchsortedfirst(bests.best_measures, measure, bests.ordering)
                insert!(bests.best_idxs, insert_idx, idx)
                insert!(bests.best_measures, insert_idx, measure)
            end
        end
    end
end

function Base.empty!(bests::RoughBest)
    empty!(bests.best_idxs)
    empty!(bests.best_measures)
end

function Base.resize!(bests::RoughBest, size)
    resize!(bests.best_idxs, size)
    resize!(bests.best_measures, size)
end

function Base.length(bests::RoughBest)
    length(bests.best_idxs)
end