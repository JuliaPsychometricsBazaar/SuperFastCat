using Base.Order: By

const PartialMeanAndCT = SMatrix{4, 2, Float32}
const zero_partial_mean_and_c = @SMatrix zeros(Float32, 4, 2)
const c_pt_idx = 1
const c_err_idx = 2 
const unnorm_mean_pt_idx = 3
const unnorm_mean_err_idx = 4

@kwdef struct Segment
    seg_idx::Int
    vals::PartialMeanAndCT=zero_partial_mean_and_c 
end

Base.isless(i::Segment, j::Segment) = isless(i.vals[c_err_idx], j.vals[c_err_idx])

@kwdef struct IntervalBestEntry
    item_bank_idx::Int
    best_measure::Float32
    num_expansions::Int
    acc::PartialMeanAndCT=zero_partial_mean_and_c
    segs::Union{Vector{Segment}, Nothing}=nothing
end

"""
Interval best has property that it is always sorted according to `ordering`.
The interval with the best worst value is at the beginning `bests[1]`. The rest
of the list is within `margin` of the best.
"""
@Base.kwdef mutable struct IntervalBest{OrderingT <: Ordering}
    const ordering::OrderingT
    best_worst_measure::Float32
    const contents::Vector{IntervalBestEntry}
end

_get_measure(ibe::IntervalBestEntry) = ibe.best_measure
_get_measure(val::Float32) = val

function IntervalBest(ordering::Ordering, capacity::Int)
    contents = IntervalBestEntry[]
    sizehint!(contents, capacity)
    IntervalBest(
        ordering=By(_get_measure, ordering),
        best_worst_measure=NaN32,
        contents=contents
    )
end

function summary(item::IntervalBestEntry)
    return "<IntervalBestEntry item_bank_idx: $(item.item_bank_idx) best_measure: $(item.best_measure) num_expansions: $(item.num_expansions)>"
end

function summarize_items(bests::IntervalBest)
    summarized_contents = join([summary(e) for e in bests.contents], " ")
    return """
    IntervalBest:
        best_worst_measure: $(bests.best_worst_measure)
        contents ($(length(bests.contents))): $(summarized_contents)
    """
end

interval_best(::KnownForwardOrderings, interval) = inf(interval)
interval_best(::KnownReverseOrderings, interval) = sup(interval)
interval_best(best::IntervalBest, interval) = interval_best(best.ordering.order, interval)
interval_worst(::KnownForwardOrderings, interval) = sup(interval)
interval_worst(::KnownReverseOrderings, interval) = inf(interval)
interval_worst(best::IntervalBest, interval) = interval_worst(best.ordering.order, interval)


function push_into_contents!(bests::IntervalBest, new_entry)
    insert_idx = searchsortedfirst(bests.contents, new_entry.best_measure, bests.ordering)
    insert!(bests.contents, insert_idx, new_entry)
    insert_idx
end

function add_to_interval_best!(bests::IntervalBest, item_bank_idx, measure)
    new_measure_worst = interval_worst(bests, measure)
    new_measure_best = interval_best(bests, measure)
    new_entry = IntervalBestEntry(item_bank_idx=item_bank_idx, best_measure=new_measure_best, num_expansions=1)
    if length(bests.contents) == 0
        # First item
        push!(bests.contents, IntervalBestEntry(item_bank_idx=item_bank_idx, best_measure=new_measure_best, num_expansions=1))
        bests.best_worst_measure = new_measure_worst
    else
        if lt(bests.ordering, new_measure_worst, bests.best_worst_measure)
            # Added item is new best-worst
            bests.best_worst_measure = new_measure_worst 
            # Evict all x where inf(x) > sup(best_measure)
            evict_idx = searchsortedlast(bests.contents, bests.best_worst_measure, bests.ordering)
            resize!(bests.contents, evict_idx)
            # Insert according to inf
            push_into_contents!(bests, new_entry)
        else
            # Try to see if we can insert it somewhere
            if lt(bests.ordering, new_measure_best, bests.best_worst_measure)
                # We want to keep it since its best value is better than the
                # current best-worst measure
                push_into_contents!(bests, new_entry)
            end
        end
    end
end

function update_measure!(bests::IntervalBest, update_idx, new_measure, acc, segs)
    # Because we assume that the interval of the measure only gets narrower
    # when it is updates, we know that the best-worst cannot get worse, and so
    # cannot be invalidated except by a better value.
    new_measure_worst = interval_worst(bests, new_measure)
    new_measure_best = interval_best(bests, new_measure)
    bests_entry = bests.contents[update_idx]
    item_bank_idx = bests_entry.item_bank_idx
    num_expansions = bests_entry.num_expansions
    new_entry = IntervalBestEntry(item_bank_idx, new_measure_best, num_expansions + 1, acc, segs)
    new_idx = nothing
    if lt(bests.ordering, new_measure_worst, bests.best_worst_measure)
        # Added item is new best-worst
        bests.best_worst_measure = new_measure_worst 
        # Evict all x where inf(x) > sup(best_measure)
        evict_idx = searchsortedlast(bests.contents, bests.best_worst_measure, bests.ordering)
        resize!(bests.contents, evict_idx)
        if update_idx <= evict_idx
            # The item we are updating has not been evicted, delete the old entry
            deleteat!(bests.contents, update_idx)
        end
        new_idx = push_into_contents!(bests, new_entry)
    else
        deleteat!(bests.contents, update_idx)
        if lt(bests.ordering, new_measure_best, bests.best_worst_measure)
            # We want to keep it since its best value is better than the
            # current best-worst measure
            new_idx = push_into_contents!(bests, new_entry)
        end
    end
    return new_idx
end

function Base.empty!(bests::IntervalBest)
    empty!(bests.contents)
end

function Base.resize!(bests::IntervalBest, size)
    resize!(bests.contents, size)
end

function Base.length(bests::IntervalBest)
    length(bests.contents)
end