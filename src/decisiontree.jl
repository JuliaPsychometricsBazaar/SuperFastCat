# TODO: Remove ability tracking from here?
Base.@kwdef struct AgendaItem
    depth::UInt32
    ability::Float32
end

Base.@kwdef mutable struct TreePosition
    max_depth::UInt
    cur_depth::UInt
    todo::PushVector{AgendaItem, Vector{AgendaItem}}
    parent_ability::Float32
end

function TreePosition(max_depth)
    TreePosition(
        max_depth=max_depth,
        cur_depth=0,
        todo=PushVector{AgendaItem}(max_depth), # depth, ability, 
        parent_ability=0f0
    )
end

function next!(state::TreePosition, lh, item_bank, question, ability)
    # Try to go deeper
    if state.cur_depth < state.max_depth
        state.parent_ability = ability
        state.cur_depth += 1
        push!(state.todo, AgendaItem(depth=state.cur_depth, ability=ability))
        push_question_response!(lh, item_bank, question, false)
        #@info "next deeper" state.cur_depth state.max_depth lh.questions lh.responses
    else
        # Try to back track
        if !isempty(state.todo)
            todo = pop!(state.todo)
            state.parent_ability = todo.ability
            state.cur_depth = todo.depth
            resize!(lh, todo.depth)
            lh.responses[todo.depth] = true
            #@info "next backtrack" state.cur_depth state.max_depth lh.questions lh.responses
        else
            # Done: break
            #@info "next done"
            return true
        end
    end
    return false
end

Base.@kwdef struct DecisionTree
    questions::Vector{UInt32}
    ability_estimates::Vector{Float32}
end

function DecisionTree(max_depth)
    @info "tree size" max_depth tree_size(max_depth) tree_size(max_depth + 1)
    DecisionTree(
        questions=Vector{UInt32}(undef, tree_size(max_depth)),
        ability_estimates=Vector{Float32}(undef, tree_size(max_depth + 1))
    )
end

function responses_idx(responses)
    (length(responses) > 0 ? evalpoly(2, responses) : 0) + 2^length(responses)
end

function Base.insert!(dt::DecisionTree, responses, ability, next_item)
    idx = responses_idx(responses)
    dt.questions[idx] = next_item
    dt.ability_estimates[idx] = ability
end

function Base.insert!(dt::DecisionTree, responses, ability)
    idx = responses_idx(responses)
    dt.ability_estimates[idx] = ability
end
