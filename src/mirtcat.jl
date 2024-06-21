Base.@kwdef struct FixedWDecisionTreeGenerationState
    # Input
    # \- Input / Item bank
    item_bank::ItemBank
    # \- Input / tolerance
    tolerance::Float32
    # State
    # \- State / Likelihood
    likelihood::ResponsesLikelihood
    # \- State / Tree position
    state_tree::TreePosition
    # Outputs
    # \- Outputs / Decision tree
    decision_tree_result::DecisionTree
    # Buffers
    # \- Buffers / Integration points
    weighted_gauss::WeightedGauss
    # \- Buffers / f(x) values at integration points
    ir_fx::Vector{Float32}
    # \- Buffers / Item-response at integration points
    item_response_quadrature_buf::Vector{Float32}
    # \- Buffers / Rough best list
    rough_best::RoughBest
end