#=
It may be possible to use TypedTables.jl/StructArrays.jl/ComponentArrays.jl
instead but at the moment they have limited interoperability and it is a bit
hard to figure out which is zero-cost + contiguous + CUDA-compatible. Let's
just do it C-style for now.
=#
# item bank
const ItemBankT = Array{Float32, 2}
const idxr_difficulty = 1
const idxr_discrimination = 2
const idxr_guess = 3
const idxr_slip = 4

# question-response
const idxr_question = 1
const idxr_response = 2

# likelihood affines
const PrecomputedLikelihoodT = Array{Float32, 2}
const idxr_x_c = 1
const idxr_x_m = 2
const idxr_lh_y_c = 3
const idxr_lh_y_m = 4

# item-response affines
const idxr_ir_neg_y_c = 3
const idxr_ir_neg_y_m = 4
const idxr_ir_pos_y_c = 5
const idxr_ir_pos_y_m = 6

