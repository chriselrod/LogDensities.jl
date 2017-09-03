module LogDensities

using SparseQuadratureGrids, StaticArrays, ConstrainedParameters, DiffBase, ForwardDiff, Optim

import  Base: show, getindex, setindex!, size, IndexStyle, convert, Val, length,
        ConstrainedParameters: type_length, param_type_length, construct, log_jacobian, update!
	Optim: Options, initial_state, update_state!, NewtonTrustRegionState

export  Data,
        parameters,
        CovarianceMatrix,
        PositiveVector,
        ProbabilityVector,
        RealVector,
        Model,
        ModelParam,
        construct,
        log_density,
        log_jacobian!,
        quad_form,
        inv_det,
        inv_root_det,
        root_det,
        log_root_det,
        trace_inverse,
        lpdf_InverseWishart,
        lpdf_normal,
        logit,
        logistic,
        update!,
        type_length,
        ModelRank,
        StaticRank,
        DynamicRank,
        Full,
        FixedRank,
        LDR,
        Dynamic,
        MarginalBuffer

include("model_utilities.jl")
include("parameter_methods.jl")
include("model.jl")

end # module
