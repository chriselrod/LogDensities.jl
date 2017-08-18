module LogDensities

using SparseQuadratureGrids, StaticArrays, ConstrainedParameters

import  Base.show,
        Base.getindex,
        Base.setindex!,
        Base.size,
        Base.IndexStyle,
        Base.+,
        Base.*,
        Base.convert,
        Base.Val,
        Base.length,
        ConstrainedParameters.type_length,
        ConstrainedParameters.param_type_length,
        ConstrainedParameters.construct,
        ConstrainedParameters.log_jacobian,
        ConstrainedParameters.update!

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
        Dynamic

include("parameter_methods.jl")
include("model.jl")

end # module
