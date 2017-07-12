module LogDensities

using SparseQuadratureGrids, TensorOperations, StaticArrays

import  Base.show,
        Base.getindex,
        Base.setindex!,
        Base.size,
        Base.IndexStyle,
        Base.+,
        Base.*,
        Base.convert

export  Data,
        parameters,
        CovarianceMatrix,
        PositiveVector,
        ProbabilityVector,
        RealVector,
        Model,
        construct,
        negative_log_density,
        negative_log_density!,
        log_jacobian!,
        quad_form,
        inv_det,
        inv_root_det,
        root_det,
        log_root_det,
        trace_inverse,
        lpdf_InverseWishart,
        lpdf_normal

include("helper_functions.jl")
include("constrained_types.jl")
include("parameter_methods.jl")
include("model.jl")
include("log_density_functions.jl")
include("models.jl")

end # module
