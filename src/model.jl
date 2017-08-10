abstract type ModelRank end
abstract type StaticRank{p} <: ModelRank end
abstract type DynamicRank <: ModelRank end
struct Full{p} <: StaticRank{p} end
struct FixedRank{p} <: StaticRank{p} end
struct Dynamic <: DynamicRank end
struct LDR{g} <: DynamicRank end
struct Model{G <: GridVessel, PF <: parameters, P <: parameters, R <: ModelRank}
  Grid::G
  Θ::PF
end

val_to_rank(::Type{SparseQuadratureGrids.Val{p}},::Type{R}) where {p, R <: StaticRank} = R{p}
FullVal(::Type{SparseQuadratureGrids.Val{p}}) where p = Full{p}
FixedRankVal(::Type{SparseQuadratureGrids.Val{p}}) where p = FixedRank{p}
default(::Type{R},::Type{P}) where {P <: parameters, R <: StaticRank} = val_to_rank(param_type_length(P), R)
default(::Type{LDR},::Type{P} where {P <: parameters}) = LDR(0.999)
complete(a::Type{UnionAll},::Type{P}) where {P <: parameters} = default(a,P)
complete(a::Type{Dynamic},::Type{P} where {P <: parameters}) = a
complete(a::Type{<:StaticRank{p} where p},::Type{P} where {P <: parameters}) = a
complete(a::Type{<:LDR{g} where g},::Type{P} where {P <: parameters}) = a
K(::Type{<:DynamicRank}, ::Type{<:AdaptiveBuild}) = Tuple{Int,Int,Int}
K(::Type{<:StaticRank}, ::Type{<:AdaptiveBuild}) = Tuple{Int,Int}
K(::Type{<:DynamicRank}, ::Type{<:aPrioriBuild}) = Tuple{Int,Vector{Int}}
K(::Type{<:StaticRank}, ::Type{<:aPrioriBuild}) = Vector{Int}

#### Idea here is to call model and perform a static dispatch, thanks to parameterizing everything deciding the dispach.
#### Resulting object will be one of several different model types.
#### Don't need a single model shell


Model(Grid::G, Θ::PF, ::Type{P}, ::Type{R}) where {G, PF, P, R} = Model{G, PF, P, R}(Grid, Θ)


function Model(::Type{P}, ::Type{R} = FullVal(param_type_length(P{Float64}))) where {P <: parameters, R <: StaticRank}
  Θ = construct(P{Float64})
  Grid = StaticGridVessel(GenzKeister, AdaptiveRaw{GenzKeister}, K(R,AdaptiveRaw{GenzKeister}), param_type_length(P{Float64}))
  Model(Grid, Θ, P, complete(R,P{Float64}))
end
function Model(::Type{P}, ::Type{R}) where {P <: parameters, R <: DynamicRank}
  Θ = construct(P{Float64})
  Grid = DynamicGridVessel(GenzKeister, AdaptiveRaw{GenzKeister}, K(R,AdaptiveRaw{GenzKeister}), param_type_length(P{Float64}))
  Model(Grid, Θ, P, complete(R,P{Float64}))
end
function Model(::Type{P}, ::Type{B}, ::Type{R} = FullVal(param_type_length(P{Float64}))) where {q, B <: GridBuild{q}, P <: parameters, R <: StaticRank}
  Θ = construct(P{Float64})
  Grid = StaticGridVessel(q, B, K(R,B), param_type_length(P{Float64}))
  Model(Grid, Θ, P, complete(R,P{Float64}))
end
function Model(::Type{P}, ::Type{B}, ::Type{R}) where {q, B <: GridBuild{q}, P <: parameters, R <: DynamicRank}
  Θ = construct(P{Float64})
  Grid = DynamicGridVessel(q, B, K(R,B), param_type_length(P{Float64}))
  Model(Grid, Θ, P, complete(R,P{Float64}))
end

function Base.show(io::IO, ::MIME"text/plain", m::Model)
  print(m.Θ)
end
