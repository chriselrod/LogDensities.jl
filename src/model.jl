abstract type ModelRank end
abstract type StaticRank{p} end
abstract type DynamicRank end
struct Full{p} <: StaticRank end
struct FixedRank{p} <: StaticRank end
struct Dynamic <: DynamicRank end
struct LDR{g} <: DynamicRank end
struct Model{G <: GridVessel, PF <: parameters, P <: parameters, R <: ModelRank}
  Grid::G
  Θ::MF
end

val_to_rank(::Type{Val{p}},::Type{R}) where {p, R <: StaticRank} = R{p}
FullVal(::Type{Val{p}}) where p = Full{p}
FixedRankVal(::Type{Val{p}}) where p = FixedRank{p}
default(::Type{R},::Type{P}) where {P <: parameters, R <: StaticRank} = val_to_rank(param_type_length(P), R)
default(::Type{LDR},::Type{P} where {P <: parameters}) = LDR(0.999)
complete(a::UnionAll,::Type{P}) where {P <: parameters} = default(a,P)
complete(a::ModelRank,::Type{P} where {P <: parameters}) = a
K(::Type{<:DynamicRank}, ::Type{<:AdaptiveBuild}) = Tuple{Int,Int,Int}
K(::Type{<:StaticRank}, ::Type{<:AdaptiveBuild}) = Tuple{Int,Int}
K(::Type{<:DynamicRank}, ::Type{<:aPrioriBuild}) = Tuple{Int,Int,Vector{Int}}
K(::Type{<:StaticRank}, ::Type{<:aPrioriBuild}) = Tuple{Int,Vector{Int}}

#### Idea here is to call model and perform a static dispatch, thanks to parameterizing everything deciding the dispach.
#### Resulting object will be one of several different model types.
#### Don't need a single model shell


Model(Grid::G, Θ::PF, ::Type{P}, ::Type{R}) where {G, PF, P, R} = Model{G, PF, P, R}(Grid, Θ)


function Model(::Type{P}, ::Type{B} = AdaptiveBuild{GenzKeister}, ::Type{R} = FullVal(param_type_length(P))) where {P <: parameters, B <: GridBuild, R <: StaticRank}
  Θ = construct(P{Float64})
  Grid = StaticGridVessel(q, B, K(R,B), param_type_length(P))
  Model(Grid, Θ, P, complete(R))
end
function Model(::Type{P}, ::Type{B}, ::Type{R}) where {P <: parameters, B <: GridBuild, R <: DynamicRank}
  Θ = construct(P{Float64})
  Grid = DynamicGridVessel(q, B, K(R,B), type_length(P))
  Model(Grid, Θ, P, complete(R))
end

function Base.show(io::IO, ::MIME"text/plain", m::Model)
  print(m.Θ)
end
