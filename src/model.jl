

struct Model{q <: QuadratureRule, P <: parameters{Float64}, B <: GridBuild}
  Grid::GridVessel{q,B}
  Θ::P
  UA::UnionAll
end


function Model(::Type{P}, ::Type{B}) where {P <: parameters, q, B <: AdaptiveBuild{q}}
  Θ = construct(P{Float64})
  p = type_length(P)
  Grid = GridVessel(q, B, Tuple{Int,Int,Int}, p)
  Model{q, P, B}(Grid, Θ, P)
end
function Model(::Type{P}, ::Type{B} = AdaptiveBuild{GenzKeister}) where {P <: parameters, q, B <: GridBuild}
  Θ = construct(P{Float64})
  p = length(Θ)
  Grid = GridVessel(q, B, Tuple{Int,Vector{Int}}, p)
  Model{q, P, B}(Grid, Θ, P)
end

function Base.show(io::IO, ::MIME"text/plain", m::Model)
  print(m.Θ)
end
