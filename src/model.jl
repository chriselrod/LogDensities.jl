

struct Model{p, q <: QuadratureRule, P <: parameters{Float64}}
  Grid::GridContainer{p,q}
  Θ::P
  UA::UnionAll
end


function Model(::Type{P}; l::Int = 6, grid_build::GridBuild{q} = AdaptiveBuild{GenzKeister}) where {P <: parameters, q}
  Θ = construct(P{Float64})
  Grid = GridContainer(length(Θ), l, q, seq)
  Model(Grid, Θ, P)
end

function Base.show(io::IO, ::MIME"text/plain", m::Model)
  print(m.Θ)
end
