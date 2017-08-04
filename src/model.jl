

struct Model{p, q <: QuadratureRule, P <: parameters{Float64}, B <: GridBuild}
  Grid::GridContainer{p,q}
  Θ::P
  UA::UnionAll
end


function Model(::Type{P}; l::Int = 6, ::Type{B} = AdaptiveBuild{GenzKeister}) where {P <: parameters, q, B <: GridBuild}
  Θ = construct(P{Float64})
  p = length(Θ)
  Grid = GridContainer(p, l, q, seq)
  Model{p, q, P, B}(Grid, Θ, P)
end

function Base.show(io::IO, ::MIME"text/plain", m::Model)
  print(m.Θ)
end
