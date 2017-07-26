

struct Model{p, q <: QuadratureRule, P <: parameters{Float64}}
  Grid::GridContainer{p,q}
  Θ::P
  UA::UnionAll
end


function Model{P <: parameters}(::Type{P}; l::Int = 6, q::DataType = GenzKeister, seq::Vector{Int} = SparseQuadratureGrids.default(q))
  Θ = construct(P{Float64})
  Grid = GridContainer(length(Θ), l, q, seq)
  Model(Grid, Θ, P)
end

function Base.show(io::IO, ::MIME"text/plain", m::Model)
  print(m.Θ)
end
