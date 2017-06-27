using LogDensities


struct BinaryClassification{T} <: parameters{T}
  x::Vector{T}
  p::ProbabilityVector{3,T}
end

m1 = Model(BinaryClassification)












include("/home/chris/.julia/v0.6/LogDensities/src/constrained_types.jl")


pv = ProbabilityVector([.04,.5,.8])


struct param{T} <: Parameters{T}
  x::Vector{T}
  pv1::ProbabilityVector{3,T}
  pv2::ProbabilityVector{4,T}
end
@generated function Base.size{T <: Parameters}(A::T)
  p = 0
  for i ∈ T.types
    p += type_length(i)
  end
  (p, )
end



po = param(ProbabilityVector(rand(3)), ProbabilityVector(rand(4)))


size(po)

@generated function Base.getindex{T <: Parameters}(A::T, i::Int)
    ...
end
@generated function Base.setindex!{T <: Parameters}(A::T, v::Real, i::Int)
    ...
end
@generated function log_jacobian{T <: Parameters}(A::T)

end



tddt = param{Float64}.types[1]
tddtf{p,T}(::ConstrainedParameters{p,T}) = p
tddtf{p,T}(::Type{ConstrainedParameters{p,T}}) = p
tddtf{p,T}(::Type{ProbabilityVector{p,T}}) = p
tddtf(tddt)
tddtf(param{Float64}.types[2])







abstract type Parameters{T} <: AbstractArray{T,1} end
abstract type ConstrainedParameters{p,T} <: Parameters{T} end
abstract type ConstrainedVector{p,T} <: ConstrainedParameters{p,T} end
@generated function Base.show{T <: Parameters}(io::IO, ::MIME"text/plain", Θ::T)
  quote
    for j in fieldnames(T)
      println(getfield(Θ, j))
    end
  end
end
Base.IndexStyle(::Parameters) = IndexLinear()

struct PositiveVector{p, T} <: ConstrainedVector{p, T}
  Θ::Vector{T}
  x::Vector{T}
end

PositiveVector{T}(x::Vector{T}) = PositiveVector{length(x), T}(log.(x), x)
Base.length{p,T}(::Type{PositiveVector{p,T}}) = p
Base.getindex(x::ConstrainedVector, i::Int) = x.x[i]
Base.show(io::IO, ::MIME"text/plain", Θ::ConstrainedVector) = print(io, Θ.x)
Base.size(x::ConstrainedVector) = size(x.x)

struct param{T} <: Parameters{T}
  pv1::PositiveVector{3,T}
  pv2::PositiveVector{4,T}
end
@generated function Base.size{T <: Parameters}(A::T)
  p = 0
  for i ∈ T.types
    p += length(i)
  end
  (p, )
end

po = param(PositiveVector(rand(3)), PositiveVector(rand(4)))
size(po)
@code_warntype size(po)

@generated function Base.size{T <: Parameters}(A::T)
  p = 0
  for i ∈ T.types
    p += length(i)
  end
  (p, )
end

@code_warntype size(po)

using BenchmarkTools

@benchmark size(po)
@benchmark size($po)

return_seven(A) = 7;

@benchmark return_seven(po)


function func_test()
  po = param(PositiveVector(rand(3)), PositiveVector(rand(4)))
  @benchmark size(po)
end

function getfield(p::param{Float64}, :baz)
  randn(5)
end

struct vector_wrap{T,P <: Parameters}
  x::Vector{T}
  po::P
end

mutable struct vector_wrap_mutable{T,P <: Parameters}
  x::Vector{T}
  po::P
end
