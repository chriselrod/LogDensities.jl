

struct BinaryClassParams{T} <: Parameters{T}
  Θ::ProbabilityVector{T,p}

end


@generated function Base.size{T <: Parameters}(A::T)
  p = 0
  for i ∈ T.types
    p += length(i)
  end
  (p, )
end

@generated function type_length()

@generated function Base.getindex(A::Parameters, i::Int)

end

@generated function Base.setindex!(A::Parameters, v::Real, i::Int)

end

@generated function log_jacobian{T <: Parameters}(A::T)
  expr = :(log_jacobian(getfield(A, 2)))
  for i ∈ 3:length(T.types)
    expr = :(log_jacobian(getfield(A, $i)) + $expr)
  end
  expr
end


@generated function construct{T <: Parameters}(::Type{T})

  field_count = length(T.types)
  indices = cumsum([type_length(T.types[i]) for i ∈ 1:field_count])
  Θ = zeros(T.parameters[1], indices[end])

  Expr(:call, T, Θ, [construct(T.types[i+1], Θ, indices[i]) for i ∈ 1:field_count-1]...)
end
function construct_no_gen{T <: Parameters}(::Type{T})

  field_count = length(T.types)
  indices = cumsum([type_length(T.types[i]) for i ∈ 1:field_count])
  Θ = zeros(T.parameters[1], indices[end])

  Expr(:call, T, Θ, [construct(T.types[i+1], Θ, indices[i]) for i ∈ 1:field_count-1]...)
end

function test_construct{T <: Parameters}(::Type{T})
  T
end
