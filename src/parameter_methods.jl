

@generated function Base.size{T <: parameters}(A::T)
  p = 0
  for i ∈ T.types
    p += type_length(i)
  end
  (p, )
end

@generated function type_length{T <: parameters}(::Type{T})
  p = 0
  for i ∈ T.types
    p += type_length(i)
  end
  p
end

function Base.getindex(A::parameters, i::Int)
  A.x[i]
end

function Base.setindex!(A::parameters, v::Real, i::Int)
  A.x[i] = v
end

@generated function log_jacobian!{T <: parameters}(A::T)
  expr = :(log_jacobian!(getfield(A, 2)))
  for i ∈ 3:length(T.types)
    expr = :(log_jacobian!(getfield(A, $i)) + $expr)
  end
  expr
end


@generated function construct{T <: parameters}(::Type{T})
  if isa(T, UnionAll)
    T2 = T{Float64}
  else
    T2 = T
  end
  field_count = length(T2.types)
  indices = cumsum([type_length(T2.types[i]) for i ∈ 1:field_count])
  Θ = zeros(T2.parameters[1], indices[end])

  Expr(:call, T2, Θ, [construct(T2.types[i+1], Θ, indices[i]) for i ∈ 1:field_count-1]...)
end


function construct{T <: parameters}(::Type{T}, A::Vararg)

  field_count = length(T.types)
  indices = cumsum([type_length(T.types[i]) for i ∈ 1:field_count])
  Θ = zeros(T.parameters[1], indices[end])

  eval(Expr(:call, T, Θ, [construct(T.types[i+1], Θ, indices[i], A[i]) for i ∈ 1:field_count-1]...))
end
function construct{T <: parameters}(::Type{T}, Θ::Vector)

  field_count = length(T.types)
  indices = cumsum([type_length(T.types[i]) for i ∈ 1:field_count])

  eval(Expr(:call, T, Θ, [construct(T.types[i+1], Θ, indices[i]) for i ∈ 1:field_count-1]...))
end



function negative_log_density{T, P <: parameters}(Θ::Vector{T}, ::Type{P}, data::Data)
  param = construct(P{T}, Θ)
  nld = -log_jacobian!(param)
  nld - Main.log_density(param, data)
end
function negative_log_density!(Θ::parameters, data::Data)
  nld = -log_jacobian!(Θ)
  nld - Main.log_density(Θ, data)
end
