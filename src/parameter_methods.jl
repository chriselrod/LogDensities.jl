struct ModelParam{p, T, P <: parameter{T}}
    x::MVector{p,T}
    Θ::P
end

@generated function Base.size(A::T) where {T <: parameters}
  p = 0
  for i ∈ T.types
    p += type_length(i)
  end
  (p, )
end

@generated function type_length(::Type{T}) where {T <: parameters}
  p = 0
  for i ∈ T.types
    p += type_length(i)
  end
  p
end
#Perhaps confusingly, while "ConstrainedParameters.param_type_length" functions return ConstrainedParameters.Val{p}, this returns SparseQuadratureGrids.Val{p}.
@generated function param_type_length(::Type{T}) where {T <: parameters}
  p = 0
  for i ∈ T.types
    p += type_length(i)
  end
  SparseQuadratureGrids.Val{p}
end

function Base.getindex(A::parameters, i::Int)
  A.x[i]
end

function Base.setindex!(A::parameters, v::Real, i::Int)
  A.x[i] = v
end

@generated function log_jacobian!(A::T) where {T <: parameters}
  expr = :(log_jacobian!(getfield(A, 2)))
  for i ∈ 3:length(T.types)
    expr = :(log_jacobian!(getfield(A, $i)) + $expr)
  end
  expr
end

function update!(A::T) where {T <: parameters}
  for i ∈ 2:length(T.types)
    update!(getfield(A, i))
  end
end

@generated function param_indices(::Type{P}) where P
    cumsum([type_length(p_i) for p_i ∈ P.types])
end

@generated function construct(::Type{P}, Θ::Vector) where {P <: parameter}

  indices = cumsum([type_length(p_i) for p_i ∈ P.types])
  tup = ntuple(i -> Expr(:call, construct, P.types[i+1], :Θ, indices[i]), length(P.types)-1)
  :( $P(Θ, $(tup...)) )
end

function construct(::Type{P}) where {T, P <: parameter{T}}
  Θ = Vector{T}(type_length(P))
  construct(P, Θ)
end


function construct(::Type{T}, A::Vararg) where {T <: parameter}
  field_count = length(T.types)
  indices = param_indices(T)
  Θ = zeros(T.parameters[end], indices[end])
  #eval(Expr(:call, T, Θ, [construct(T.types[i+1], Θ, indices[i], A[i]) for i ∈ 1:field_count-1]...))
  T(Θ, (construct(T.types[i+1], Θ, indices[i], A[i]) for i ∈ 1:field_count-1)...)
end


function log_density(Θ::Vector{T}, ::Type{P}, data) where {T, P <: parameters}
  param = construct(P{T}, Θ)
  update!(param)
  ld = log_jacobian!(param)
  ld + Main.log_density(param, data)
end
function log_density!(Θ::parameters, data)
  ld = log_jacobian!(Θ)
  ld + Main.log_density(Θ, data)
end
