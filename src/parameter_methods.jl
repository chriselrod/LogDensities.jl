struct Val{p} end

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


@generated function construct(::Type{T}) where {T <: parameters}
  if isa(T, UnionAll)
    T2 = T{Float64}
  else
    T2 = T
  end
  field_count = length(T2.types)
  indices = cumsum([type_length(T2.types[i]) for i ∈ 1:field_count])
  Θ = zeros(T2.parameters[end], indices[end])

  Expr(:call, T2, Θ, [construct(T2.types[i+1], Θ, indices[i]) for i ∈ 1:field_count-1]...)
end


function construct(::Type{T}, A::Vararg) where {T <: parameters}

  field_count = length(T.types)
  indices = cumsum([type_length(T.types[i]) for i ∈ 1:field_count])
  Θ = zeros(T.parameters[end], indices[end])

  eval(Expr(:call, T, Θ, [construct(T.types[i+1], Θ, indices[i], A[i]) for i ∈ 1:field_count-1]...))
end
function construct(::Type{T}, Θ::Vector) where {T <: parameters}

  field_count = length(T.types)
  indices = cumsum([type_length(T.types[i]) for i ∈ 1:field_count])

  eval(Expr(:call, T, Θ, [construct(T.types[i+1], Θ, indices[i]) for i ∈ 1:field_count-1]...))
end

julia> struct ts{T, B <: AbstractArray{T}}
           t::T
           v::B
       end

function negative_log_density(Θ::Vector{T}, ::Type{P}, data::Data) where {T, P <: parameters}
  param = construct(P{T}, Θ)
  nld = -log_jacobian!(param)
  nld - Main.log_density(param, data)
end
function negative_log_density!(Θ::parameters, data::Data)
#  update!(Θ)
  nld = -log_jacobian!(Θ)
  nld - Main.log_density(Θ, data)
end
