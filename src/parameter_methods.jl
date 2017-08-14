struct ModelParam{p, T, V <: AbstractArray{T,1}, P <: Tuple} <: parameter
    x::V
    Θ::P
end
function ModelParam(::Type{Val{p}}, x::V, Θ::P) where {p, T, V <: AbstractArray{T,1}, P <: Tuple}
    ModelParam{p, T, V, P}(x, Θ)
end

log_jacobian(A::ModelParam) = sum(log_jacobian.(A.Θ))
update!(A::ModelParam) = update!.(A.Θ)
Base.size(A::ModelParam{p}) where p = ( p, )
Base.length(A::ModelParam{p}) where p = p
Base.@pure param_length(x::Tuple) = Val{sum(length.(x))}
@generated indices(x::Tuple) = cumsum(length.(x))
construct(Θ::Tuple, ::Type{Val{p}} = param_length(Θ)) where p = construct(SlidingVec(zeros(p),0,p), Θ, Val{p})
function construct(x::V, Θ::P, ::Type{Val{p}} = param_length(Θ)) where {p, T, V <: AbstractVector{T}, P <: Tuple}
    ModelParam{p, T, V, P}(x, construct.(Θ, (x,), indices(x)))
end


function log_density(x::AbstractVector{T}, θ::Tuple, data, ::Type{Val{p}} = param_length(θ)) where {p, T}
  param = construct(x, P, Val{p})
  ld = log_jacobian!(param)
  ld + Main.log_density(data, param.Θ...)
end
function log_density!(Θ::ModelParam, data)
  ld = log_jacobian!(Θ)
  ld + Main.log_density(data, Θ.Θ...)
end
