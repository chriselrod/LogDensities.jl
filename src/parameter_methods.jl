struct ModelParam{p, T, V <: AbstractArray{T,1}, P <: Tuple} <: parameter{T}
    v::V
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
@generated cum_inds(x::Tuple{P}) where P = tuple(cumsum([ConstrainedParameters.type_length.(P)...])...)
construct(Θ::Tuple, ::Type{Val{p}} = param_length(Θ)) where p = construct(SlidingVec(zeros(p,1),0,p), Θ, Val{p})
sconstruct(Θ::Tuple, ::Type{Val{p}} = param_length(Θ)) where p = construct(SVector{p}(zeros(p)), Θ, Val{p})
function construct(x::V, Θ::P, ::Type{Val{p}} = param_length(Θ)) where {p, T, V <: AbstractVector{T}, P <: Tuple}
    ModelParam(Val{p}, x, construct.(Θ, (x,), cum_inds(Θ)))
end


function log_density(x::AbstractVector, θ::Tuple, data, ::Type{p} = param_length(θ)) where p
    param = construct(x, θ, p)
    log_jacobian(param) + Main.log_density(data, param.Θ...)
end
function log_density(Θ::ModelParam, data)
    log_jacobian(Θ) + Main.log_density(data, Θ.Θ...)
end
function log_density(x::AbstractVector, Θ::ModelParam, data)
    Θ.v .= x
    update!(Θ)
    log_jacobian(Θ) + Main.log_density(data, Θ.Θ...)
end
