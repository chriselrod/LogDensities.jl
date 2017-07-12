
abstract type Data end
abstract type parameters{T,N} <: AbstractArray{T,N} end


abstract type Constrainedparameters{p,T,N} <: parameters{T,N} end
log_jacobian!{T}(A::AbstractArray{T}) = zero(T)
Base.IndexStyle(::parameters) = IndexLinear()

type_length{T}(::Type{Vector{T}}) = 0

function Base.show{T <: parameters}(io::IO, ::MIME"text/plain", Θ::T)
  for j in 2:length(fieldnames(T))
    println(getfield(Θ, j))
  end
end
function Base.show{T <: parameters}(io::IO, Θ::T)
  for j in 2:length(fieldnames(T))
    println(getfield(Θ, j))
  end
end

abstract type SquareMatrix{p, T} <: Constrainedparameters{p, T, 2} end

update!(Θ::Constrainedparameters) = nothing

@generated function Base.size{T <: SquareMatrix}(A::T)
  p = T.parameters[1]
  l = round(Int, p * (p + 1) / 2)
  (l, )
end

function Base.size{p}(A::SquareMatrix{p,<:Real})
  (p, p)
end

abstract type UpperTriangle{p,T} <: AbstractArray{T,2} end
struct UpperTriangleView{p,T} <: UpperTriangle{p,T}
  diag::Vector{T}
  off_diag::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},true}
end
struct UpperTriangleVector{p,T} <: UpperTriangle{p,T}
  diag::Vector{T}
  off_diag::Vector{T}
end
Base.IndexStyle(::UpperTriangle) = IndexLinear()
Base.getindex(A::UpperTriangle, i::Int) = A.off_diag[i]
function Base.setindex!(A::UpperTriangle, v, i::Int)
  A.off_diag[i] = v
end
function Base.size{p}(::UpperTriangle{p,Float64})
  (p,p)
end
sub2triangle(i_1::Int, i_2::Int) = i_1 + round(Int, i_2*(i_2-1)/2)
function Base.getindex(A::UpperTriangle, i_1::Int, i_2::Int)
  if i_1 == i_2
    A.diag[i_1]
  else
    A.off_diag[sub2triangle(i_1, i_2-1)]
  end
end
function Base.setindex!{T,p}(A::UpperTriangle{p,T}, v::T, i1::Int, i2::Int)
  if i_1 == i_2
    A.diag[i_1] = v
  else
    A.off_diag[sub2triangle(i_1, i_2-1)] = v
  end
end

struct CovarianceMatrix{p, T <: Real} <: SquareMatrix{p, T}
  Λ::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},true}#length p
  U::UpperTriangleVector{p,T}
  Σ::Symmetric{T,Array{T, 2}}
  U_inverse::UpperTriangleView{p,T}
end
function CovarianceMatrix(p, Σ::Symmetric = Symmetric(Array{Float64}(p,p)))
  Θ = CovarianceMatrix{p, Float64}(Array{Float64,1}(p), UpperTriangle(Array{Float64}(p,p)), Σ, UpperTriangle(Array{Float64}(p,p)), round(Int, p * (p+1)/2))
  update_U!(Θ)
  Θ
end
function CovarianceMatrix(T::DataType, p, Σ::Symmetric = Symmetric(Array{T}(p,p)))
  Θ = CovarianceMatrix{p, T}(Array{T,1}(p), UpperTriangle(Array{T}(p,p)), Σ, UpperTriangle(Array{T}(p,p)), round(Int, p * (p+1)/2))
  update_U!(Θ)
  Θ
end
function update_U_inverse!(Θ::CovarianceMatrix)
  Θ.U_inverse.diag .= exp.(Θ.Λ)
end
function construct{p, T}(A::Type{CovarianceMatrix{p,T}}, Θv::Vector{T}, i::Int)
  Λ = view(Θv, i + (1:p))
  U = UpperTriangleVector{p,T}(Array{T}(p), Array{T}(round(Int,p*(p-1)/2)))
  U_inverse = UpperTriangleView{p,T}(exp.(Λ), view(Θv, i + (1+p:type_length(A))))
  CovarianceMatrix{p, T}(Λ, U, Symmetric(Array{T}(p,p)), U_inverse)
end
function construct{p, T}(A::Type{CovarianceMatrix{p,T}}, Θv::Vector{T}, i::Int, CovMat::Array{T,2})
  construct{p, T}(A, Θv, i, Symmetric(CovMat))
end
function construct{p, T}(A::Type{CovarianceMatrix{p,T}}, Θv::Vector{T}, i::Int, CovMat::Symmetric{T,Array{T,2}})
  if CovMat.uplo != 'U'
    CovMat = Symmetric(CovMat.data')
  end
  Λ = view(Θv, i + (1:p))
  U = UpperTriangleVector{p,T}(Array{T}(p), Array{T}(round(Int,p*(p-1)/2)))
  U_inverse = UpperTriangleView{p,T}(exp.(Λ), view(Θv, i + (1+p:type_length(A))))
  Θ = CovarianceMatrix{p, T}(Λ, U, CovMat, U_inverse)
  set_Σ!(Θ)
  Θ
end


#@generated function Base.length{p,T}(::Type{CovarianceMatrix{p,T}})
#  round(Int, p*(p+1)/2)
#end
function log_jacobian!{p, T}(Θ::CovarianceMatrix{p, T})
  update_U_inverse!(Θ)
  l_jac = zero(T)
  for i ∈ 1:p
    l_jac -= (p + i) * Θ.Λ[i]
  end
  l_jac
end
function chol!{p,T}(U::UpperTriangle{p,T}, Σ::Symmetric{T,Array{T, 2}})
  for i ∈ 1:p
    U[i,i] = Σ[i,i]
    for j ∈ 1:i-1
      U[j,i] = Σ[j,i]
      for k ∈ 1:j-1
        U[j,i] -= U[k,i] * U[k,j]
      end
      U[j,i] /= U[j,j]
      U[i,i] -= U[j,i]^2
    end
    U[i,i] = √U[i,i]
  end
end
###This happens when someone sets an index of the covariance matrix.
function calc_U_from_Σ!{p,T}(Θ::CovarianceMatrix{p, T})
  chol!(Θ.U, Θ.Σ)
end
function calc_Σij!(Θ::CovarianceMatrix, i::Int, j::Int)
  Θ.Σ.data[j,i] = Θ.U[1,i] * Θ.U[1,j]
  for k ∈ 2:j
    Θ.Σ.data[j,i] += Θ.U[k,i] * Θ.U[k,j]
  end
  Θ.Σ.data[j,i]
end
function calc_Σ!{p}(Θ::CovarianceMatrix{p,<:Real})### When would I actually want this???
  for i ∈ 1:p, j ∈ 1:i
    calc_Σij!(Θ, i, j)
  end
end
function calc_invΣij{p,T}(Θ::CovarianceMatrix{p,T}, i::Int, j::Int)
  out = zero(T)
  for k ∈ i:p
    out += Θ.U_inverse[i,k] * Θ.U_inverse[j,k]
  end
  out
end
function inv!{p,T}(U_inverse::UpperTriangle{p,T}, U::UpperTriangle{p,T})
  for i ∈ 1:p
    U_inverse.diag[i] = 1 / U.diag[i]
    for j ∈ i+1:p
      triangle_index = sub2triangle(i,j-1)
      U_inverse.off_diag[triangle_index] = U[i,j] * U_inverse.diag[i]
      for k ∈ i+1:j-1
        U_inverse.off_diag[triangle_index] += U[k,j] * U_inverse[i,k]
      end
      U_inverse.off_diag[triangle_index] /= -U.diag[j]
    end
  end
end
function calc_U_inverse_from_U!(Θ::CovarianceMatrix)
  inv!(Θ.U_inverse, Θ.U)
  Θ.Λ .= log.(Θ.U_inverse.diag)
end
function calc_U_from_U_inverse!(Θ::CovarianceMatrix)
  inv!(Θ.U, Θ.U_inverse)
end
function set_Σ!(Θ::CovarianceMatrix)
  calc_U_from_Σ!(Θ)
  calc_U_inverse_from_U!(Θ)
end
function update_Σ!(Θ::CovarianceMatrix)
  calc_U_from_U_inverse!(Θ)
  calc_Σ!(Θ)
end

#Note, accessing the covariance matrix brings you here, where you calculate Σij; if you want access to the cached value you need to reference Θ.Σ[i,j]. Note that the cache is not updated often.
function Base.getindex(Θ::CovarianceMatrix, i::Int, j::Int)
  i > j ? calc_Σij!(Θ,i, j) : calc_Σij!(Θ,j, i)
end
function Base.getindex{p,T}(Θ::CovarianceMatrix{p,T}, k::Int)
  Θ[ind2sub((p,p), k)...]
end
#Strongly discouraged from calling the following method. But...if you have to, it is here.
function Base.setindex!{p,T}(Θ::CovarianceMatrix{p,T}, v::T, k::Int)
  Θ[ind2sub((p,p), k)] = v
end
function Base.setindex!{p,T}(Θ::CovarianceMatrix{p,T}, v::T, i::Int, j::Int)
  update_Σ!(Θ)
  i > j ? setindex!(Θ.Σ.data, v, j, i) : setindex!(Θ.Σ.data, v, i, j)
  set_Σ!(Θ)
end


function quad_form(x::Vector{Real}, Θ::CovarianceMatrix)
  out = (x[1] * Θ.U_inverse.diag[1])^2
  for i ∈ 2:p
    dot_prod = x[i] * Θ.U_inverse.diag[i]
    triangle_index = round(Int, (i-1)*(i-2)/2)
    for j ∈ 1:i-1
      dot_prod += x[j] * Θ.U_inverse.off_diag[triangle_index + j]
    end
    out += dot_prod^2
  end
  out
end
function trace_AΣinv{p,T}(A::AbstractArray{<:Real}, Σ::CovarianceMatrix{p,T})
  2*htrace_AΣinv(A,Σ)
end
function htrace_AΣinv{p,T}(A::AbstractArray{<:Real}, Σ::CovarianceMatrix{p,T})
  out = zero(T)
  for i ∈ 1:p
    out += A[i,i] * calc_invΣij(Σ, i, i) / 2
    for j ∈ 1:i-1
      out += A[j,i] * calc_invΣij(Σ, i, j)
    end
  end
  out
end
Base.det(U::UpperTriangle) = prod(U.diag)
Base.logdet(U::UpperTriangle) = sum(log, U.diag)
Base.trace(U::UpperTriangle) = sum(U.diag)
Base.det(Θ::CovarianceMatrix) = det(Θ.U_inverse)^-2
Base.logdet(Θ::CovarianceMatrix) = 2hlogdet(Θ)
inv_det(Θ::CovarianceMatrix) = det(Θ.U_inverse)^2
inv_root_det(Θ::CovarianceMatrix) = det(Θ.U_inverse)
root_det(Θ::CovarianceMatrix) = 1/det(Θ.U_inverse)
hlogdet(Θ::CovarianceMatrix) = -sum(Θ.Λ)
nhlogdet(Θ::CovarianceMatrix) = sum(Θ.Λ)
function trace_inverse{p,T}(Θ::CovarianceMatrix{p,T})
  out = 0
  for i ∈ 1:p
    out += calc_invΣij(Θ, i, i)
  end
  out
end
function Base.:+(Θ::CovarianceMatrix, A::AbstractArray{Real,2})
  update_Σ!(Θ)
  Θ.Σ + A
end
function Base.:+(A::AbstractArray{Real,2}, Θ::CovarianceMatrix)
  update_Σ!(Θ)
  Θ.Σ + A
end
#Would you want to output a regular matrix, a symmetric matrix, or a covariance matrix?
function Base.:+{p,T}(Θ_1::CovarianceMatrix{p,T}, Θ_2::CovarianceMatrix{p,T})
  update_Σ!(Θ_1)
  update_Σ!(Θ_2)
  #CovarianceMatrix(T, p, Θ_1.Σ + Θ_2.Σ)
  Θ_1.Σ + Θ_2.Σ
end
function Base.:*(Θ::CovarianceMatrix, A::AbstractArray)
  update_Σ!(Θ)
  Θ.Σ * A
end
function Base.:*(A::AbstractArray, Θ::CovarianceMatrix)
  update_Σ!(Θ)
  A * Θ.Σ
end
function Base.:*{p,T}(Θ_1::CovarianceMatrix{p,T}, Θ_2::CovarianceMatrix{p,T})
  update_Σ!(Θ_1)
  update_Σ!(Θ_2)
  #CovarianceMatrix(T, p, Θ_1.Σ + Θ_2.Σ)
  Θ_1.Σ * Θ_2.Σ
end
function Base.show(io::IO, ::MIME"text/plain", Θ::CovarianceMatrix)
  update_Σ!(Θ)
  println(Θ.Σ)
end
@generated type_length{p,T}(::Type{CovarianceMatrix{p,T}}) = round(Int, p * (p+1)/2)
function Base.convert{p,T}(::Type{Symmetric{T,Array{T,2}}}, A::CovarianceMatrix{p,T})
  update_Σ!(A)
  A.Σ
end
function Base.convert{p,T}(::Type{Array{T,2}}, A::CovarianceMatrix{p,T})
  convert(Array{T,2}, convert(Symmetric{T,Array{T,2}}, A))
end

abstract type ConstrainedVector{p,T} <: Constrainedparameters{p,T,1} end
Base.:+(x::ConstrainedVector, y::Vector) = x.x .+ y
Base.:+(y::Vector, x::ConstrainedVector) = x.x .+ y
Base.:+(x::ConstrainedVector, y::Real) = x.x .+ y
Base.:+(y::Real, x::ConstrainedVector) = x.x .+ y
Base.:+(y::ConstrainedVector, x::ConstrainedVector) = x.x .+ y.x
Base.:-(x::ConstrainedVector, y::Vector) = x.x .- y
Base.:-(y::Vector, x::ConstrainedVector) = y .- x.x
Base.:-(x::ConstrainedVector, y::Real) = x.x .- y
Base.:-(y::Real, x::ConstrainedVector) = y .- x.x
Base.:-(x::ConstrainedVector, y::ConstrainedVector) = x.x .- y.x
Base.:*(A::AbstractArray{<:Real,2}, x::ConstrainedVector) = A * x.x
Base.:*(x::ConstrainedVector, A::AbstractArray{<:Real,2}) = x.x * A
Base.convert(::Type{Vector}, A::ConstrainedVector) = A.x
Base.show(io::IO, ::MIME"text/plain", Θ::ConstrainedVector) = println(Θ.x)
Base.size(x::ConstrainedVector) = size(x.x)
Base.getindex(x::ConstrainedVector, i::Int) = x.x[i]

struct PositiveVector{p, T} <: ConstrainedVector{p, T}
  Θ::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},true}
  x::Vector{T}
end
PositiveVector{T}(x::Vector{T}) = PositiveVector{length(x), T}(log.(x), x)
function log_jacobian!(x::PositiveVector)
  x.x .= exp.(x.Θ)
  sum(x.Θ)
end
type_length{p,T}(::Type{PositiveVector{p,T}}) = p
Base.getindex(x::PositiveVector, i::Int) = exp(x.Θ[i])
function Base.setindex!(x::PositiveVector, v::Real, i::Int)
  x.x[i] = v
  x.Θ[i] = log(v)
end
function construct{p, T}(::Type{PositiveVector{p,T}}, Θv::Vector{T}, i::Int)
  v = view(Θv, i + (1:p))
  PositiveVector{p, T}(v, exp.(v))
end
function construct{p, T}(::Type{PositiveVector{p,T}}, Θv::Vector{T}, i::Int, vals::Vector{T})
  pv = PositiveVector{p, T}(view(Θv, i + (1:p)), vals)
  pv.Θ .= log.(vals)
  pv
end



struct ProbabilityVector{p, T} <: ConstrainedVector{p, T}
  Θ::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},true}
  x::Vector{T}
end
ProbabilityVector{T}(Θ::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},true}) = ProbabilityVector{length(x), T}(logit.(x), x)
function log_jacobian!(x::ProbabilityVector)
  x.x .= logistic.(x.Θ)
  sum(log.(x.x) .+ log.(1 .- x.x))
end
type_length{p,T}(::Type{ProbabilityVector{p,T}}) = p
Base.getindex(x::ProbabilityVector, i::Int) = logistic(x.Θ[i])
function Base.setindex!(x::ProbabilityVector, v::Real, i::Int)
  x.x[i] = v
  x.Θ[i] = logit(v)
end
function construct{p, T}(::Type{ProbabilityVector{p,T}}, Θv::Vector{T}, i::Int)
  v = view(Θv, i + (1:p))
  ProbabilityVector{p, T}(v, logistic.(v))
end
function construct{p, T}(::Type{ProbabilityVector{p,T}}, Θv::Vector{T}, i::Int, vals::Vector{T})
  pv = ProbabilityVector{p, T}(view(Θv, i + (1:p)), vals)
  pv.Θ .= logit.(vals)
  pv
end

struct RealVector{p, T} <: ConstrainedVector{p, T}
  x::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},true}
end
function Base.setindex!(x::RealVector, v::Real, i::Int)
  x.x[i] = v
end
@generated log_jacobian!{p, T}(x::RealVector{p, T}) = zero(T)

type_length{p,T}(::Type{RealVector{p,T}}) = p
function construct{p, T}(::Type{RealVector{p,T}}, Θ::Vector{T}, i::Int)
  RealVector{p, T}(view(Θ, i + (1:p)))
end
function construct{p, T}(::Type{RealVector{p,T}}, Θ::Vector{T}, i::Int, vals::Vector{T})
  rv = RealVector{p, T}(view(Θ, i + (1:p)))
  rv.x .= vals
  rv
end

struct Buffer{p,T,N} <: Constrainedparameters{p,T,N}
  x::Array{T,N}
end
function Base.getindex(x::Buffer, i::Int)
  x.x[i]
end
function Base.setindex!(x::Buffer, v::Real, i::Int)
  x.x[i] = v
end
@generated log_jacobian!{p,T,N}(x::Buffer{p,T,N}) = zero(T)

type_length(::Type{Buffer}) = 0
@generated function construct{p, T, N}(::Type{Buffer{p,T,N}}, Θ::Vector{T}, i::Int)
  Buffer{p,T,N}(Array{T,N}(fill(p,N)...))
end
function construct{p, T, N}(::Type{Buffer{p,T,N}}, Θ::Vector{T}, i::Int, vals::Array{T,N})
  Buffer{p,T,N}(vals)
end


#struct LowerBoundVector{p, T}
#  Θ::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},true}
#  x::Vector{T}
#  L::Vector{T}
#end
