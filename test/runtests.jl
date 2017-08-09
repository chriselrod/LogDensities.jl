using LogDensities
using Base.Test
using ForwardDiff

struct TestObj{p,T} <: parameters{T,1}
  x::Vector{T}
  prob::ProbabilityVector{p,T}
  pos::PositiveVector{p,T}
  real::RealVector{p,T}
  cov::CovarianceMatrix{p,T}
end

function inv_transform(::Type{TestObj{p,T2} where T2}, x::Vector{T}) where {T <: Real, p}
  Θ = construct(TestObj{p,T}, x)
  ConstrainedParameters.update_Σ!(Θ.cov)
  out = similar(x)
  k = 3p
  for i ∈ 1:p
    out[i] = Θ.prob[i]
    out[i+p] = Θ.pos[i]
    out[i+2p] = Θ.real[i]
    for j ∈ 1:i
      k += 1
      out[k] = Θ.cov.Σ[j,i]
    end
  end
  out
end

p = 2
m1 = Model(TestObj{p})
@testset begin
  @test type_length(typeof(m1.Θ)) == length(m1.Θ) == length(m1.Θ.x) == round(Int, p*(3 + (p+1)/2))

  x = randn(length(m1.Θ))
  copy!(m1.Θ.x, x)
  update!(m1.Θ)

  #The CovarianceMatrix log_jacobian! function drops the constant p*log(2) term.
  @test log_jacobian!(m1.Θ) + p*log(2) ≈ logabsdet(ForwardDiff.jacobian(x -> inv_transform(TestObj{2}, x), x))[1]
end
