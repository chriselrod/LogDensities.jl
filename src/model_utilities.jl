

struct MarginalBuffer
    ϕ::Vector{Float64}
    θ::Vector{Float64}
    β::Vector{Float64}
    ind::Vector{Int}
    w::Vector{Float64}
    v::Vector{Float64}
    V::Matrix{Float64}
    Vβ::Vector{Float64}
    Vfβ::Vector{Float64}
    δ::Vector{Float64}
    nbuff1::Vector{Float64}
    ∇β::Vector{Float64}
    Λ::Vector{Float64}
    Q::Vector{Float64}
    α::Matrix{Float64}
    ∇α::Vector{Float64}
    init::Vector{Float64}
    n::Int
end

function MarginalBuffer(n::Int)
    ϕ = Vector{Float64}(9)
    θ = Vector{Float64}(7)#Representations from nested cubic.
    β = Vector{Float64}(10)#Representation multiplied by V.
    ind = Vector{Int}(n)
    w = Vector{Float64}(n)
    v = Vector{Float64}(n)
    V =  ones(10,n)
    Vβ = Vector{Float64}(n)
    Vfβ = Vector{Float64}(n)
#    VFβ = Vector{Float64}(n)
    δ = Vector{Float64}(n)
    nbuff1 = Vector{Float64}(n)
    ∇β = Vector{Float64}(10)
    Λ = Vector{Float64}(2)
    Q = Vector{Float64}(3)
    α = zeros(7,10)
    ∇α = Vector{Float64}(7)
    init = [0.0, 0.0, 0.0, 0.0, -2.0, 0.5, -0.2, -5.0, 3.5]

    MarginalBuffer(ϕ, θ, β, ind, w, v, V, Vβ, Vfβ, δ, nbuff1, ∇β, Λ, Q, α, ∇α, init, n)
end
