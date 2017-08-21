

struct MarginalBuffer
    ϕ::Vector{Float64}
    θ::Vector{Float64}
    β::Vector{Float64}
    ind::Vector{Int}
    w::Vector{Float64}
    v::Matrix{Float64}
    V::Matrix{Float64}
    Vβ::Vector{Float64}
end

function MarginalBuffer(n::Int)
    ϕ = zeros(9)
    θ = Vector{Float64}(6)#Representations from nested cubic.
    β = Vector{Float64}(10)#Representation multiplied by V.
    ind = Vector{Int}(n)
    w = Vector{Float64}(n)
    v = Vector{Float64}(n)
    V =  ones(10,n)
    Vβ = Vector{Float64}(n)
    Vfβ = Diagonal(Vector{Float64}(n))
#    VFβ = Vector{Float64}(n)
    δ = Vector{Float64}(n)
    nbuff1 = Vector{Float64}(n)
    nbuff2 = Vector{Float64}(n)
    ∇β = Vector{Float64}(10)
    Λ = Vector{Float64}(2)
    Q = Vector{Float64}(3)
    α = zeros(7,10)
    α[64] = 1.0
    ∇α = Vector{Float64}(7)

    MarginalBuffer(ϕ, θ, β, ind, w, v, V, Vβ, n)
end
