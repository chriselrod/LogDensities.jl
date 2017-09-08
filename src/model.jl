abstract type ModelRank end
abstract type StaticRank{p} <: ModelRank end
abstract type DynamicRank <: ModelRank end
struct Full{p} <: StaticRank{p} end
struct FixedRank{p} <: StaticRank{p} end
struct Dynamic <: DynamicRank end
struct LDR{g} <: DynamicRank end
struct Model{G <: GridVessel, MP <: ModelParam, P <: Tuple, R <: ModelRank, B <: ModelDiffBuffer}
	Grid::G
	Θ::MP
    ϕ::P
    MarginalBuffers::Dict{Function,MarginalBuffer}
	diff_buffer::B
end
function Model(Grid::G, Θ::MP, ϕ::P, MarginalBuffers::Dict{Function,MarginalBuffer}, diff_buffer::B, ::Type{R}) where {G, MP, P, R, B}
    Model{G, MP, P, R, B}(Grid, Θ, ϕ, MarginalBuffers, diff_buffer)
end
#Commented out model options instead of deleting it, to serve as a reminder.
#It would probably be a relatively convenient API change.
#struct ModelOptions{R <: ModelRank, T <: Bool}
#    MR::R
#    cache::Val{T}
#end
#function ModelOptions(; MR = Full())
#
#end


complete(::Type{LDR},::P where P <: ModelParam) = LDR{0.999}
complete(::Type{LDR{g}},::P where P <: ModelParam) where g = LDR{g}
complete(::Type{R},::P) where {p, P <: ModelParam{p}, R <: StaticRank} = R{p}
complete(::Type{Dynamic},::P where {P <: ModelParam}) = Dynamic
complete(::Type{R},::P where {P <: ModelParam}) where {p, R <: StaticRank{p}} = R
K(::Type{<:DynamicRank}, ::Type{<:AdaptiveBuild}) = Tuple{Int,Int,Int}
K(::Type{<:StaticRank}, ::Type{<:AdaptiveBuild}) = Tuple{Int,Int}
K(::Type{<:DynamicRank}, ::Type{<:aPrioriBuild}) = Tuple{Int,Vector{Int}}
K(::Type{<:StaticRank}, ::Type{<:aPrioriBuild}) = Vector{Int}

#### Idea here is to call model and perform a static dispatch, thanks to parameterizing everything deciding the dispach.
#### Resulting object will be one of several different model types.
#### Don't need a single model shell



Base.@pure SmallModel(::Type{Val{p}}) where p = Val{p < 13}

function Model(ϕ::Tuple, ::Type{B} = AdaptiveRaw{GenzKeister}, ::Type{R} = Full) where {R <: ModelRank, B <: RawBuild}
    Θ = construct(ϕ)
    Model(Θ, ϕ, complete(R, Θ), B)
end
function Model(ϕ::Tuple, ::Type{B}, ::Type{R} = Full) where {R <: ModelRank, B <: CacheBuild}
    Θ = sconstruct(ϕ)
    Model(Θ, ϕ, complete(R, Θ), B)
end
function Model(θ::MP, ϕ::Tuple, ::Type{R}, ::Type{B}) where {p, MP <: ModelParam{p}, R <: StaticRank, B}
    Model(θ, ϕ, R, B, SmallModel(Val{p}))
end
function Model(θ::MP, ϕ::Tuple, ::Type{R}, ::Type{B}, ::Type{Val{true}}) where {p, d, MP <: ModelParam{p}, R <: StaticRank{d}, q, B <: RawBuild{q}}
    Grid = DynamicGridVessel(q, B, K(R, B), Val{p}, Vector{SVector{d, Float64}}, MatrixVecSVec{p, Float64})
    Model( Grid, θ, ϕ, R )
end
function Model(θ::MP, ϕ::Tuple, ::Type{R}, ::Type{B}, ::Type{Val{true}}) where {p, d, MP <: ModelParam{p}, R <: StaticRank{d}, q, B <: CacheBuild{q}}
    Grid = DynamicGridVessel(q, B, K(R, B), Val{p}, Vector{SVector{d, Float64}}, Vector{SVector{p, Float64}})
    Model( Grid, θ, ϕ, R )
end

function Model(θ::MP, ϕ::Tuple, ::Type{R}, ::Type{B}, ::Type{V}) where {p, MP <: ModelParam{p}, R <: ModelRank, q, B <: RawBuild{q}, V <: Val}
    Grid = DynamicGridVessel(q, B, K(R, B), Val{p}, Matrix{Float64}, Matrix{Float64})
    Model( Grid, θ, ϕ, R )
end
function Model(θ::MP, ϕ::Tuple, ::Type{R}, ::Type{B}, ::Type{V}) where {p, MP <: ModelParam{p}, R <: ModelRank, q, B <: CacheBuild{q}, V <: Val}
    Grid = DynamicGridVessel(q, B, K(R, B), Val{p}, Matrix{Float64}, MatrixVecSVec{p, Float64})
    Model( Grid, θ, ϕ, R )
end

function Model(Grid::G, Θ::MP, ϕ::P, ::Type{R}) where {p, G, MP <: ModelParam{p}, P <: Tuple, R}
    Model(Grid, Θ, ϕ, Dict{Function,MarginalBuffer}(), ModelDiffBuffer(ϕ, Val{p}), R)
end



MarginalBuffer(M::Model, f::Function, n::Int) = get!( () -> MarginalBuffer(n), m.mc, (f, n) )


function Base.show(io::IO, ::MIME"text/plain", m::Model)
  print(m.Θ)
end
