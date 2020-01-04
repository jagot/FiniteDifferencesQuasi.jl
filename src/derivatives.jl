# * Derivatives

# ** Three-point stencils
α(::FiniteDifferences{T}) where T = one(T)
β(::FiniteDifferences{T}) where T = one(T)

# Variationally derived coefficients for three-point stencil
α( ::RadialDifferences,    j::Integer)         = j^2/(j^2 - 1/4)
β(B::RadialDifferences{T}, j::Integer) where T = (j^2 - j + 1/2)/(j^2 - j + 1/4) + (j == 1 ? B.δβ₁ : zero(T))

α( ::NumerovFiniteDifferences{T}) where T = one(T)
β(B::NumerovFiniteDifferences{T}, j::Integer) where T = one(T) + (j == 1 ? B.δβ₁ : zero(T))

α(B::RadialDifferences) = α.(Ref(B), B.j[1:end-1])
β(B::Union{RadialDifferences,NumerovFiniteDifferences}) = β.(Ref(B), B.j)

# ** Dispatch aliases

const FlatFirstDerivative{Basis<:AbstractFiniteDifferences} = Mul{<:Any, <:Tuple{
    <:QuasiAdjoint{<:Any, <:Basis},
    <:Derivative,
    <:Basis}}
const LazyFirstDerivative{Basis<:AbstractFiniteDifferences} = Mul{<:Any, <:Tuple{
    <:Mul{<:Any, <:Tuple{
        <:QuasiAdjoint{<:Any, <:Basis},
        <:Derivative}},
    <:Basis}}

const FirstDerivative{Basis} = Union{FlatFirstDerivative{Basis}, LazyFirstDerivative{Basis}}

const FlatSecondDerivative{Basis<:AbstractFiniteDifferences} = Mul{<:Any, <:Tuple{
    <:QuasiAdjoint{<:Any, <:Basis},
    <:QuasiAdjoint{<:Any, <:Derivative},
    <:Derivative,
    <:Basis}}
const LazySecondDerivative{Basis<:AbstractFiniteDifferences} = Mul{<:Any, <:Tuple{
    <:Mul{<:Any, <:Tuple{
        <:Mul{<:Any, <:Tuple{
            <:QuasiAdjoint{<:Any, <:Basis}, <:QuasiAdjoint{<:Any, <:Derivative}}},
        <:Derivative}},
    <:Basis}}

const SecondDerivative{Basis} = Union{FlatSecondDerivative{Basis},LazySecondDerivative{Basis}}

const FirstOrSecondDerivative{Basis} = Union{FirstDerivative{Basis},SecondDerivative{Basis}}

# ** Materialization

function copyto!(dest::Tridiagonal{T}, M::FirstDerivative{<:AbstractFiniteDifferences}) where T
    axes(dest) == axes(M) || throw(DimensionMismatch("axes must be same"))

    B = last(M.args)

    # Central difference approximation
    a = α(B)/2step(B)
    dest.dl .= -a
    dest.d .= zero(T)
    dest.du .= a

    if B isa NumerovFiniteDifferences && first(B.j) == 1
        dest.d[1] += B.λ
    end

    dest
end

function similar(M::FirstDerivative, ::Type{T}) where T
    B = last(M.args)
    n = size(B,2)
    Tridiagonal(Vector{T}(undef, n-1), Vector{T}(undef, n), Vector{T}(undef, n-1))
end

function copyto!(dest::SymTridiagonal{T}, M::SecondDerivative{<:AbstractFiniteDifferences}) where T
    axes(dest) == axes(M) || throw(DimensionMismatch("axes must be same"))

    B = last(M.args)
    δ² = step(B)^2

    dest.dv .= -2β(B)/δ²
    dest.ev .= α(B)/δ²

    dest
end

function similar(M::SecondDerivative, ::Type{T}) where T
    B = last(M.args)
    n = size(B,2)
    SymTridiagonal(Vector{T}(undef, n), Vector{T}(undef, n-1))
end

materialize(M::FirstOrSecondDerivative) = copyto!(similar(M, eltype(M)), M)

function NumerovDerivative(::Type{T}, Δ::Tri, ∇::FirstDerivative) where {T,Tri}
    B = last(∇.args)
    # M₁ = Diagonal{T}(I, size(B,2))

    M₁ = similar(Δ)
    M₁.dl .= 1
    M₁.d .= 4
    M₁.du .= 1

    # Eq. (20ff), Muller (1999), λ = λ′ = √3 - 2, but only if
    # `singular_origin==true`.
    B.j[1] == 1 && (M₁.d[1] += B.λ)

    M₁ /= 6one(T)

    NumerovDerivative(Δ, M₁, one(T))
end

function NumerovDerivative(::Type{T}, Δ::Tri, ∇²::SecondDerivative) where {T,Tri}
    B = last(∇².args)

    M₂ = similar(Δ)
    M₂.dv .= 10
    M₂.ev .= 1

    # Eq. (17), Muller (1999)
    B.j[1] == 1 && (M₂.dv[1] -= 2B.δβ₁)

    M₂ /= -6one(T)
    NumerovDerivative(Δ, M₂, -2one(T))
end

materialize(M::FirstOrSecondDerivative{NumerovFiniteDifferences{T}}) where T =
    NumerovDerivative(T, copyto!(similar(M, eltype(M)), M), M)
