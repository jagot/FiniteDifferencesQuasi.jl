module FiniteDifferencesQuasi

import Base: eltype, axes, size, ==, getindex, checkbounds, copyto!, similar, show, step
import Base.Broadcast: materialize

using ContinuumArrays
import ContinuumArrays: ℵ₁, Derivative
import ContinuumArrays.QuasiArrays: AbstractQuasiMatrix, QuasiAdjoint

using IntervalSets

using LazyArrays
import LazyArrays: Mul2

using LinearAlgebra
import LinearAlgebra: Matrix

using Printf

abstract type AbstractFiniteDifferences{T} <: AbstractQuasiMatrix{T} end

eltype(::AbstractFiniteDifferences{T}) where T = T

axes(B::AbstractFiniteDifferences{T}) where T = (first(locs(B))..last(locs(B)), Base.OneTo(length(locs(B))))
size(B::AbstractFiniteDifferences) = (ℵ₁, length(locs(B)))

==(A::AbstractFiniteDifferences,B::AbstractFiniteDifferences) = locs(A) == locs(B)

getindex(B::AbstractFiniteDifferences{T}, x::Real, i::Integer) where T =
    x == locs(B)[i] ? one(T) : zero(T)

# * Mass matrix
function materialize(M::Mul2{<:Any,<:Any,<:QuasiAdjoint{<:Any,<:FD},<:FD}) where {T,FD<:AbstractFiniteDifferences{T}}
    Ac, B = M.factors
    axes(Ac,2) == axes(B,1) || throw(DimensionMismatch("axes must be same"))
    A = parent(Ac)
    A == B || throw(ArgumentError("Cannot multiply functions on different grids"))
    Diagonal(ones(T, size(A,2)))
end

# * Cartesian finite differences
struct FiniteDifferences{T} <: AbstractFiniteDifferences{T}
    j::UnitRange{Integer}
    Δx::T
end
FiniteDifferences(n::Integer, Δx::T) where {T<:Real} =
    FiniteDifferences{T}(1:n, Δx)

locs(B::FiniteDifferences) = B.j*B.Δx
step(B::FiniteDifferences{T}) where {T} = B.Δx

show(io::IO, B::FiniteDifferences{T}) where {T} =
    write(io, "Finite differences basis {$T} on $(axes(B,1)) with $(size(B,2)) points spaced by Δx = $(B.Δx)")


# * Radial finite differences
# Specialized finite differences for the case where there is
# Dirichlet0 boundary condition at r = 0.
struct RadialDifferences{T} <: AbstractFiniteDifferences{T}
    j::Base.OneTo{Integer}
    ρ::T
    Z::T
    δβ₁::T # Correction used for bare Coulomb potentials, Eq. (22) Schafer2009
end

RadialDifferences(n::I, ρ::T, Z::T=one(T)) where {I<:Integer, T} =
    RadialDifferences{T}(Base.OneTo(n), ρ, Z, Z*ρ/8 * (one(T) + Z*ρ))

locs(B::RadialDifferences{T}) where T = (B.j .- one(T)/2)*B.ρ
step(B::RadialDifferences{T}) where {T} = B.ρ

==(A::RadialDifferences,B::RadialDifferences) = locs(A) == locs(B) && A.Z == B.Z && A.δβ₁ == B.δβ₁

show(io::IO, B::RadialDifferences{T}) where T =
    write(io, "Radial finite differences basis {$T} on $(axes(B,1)) (formally 0..$(rightendpoint(axes(B,1)))) with $(size(B,2)) points spaced by ρ = $(B.ρ)")

# * Derivatives

α(::FiniteDifferences{T}) where T = one(T)
β(::FiniteDifferences{T}) where T = one(T)

# Variationally derived coefficients for three-point stencil
α( ::RadialDifferences,    j::Integer)         = j^2/(j^2 - 1/4)
β(B::RadialDifferences{T}, j::Integer) where T = (j^2 - j + 1/2)/(j^2 - j + 1/4) + (j == 1 ? B.δβ₁ : zero(T))

α(B::RadialDifferences) = α.(Ref(B), B.j[1:end-1])
β(B::RadialDifferences) = α.(Ref(B), B.j)

const FirstDerivative{Basis<:AbstractFiniteDifferences} = Mul{<:Tuple,<:Tuple{<:QuasiAdjoint{<:Any,<:Basis},<:Derivative,<:Basis}}
const SecondDerivative{Basis<:AbstractFiniteDifferences} = Mul{<:Tuple,<:Tuple{<:QuasiAdjoint{<:Any,<:Basis},<:QuasiAdjoint{<:Any,<:Derivative},<:Derivative,<:Basis}}
const FirstOrSecondDerivative = Union{FirstDerivative,SecondDerivative}

function copyto!(dest::Tridiagonal{T}, M::FirstDerivative) where T
    axes(dest) == axes(M) || throw(DimensionMismatch("axes must be same"))
    
    B = last(M.factors)
    a = α(B)/2step(B)
    
    dest.dl .= -a
    dest.d .= zero(T)
    dest.du .= a
    
    dest
end

function similar(M::FirstDerivative, ::Type{T}) where T
    B = last(M.factors)
    n = size(B,2)
    Tridiagonal(Vector{T}(undef, n-1), Vector{T}(undef, n), Vector{T}(undef, n-1))
end

function copyto!(dest::SymTridiagonal{T}, M::SecondDerivative) where T
    axes(dest) == axes(M) || throw(DimensionMismatch("axes must be same"))
    
    B = last(M.factors)
    
    dest.dv .= -2β(B)/step(B)^2
    dest.ev .= α(B)/step(B)^2
    
    dest
end

function similar(M::SecondDerivative, ::Type{T}) where T
    B = last(M.factors)
    n = size(B,2)
    SymTridiagonal(Vector{T}(undef, n), Vector{T}(undef, n-1))
end

materialize(M::FirstOrSecondDerivative) = copyto!(similar(M, eltype(M)), M)

export FiniteDifferences, RadialDifferences, Derivative

end # module
