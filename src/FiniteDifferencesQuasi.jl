module FiniteDifferencesQuasi

import Base: eltype, axes, size, ==, getindex, checkbounds, copyto!, similar, show, step
import Base.Broadcast: materialize

using ContinuumArrays
import ContinuumArrays: Basis, ℵ₁, Derivative, Inclusion
import ContinuumArrays.QuasiArrays: AbstractQuasiMatrix, QuasiAdjoint, MulQuasiArray

using IntervalSets

using LazyArrays
import LazyArrays: ⋆, Mul2

using LinearAlgebra
import LinearAlgebra: Matrix, dot
using SparseArrays

using Printf

abstract type AbstractFiniteDifferences{T,I<:Integer} <: Basis{T} end

eltype(::AbstractFiniteDifferences{T}) where T = T

axes(B::AbstractFiniteDifferences{T}) where T = (Inclusion(leftendpoint(B)..rightendpoint(B)), Base.OneTo(length(locs(B))))
size(B::AbstractFiniteDifferences) = (ℵ₁, length(locs(B)))

==(A::AbstractFiniteDifferences,B::AbstractFiniteDifferences) = locs(A) == locs(B)

tent(x, x₀::T, δx::T) where T =
    one(T) - abs(clamp((x-x₀)/δx, -one(T), one(T)))

getindex(B::AbstractFiniteDifferences, x::Real, i::Integer) =
    tent(x, locs(B)[i], step(B))

"""
    within_interval(x, interval)

Return the indices of the elements of `x` that lie within the given
closed `interval`.
"""
function within_interval(x::AbstractRange, interval::ClosedInterval)
    a = leftendpoint(interval)
    b = rightendpoint(interval)
    δx = step(x)
    max(ceil(Int, (a-x[1])/δx),1):min(floor(Int, (b-x[1])/δx)+1,length(x))
end

function getindex(B::AbstractFiniteDifferences{T}, x::AbstractRange, sel::AbstractVector) where T
    l = locs(B)
    δx = step(B)
    χ = spzeros(T, length(x), length(sel))
    for j in sel
        x₀ = l[j]
        for i ∈ within_interval(x, x₀ - δx..x₀ + δx)
            χ[i,j] = tent(x[i], x₀, δx)
        end
    end
    χ
end

getindex(B::AbstractFiniteDifferences{T}, x::AbstractRange, ::Colon) where T =
    getindex(B, x, axes(B,2))

# * Types

const FDArray{T,N,B<:AbstractFiniteDifferences} = MulQuasiArray{T,N,<:Mul{<:Any,<:Tuple{<:B,<:AbstractArray{T,N}}}}
const FDVector{T,B<:AbstractFiniteDifferences} = FDArray{T,1,B}
const FDMatrix{T,B<:AbstractFiniteDifferences} = FDArray{T,2,B}
const FDVecOrMat{T,B<:AbstractFiniteDifferences} = Union{FDVector{T,B},FDMatrix{T,B}}

const FDOperator{T,B<:AbstractFiniteDifferences,M<:AbstractMatrix} = MulQuasiArray{T,2,<:Mul{<:Any,<:Tuple{<:B,<:M,<:QuasiAdjoint{<:Any,<:B}}}}

const FDMatrixElement{T,B<:AbstractFiniteDifferences,M<:AbstractMatrix,V<:AbstractVector} =
    MulQuasiArray{T,0,<:Mul{<:Any,
                            <:Tuple{<:Adjoint{<:Any,<:V},<:QuasiAdjoint{<:Any,<:B},
                                    <:B,<:M,<:QuasiAdjoint{<:Any,<:B},
                                    <:B,<:V}}}

const FDInnerProduct{T,U,B<:AbstractFiniteDifferences{U},V1<:AbstractVector{T},V2<:AbstractVector{T}} =
    Mul{<:Any, <:Tuple{<:Adjoint{<:Any,<:V1},<:QuasiAdjoint{<:Any,<:B},<:B,<:V2}}

const LazyFDInnerProduct{FD<:AbstractFiniteDifferences} = Mul{<:Any,<:Tuple{
    <:Mul{<:Any, <:Tuple{
        <:Adjoint{<:Any,<:AbstractVector},
        <:QuasiAdjoint{<:Any,<:FD}}},
    <:Mul{<:Any, <:Tuple{
        <:FD,
        <:AbstractVector}}}}

# * Mass matrix
function materialize(M::Mul{<:Any,<:Tuple{<:QuasiAdjoint{<:Any,<:FD},<:FD}}) where {T,FD<:AbstractFiniteDifferences{T}}
    Ac, B = M.args
    axes(Ac,2) == axes(B,1) || throw(DimensionMismatch("axes must be same"))
    A = parent(Ac)
    A == B || throw(ArgumentError("Cannot multiply functions on different grids"))
    Diagonal(ones(T, size(A,2)))
end

# * Norms

_norm(B::AbstractFiniteDifferences, c::AbstractArray, p::Real=2) =
    norm(c, p)*(step(B)^(inv(p)))

LinearAlgebra.norm(v::FDVecOrMat, p::Real=2) = _norm(v.applied.args..., p)
LinearAlgebra.norm(v::Mul{<:Any, <:Tuple{<:AbstractFiniteDifferences, <:AbstractArray}},
                   p::Real=2) = _norm(v.args..., p)

function LinearAlgebra.normalize!(v::FDVecOrMat, p::Real=2)
    v.applied.args[2][:] /= norm(v,p)
    v
end

function LinearAlgebra.normalize!(v::Mul{<:Any, <:Tuple{<:AbstractFiniteDifferences, <:AbstractArray}},
                                  p::Real=2)
    v.args[2][:] /= norm(v,p)
    v
end

# * Inner products

function _inner_product(a::Adjoint{<:Any,<:AbstractVector}, A::QuasiAdjoint{<:Any,<:FD},
                        B::FD, b::AbstractVector) where {FD<:AbstractFiniteDifferences}
    axes(A.parent) == axes(B) || throw(ArgumentError("Incompatible axes"))
    a*b*step(B)
end

LazyArrays.materialize(inner_product::FDInnerProduct{T,U,FD}) where {T,U,FD<:AbstractFiniteDifferences{U}} =
    _inner_product(inner_product.args...)

function LazyArrays.materialize(inner_product::LazyFDInnerProduct{FD}) where {FD<:AbstractFiniteDifferences}
    aA,Bb = inner_product.args
    _inner_product(aA.args..., Bb.args...)
end

function LazyArrays.materialize(s::Mul{<:Any, <:Tuple{
                <:Mul{<:Any, <:Tuple{
                    <:Adjoint{<:Any,<:AbstractVector},
                    <:ContinuumArrays.QuasiArrays.QuasiAdjoint{<:Any, <:FD}}},
                <:Mul{<:Any, <:Tuple{
                    <:FD,
                    <:Diagonal,
                    <:ContinuumArrays.QuasiArrays.QuasiAdjoint{<:Any, <:FD}}},
                <:Mul{<:Any, <:Tuple{
                    <:FD,
                    <:AbstractVector}}}}) where {FD<:AbstractFiniteDifferences}
    a,o,b = s.args
    axes(a.args[2].parent) == axes(o.args[1]) &&
        axes(o.args[3].parent) == axes(b.args[1]) ||
        throw(ArgumentError("Incompatible axes"))
    av = first(a.args)
    ov = o.args[2].diag
    bv = last(b.args)
    v = zero(promote_type(eltype(av),eltype(ov),eltype(bv)))
    @inbounds for i in eachindex(bv)
        v += av[i]*ov[i]*bv[i]
    end
    v*step(first(b.args))
end

# * Various finite differences
# ** Cartesian finite differences
struct FiniteDifferences{T,I} <: AbstractFiniteDifferences{T,I}
    j::UnitRange{I}
    Δx::T
end
FiniteDifferences(j::UnitRange{I}, Δx::T) where {T<:Real,I<:Integer} =
    FiniteDifferences{T,I}(j, Δx)

FiniteDifferences(n::I, Δx::T) where {T<:Real,I<:Integer} =
    FiniteDifferences(1:n, Δx)

locs(B::FiniteDifferences) = B.j*B.Δx
step(B::FiniteDifferences{T}) where {T} = B.Δx

IntervalSets.leftendpoint(B::FiniteDifferences) = (B.j[1]-1)*step(B)
IntervalSets.rightendpoint(B::FiniteDifferences) = (B.j[end]+1)*step(B)

show(io::IO, B::FiniteDifferences{T}) where {T} =
    write(io, "Finite differences basis {$T} on $(axes(B,1).domain) with $(size(B,2)) points spaced by Δx = $(B.Δx)")

# ** Radial finite differences
# Specialized finite differences for the case where there is
# Dirichlet0 boundary condition at r = 0.
struct RadialDifferences{T,I} <: AbstractFiniteDifferences{T,I}
    j::Base.OneTo{I}
    ρ::T
    Z::T
    δβ₁::T # Correction used for bare Coulomb potentials, Eq. (22) Schafer2009

    RadialDifferences(n::I, ρ::T, Z=one(T),
                      δβ₁=Z*ρ/8 * (one(T) + Z*ρ)) where {I<:Integer, T} =
        new{T,I}(Base.OneTo(n), ρ, convert(T,Z), convert(T,δβ₁))
end

"""
    RadialDifferences(rₘₐₓ::T, n::I, args...)

Convenience constructor for [`RadialDifferences`](@ref) covering the
open interval `(0,rₘₐₓ)` with `n` grid points.
"""
RadialDifferences(rₘₐₓ::T, n::I, args...) where {I<:Integer, T} =
    RadialDifferences(n, rₘₐₓ/(n+one(T)/2), args...)

locs(B::RadialDifferences{T}) where T = (B.j .- one(T)/2)*B.ρ
step(B::RadialDifferences{T}) where {T} = B.ρ

IntervalSets.leftendpoint(B::RadialDifferences{T}) where T = zero(T)
IntervalSets.rightendpoint(B::RadialDifferences{T}) where T = (B.j[end]+one(T)/2)*step(B)

==(A::RadialDifferences,B::RadialDifferences) = locs(A) == locs(B) && A.Z == B.Z && A.δβ₁ == B.δβ₁

show(io::IO, B::RadialDifferences{T}) where T =
    write(io, "Radial finite differences basis {$T} on $(axes(B,1).domain) with $(size(B,2)) points spaced by ρ = $(B.ρ)")


# ** Numerov finite differences
#=
This is an implementation of finite difference scheme described in

- Muller, H. G. (1999). An Efficient Propagation Scheme for the
  Time-Dependent Schrödinger equation in the Velocity Gauge. Laser
  Physics, 9(1), 138–148.

where the first derivative is approximated as

\[\partial f =
\left(1+\frac{h^2}{6}\Delta_2\right)^{-1}
\Delta_1 f
\equiv
M_1^{-1}\tilde{\Delta}_1 f,\]
where
\[M_1 \equiv
\frac{1}{6}
\begin{bmatrix}
4+\lambda' & 1 &\\
1 & 4 & 1\\
& 1 & 4 & \ddots\\
&&\ddots&\ddots\\
\end{bmatrix},\]
and
\[\tilde{\Delta}_1 \equiv
\frac{1}{2h}
\begin{bmatrix}
\lambda & 1 &\\
-1 &  & 1\\
& -1 &  & \ddots\\
&&\ddots&\ddots\\
\end{bmatrix},\]

where \(\lambda=\lambda'=\sqrt{3}-2\) for problems with a singularity
at the boundary \(r=0\) and zero otherwise; and the second derivative
as

\[\partial^2 f =
\left(1+\frac{h^2}{12}\Delta_2\right)^{-1}
\Delta_2 f
\equiv
-2M_2^{-1}\Delta_2 f,\]
where
\[M_2 \equiv
-\frac{1}{6}
\begin{bmatrix}
10-2\delta\beta_1 & 1 &\\
1 & 10 & 1\\
& 1 & 10 & \ddots\\
&&\ddots&\ddots\\
\end{bmatrix},\]
and
\[\Delta_2 \equiv
-\frac{1}{h^2}
\begin{bmatrix}
-2(1+\delta\beta_1) & 1 &\\
1 & -2 & 1\\
& 1 & -2 & \ddots\\
&&\ddots&\ddots\\
\end{bmatrix},\]

where, again, \(\delta\beta_1 = -Zh[12-10Zh]^{-1}\) is a correction
introduced for problems singular at the origin.

=#

struct NumerovFiniteDifferences{T,I} <: AbstractFiniteDifferences{T,I}
    j::UnitRange{I}
    Δx::T
    # Used for radial problems with a singularity at r = 0.
    λ::T
    δβ₁::T
end

function NumerovFiniteDifferences(j::UnitRange{I}, Δx::T, singular_origin::Bool=false, Z=zero(T)) where {T<:Real,I<:Integer}
    λ,δβ₁ = if singular_origin
        first(j) == 1 ||
            throw(ArgumentError("Singular origin correction only valid when grid starts at Δx (i.e. `j[1] == 1`)"))
        # Eqs. (20,17), Muller (1999)
        (√3 - 2),(-Z*Δx/(12 - 10Z*Δx))
    else
        zero(T),zero(T)
    end
    NumerovFiniteDifferences{T,I}(j, Δx, λ, δβ₁)
end

NumerovFiniteDifferences(n::I, Δx::T, args...) where {T<:Real,I<:Integer} =
    NumerovFiniteDifferences(1:n, Δx, args...)

locs(B::NumerovFiniteDifferences) = B.j*B.Δx
step(B::NumerovFiniteDifferences{T}) where {T} = B.Δx

IntervalSets.leftendpoint(B::NumerovFiniteDifferences) = (B.j[1]-1)*step(B)
IntervalSets.rightendpoint(B::NumerovFiniteDifferences) = (B.j[end]+1)*step(B)

show(io::IO, B::NumerovFiniteDifferences{T}) where {T} =
    write(io, "Numerov finite differences basis {$T} on $(axes(B,1).domain) with $(size(B,2)) points spaced by Δx = $(B.Δx)")

mutable struct NumerovDerivative{T,Tri,Mat,MatFact} <: AbstractMatrix{T}
    Δ::Tri
    M::Mat
    M⁻¹::MatFact
    c::T
end
NumerovDerivative(Δ::Tri, M::Mat, c::T) where {T,Tri,Mat} =
    NumerovDerivative(Δ, M, factorize(M), c)

Base.show(io::IO, ∂::ND) where {ND<:NumerovDerivative} =
    write(io, "$(size(∂,1))×$(size(∂,2)) $(ND)")

Base.show(io::IO, ::MIME"text/plain", ∂::ND) where {ND<:NumerovDerivative} =
    show(io, ∂)

Base.size(∂::ND, args...) where {ND<:NumerovDerivative} = size(∂.Δ, args...)
Base.eltype(∂::ND) where {ND<:NumerovDerivative} = eltype(∂.Δ)
Base.axes(∂::ND, args...) where {ND<:NumerovDerivative} = axes(∂.Δ, args...)

Base.:(*)(∂::ND,a::T) where {T<:Number,ND<:NumerovDerivative} =
    NumerovDerivative(∂.Δ, ∂.M, ∂.M⁻¹, ∂.c*a)

Base.:(*)(a::T,∂::ND) where {T<:Number,ND<:NumerovDerivative} =
    ∂ * a

Base.:(/)(∂::ND,a::T) where {T<:Number,ND<:NumerovDerivative} =
    ∂ * inv(a)

function LinearAlgebra.mul!(y::Y, ∂::ND, x::X) where {Y<:AbstractVector,
                                                      ND<:NumerovDerivative,
                                                      X<:AbstractVector}
    mul!(y, ∂.Δ, x)
    ldiv!(∂.M⁻¹, y)
    lmul!(∂.c, y)
    y
end

function Base.copyto!(y::Y, ∂::Mul{<:Any,Tuple{<:NumerovDerivative, X}}) where {X<:AbstractVector,Y<:AbstractVector}
    C = ∂.C
    mul!(C, ∂.A, ∂.B)
    lmul!(∂.α, C)
    C
end

for op in [:(+), :(-)]
    for Mat in [:Diagonal, :Tridiagonal, :SymTridiagonal, :UniformScaling]
        @eval begin
            function Base.$op(∂::ND, B::$Mat) where {ND<:NumerovDerivative}
                B̃ = inv(∂.c)*∂.M*B
                NumerovDerivative($op(∂.Δ, B̃), ∂.M, ∂.M⁻¹, ∂.c)
            end
        end
    end
end

struct NumerovFactorization{TriFact,Mat}
    Δ⁻¹::TriFact
    M::Mat
end

Base.size(∂⁻¹::NF, args...) where {NF<:NumerovFactorization} = size(∂⁻¹.M, args...)
Base.eltype(∂⁻¹::NF) where {NF<:NumerovFactorization} = eltype(∂⁻¹.M)

LinearAlgebra.factorize(∂::ND) where {ND<:NumerovDerivative} =
    NumerovFactorization(factorize(∂.c*∂.Δ), ∂.M)

function LinearAlgebra.ldiv!(y, ∂⁻¹::NF, x) where {NF<:NumerovFactorization}
    mul!(y, ∂⁻¹.M, x)
    ldiv!(∂⁻¹.Δ⁻¹, y)
    y
end

# * Scalar operators

Matrix(f::Function, B::AbstractFiniteDifferences{T}) where T = Diagonal(f.(locs(B)))
Matrix(::UniformScaling, B::AbstractFiniteDifferences{T}) where T = Diagonal(ones(T, size(B,2)))

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

# * Projections

dot(B::FD, f::Function) where {T,FD<:AbstractFiniteDifferences{T}} = f.(locs(B))
# Vandermonde interpolation for finite differences is equivalent to
# evaluating the function on the grid points, since the basis
# functions are orthogonal and there is no overlap between adjacent
# basis functions.
Base.:(\)(B::FD, f::Function) where {T,FD<:AbstractFiniteDifferences{T}} = dot(B, f)

# * Densities

function Base.Broadcast.broadcasted(::typeof(*), a::M, b::M) where {T,N,FD<:AbstractFiniteDifferences,M<:FDArray{T,N,FD}}
    axes(a) == axes(b) || throw(DimensionMismatch("Incompatible axes"))
    A,ca = a.applied.args
    B,cb = b.applied.args
    A == B || throw(DimensionMismatch("Incompatible bases"))
    # We want the first MulQuasiArray to be conjugated, if complex
    c = conj.(ca) .* cb
    A*c
end

struct FDDensity{T,B<:AbstractFiniteDifferences,
                 U<:AbstractVecOrMat{T},V<:AbstractVecOrMat{T}}
    R::B
    u::U
    v::V
end

function _FDDensity(Ra::AbstractFiniteDifferences, ca::AbstractVecOrMat,
                    Rb::AbstractFiniteDifferences, cb::AbstractVecOrMat)
    Ra == Rb || throw(DimensionMismatch("Incompatible bases"))
    FDDensity(Ra, ca, cb)
end

function Base.copyto!(cρ::AbstractVecOrMat{T}, ld::FDDensity{T,R}, Rρ::R) where {T,R}
    Rρ == ld.R || throw(DimensionMismatch("Incompatible bases"))
    size(cρ) == size(ld.u) || throw(DimensionMismatch("Incompatible sizes"))
    # We want the first MulQuasiArray to be conjugated, if complex
    cρ .= conj.(ld.u) .* ld.v
    cρ
end

function Base.Broadcast.broadcasted(::typeof(⋆), a::V₁, b::V₂) where {T,B<:AbstractFiniteDifferences,V₁<:FDVecOrMat{T,B},V₂<:FDVecOrMat{T,B}}
    axes(a) == axes(b) || throw(DimensionMismatch("Incompatible axes"))
    _FDDensity(a.applied.args..., b.applied.args...)
end

function Base.copyto!(ρ::FDVecOrMat{T,R}, ld::FDDensity{T,R}) where {T,R}
    copyto!(ρ.applied.args[2], ld, ρ.applied.args[1])
    ρ
end

function Base.Broadcast.broadcasted(::typeof(⋆), a::V₁, b::V₂) where {T,B<:AbstractFiniteDifferences,
                                                                      V₁<:Mul{<:Any, <:Tuple{B,<:AbstractVector{T}}},
                                                                      V₂<:Mul{<:Any, <:Tuple{B,<:AbstractVector{T}}}}
    axes(a) == axes(b) || throw(DimensionMismatch("Incompatible axes"))
    _FDDensity(a.args..., b.args...)
end

function Base.copyto!(ρ::Mul{<:Any, <:Tuple{R,<:AbstractVector{T}}}, ld::FDDensity{T,R}) where {T,R}
    copyto!(ρ.args[2], ld, ρ.args[1])
    ρ
end

# * Exports

export AbstractFiniteDifferences, FiniteDifferences, RadialDifferences, NumerovFiniteDifferences, Derivative, dot

end # module
