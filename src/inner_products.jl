# * Norms

_norm(B::AbstractFiniteDifferences, c::AbstractArray, p::Real=2) =
    norm(c, p)*(step(B)^(inv(p)))

LinearAlgebra.norm(v::FDVecOrMat, p::Real=2) = _norm(v.args..., p)
LinearAlgebra.norm(v::Mul{<:Any, <:Tuple{<:AbstractFiniteDifferences, <:AbstractArray}},
                   p::Real=2) = _norm(v.args..., p)

function LinearAlgebra.normalize!(v::FDVecOrMat, p::Real=2)
    v.args[2][:] /= norm(v,p)
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
