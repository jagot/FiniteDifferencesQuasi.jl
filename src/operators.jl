# * Scalar operators

@simplify function *(Ac::QuasiAdjoint{<:Any,<:AbstractFiniteDifferences},
                     D::QuasiDiagonal,
                     B::AbstractFiniteDifferences)
    A = parent(Ac)
    A == B || throw(ArgumentError("Cannot multiply functions on different grids"))

    Diagonal(getindex.(Ref(D.diag), locs(B)))
end
