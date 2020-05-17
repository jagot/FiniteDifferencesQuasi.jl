module FiniteDifferencesQuasi

function __init__()
    @warn "The FiniteDifferencesQuasi.jl package has been deprecated in favour of JuliaApproximation/CompactBases.jl"
    nothing
end

import Base: eltype, axes, size, ==, getindex, checkbounds, copyto!, similar, show, step
import Base.Broadcast: materialize

using ContinuumArrays
import ContinuumArrays: Basis, ℵ₁, Derivative, Inclusion, @simplify

using QuasiArrays
import QuasiArrays: AbstractQuasiMatrix, QuasiAdjoint, MulQuasiArray,
    PInvQuasiMatrix, InvQuasiMatrix, QuasiDiagonal,
    BroadcastQuasiArray, SubQuasiArray

using IntervalSets

using LazyArrays

using LinearAlgebra
import LinearAlgebra: Matrix, dot
using SparseArrays
using BandedMatrices
using FillArrays

using Printf

include("restricted_bases.jl")
include("finite_differences.jl")
include("inner_products.jl")
include("operators.jl")
include("derivatives.jl")
include("densities.jl")

export AbstractFiniteDifferences, FiniteDifferences,
    RadialDifferences, StaggeredFiniteDifferences,
    NumerovFiniteDifferences,
    Derivative, dot, QuasiDiagonal

end # module
