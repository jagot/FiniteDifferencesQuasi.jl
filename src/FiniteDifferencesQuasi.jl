module FiniteDifferencesQuasi

import Base: eltype, axes, size, ==, getindex, checkbounds, copyto!, similar, show, step
import Base.Broadcast: materialize

using ContinuumArrays
import ContinuumArrays: Basis, ℵ₁, Derivative, Inclusion, @simplify

using QuasiArrays
import QuasiArrays: AbstractQuasiMatrix, QuasiAdjoint, MulQuasiArray,
    PInvQuasiMatrix, InvQuasiMatrix, QuasiDiagonal,
    BroadcastQuasiArray

using IntervalSets

using LazyArrays

using LinearAlgebra
import LinearAlgebra: Matrix, dot
using SparseArrays

using Printf

include("finite_differences.jl")
include("inner_products.jl")
include("operators.jl")
include("derivatives.jl")
include("densities.jl")

export AbstractFiniteDifferences, FiniteDifferences, RadialDifferences, NumerovFiniteDifferences,
    Derivative, dot, QuasiDiagonal

end # module
