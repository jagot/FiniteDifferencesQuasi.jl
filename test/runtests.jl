using FiniteDifferencesQuasi
import FiniteDifferencesQuasi: FirstDerivative, SecondDerivative, locs, FDDensity
using IntervalSets
using ContinuumArrays
import ContinuumArrays: ℵ₁, materialize
using LinearAlgebra
using LazyArrays
import LazyArrays: ⋆
using Test

@testset "Scalar operators" begin
    B = FiniteDifferences(5,1.0)
    x² = Matrix(x -> x^2, B)
    @test x² isa Diagonal

    v = ones(5)
    @test x²*v == locs(B).^2
end

@testset "Inner products" begin
    rₘₐₓ = 300
    ρ = rand()
    N = ceil(Int, rₘₐₓ/ρ + 1/2)
    R = RadialDifferences(N, ρ)
    B = FiniteDifferences(N, 1.0)

    for T in [Float64,ComplexF64]
        v = R*rand(T, size(R,2))
        normalize!(v)

        @test norm(v) ≈ 1.0
        @test v'⋆v isa FiniteDifferencesQuasi.FDInnerProduct{T,Float64,RadialDifferences{Float64,Int}}
        @test v'v ≈ 1.0

        w = B*rand(T, size(B,2))
        normalize!(w)

        @test norm(w) ≈ 1.0
        @test w'⋆w isa FiniteDifferencesQuasi.FDInnerProduct{T,Float64,FiniteDifferences{Float64,Int}}
        @test w'w ≈ 1.0
    end
end

@testset "Derivatives" begin
    B = FiniteDifferences(5,1.0)
    D = Derivative(axes(B,1))

    ∇ = B'⋆D⋆B
    ∇² = B'⋆D'⋆D⋆B

    @test ∇ isa FirstDerivative
    @test ∇² isa SecondDerivative

    ∂ = materialize(∇)
    T = materialize(∇²)

    @test ∂ isa Tridiagonal
    @test T isa SymTridiagonal

    @test all(diag(∂) .== 0)
    @test all(diag(∂,1) .== 0.5)
    @test all(diag(∂,-1) .== -0.5)

    @test all(diag(T) .== -2)
    @test all(diag(T,1) .== 1)
    @test all(diag(T,-1) .== 1)
end

@testset "Projections" begin
    R = FiniteDifferences(20,1.0)
    r = locs(R)
    χ = R*R[r,:]'

    fu = r -> r^2*exp(-r)
    u = R*dot(R, fu)
    @test norm(χ'u - fu.(r)) == 0
    fv = r -> r^6*exp(-r)
    v = R*dot(R, fv)
    @test norm(χ'v - fv.(r)) == 0
end

@testset "Densities" begin
    R = FiniteDifferences(20,1.0)
    r = locs(R)
    χ = R*R[r,:]'

    fu = r -> r^2*exp(-r)
    u = R*dot(R, fu)

    fv = r -> r^6*exp(-r)
    v = R*dot(R, fv)

    w = u .* v
    fw = r -> fu(r)*fv(r)

    @test norm(χ'w - fw.(r)) == 0

    @testset "Lazy densities" begin
        uv = u .⋆ v
        @test uv isa FDDensity

        w′ = similar(u)
        copyto!(w′, uv)
        @test norm(χ'w′ - fw.(r)) == 0

        uu = R*repeat(u.mul.factors[2],1,2)
        vv = R*repeat(v.mul.factors[2],1,2)
        uuvv = uu .⋆ vv
        ww′ = similar(uu)
        copyto!(ww′, uuvv)

        @test norm(χ'ww′ .- fw.(r)) == 0
    end
end

# This tests that the discretization of the Laplacian, especially near
# the origin where the potential term is singular.
@testset "Hydrogen bound states" begin
    rₘₐₓ = 300
    ρ = 0.25
    N = ceil(Int, rₘₐₓ/ρ + 1/2)

    R = RadialDifferences(N, ρ)

    D = Derivative(Base.axes(R,1))
    ∇² = R'⋆D'⋆D⋆R
    Tm = materialize(∇²)
    Tm /= -2

    V = Matrix(r -> -1/r, R)

    H = Tm + V

    ee = eigen(H)

    n = 1:10
    λₐ = -inv.(2n.^2)

    abs_error = ee.values[n] - λₐ
    rel_error = abs_error ./ abs.(1e-10 .+ abs.(λₐ))

    @test abs_error[1] < 3e-5
    @test all(rel_error .< 1e-3)

    r = locs(R)
    # Table 2.2 Foot (2005)
    Rₐ₀ = [2exp.(-r),
           -(1/2)^1.5 * 2 * (1 .- r/2).*exp.(-r/2), # Why the minus?
           (1/3)^1.5 * 2 * (1 .- 2r/3 .+ 2/3*(r/3).^2).*exp.(-r/3)]
    expected_errors = [1e-3,1e-3,2e-3]

    for i = 1:3
        v = ee.vectors[:,i]
        N = norm(v)*√ρ
        # The sign from the diagonalization is arbitrary; make max lobe positive
        N *= sign(v[argmax(abs.(v))])
        abs_error = v/N .- r.*Rₐ₀[i]
        @test norm(abs_error)/abs(1e-10+norm(r.*Rₐ₀[i])) < expected_errors[i]
    end
end
