using FiniteDifferencesQuasi
import FiniteDifferencesQuasi: FirstDerivative, SecondDerivative, locs
using IntervalSets
using ContinuumArrays
import ContinuumArrays: ℵ₁, materialize
using LinearAlgebra
using LazyArrays
import LazyArrays: ⋆
using Test

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
end
