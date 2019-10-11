using FiniteDifferencesQuasi
import FiniteDifferencesQuasi: FirstDerivative, SecondDerivative, locs, FDDensity
using IntervalSets
using QuasiArrays
using ContinuumArrays
import ContinuumArrays: ℵ₁, materialize
import ContinuumArrays.QuasiArrays: AbstractQuasiArray,  AbstractQuasiMatrix, MulQuasiArray
using LinearAlgebra
using SparseArrays
using LazyArrays
import LazyArrays: ⋆
using Test

@testset "Basis functions" begin
    rₘₐₓ = 10
    ρ = rand()
    N = ceil(Int, rₘₐₓ/ρ + 1/2)

    for B in [RadialDifferences(N, ρ), FiniteDifferences(N, 1.0)]
        x = locs(B)
        χ = B[x, :]
        @test χ isa AbstractSparseMatrix
        @test χ ≈ Diagonal(ones(N))
    end
end

@testset "Mass matrices and inverses" begin
    R = FiniteDifferences(20,0.2)
    @testset "Mass matrix" begin
        @test R'R == step(R)*I
    end
    @testset "Inverses" begin
        R⁻¹ = pinv(R)
        @test R⁻¹*R == I
        @test R*R⁻¹ == I

        cu = rand(size(R,2))
        cv = rand(size(R,2))
        cuv = [cu cv]

        u = R*cu
        v = R*cv
        uv = R*cuv

        @test R⁻¹*u === cu
        @test R⁻¹*v === cv
        @test R⁻¹*uv === cuv

        ut = u'
        # Fails with: ERROR: MethodError: no method matching axes(::UniformScaling{Float64}, ::Int64)
        # @test ut*R⁻¹' === ut.args[1]
    end
end

@testset "Scalar operators" begin
    B = FiniteDifferences(5,1.0)
    x = axes(B,1)
    x² = B'QuasiDiagonal(x.^2)*B
    @test x² isa Diagonal

    v = ones(5)
    @test x²*v == locs(B).^2

    y = Inclusion(0.0..5.0)
    @test_throws DimensionMismatch B'QuasiDiagonal(y.^2)*B
end

@testset "Inner products" begin
    rₘₐₓ = 300
    ρ = rand()
    N = ceil(Int, rₘₐₓ/ρ + 1/2)

    for B in [RadialDifferences(N, ρ), FiniteDifferences(N, 1.0)]
        S = B'B
        for T in [Float64,ComplexF64]
            vv = rand(T, size(B,2))
            v = B*vv
            lv = B ⋆ vv
            normalize!(v)

            @test norm(v) ≈ 1.0
            @test applied(*, v'.args..., v.args...) isa FiniteDifferencesQuasi.FDInnerProduct # {T,Float64,RadialDifferences{Float64,Int}}
            @test v'v ≈ 1.0
            @test vv'S*vv ≈ 1.0

            lazyip = lv' ⋆ lv

            @test lazyip isa FiniteDifferencesQuasi.LazyFDInnerProduct
            @test materialize(lazyip) ≈ 1.0
        end
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

@testset "Function interpolation" begin
    R = FiniteDifferences(20,1.0)
    R⁻¹ = pinv(R)
    r̃ = locs(R)
    χ = R[r̃,:]

    r = axes(R,1)

    y = Inclusion(0.0..5.0)
    @test_throws DimensionMismatch R \ y.^2

    fu = r -> r^2*exp(-r)
    u = R*(R\fu.(r))
    @test norm(χ * (R⁻¹*u) - fu.(r̃)) == 0

    fv = r -> r^6*exp(-r)
    v = R*(R\fv.(r))
    @test norm(χ * (R⁻¹*v) - fv.(r̃)) == 0
end

@testset "Densities" begin
    R = FiniteDifferences(20,1.0)
    R⁻¹ = pinv(R)
    r̃ = locs(R)
    χ = R[r̃,:]

    r = axes(R,1)

    fu = r -> r^2*exp(-r)
    u = R*(R\fu.(r))

    fv = r -> r^6*exp(-r)
    v = R*(R\fv.(r))

    w = u .* v
    fw = r -> fu(r)*fv(r)

    @test norm(χ * (R⁻¹*w) - fw.(r̃)) == 0

    y = R*rand(ComplexF64, size(R,2))
    y² = y .* y
    @test all(isreal.(R⁻¹*y²))
    @test all(R⁻¹*y² .== abs2.(R⁻¹*y))

    @testset "Lazy densities" begin
        uv = u .⋆ v
        @test uv isa FDDensity

        w′ = similar(u)
        copyto!(w′, uv)
        @test norm(χ * (R⁻¹*w′) - fw.(r̃)) == 0

        uu = R*repeat(R⁻¹*u,1,2)
        vv = R*repeat(R⁻¹*v,1,2)
        uuvv = uu .⋆ vv
        ww′ = similar(uu)
        copyto!(ww′, uuvv)

        @test norm(χ * (R⁻¹*ww′) .- fw.(r̃)) == 0

        yy = y .⋆ y
        @test yy isa FDDensity
        wy = similar(y)
        copyto!(wy, yy)
        @test all(isreal.(R⁻¹*wy))
        @test all(R⁻¹*wy .== abs2.(R⁻¹*y))
    end
end

include("derivative_accuracy_utils.jl")

@testset "Derivative accuracy" begin
    Ns = 2 .^ (5:20)

    d = 1.0
    f,g,h,a,b = derivative_test_functions(d)

    for (order,B) in [(2,FiniteDifferences),
                      (4,NumerovFiniteDifferences)]
        ϵg,ϵh,pg,ph = compute_derivative_errors(a, b, Ns, B, f, g, h)

        @test isapprox(pg, order, atol=0.03) || pg > order
        @test isapprox(ph, order, atol=0.03) || ph > order
    end
    let (order,B) = (2,RadialDifferences)
        dd = b-a
        ϵg,ϵh,pg,ph = compute_derivative_errors(0, dd, Ns, B, x -> f(x-dd/2), x -> g(x-dd/2), x -> h(x-dd/2))

        @test isapprox(pg, order, atol=0.03) || pg > order
        @test isapprox(ph, order, atol=0.03) || ph > order
    end
end


@testset "Particle in a box" begin
    @testset "Eigenvalues convergence rate" begin
        Ns = 2 .^ (7:14)
        nev = 3
        L = 1.0
        for (order,B) in [(2,FiniteDifferences),
                          (4,NumerovFiniteDifferences)]
            ϵλ,slopes,elapsed = compute_diagonalization_errors(test_fd_particle_in_a_box,
                                                               B, Ns, L, nev)
            for p in slopes
                @test isapprox(p, order, atol=0.1) || p > order
            end
        end
    end
end

# This tests the discretization of the Laplacian, especially near
# the origin where the potential term is singular.
@testset "Hydrogen bound states" begin
    @testset "Eigenvalues and eigenvectors" begin
        rₘₐₓ = 300
        ρ = 0.25
        N = ceil(Int, rₘₐₓ/ρ + 1/2)

        R = RadialDifferences(N, ρ)
        r = axes(R, 1)

        D = Derivative(Base.axes(R,1))
        ∇² = R'⋆D'⋆D⋆R
        Tm = materialize(∇²)
        Tm /= -2

        V = R'QuasiDiagonal(-inv.(r))*R

        H = Tm + V

        ee = eigen(H)

        n = 1:10
        λₐ = -inv.(2n.^2)

        abs_error = ee.values[n] - λₐ
        rel_error = abs_error ./ abs.(1e-10 .+ abs.(λₐ))

        @test abs_error[1] < 3e-5
        @test all(rel_error .< 1e-3)

        r̃ = locs(R)
        # Table 2.2 Foot (2005)
        Rₐ₀ = [2exp.(-r̃),
               -(1/2)^1.5 * 2 * (1 .- r̃/2).*exp.(-r̃/2), # Why the minus?
               (1/3)^1.5 * 2 * (1 .- 2r̃/3 .+ 2/3*(r̃/3).^2).*exp.(-r̃/3)]
        expected_errors = [1e-3,1e-3,2e-3]

        for i = 1:3
            v = ee.vectors[:,i]
            N = norm(v)*√ρ
            # The sign from the diagonalization is arbitrary; make max lobe positive
            N *= sign(v[argmax(abs.(v))])
            abs_error = v/N .- r̃.*Rₐ₀[i]
            @test norm(abs_error)/abs(1e-10+norm(r̃.*Rₐ₀[i])) < expected_errors[i]
        end
    end

    @testset "Eigenvalues convergence rate" begin
        Ns = 2 .^ (7:14)
        nev = 3
        rₘₐₓ = 100.0
        Z = 1.0
        ℓ = 0

        for (order,B) in [(2,RadialDifferences),
                          (4,NumerovFiniteDifferences)]
            ϵλ,slopes,elapsed = compute_diagonalization_errors(test_singular_fd_scheme,
                                                               B, Ns, rₘₐₓ, Z, ℓ, nev)
            for p in slopes
                @test isapprox(p, order, atol=0.04) || p > order
            end
        end
    end
end
