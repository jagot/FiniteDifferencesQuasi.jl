using FiniteDifferencesQuasi
import FiniteDifferencesQuasi: FirstDerivative, SecondDerivative, locs, FDDensity
using IntervalSets
using QuasiArrays
using ContinuumArrays
import ContinuumArrays: ℵ₁, materialize
import ContinuumArrays.QuasiArrays: AbstractQuasiArray,  AbstractQuasiMatrix, MulQuasiArray
using LinearAlgebra
using BandedMatrices
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

    @testset "Basis functions in restricted bases" begin
        N = 100
        L = 20.0
        δx = L/(N+1)

        B = FiniteDifferences((1:N) .- div(N,2), δx)
        xx = axes(B,1)
        a = xx.domain
        x = range(a.left, stop=a.right, length=max(N,101))
        χ = B[x,:]
        @test χ isa AbstractSparseMatrix

        sell = 1:round(Int, 0.6N)
        selv = round(Int, 0.4N):N

        Bl = B[:,sell]
        Bv = B[:,selv]

        χl = Bl[x, :]
        @test χl isa AbstractSparseMatrix
        @test χl == χ[:,sell]
        χv = Bv[x, :]
        @test χv == χ[:,selv]

        @test Bv[x, 4:8] == χ[:,selv[4:8]]
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

    @testset "Restricted bases" begin
        N = 10
        ρ = 1.0

        R = FiniteDifferences(N, ρ)
        r = axes(R, 1)

        sel = 3:6
        sel2 = 8:10

        R̃ = R[:, sel]
        R′ = R[:, sel2]

        x = QuasiDiagonal(r)

        apply_obj = applied(*, R', x, R)
        @test LazyArrays.ApplyStyle(*, typeof(R'), typeof(x), typeof(R)) ==
            FiniteDifferencesQuasi.FiniteDifferencesStyle()

        A = apply(*, R', x, R)
        @test A isa Diagonal

        a = apply(*, R̃', x, R)
        @test a isa BandedMatrix
        @test a == Matrix(A)[sel,:]
        @test bandwidths(a) == (-2,2)

        a = apply(*, R', x, R̃)
        @test a isa BandedMatrix
        @test a == Matrix(A)[:,sel]
        @test bandwidths(a) == (2,-2)

        a = apply(*, R̃', x, R̃)
        @test a isa Diagonal
        @test a == Matrix(A)[sel,sel]

        # Non-overlapping restrictions
        a = apply(*, R̃', x, R′)
        @test a isa BandedMatrix
        @test size(a) == (length(sel), length(sel2))
        @test iszero(a)
        @test bandwidths(a) == (5,-5)
    end
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

    @testset "Restricted bases" begin
        N = 100
        n = 10
        dr = 0.1

        R = RadialDifferences(N, dr)
        R̃ = R[:,1:n]

        f = 3

        ϕ = applied(*, R, f*ones(N))
        ϕ̃ = applied(*, R̃, f*ones(n))

        @test norm(ϕ) ≈ f*√(N*dr)
        @test norm(ϕ̃) ≈ f*√(n*dr)

        @test apply(*, ϕ', ϕ) == (f^2*N*dr)
        @test apply(*, ϕ̃', ϕ̃) == (f^2*n*dr)

        normalize!(ϕ)
        @test norm(ϕ) ≈ 1.0
        normalize!(ϕ̃)
        @test norm(ϕ̃) ≈ 1.0
    end
end

@testset "Derivatives" begin
    @testset "Normal finite differences" begin
        B = FiniteDifferences(5,1.0)
        D = Derivative(axes(B,1))

        ∇ = applied(*, B', D, B)
        ∇² = applied(*, B', D', D, B)

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

    @testset "Restricted bases" begin
        N = 10
        ρ = 1.0

        R = RadialDifferences(N, ρ)
        r = axes(R, 1)
        D = Derivative(r)

        ∇ = apply(*, R', D, R)
        ∇² = apply(*, R', D', D, R)

        for (sela,selb) in Iterators.product([1:10, 3:6, 8:10, 5:10, 4:5], [1:10, 3:6, 8:10, 5:10, 4:5])
            Ra = R[:,sela]
            Rb = R[:,selb]
            @test LazyArrays.ApplyStyle(*, typeof(Ra'), typeof(D), typeof(Rb)) ==
                FiniteDifferencesQuasi.FiniteDifferencesStyle()

            ∂ = apply(*, Ra', D, Rb)
            @test ∂ == Matrix(∇)[sela,selb]
            if sela == selb
                @test ∂ isa Tridiagonal
            else
                @test ∂ isa BandedMatrix
                @test length(bandrange(∂)) == 3
            end

            ∂² = apply(*, Ra', D', D, Rb)
            @test ∂² == Matrix(∇²)[sela,selb]
            if sela == selb
                @test ∂² isa SymTridiagonal
            else
                @test ∂² isa BandedMatrix
                @test length(bandrange(∂²)) == 3
            end
        end
    end

    @testset "Numerov finite differences" begin
        N = 10
        ρ = 0.3
        Z = 1.0
        @testset "Singular origin=$(singular_origin)" for singular_origin = [true, false]
            R = NumerovFiniteDifferences(1:N, ρ, singular_origin, Z)
            r = axes(R, 1)
            D = Derivative(r)

            λ,δβ₁ = if singular_origin
                √3 - 2, -Z*ρ*inv(12 - 10*Z*ρ)
            else
                0,0
            end

            ∂ = apply(*, R', D, R)
            @test ∂ isa FiniteDifferencesQuasi.NumerovDerivative

            @test ∂.Δ isa Tridiagonal
            @test ∂.Δ[1,1] ≈ λ/2ρ
            @test all(iszero, ∂.Δ.d[2:end])
            @test all(∂.Δ.dl .≈ -1/2ρ)
            @test all(∂.Δ.du .≈ 1/2ρ)

            @test ∂.M isa Tridiagonal
            @test ∂.M[1,1] ≈ (4+λ)/6
            @test all(∂.M.d[2:end] .≈ 4/6)
            @test all(∂.M.dl .≈ 1/6)
            @test all(∂.M.du .≈ 1/6)


            ∂² = apply(*, R', D', D, R)
            @test ∂ isa FiniteDifferencesQuasi.NumerovDerivative

            @test ∂².Δ isa SymTridiagonal
            @test ∂².Δ[1,1] ≈ -2*(1+δβ₁)/ρ^2
            @test all(∂².Δ.dv[2:end] .≈ -2/ρ^2)
            @test all(∂².Δ.ev .≈ 1/ρ^2)

            @test ∂².M isa SymTridiagonal
            @test ∂².M[1,1] ≈ -(10-2δβ₁)/6
            @test all(∂².M.dv[2:end] .≈ -10/6)
            @test all(∂².M.ev .≈ -1/6)
        end
    end
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

    @testset "Restricted bases" begin
        N = 100
        n = 10
        dr = 0.1

        R = RadialDifferences(N, dr)
        R̃ = R[:,1:n]

        f = 3

        ϕ = applied(*, R, f*ones(N))
        ϕ̃ = applied(*, R̃, f*ones(n))

        ρ = similar(ϕ)
        ρ̃ = similar(ϕ̃)

        @test ϕ .⋆ ϕ isa FiniteDifferencesQuasi.FDDensity
        @test ϕ̃ .⋆ ϕ̃ isa FiniteDifferencesQuasi.FDDensity

        copyto!(ρ, ϕ .⋆ ϕ)
        @test ρ.args[2] == f^2*ones(N)
        copyto!(ρ̃, ϕ̃ .⋆ ϕ̃)
        @test ρ̃.args[2] == f^2*ones(n)
    end
end

include("derivative_accuracy_utils.jl")

@testset "Derivative accuracy" begin
    Ns = 2 .^ (5:20)

    d = 1.0
    f,g,h,a,b = derivative_test_functions(d)

    @testset "kind = $B" for (order,B) in [(2,FiniteDifferences),
                                           (4,NumerovFiniteDifferences)]
        ϵg,ϵh,pg,ph = compute_derivative_errors(a, b, Ns, B, f, g, h)

        @test isapprox(pg, order, atol=0.03) || pg > order
        @test isapprox(ph, order, atol=0.03) || ph > order
    end
    @testset "kind = RadialDifferences" begin
        let (order,B) = (2,RadialDifferences)
            dd = b-a
            ϵg,ϵh,pg,ph = compute_derivative_errors(0, dd, Ns, B, x -> f(x-dd/2), x -> g(x-dd/2), x -> h(x-dd/2))

            @test isapprox(pg, order, atol=0.03) || pg > order
            @test isapprox(ph, order, atol=0.03) || ph > order
        end
    end
end


@testset "Particle in a box" begin
    @testset "Eigenvalues convergence rate" begin
        Ns = 2 .^ (7:14)
        nev = 3
        L = 1.0
        @testset "kind = $B" for (order,B) in [(2,FiniteDifferences),
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
        ∇² = apply(*, R', D', D, R)
        Tm = ∇² / -2
        @test Tm isa SymTridiagonal

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

        @testset "kind = $B" for (order,B) in [(2,RadialDifferences),
                                               (4,NumerovFiniteDifferences)]
            ϵλ,slopes,elapsed = compute_diagonalization_errors(test_singular_fd_scheme,
                                                               B, Ns, rₘₐₓ, Z, ℓ, nev)
            for p in slopes
                @test isapprox(p, order, atol=0.04) || p > order
            end
        end
    end
end
