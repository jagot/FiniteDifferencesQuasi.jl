using ArnoldiMethod

function test_fd_derivatives(a, b, N, ::Type{B}, f::Function, g::Function, h::Function) where {B<:AbstractQuasiMatrix}
    L = b-a
    Δx = L/(N+1)
    j = (1:N) .+ round(Int, a/Δx)

    R = B(j, Δx)
    D = Derivative(axes(R,1))

    ∇ = R'D*R
    ∇² = R'D'D*R

    r = FiniteDifferencesQuasi.locs(R)

    fv = f.(r)
    gv = similar(fv)
    hv = similar(fv)

    mul!(gv, ∇, fv)
    mul!(hv, ∇², fv)

    δg = gv-g.(r)
    δh = hv-h.(r)

    r,fv,gv,hv,δg,δh,step(R)
end

function error_slope(loghs,ϵ)
    # To avoid the effect of round-off errors on the order
    # estimation.
    i = argmin(abs.(ϵ)) - 1
    println(i)
    
    ([loghs[1:i] ones(i)] \ log10.(abs.(ϵ[1:i])))[1]
end

function compute_derivative_errors(a, b, Ns, ::Type{B}, f::Function, g::Function, h::Function) where {B<:AbstractQuasiMatrix}
    errors = map(Ns) do N
        r,fv,gv,hv,δg,δh,Δx = test_fd_derivatives(a, b, N, B, f, g, h)

        [norm(δg)*Δx norm(δh)*Δx]
    end |> e -> vcat(e...)

    ϵg = errors[:,1]
    ϵh = errors[:,2]

    loghs = log10.(1.0 ./ Ns)
    pg = error_slope(loghs, ϵg)
    ph = error_slope(loghs, ϵh)

    ϵg,ϵh,pg,ph
end

function norm_rot!(v)
    normalize!(v)
    vc = v.mul.factors[2]
    vc[:] *= sign(vc[1])
    v
end

struct ShiftInvert{M}
    A⁻¹::M
end

Base.size(S::ShiftInvert, args...) = size(S.A⁻¹, args...)
Base.eltype(S::ShiftInvert) = eltype(S.A⁻¹)

LinearAlgebra.mul!(y, S::ShiftInvert, x) =
    ldiv!(y, S.A⁻¹, x)

function diagonalize_fd_hamiltonian(H, R::B, nev, σ; method=:arnoldi_shift_invert) where {B<:AbstractQuasiMatrix}
    A,target = if method == :arnoldi_shift_invert
        ShiftInvert(factorize(H - σ*I)),LR()
    else
        H, SR()
    end
    schur,history = partialschur(A, nev=nev, which=target)
    println(history)
    
    ϕ = [norm_rot!(R*schur.Q[:,j]) for j = 1:nev]
    
    θ = diag(schur.R)
    λ = if method == :arnoldi_shift_invert
        σ .+ inv.(θ)
    else
        θ
    end
    
    λ,ϕ
end

function get_kinetic_operator(R::B) where {B<:AbstractQuasiMatrix}
    D = Derivative(Base.axes(R,1))
    Tm = R'D'D*R
    Tm /= -2
end

function test_fd_particle_in_a_box(::Type{B}, N, L, nev; kwargs...) where {B<:AbstractQuasiMatrix}
    R = B(1:N, L/(N+1))
    
    Tm = get_kinetic_operator(R)
    λ,ϕ = diagonalize_fd_hamiltonian(Tm, R, nev, 0.0; kwargs...)
    
    r = FiniteDifferencesQuasi.locs(R)
    
    n = 1:nev
    δλ = λ - n.^2*π^2/(2L^2)

    # Could/should also test eigenvectors
    
    λ,ϕ,r,R,δλ
end

function test_singular_fd_scheme(::Type{B}, N, rₘₐₓ, Z, ℓ, nev; kwargs...) where {B<:AbstractQuasiMatrix}
    R = if B == RadialDifferences
        ρ = rₘₐₓ/(N-0.5)
        RadialDifferences(N, ρ, Z)
    else
        ρ = rₘₐₓ/(N+1)
        NumerovFiniteDifferences(1:N, ρ, true, Z)
    end
    display(R)
    println()

    n = 1:nev
    λₐ = -inv.(2(n .+ ℓ).^2)

    Tm = get_kinetic_operator(R)    
    V = Matrix(r -> -1/r + ℓ*(ℓ+1)/2r^2, R)
    H = Tm + V
    λ,ϕ = diagonalize_fd_hamiltonian(H, R, nev, 1.1λₐ[1]; kwargs...)

    r = FiniteDifferencesQuasi.locs(R)

    δλ = λ - λₐ

    # Could/should also test eigenvectors
    
    λ,ϕ,r,R,δλ
end

function compute_diagonalization_errors(f::Function, ::Type{B}, Ns, args...; kwargs...) where B
    errors = map(Ns) do N
        println("N = $N")
        t = time()
        λ,ϕ,r,R,δλ = f(B, N, args...; kwargs...)
        elapsed = time()-t
        vcat(δλ,elapsed)'
    end |> e -> vcat(e...)

    errors,elapsed = errors[:,1:end-1],errors[:,end]

    loghs = log10.(inv.(Ns))
    slopes = map(1:size(errors,2)) do j
        error_slope(loghs, errors[:,j])
    end
    
    errors,slopes,elapsed    
end
