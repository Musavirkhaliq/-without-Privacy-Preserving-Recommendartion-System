
module Elbmf

using Random, LinearAlgebra, CUDA
using CUDA.CUSPARSE
using CUDA: norm
using Random


integrality_gap_elastic(e, kappa, lambda) = min(kappa * abs(e) + lambda * e^2, kappa * abs(e - 1) + lambda * (e - 1)^2)
regularizer_elbmf(x, l1reg, l2reg)        = sum(e -> integrality_gap_elastic(e, l1reg, l2reg), x)
proxel_1(x, k, l)                         = (x <= 0.5 ? (x - k * sign(x)) : (x - k * sign(x - 1) + l)) / (1 + l)
proxelp(x, k, l)                          = max(proxel_1(x, k, l), zero(x))
prox_elbmf!(X, k, l)                      = X .= proxelp.(X, k, l)
proxelb(x, k, l)                          = clamp(proxel_1(x, k, l), zero(x), one(x))
prox_elbmf_box!(X, k, l)                  = X .= proxelb.(X, k, l)

mutable struct ElasticBMF{T}
    l1reg::T
    l2reg::T
end

struct PALM end
struct iPALM{T} beta::T end

rounding!(fn::ElasticBMF, _, args...) = foreach(X -> X .= round.(clamp.(prox_elbmf!(X, 0.5, 1e20), 0, 1)), args)
rounding!(fn, _...)                   = nothing

apply_rate!(fn::ElasticBMF, fn0, nu) = fn.l2reg = fn0.l2reg * nu
apply_rate!(fn, _...)                = nothing

gpu(X) = CUDA.adapt(CuArray, X)
cpu(X) = CUDA.adapt(Array, X)

function reducemf_impl!(prox!, ::Type{PALM},  A, U, V)
    VVt, AVt = V * V', A * V'
    grad(x) = x * VVt .- AVt
    L = max(norm(VVt), 1e-4)

    step_size = 1 / (1.1 * L)
    U .= U .- grad(U) * step_size
    prox!(U, step_size)
end
function reducemf_impl!(prox!, opt::iPALM,  A, U, V, U_)
    VVt, AVt = V * V', A * V'
    grad(x) = x * VVt .- AVt
    L = max(norm(VVt), 1e-4)

    @. U = U + opt.beta * (U - U_)
    @. U_ = U

    step_size = 2 * (1 - opt.beta) / (1 + 2 * opt.beta) / L
    U .= U .- grad(U) * step_size
    prox!(U, step_size)
end

reducemf!(fn::ElasticBMF, opt::Type{PALM}, A, U, V)      = reducemf_impl!((x, alpha) -> prox_elbmf!(x, fn.l1reg * alpha, fn.l2reg * alpha), opt, A, U, V) 
reducemf!(fn::ElasticBMF, opt::iPALM, A, U, V, U_)       = reducemf_impl!((x, alpha) -> prox_elbmf!(x, fn.l1reg * alpha, fn.l2reg * alpha), opt, A, U, V, U_) 

function factorize_palm!(
    fn::ElasticBMF,
    X,
    U,
    V,
    regularization_rate,
    maxiter,
    tol;
    callback = nothing
)
    ell = typemax(tol)
    fn0 = deepcopy(fn)

    for t = 1:maxiter
        fn.l2reg = fn0.l2reg * regularization_rate(t - 1)

        reducemf!(fn, PALM, X, U, V)
        reducemf!(fn, PALM, X', V', U')

        ell, ell0 = norm(X .- U * V)^2, ell

        (callback !== nothing) && callback((U, V), ell)
        (abs(ell - ell0) < tol) && break
    end
    fn = fn0
    U, V
end

function factorize_ipalm!(
    fn::ElasticBMF,
    X,
    U,
    V,
    regularization_rate,
    maxiter,
    tol,
    beta;
    callback = nothing,
)
    if beta == 0
        return factorize_palm!(fn, X, U, V, regularization_rate, maxiter, tol; callback=callback)
    end

    ell = typemax(tol)
    fn0 = deepcopy(fn)

    ipalm = iPALM(beta)
    U_    = copy(U)
    Vt_   = copy(V')

    for t = 1:maxiter
        fn.l2reg = fn0.l2reg * regularization_rate(t - 1)

        reducemf!(fn, ipalm, X, U, V, U_)
        reducemf!(fn, ipalm, X', V', U', Vt_)

        ell, ell0 = norm(X .- U * V)^2, ell

        (callback !== nothing) && callback((U, V), ell)
        (abs(ell - ell0) < tol) && break
    end
    fn = fn0
    U, V
end

# you can achieve potentially better performance by using SGD
function batched_factorize_ipalm!(
    fn::ElasticBMF,
    X,
    U,  
    V,
    regularization_rate,
    maxiter,
    tolerance,
    beta,
    batchsize;
    callback = nothing
)
    ell = typemax(tolerance)
    fn0 = deepcopy(fn)

    ipalm = iPALM(beta)
    U_    = beta == 0 ? nothing : copy(U)        
    H     = gpu(V)
    Ht_   = copy(H')

    for t = 1:maxiter
        ell, ell0 = 0, ell

        fn.l2reg = fn0.l2reg * regularization_rate(t - 1)

        foreach(Iterators.partition(1:size(X, 1), batchsize)) do batch
            A, W = gpu(view(X, batch, :)), gpu(view(U, batch, :))
            
            if !isnothing(U_)
                W_ = gpu(view(U_, batch, :))
                reducemf!(fn, ipalm, A, W, H, W_)
                reducemf!(fn, ipalm, A', H', W', Ht_)
                view(U_, batch, :) .= cpu(W_)
            else
                reducemf!(fn, PALM, A, W, H)
                reducemf!(fn, PALM, A', H', W')
            end
            ell += norm(A .- W * H)^2
            view(U, batch, :) .= cpu(W)
        end
        (callback !== nothing) && callback((U, H), fn)
        (abs(ell - ell0) < tolerance) && break
    end
    V .= cpu(H)
    fn = fn0
    U, V
end

function init_factorization(A, k; T = eltype(A))
    n, m = size(A)
    if isa(A, CUDA.CuArray) || isa(A, CUDA.CUSPARSE.CuSparseMatrix)
        CUDA.rand(T, n, k), CUDA.rand(T, k, m)
    else
        rand(T, n, k), rand(T, k, m)
    end
end

function elbmf(
    A,
    ncomponents,
    l1reg,
    l2reg,
    regularization_rate, # = t -> c^t 
    maxiter,
    tolerance,
    beta                 = 0.0, # inertial disabled by default
    batchsize            = size(A, 1),
    with_rounding        = true;
    callback             = nothing,
    args...,
)
    U, V = init_factorization(A, ncomponents; T = (A isa BitArray || eltype(A) == Bool) ? Float64 : eltype(A))

    fn   = ElasticBMF(l1reg, l2reg)
    U, V = if batchsize >= size(A, 1)
        factorize_ipalm!(fn, A, U, V, regularization_rate, maxiter, tolerance, beta; callback=callback)
    elseif CUDA.has_cuda_gpu()
        batched_factorize_ipalm!(fn, A, U, V, regularization_rate, maxiter, tolerance, beta, batchsize; callback=callback)
    else
        throw(ErrorException("Error: minibatches and require CUDA, which is unavailable"))
    end

    with_rounding && rounding!(fn, A, U, V)

    U, V
end

export elbmf

end
