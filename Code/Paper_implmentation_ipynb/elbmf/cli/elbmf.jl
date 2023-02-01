using ArgParse, Elbmf, DelimitedFiles, LinearAlgebra, CUDA

gridsearch(loss, U, V; steps = 200) =
    mapreduce(
        (s, t) -> (s[3] <= t[3]) ? s : t,
        Base.Iterators.product(range(1 - 1e-5, 1e-5, steps), range(1 - 1e-5, 1e-5, steps));
        init = (0, 0, Inf),
    ) do (x, y)
        x, y, loss(U .>= x, V .>= y)
    end
function gridsearch!(X, U, V; steps = 200, objective = (u, v) -> norm(X .- ((u * v) .!= 0))^2)
    U .= clamp.(U, 0, 1)
    V .= clamp.(V, 0, 1)
    th_u, th_v, _ = gridsearch(objective, U, V; steps = steps)
    U .= U .>= th_u
    V .= V .>= th_v
    nothing
end

# This specialization has been missing in the CUDA.jl in version that I've used
# LinearAlgebra.norm(A::Adjoint{T, CuArray{T, 2, CUDA.Mem.DeviceBuffer}}, p::Core.Real=2) where T = LinearAlgebra.norm(A', p)

function main()
    s = ArgParseSettings(description = "Elbmf CLI")
    @add_arg_table! s begin
        "-i", "--input-data"
            help = "Boolean input matrix in tsv format"
            required = true
        "-o", "--output-factors"
            help = "The output factor matrices (as '<output-factors-argument>'.{lhs,rhs}.tsv) in tsv"
            required = true
        "--dont-round-on-convergence"
            action = :store_true
            help = "Enable to prevent elbmf to round factor matrices upon convergence"
        "--gridsearch"
            action = :store_true
            help = "Elbmf DOES NOT use this argument. This is a last-resort option if it is complicated to tune hyperparameters (see Readme.md)"
        "--cuda"
            action = :store_true
            help = "Enables CUDA support, which is assumed to be true if '--batchsize' is set"
        "-k", "--ncomponents"
            arg_type = Int
            required = true
            help = "The number of components (rank) of the factorization"
        "--maxiter"
            arg_type = Int
            required = true
            help = "The maximum number of iterations"
        "--inertial"
            arg_type = Float64
            help = "The inertial coefficient 'beta' for iPALM"
            default = 0.0
            required = false
        "--batchsize"
            arg_type = Int64
            required = false
            help = "Compute gradient in minibatches (if the matrices do not fit into the GPU memory (i.e., if you observe an CUDA Memory error)"
        "--l1reg"
            arg_type = Float64
            required = true
            help = "l1 regularization coefficient"
        "--l2reg"
            arg_type = Float64
            required = true
            help = "l2 regularization coefficient"
        "-c", "--regularization-rate-coeff"
            arg_type = Float64
            required = true
            help = "base coefficient of an _exponential_ regularization rate (usually: \\gamma = 1 + \\epsilon \\geq 1)."
        "--tolerance"
            arg_type = Float64
            required = true
            help = "convergence criteria threshold for the absolute delta in loss gain"
    end
    opts = parse_args(s)

    if opts["gridsearch"]
        println("INFO: Grid search is enabled.\nNOTE: Elbmf, as used in our paper, does **not** rely on grid search.")
        opts["dont-round-on-convergence"] = true
    end

    if !isnothing(opts["batchsize"])
        println("INFO: the option <batchsize> is not production-ready.")
    end

    c = opts["regularization-rate-coeff"]

    if c <= 1
        println("WARNING: regularization-rate-coeff (c) = $c <= 1, usually we want \\gamma = 1 + \\epsilon \\geq 1, however. See Readme.md or Elbmf.pdf")
    end

    X = Float64.(readdlm(opts["input-data"], '\t', Int8))
    if CUDA.functional() && !isnothing(opts["batchsize"]) && opts["cuda"]
        X = CUDA.adapt(CuArray, X)
    end

    U, V = Elbmf.elbmf(X,
                       opts["ncomponents"],
                       opts["l1reg"],
                       opts["l2reg"],
                       t -> c^t,
                       opts["maxiter"],
                       opts["tolerance"],
                       opts["inertial"],
                       isnothing(opts["batchsize"]) ? size(X, 1) : opts["batchsize"],
                       !opts["dont-round-on-convergence"])
    
    if opts["gridsearch"] 
        # !! gridsearch was *NOT* used by Elbmf in the paper; we simply included this for your convenience.
        println("INFO: starting 2D grid search (which might take some time).")
        gridsearch!(X, U, V; steps=200)
    end

    if !opts["dont-round-on-convergence"] || opts["gridsearch"] 
        # only converts the data-type of U and V
        U, V = Int.(U), Int.(V)
    end
    writedlm(opts["output-factors"] * ".lhs.tsv", U, '\t')
    writedlm(opts["output-factors"] * ".rhs.tsv", V, '\t')
end

main()
