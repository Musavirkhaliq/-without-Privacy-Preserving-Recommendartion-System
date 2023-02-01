# Efficiently Factorizing Boolean Matrices using Proximal Gradient Descent

This online supplementary containing the official implementation of Efficiently Factorizing Boolean Matrices using Proximal Gradient Descent. 

Please consider citing us, for example, by using

```bibtex
@inproceedings{dalleiger2022efficiently,
    title={Efficiently Factorizing Boolean Matrices using Proximal Gradient Descent},
    author={Sebastian Dalleiger and Jilles Vreeken},
    booktitle={Thirty-Sixth Conference on Neural Information Processing Systems (NeurIPS)},
    year={2022}
}
```

## Requirements

To install requirements:

```
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.add.(["ArgParse", "DelimitedFiles"])'
```

>📋  To install the Julia Language, please consider consulting the [official installation guide](https://julialang.org/downloads/).

## Factorization

To factorize a Boolean input matrix, either use the REPL or consider the 
```julia --project=. -O3 ./cli/elbmf.jl``` 
command line interface, which takes the following arguments:
```
  -i, --input-data INPUT-DATA         Boolean input matrix in tsv format
  -o, --output-factors OUTPUT-FACTORS The output factor matrices (as <OUTPUT-FACTORS>.{lhs,rhs}.tsv) in tsv
  -k, --ncomponents NCOMPONENTS       The number of components (rank) of the factorization (type: Int64)
  --maxiter MAXITER                   The maximum number of iterations (type: Int64)
  --l1reg L1REG                       l1 regularization coefficient (type: Float64)
  --l2reg L2REG                       l2 regularization coefficient (type: Float64)
  -c, --regularization-rate-coeff C   The base coefficient of an _exponential_ regularization rate (usually: \gamma = 1 + \epsilon \geq 1). (type: Float64)
  --tolerance TOLERANCE               The threshold of the absolute delta in loss gain which determines convergence (type: Float64)
  --dont-round-on-convergence         Enable to prevent elbmf to round factor matrices upon convergence
  [--gridsearch]                      Elbmf DOES NOT use this argument. This is a last-resort option if it is complicated to tune hyperparameters (see Readme.md)
  [--cuda]                            Enables CUDA support, which is assumed to be true if '--batchsize' is set
  [--inertial INERTIAL]               Inertial coefficient 'beta' for iPALM (type:Float64, default: 0.0)
  [--batchsize BATCHSIZE]             Compute gradient in minibatches (if the matrices do not fit into the GPU memory (i.e., if you observe an CUDA Memory error) (type: Int64)
```

>❗  To results in Boolean factors which approximate the input well, Elbmf requires properly chosen hyperparameters. 

📋  The number for ```--regularization-rate-coeff``` denotes the base of an exponential rate $\gamma^t$, where $t$ is the current iteration number and $\gamma \geq 1$, e.g., $\gamma = 1+\epsilon$.<br/>
❗  **If you observe a bad reconstruction**, you might want to tune the $\gamma$ parameter (see also ```--maxiter```, ```--l1reg```, ```--l2reg```, and ```--ncomponents```, which all affects Elbmf's performance). 
In case you require a *different rate function*, please feel free to update the code in ```cli/elbmf.jl```.

📋  Although we usually want to use it, *not* rounding helps with assessing the effects of our hyperparameters. In particular, it allows us to validate whether the configuration suffices to return an almost-Boolean solutions.
To prevent rounding, you can use the flag ```--dont-round-on-convergence```, which prevents Elbmf from rounding the final factors to a Boolean solution upon convergence. <br/> 
❗ One use case, for example, is when you observe a terrible reconstruction. Then, you might want to set ```--dont-round-on-convergence``` and tune hyperparameters, until you observe a Boolean or almost-Boolean result, after which you can remove this flag again.

📋  Despite the fact that we do **not** combine grid search with Elbmf in our paper, we still include grid search as a last-resort post-processing procedure for your convenience, when it is very hard to find good candidates for the parameters above (which it should not be). 
Regardless of our paper, you can enable post-processing via the flag ```--gridsearch``` (higher precedence than ```--dont-round-on-convergence```), to jointly search for rounding-thresholds for $U$ and $V$, by means of 2D grid search. <br/>
📋  The argument to ```---batchsize``` is experimental and not production-ready. Please consider combining Elbmf with e.g. Fluxml, PyTorch, or JAX.
>❗  Note that Julia has a very long start-up time and first-function-call time, as it usually compiles the code before running it. Measuring the time thus is tricky, especially from the command line.

## Synthetic Data Generation

To generate synthetic data in our experiments, see ```julia --project=. ./cli/generate.jl```, which takes the following arguments:
```
  -o, --output OUTPUT     The output matrix as tsv
  -n, --height HEIGHT     The height of the random matrix (type: Int64)
  -m, --width WIDTH       The width of the random matrix (type: Int64)
  -k, --numtiles NUMTILES The number of tiles (type: Int64)
  --minwidth MINWIDTH     The min width of a tile (type: Int64)
  --maxwidth MAXWIDTH     The max width of a tile (type: Int64)
  [--seed SEED]           The random seed (type: Int64)
  [--allowoverlap]        Allow tiles to overlap
  --additivenoise LEVEL   The level of additive noise (type: Float32, 0 <= LEVEL <= 1) 
```

## Contributing

This research-code is under **MIT License**, see the accompanying `License` file.
As this implementation exists for archival purposes, there will be no further development in this repository.

If you are interested in a python implementation or if you have further questions or comments, please feel free to reach out to Sebastian Dalleiger ([E-Mail](sdalleig@mpi-inf.mpg.de), [ORCiD](https://orcid.org/0000-0003-1915-1709)).
