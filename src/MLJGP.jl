module MLJGP

# for building GPs
using Random
using LinearAlgebra
using Statistics
using Distributions
using KernelFunctions
using AbstractGPs

# for optimizing them
using ParameterHandling
using Optim
using Zygote

# for MLJ
import MLJModelInterface
import MLJModelInterface: Continuous, Multiclass, metadata_model, metadata_pkg
import Random.GLOBAL_RNG

## CONSTANTS
const MMI = MLJModelInterface
const PKG = "MLJGP"



export θ_default
export default_kernel


# -------------------------------------------------------------------------
#         Kernel Building
# -------------------------------------------------------------------------

θ_default = (σf² = positive(1.0), ℓ = positive(1.0))
  
# this is what the user should supply for MLJ instead of just a kernel
function default_kernel(θ::NamedTuple)
    return  θ.σf²  * (SqExponentialKernel() ∘ ScaleTransform(1/2(θ.ℓ)^2))
end

# note: we need to figure out a reasonable way to do these constraints.
MMI.@mlj_model mutable struct GPR <: MMI.Probabilistic
    # μ::Union{Float64, Function} = 0.0  # can either pass a constant mean or a custom function
    μ::Float64 = 0.0::(_ ≥ 0)
    k::Function = default_kernel
    θ_init::NamedTuple = θ_default
    σ²::Float64 = 1e-6::(_ > 0)
    optimizer::Optim.AbstractOptimizer = LBFGS()
end


function MLJModelInterface.fit(gpr::GPR, verbosity, X, y)
    Xmatrix = MMI.matrix(X)'  # make matrix p × n for efficiency
    nfeatures = size(Xmatrix, 1)

    # augment θ_init to include σ²
    θ_init_full = (gpr.θ_init..., σ²=positive(gpr.σ²))
    flat_θᵢ, unflatten = ParameterHandling.value_flatten(θ_init_full)

    # define loss function as minus marginal log-likelihood
    function objective(θ::NamedTuple)
        k = gpr.k(θ)  # rebuild kernel function with current params
        f = GP(gpr.μ, k)   # build AbstractGP using our μ and k
        fₓ = f(Xmatrix, θ.σ²)

        return -logpdf(fₓ, y)  # return minus marginal log-likelihood
    end

    # default option uses finite diff methods
    if verbosity > 0
      opt = Optim.optimize(
          objective ∘ unflatten,
          θ -> only(Zygote.gradient(objective ∘ unflatten, θ)),
          flat_θᵢ,
          gpr.optimizer,
          Optim.Options(show_trace = true);
          inplace=false,
        )
    else
        opt = Optim.optimize(
            objective ∘ unflatten,
            θ -> only(Zygote.gradient(objective ∘ unflatten, θ)),
            flat_θᵢ,
            gpr.optimizer,
            inplace=false,
        )
    end


    θ_best = unflatten(opt.minimizer)

    f = GP(gpr.μ, gpr.k(θ_best))
    fₓ = f(Xmatrix, θ_best.σ²)
    p_fₓ = posterior(fₓ, y)  # <-- this is our fitresult as it let's us do everything we need

    # generate new θ for fitresult
    θ_out = [p for p ∈ pairs(θ_best) if p[1] != :σ²]


    fitresult = (p_fₓ, θ_out, θ_best.σ²)

    # 3. collect results
    cache  = nothing
    # put class labels in the report
    report = (;
              summary=Optim.summary(opt),
              minimizer=Optim.minimizer(opt),
              minimum=Optim.minimum(opt),
              iterations=Optim.iterations(opt),
              converged=Optim.converged(opt),
              )

    return (fitresult, cache, report)
end


MMI.fitted_params(gpr::GPR, fitresult) = (;θ_best=fitresult[2], σ²=fitresult[3])


function MMI.predict(gpr::GPR, fitresult, X)
    p_fₓ, θ, σ² = fitresult
    Xdata = MMI.matrix(X)'

    fₓ = p_fₓ(Xdata)
    return marginals(fₓ)
end


function MMI.predict_mean(gpr::GPR, fitresult, X)
    p_fₓ, θ, σ² = fitresult
    Xdata = MMI.matrix(X)'

    fₓ = p_fₓ(Xdata)

    return mean(fₓ)
end


function MMI.predict_mode(gpr::GPR, fitresult, X)
    # return the coordinates of the bmu for each instance
    MMI.predict_mean(gpr, fitresult, X)
end



MMI.metadata_pkg.(GPR,
                  name = "MLJGP",
                  uuid = "e2c51686-8c13-4540-892e-18cf079a7e1a", # see your Project.toml
                  url  = "https://github.com/john-waczak/MLJGP.jl",  # URL to your package repo
                  julia = true,          # is it written entirely in Julia?
                  license = "MIT",       # your package license
                  is_wrapper = false,    # does it wrap around some other package?
)


# Then for each model,
MMI.metadata_model(GPR,
                   input=Union{AbstractMatrix{MMI.Continuous}, MMI.Table(MMI.Continuous)},
                   target=AbstractVector{MMI.Continuous},
                   supports_weights = false,
                   descr   = "A simple interface to Gaussian Processes in MLJ with probabilistic outputs.",
                   load_path    = "$(PKG).GPR"
                   )



# ------------ documentation ------------------------


const DOC_GPR = "[Gaussian Process Regression](https://gaussianprocess.org/gpml/chapters/RW2.pdf)"


"""
$(MMI.doc_header(GPR))
MLJGPR implements $(DOC_GPR), a non-parametric, non-linear regression model for supervised machine learning utilizing tools from the [JuliaGaussianProcesses](https://juliagaussianprocesses.github.io/) organization.
# Training data
In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)
where
- `X`: an `AbstractMatrix` or `Table` of input features whose columns are of scitype `Continuous.`
Train the machine with `fit!(mach, rows=...)`.
- `y`: a `Vector` of target variables of scitype `Continuous`.
# Hyper-parameters
- `μ=0`: Constant value to use for mean function of Gaussian Process.
- `k=default_kernel`: A function `k(θ)` which takes parameters `θ` and returns a `KernelFunction`. `default_kernel` is the classic RBF kernel with variance `σf²`, and length scale `ℓ`
- `θ_init=θ_default`: Default parameters to initialize the optimization. Defaults to `θ_default = (1.0, 1.0)` for the default kernel.
- `σ²=1e-6`: Measurement noise (variance). Must be greater than `0` to ensure stability of internal Cholesky factorization.
- `optimizer:LBFGS()`: Optimizer from `Optim.jl`.
# Operations
- `predict(mach, X)`:  Returns a vector of normal distributions for each predicted target.
- `predict_mean(mach, X)`: Return a vector of means from the distribution of predicted targets.
- `predict_mode(mach, X)`: Return a vector of modes from the distribution of predicted targets.
# Fitted parameters
The fields of `fitted_params(mach)` are:
- `θ_best`: A named tuple of best parameters found during GPR fit.
- `σ²`: The best fit for the measurement variance.
# Report
The fields of `report(mach)` are:
- `summary`: A summary of results of the optimization.
- `minimizer`: The parameters that minimized the marginal log-likelihood for the GPR model.
- `minimum`: The minimum value of minus the marginal log-likelihood during optimization.
- `iterations`: The number of steps taken by the optimizer
- `converged`: Whether or not the optimization scheme converged in the allotted number of iterations.
# Examples
```
using MLJ
gpr = @load GPR pkg=MLJGP
model = gpr()
X, y = make_regression(50, 3) # synthetic data
mach = machine(model, X, y) |> fit!
p_y = predict(mach, X)
ŷ = predict_mean(mach, X)
rpt = report(mach)
```
"""
GPR



end
