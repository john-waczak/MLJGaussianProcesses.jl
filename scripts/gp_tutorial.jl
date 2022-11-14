using LinearAlgebra
using Distributions
using Random
using Plots
using Tables
using MLJBase
using KernelFunctions
using AbstractGPs
using Statistics

# X, y = make_regression(200, 2)
# typeof(X)
# typeof(y)
# keys(X)  # a table of features
# y  # a vector of targets
# p1 = scatter(X.x1, X.x2, zcolor=y)


function make_training_data(n, ν)
    xs = reshape(collect(range(0.0, stop=1.0, length=500)), (1,500))

    y(x) = (exp(-x/(0.5)^2) * sin(2π*ν*x)) + (0.3)^2*(rand()-0.5)
    ytruth(x) = exp(-x/(0.5)^2) * sin(2π*ν*x)

    X = Tables.table(rand(1,n)', header=[:x])
    Xtrue = Tables.table(xs', header=[:x])

    y = y.(X.x)
    ytrue = ytruth.(Xtrue.x)

    return X, y, Xtrue, ytrue
end

X, y, Xtrue, ytrue = make_training_data(50, 10)


scatter(X.x, y, color=:red, label="noisy data")
plot!(Xtrue.x, ytrue, color=:blue, label="true function")
xlabel!("x")
ylabel!("y")

ℓ = 0.1
k = 1.0 * (SqExponentialKernel() ∘ ScaleTransform(1/2ℓ^2))
# let's test out the kernels interface
K = kernelmatrix(k, Tables.matrix(X)')

# 1. construct a AbstractGP with mean 0 and cov k(x,x′)
f = GP(k)
# 2. Create FiniteGP, i.e. MVN by applying GP to training data w/ noise σ²=0.1
fₓ = f(Tables.matrix(X)', (0.1)^2)

# let's examine this:
fₓ.x.X  # the underlying data
fₓ.x  # the internal ColVecs representation (needed for computing kernel matrix K later)
fₓ.Σy

# compute log-marginal likelihood p(y|X), i.e. probability of targets given features, hyperparameters, etc...
logpdf(fₓ, y)


# compute the posterior distribution i.e. p(ynew | Xnew, X, y)
p_fₓ = posterior(fₓ, y)



# here's how you might try to do prediction
ypred = mean(p_fₓ(Tables.matrix(Xtrue)'))

# alternatively, we can compute the infividual marginal distributions for each point w/o computing full covariance matrix on test set
p_pred = marginals(p_fₓ(Tables.matrix(Xtrue)'));
y_pred = mean.(p_pred)
σ_pred = std.(p_pred)



p1 = plot(X.x, y, seriestype=:scatter, color=:red, label="noisy data")
plot!(Xtrue.x, ytrue, color=:black, linestyle=:dash, label="true function")
plot!(Xtrue.x, y_pred, color=:blue, linewidth=3, label="fit")
plot!(Xtrue.x, y_pred .+ 2 .* σ_pred, c=:gray, label="")
plot!(Xtrue.x, y_pred .- 2 .* σ_pred, fillrange = y_pred .+ 2 .* σ_pred, fillalpha = 0.25, c = :gray, label = "2σ Confidence band")
xlabel!("x")
ylabel!("y")
title!("Vanilla GPR")


# so how do we pick the right hyperparameter values?
# we optimize the log-marginal likelihood, i.e. the probability of observing the targets given
# our choice of hyperparameters
using Optim

softplus(x) = log(1+exp(x))
softplusinv(x) = log(exp(x)-1)


function loss_function(x, y)
    function negativelogmarginallikelihood(params)
        kernel =
            softplus(params[1]) * (SqExponentialKernel() ∘ ScaleTransform(softplus(params[2])))
        f = GP(kernel)
        fx = f(x, softplus(params[3]))
        return -logpdf(fx, y)
    end
    return negativelogmarginallikelihood
end

# initial guess
# θ₀ = [(0.25)^2, rand(2)...]
# θ₀ = softplusinv.([1.0, 0.1, 0.1^2])
θ₀ = [1.0, 1/(2*ℓ^2), (0.25)^2]
# perform optimization
using BenchmarkTools
@btime Optim.optimize(loss_function(Tables.matrix(X)', y), θ₀, LBFGS(); autodiff=:forward)
@btime Optim.optimize(loss_function(Tables.matrix(X)', y), θ₀, LBFGS())  # default uses finite diff methods

opt = Optim.optimize(loss_function(Tables.matrix(X)', y), θ₀, LBFGS(); autodiff=:forward)
params_best = opt.minimizer
softplus.(params_best)


kernel_best =  softplus(params_best[1]) * (SqExponentialKernel() ∘ ScaleTransform(softplus(params_best[2])))
f = GP(kernel_best)
#fₓ = f(Tables.matrix(X)', softplus(params_best[3]))
fₓ = f(Tables.matrix(X)', softplus(params_best[3]))
p_fₓ = posterior(fₓ, y)
p_pred = marginals(p_fₓ(Tables.matrix(Xtrue)'));
y_pred = mean.(p_pred)
σ_pred = std.(p_pred)



p2 = plot(X.x, y, seriestype=:scatter, color=:red, label="noisy data")
plot!(Xtrue.x, ytrue, color=:black, linestyle=:dash, label="true function")
plot!(Xtrue.x, y_pred, color=:blue, linewidth=3, label="fit")
plot!(Xtrue.x, y_pred .+ 2 .* σ_pred, c=:gray, label="")
plot!(Xtrue.x, y_pred .- 2 .* σ_pred, fillrange = y_pred .+ 2 .* σ_pred, fillalpha = 0.25, c = :gray, label = "2σ Confidence band")
xlabel!("x")
ylabel!("y")
title!("HPO GPR")


p3 = scatter(ytrue, y_pred, xlabel="truth", ylabel="prediction")

plot(p1, p2,)

