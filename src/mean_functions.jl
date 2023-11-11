"""
    LinearMean{coefType,meanType} <: AbstractGPs.MeanFunction

Simple implementation of a linear mean function `β⋅x + μ` where `x`
is the feature vector, `β` are the linear coefficients, and `μ` is
a constant offset term.
"""
struct LinearMean{coefType,meanType} <: AbstractGPs.MeanFunction
    β::coefType
    μ::meanType
end

AbstractGPs.mean_vector(mf::LinearMean, vecs::ColVecs) = reshape(mf.β,1,:)*vecs.X .+ mf.μ
AbstractGPs.mean_vector(mf::LinearMean, vecs::RowVecs) = vecs.X*reshape(mf.β,:,1) .+ mf.μ

function default_linear_mean(θ)
    return LinearMean(θ.β, θ.μ)
end

function mean_function_initializer(::typeof(default_linear_mean), rng::AbstractRNG)
    function init(X, y)
        return (β = randn(rng, size(X,2)), μ = zero(eltype(y)))
    end
end

function default_zero_mean(θ::NamedTuple)
    return ZeroMean()
end

function mean_function_initializer(::typeof(default_zero_mean), ::AbstractRNG)
    init_zero_mean(X, y) = (;)
end

function default_const_mean(θ::NamedTuple)
    return ConstMean(θ.μ)
end

function mean_function_initializer(::typeof(default_const_mean), ::AbstractRNG)
    init_const_mean(X, y) = (μ=zero(eltype(y)),)
end
