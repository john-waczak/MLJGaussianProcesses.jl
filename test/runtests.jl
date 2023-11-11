using MLJGaussianProcesses

using AbstractGPs
using Distributions
using KernelFunctions
using MLJBase
using ParameterHandling
using StableRNGs
using StatisticalMeasures
using Test


stable_rng() = StableRNGs.StableRNG(1234)


@testset "default kernel builders" begin

    flat_θᵢ, unflatten = ParameterHandling.value_flatten(θ_default)
    initial_params = ParameterHandling.value(θ_default)
    @test typeof(default_kernel(initial_params)) <: KernelFunctions.Kernel

end

@testset "default mean builders" begin

    @testset "zero mean" begin
        flat_θᵢ, unflatten = ParameterHandling.value_flatten(θ_default)
        initial_params = unflatten(flat_θᵢ)
        @test typeof(default_zero_mean(initial_params)) <: AbstractGPs.ZeroMean
    end

    @testset "constant mean" begin
        θ_kernel = θ_default
        θ_mean = (μ=1.0,)
        flat_θᵢ, unflatten = ParameterHandling.value_flatten(merge(θ_kernel, θ_mean))
        initial_params = unflatten(flat_θᵢ)
        mean_function = default_const_mean(initial_params)
        @test typeof(mean_function) <: AbstractGPs.ConstMean
        @test mean_function.c == 1.0
    end

    @testset "linear mean" begin
        θ_kernel = θ_default
        θ_mean = (β=[0.5,0.5],μ=1.0)
        flat_θᵢ, unflatten = ParameterHandling.value_flatten(merge(θ_kernel, θ_mean))
        initial_params = unflatten(flat_θᵢ)
        mean_function = default_linear_mean(initial_params)
        @test typeof(mean_function) <: LinearMean
        @test mean_function.β == [0.5,0.5]
        @test mean_function.μ == 1.0
        @test mean_vector(mean_function, ColVecs(ones(2,1)))[1] == 2.0
    end

end

@testset "MLJ interface" begin

    X, y = make_regression(100, 3, rng=stable_rng());

    gpr = GPR()

    m = machine(gpr, X, y)
    res = fit!(m)

    rpt = report(m)
    @test Set([:summary, :minimizer, :minimum, :iterations, :converged]) == Set(keys(rpt))

    fp = fitted_params(m)
    @test Set([:θ_best, :σ²]) == Set(keys(fp))


    p_y = predict(m, X)
    ŷ = predict_mean(m, X)
    @test typeof(p_y[1]) <: Distributions.Normal
    r2 = rsq(y, ŷ)
    @test isapprox(r2, 1.0, atol=0.00001)  #

    y_mode = predict_mode(m, X)
    @test all(ŷ .== y_mode)

end
