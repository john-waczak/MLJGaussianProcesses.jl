using MLJGP
using Test
using KernelFunctions
using ParameterHandling
using Distributions
using MLJBase
using StableRNGs


stable_rng() = StableRNGs.StableRNG(1234)


@testset "default kernel builders" begin

    flat_θᵢ, unflatten = ParameterHandling.value_flatten(θ_default)
    initial_params = ParameterHandling.value(θ_default)
    @test typeof(default_kernel(initial_params)) <: KernelFunctions.Kernel

end



@testset "MLJ" begin

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
