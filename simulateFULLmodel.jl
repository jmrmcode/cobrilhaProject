using StatsPlots
using StatsBase
using Turing
using StatsFuns: logistic
using StatsModels
using SpecialFunctions
using ReverseDiff
using LinearAlgebra
using Distributed
using SentinelArrays
using StaticArrays
using Random:seed!
seed!(123)
## simulate scalars
N = 500

m_δ = 0
v_δ = 1
x = rand(Normal(0, 1), N)
x2 = rand(Normal(2, 0.5), N)
ω1 = 0.1
ω2 = 0.9
δ = Array{Float64,1}(undef, N)

for i in 1:N

    δ[i] = exp((-0.5 / v_δ) * ((ω1 * x[i] + ω2 * x2[i]) - m_δ)^2)

end

#=
m_δ2 = 0
v_δ2 = 1
x2 = rand(Normal(0, 1), N)
δ2 = Array{Float64,1}(undef, N)

for i in 1:N

    δ2[i] = exp((-0.5 / v_δ2) * (x2[i] - m_δ2)^2)

end

m_δ3 = 0
v_δ3 = 1
x3 = rand(Normal(0, 1), 1000)
δ3 = Array{Float64,1}(undef, 1000)

for i in 1:1000

    δ3[i] = exp((-0.5 / v_δ3) * (x3[i] - m_δ3)^2)

end
=#

## simulate ar(1) * scalar

function simulate_ar(φ, n, δ)

	 y = [rand(Binomial(5, 0.5)) for i = 1:n]

	 for i in 1:(n-1)

	     y[i+1] = rand(Binomial(5, logistic(φ * y[i]) * δ[i+1]))
	 end

	 return y
end

φ = 0.7
etaa = simulate_ar(φ, N, δ)

## model

@model simul(y, x, x2) = begin

# priors
φ ~ Uniform(-1, 1)
v_δ ~ Uniform(0, 30)
m_δ ~ Truncated(Normal(0, 0.01), -1, 1)
ω ~ Dirichlet(2, 1.0)

# Binomial
for i in 2:N

	δ = exp((-0.5 / v_δ) * ((ω[1] * x[i] + ω[2] * x2[i]) - m_δ)^2)

	p = logistic(φ * y[i-1]) * δ
    y[i] ~ Binomial(5, p)

end # for

end # begin

chn = sample(simul(etaa, x, x2), NUTS(), 3000)
plot(chn)

write("U:/juanmi/cobrilha_project/code/chn.json", chn)
chn = read("U:/juanmi/cobrilha_project/code/chn.json", Chains)
