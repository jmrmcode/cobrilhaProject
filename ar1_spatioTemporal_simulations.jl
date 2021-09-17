####################*************      SIMULATE DATA      **************##########################
using CategoricalArrays

# 10 spatial replicates x 10 temporal replicates across 4 sites; total = 400 spatio-temporal observations
## ar(1)
φ = [0.6]#, 0.1, -0.5, 0.4] # 4 sites
p1 = Array{Float64,3}(undef, 10, 100, 1)
for i in 1:1

Q = fill(-1*φ[i], 10, 10) # 10 years
Q[diagind(Q)] .= 1 + (φ[i])^2
Q[1, 1] = 1; Q[10, 10] = 1
Q = Array(Tridiagonal(Q))
Σ = Symmetric(inv(Matrix(Q)))

# Normal dist is parameterized by the sd
# u₂ ~ MvNormal(zeros(0), sqrt.(Σ))

ar1Devations = rand( MvNormal(ones(10), Σ), 100) # 100 observations per site and year
# cor(ar1Devations[8, :], ar1Devations[9, :])
# var(ar1Devations[10,:])
β0 = 1
p_1 = logistic.(β0 .+ ar1Devations)

p1[:,:,i] = p_1
end

# scalar
m_δ = 0
v_δ = 1
δ = Array{Float64,3}(undef, 10, 100, 1)
for i in 1:1

for j in 1:10

    for h in 1:100

    δ[j,h,i] = exp((-0.5 / v_δ) * (rand(Normal(0, 1), 1)[1] - m_δ)^2)
end
end
end

# total probability
p = Array{Float64,3}(undef, 10, 100, 1)
for i in 1:1

for j in 1:10

    p[j,:,i] = p1[j,:,i] #.* δ[j,:,i]
end
end

# observations
# pp = reshape(p, 400)
#y = Array{Float64,1}(undef, 400)
y = Array{Float64,3}(undef, 10, 100, 1)
for i in 1:1
for j in 1:10
for h in 1:100
    y[j,h,i] = rand(Binomial(5, p[j,h,i]), 1)[1]
end
end
end




## model

# X, a block matrix having 4 10x10 matrices
X = ones(Int64, 1000, 1, 1)

# Z, a block matrix having 4 100 x 10 matrices
year = repeat(1:10, 100)
year = CategoricalArray(repeat(1:10, 100))
Z = Array{Float64,3}(undef, 1000, 10, 1)
for s in 1:1
    df = DataFrame(y = reshape(y[:,:,s], 1000), year = year)
    Z1 = ModelFrame(@formula(y ~ -1 + year), df)
    Z1 = ModelMatrix(Z1)
    Z[:,:,s] = Z1.m
end

y = reshape(y[:,:,1], 1000)

@model simul(y, year) = begin

# priors
β0 ~ Normal(0, 10)
σ ~ truncated(Cauchy(0, 2), 0, Inf)
# φ ~ Uniform(-1, 1)

for s in 1:1 # 4 sites

# covariance matrix, build one per site
    Q = fill(-1*φ[s], 10, 10)
    Q[diagind(Q)] .= 1 + φ[s]^2
    Q[1, 1] = 1; Q[10, 10] = 1
    Q = Array(Tridiagonal(Q))
    #Σ = Symmetric(inv(Matrix(Q)))

# prior for the ar1 term at site s
# u ~ MvNormal(zeros(10), Symmetric(inv(Matrix(Q))))
u ~ filldist(Normal(0, σ), 10)

# logistic(Xβ0 + Zu); 100x1 * 1 + 100x10 * 10x1 = 100x1
#p1 = Array{Any}(undef, 1000)
#p1 = map.(logistic, X[:,:,s] .* β0 + Z[:,:,s] * u)


# Binomial
for i in 1:1000
    p1 = logistic(β0 + u[year[i]])
    y[i] ~ Binomial(5, p1)
end

end # sites
end # begin

chn = sample(simul(y, year), NUTS(), 3000)
plot(chn)
