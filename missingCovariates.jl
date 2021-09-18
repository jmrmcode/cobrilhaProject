
## covariates model
# the code below tries to run a model for every covariate having missing values and replace them with the estimate.
# Basically the idea is to estimate the posterior of the covariate and then estimate missing values. However, this is a bit idiot because we already know that covariates follow Normal (0, 1)
# Also, it is slower than the alternative approach which is to sample fro Normal (0, 1) inside the full model code (see scalarModels.jl).
# and sample directly from a Normal(0, 1)

# pptAPR3LAG
@model function gdemo(x, ::Type{T} = Float64) where {T}
    s² ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s²))
    for i in eachindex(Array(x))
        x[i] ~ Normal(m, sqrt(s²))
    end
end

# x[1] is a parameter, but x[2] is an observation
model = gdemo(Array(df.pptAPR3LAG))
c = sample(model, NUTS(), 3000)

# get posterior means of the predicted values]
using Statistics
sumstats = summarize(c, mean)
means = sumstats[3:16, :mean]
# replace missing with the predicted values
df.pptAPR3LAG[findall(ismissing, df.pptAPR3LAG)] .= means
findall(ismissing, df.pptAPR3LAG)

# Deviance Information Criterion (DIC) and number of effective parameter
    niter = size(c)[1]
    lp = zeros(niter) # resulting logpdf values
    for i = 1:3000
        lp[i] = sum(map((m,s) -> logpdf.(Normal(m, s), df.pptAPR3LAG), Array(c[:,2,:]), Array(c[:,1,:]))[i]) # Compute the sum of log-densities at each iteration of MCMC output for stochastic nodes.
    end

    # D = Deviance = -2 * loglikehood
    D = -2 * lp
    D̄ = mean(D)
    # effective number of parameters pᵩ = D̄ + D_θ_hat = mean deviamce + deviance of the parameters mean
    D_θ_hat = -2 * sum(map((m,s) -> logpdf.(Normal(m, s), df.pptAPR3LAG), mean(Array(c[:,2,:])), mean(Array(c[:,1,:]))))
    pᵩ = D̄ + D_θ_hat
    # deviance information criterion (DIC)
    DIC = D̄ + pᵩ

# JLSO saves also metadata to import later to different Julia versions. However, the line below does not work
# JLSO.save("U:/juanmi/cobrilha_project/models/test.jlso", c)
# The two lines below only make sense to export Julia objects between the same Julia versions
Serialization.serialize("U:/juanmi/cobrilha_project/models/c.jls",c) # export
Serialization.deserialize("U:/juanmi/cobrilha_project/models/c.jls") # import


##
# pptAPR3LAG
@model function gdemo(x, ::Type{T} = Float64) where {T}
    s² ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s²))
    for i in eachindex(Array(x))
        x[i] ~ Normal(m, sqrt(s²))
    end
end

# x[1] is a parameter, but x[2] is an observation
model = gdemo(Array(df.pptAPR3LAG))
c = sample(model, NUTS(), 3000)
# get posterior means of the predicted values]
using Statistics
sumstats = summarize(c, mean)
means = sumstats[3:13, :mean]
# replace missing with the predicted values
df.pptAPR3LAG[findall(ismissing, df.pptAPR3LAG)] .= means
findall(ismissing, df.pptAPR3LAG)
