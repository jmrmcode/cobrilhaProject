
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
means = sumstats[3:13, :mean]
# replace missing with the predicted values
df.pptAPR3LAG[findall(ismissing, df.pptAPR3LAG)] .= means
findall(ismissing, df.pptAPR3LAG)
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
