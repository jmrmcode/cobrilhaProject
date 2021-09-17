using CSV
using StatsPlots
using StatsBase
using DataFrames
using Turing
using StatsFuns: logistic
using StatsModels
using SpecialFunctions
using ReverseDiff
using LinearAlgebra
# using LazyArrays

df = CSV.read("D:/UM_Lab/Portugal/data/dfFULLCobrilha_propertyLevel.csv", DataFrame; missingstring = "NA")
df = df[!, Not(r"vci")] # remove all the columns having vci
df = df[!, Not(r"spdei")]
df = df[!, Not(r"tmmn")]
df = df[!, Not(r"tmmx")]
df = df[!, Not(r"ppt")]
df = df[!, Not(r"PDSI")]
df = dropmissing(df, disallowmissing=true) # remove rows having missing data

df.yearSt = string.(df.year)
df.CobrilhaIncCopy = df.CobrilhaInc
df.propertySt = string.(df.property)
df.scaledEspe9y = standardize(ZScoreTransform, df.Espe9yMean; dims = 1)


# dataYear.st = string.(dataYear.yearInt)
# replace missing values with 0
[df[ismissing.(df[!,i]), i] .= 0 for i in names(df) if Base.nonmissingtype(eltype(df[!,i])) <: Number]

pwd() # Get the current working directory
names(df)
first(df, 10)
last(df, 6)
unique(df.property)

# model matrix for covariates values
mf = ModelFrame(@formula(CobrilhaInc ~ 1 + scaledEspe9y), df)
X = ModelMatrix(mf)
# model matrix for random effects
# zf = ModelFrame(@formula(CobrilhaInc ~ -1 + scaledEspe9y & property), df)
# Z₁ = ModelMatrix(zf)
# model matrix for other random effects
Psif = ModelFrame(@formula(CobrilhaInc ~ -1 + yearSt), df) # yearLetter
Z₂ = ModelMatrix(Psif)
# check Z is ok
findall(x->x≠0, Z.m[1, :]) # where the first row is not zero
levels(df.property)[50] # which property code corresponds to 50
df.property[1] # the property code of the first row in df

# incidence and errors
# √(p̂ᵢ(1 - p̂ᵢ) / Nᵢ)
pj = df.CobrilhaInc ./ df.Ntrials
error = @. sqrt((pj * (1 - pj) / df.Ntrials))
# plot out the error
scatter(
  df.dfEspe9yMean, pj,
  # yerror= error,
  legend = false,
  ylim = (0, 1),
  ylab = "Incidence",
  xlab = "Thickness (cm)")

  ## model

## MODEL:      yᵢ ~ binomial(Nᵢ, logit⁻¹(β0 + β1 * xᵢ))

  @model logistic_model(X, Z₁, y, N) = begin
    # parameters
    # priors for the fixed effects
    β ~ filldist(Normal(0, 10), 2)
    #β ~ arraydist([Normal(0, 10), Normal(0, 10), Uniform(-1, +1)])
    # Prior for variance of group effects (hyperparameter)
    s² ~ InverseGamma(2, 3)
    s = sqrt(s²)

     # Prior for group effects.
    u₁ ~ filldist(Normal(0, s), 311)
    # u₂ ~ filldist(Normal(0, s), 311)

    # model

      p = logistic.(X * β + Z₁ * u₁) # + Z₂ * u₂
      y .~ Binomial.(N, p)
      # y ~ arraydist(LazyArray(@~ Binomial.(N, p)))

  end

## sampler
Turing.setadbackend(:forwarddiff) # forwarddiff  reversediff tracker

chain = sample(logistic_model(X.m, Z₁.m, df.CobrilhaInc, df.Ntrials), NUTS(), 2000)
chain = sample(logistic_model(X.m, Z₁.m, df.CobrilhaInc, df.Ntrials), NUTS(), MCMCThreads(), 2000, 3)

iterations = 2000
ϵ = 0.1
τ = 5
outcome = mapreduce(c -> sample(logistic_model(X.m, df.CobrilhaInc, df.Ntrials),
Gibbs(
    NUTS()), iterations, drop_warmup = false, progress = true, verbose = true), chainscat, 1:3)

plot(outcome)
describe(chn)

# extract just the parameters from the chain object
Chains(outcome, :φ)
# access the names of all parameters in a chain that belong to the group :name
namesingroup(chn, :φ)
println(chn.name_map.parameters)
# returns a subset of the chain chain with all parameters in the group :name
gg = group(chn, :y)
# access parameters
bb = get(chn, :β)

# access parameter that belong to the group :name
bb.β[1]
mean(bb.β[1])
plot(gg)


##
β0_post = median(chn[:β0])
β1_post = median(chn[:β1])

# iterator for distance from hole calcs
xrng = standardize(ZScoreTransform, df.dfEspe9yMean; dims = 1)
post_lines = [logistic(β0_post + β1_post * x) for x = xrng]

# 50 draws from the posterior
β0_samp = StatsBase.sample(chn[:β0], 50)
β1_samp = StatsBase.sample(chn[:β1], 50)

post_samp = [logistic(β0_samp[i] + β1_samp[i] * x) for x = xrng, i = 1:50]

plot!(df.dfEspe9yMean, post_samp, alpha = 0.5, color = :gray) # add uncertainty samples
plot!(df.dfEspe9yMean, post_lines, color = :black) # add median

## MODEL having the scalar

@model logistic_model(y, Espe9yMean, N) = begin
  # parameters
  # priors for the scalar parameters
  v_δ ~ Uniform(0,10)
  m_δ ~ Normal(0,20)
#  v_γ ~ Uniform(0, 10)
#  m_γ ~ Normal(1, 20)
  # m ~ TruncatedNormal(mu, sigma, l, u)

  # model
  for i in eachindex(y)

    # Gaussian scalar function
      δ = exp((-0.5 / v_δ) * (Espe9yMean[i] - m_δ)^2)
    #  γ = exp((-0.5 / v_γ) * (dfEspe9yMean[i] - m_γ)^2)
# binomial model

    p = 1 * δ #* γ
    y[i] ~ Binomial(N[i], p)

  end
end

  chn = sample(logistic_model(df.CobrilhaInc, df.Espe9yMean, df.Ntrials), NUTS(), 3000)
plot(chn)
# make predictions
m_lin_reg_test = logistic_model(Vector{Union{Missing, Float64}}(undef, length(df.CobrilhaInc)), df.Espe9yMean, df.Ntrials);#repeat([mean(df.tmmnCurrentYear)], length(df.CobrilhaInc))
pp = predict(m_lin_reg_test, chn)
# Get the mean predicted values.
ys_pred = collect(vec(mean(pp.value; dims = 1))) ./ df.Ntrials
# Get the prediction error:
# errors = df.CobrilhaInc - ys_pred

# plot the scalar
m = mean(get(chn, :m_δ).m_δ)
v = mean(get(chn, :v_δ).v_δ)
function scalavrF(x)
  exp((-0.5 / v) * (x - m)^2)
end

means = group(pp, :y)
stdd = mapslices(std, means.value.data; dims = 1)[sortperm(df.Espe9yMean)]
plot(sort(df.Espe9yMean), scalavrF.(sort(df.Espe9yMean)), seriestype = :line, xlabel = "Thickness (cm)", label = "", ylabel = "Performance (% max)", linecolor = "black") # yerror = stdd,
# plot!(sort(df.dfEspe9yMean), (ys_pred[sortperm(df.dfEspe9yMean)] ./ maximum(ys_pred)), yerror = stdd[sortperm(df.dfEspe9yMean)]./ df.Ntrials, seriestype = :scatter, label ="two")
plot!(sort(df.Espe9yMean), (ys_pred[sortperm(df.Espe9yMean)] ./ maximum(ys_pred)), yerror = stdd[sortperm(df.Espe9yMean)]./ df.Ntrials,
seriestype = :scatter, label ="", color = "gray", alpha = 0.5)

savefig("thickness2.png")
# save MCMC chains as JSON
write("D:/UM_Lab/Portugal/models/espe9y_propertyLevel.json", chn)
newChains = read("D:/UM_Lab/Portugal/models/espe9y_propertyLevel.json", Chains)
# newChains = newChains[1000:2000,:,:]
plot(newChains, seriestype = :traceplot)

##
  @model logistic_model(y, x, N) = begin
    # parameters
    β0 ~ Normal(0, 10)
    β1 ~ Normal(0, 10)

    # model
    for i in eachindex(y)

  # logistic model

      p = logistic(β0 + β1 * x[i])
      y[i] ~ Binomial(N[i], p)

    end
  end
  ## sampler
  chn = sample(logistic_model(df.CobrilhaInc, df.scaledEspe9y, df.Ntrials), NUTS(), 2000)

function plogis(rho)
  exp(rho) / (1 + exp(rho))
end
plogis(0.0004)

## AR(1)
@model logistic_model(Z₂, y, N) = begin
  # priors for the fixed effects
#  β ~ filldist(Normal(0, 10), 2)
   # Prior for group effects
#  σ² ~ Uniform(0, 10) # Non-informstive prior, see  7.1 Prior distributions for variance parameters in Prior distributions for variance parameters in hierarchical models by Andrew Gelman
#  prec = 1 / σ²

  φ ~ Uniform(-0.99, 0.99)
   # declare and populate the inverse covariance matrix. Code to recycle at some point Σ = diagm(0.1*ones(3))
  Q = fill(-1*φ, 18, 18) # 18 18
  Q[diagind(Q)] .= 1 + φ^2
  Q[1, 1] = 1; Q[18, 18] = 1 # 18 18
  Q = Array(Tridiagonal(Q))
  Σ = Symmetric(inv(Matrix(Q)))

# Normal dist is parameterized by the sd
   u₂ ~ MvNormal(zeros(18), Σ) # 18 (years)
  # u₂ ~ MvNormal(zeros(18), sqrt(σ²)*diagm(ones(18)))
  # model

    p = logistic.(Z₂ * u₂)# X * β +
    y .~ Binomial.(N, p)
end

  chn = sample(logistic_model(Z₂.m, df.CobrilhaInc, 1), NUTS(), 3000) #df.Ntrials
  chn1 = chn[301:2000,:,:]
  gg = group(chn, :φ)
  # access parameters
  bb = get(gg, :φ)
  mean(bb.φ)
  # access parameter that belong to the group :name
  mean(gg.φ)
  bbb = get(chn, :σ²)
  # access parameter that belong to the group :name
  mean(bbb.σ²)
  plot(gg)
## iid model
@model logistic_model(X, Z₂, y, N) = begin
  # priors for the fixed effects
  β ~ filldist(Normal(0, 10), 2)
   # Prior for group effects
  σ² ~ Uniform(0, 100) # Non-informstive prior, see  7.1 Prior distributions for variance parameters in Prior distributions for variance parameters in hierarchical models by Andrew Gelman

  # Normal dist is parameterized by the sd
  u₂ ~ MvNormal(zeros(311), σ²*diagm(ones(311))) # MvNormal internally applies abs2(σ)
  # model

    p = logistic.(X * β + Z₂ * u₂)
    y .~ Binomial.(N, p)
    return σ²
end
  chn = sample(logistic_model(X.m, Z₂.m, df.CobrilhaInc, df.Ntrials), NUTS(), 1000)
describe(chn)
bb = get(chn, :σ²)

# access parameter that belong to the group :name
mean(bb.σ²)
  gg = group(chn, :σ²)
  plot(gg)
