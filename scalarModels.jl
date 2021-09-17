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
using Distributed
using SentinelArrays
using StaticArrays

# TREE LEVEL
#= df = CSV.read("D:/UM_Lab/Portugal/data/dfFULLCobrilha_treeScaleCLEANEDbyESPE9Y.csv", DataFrame; missingstring = "NA")[:, [:Espe9y,:property, :year, :CobrilhaInc, :pptAPR3LAG, :tmmnJAN3LAG, :tmmxOCT2LAG, :vciJUL0LAG, :spdeiJAN] ] # espe9y tree level
df = dropmissing(df, disallowmissing=true)
df.spdeiJAN = standardize(ZScoreTransform, df.spdeiJAN; dims = 1)
df.Espe9y = standardize(ZScoreTransform, df.Espe9y; dims = 1)
df.pptAPR3LAG = standardize(ZScoreTransform, df.pptAPR3LAG; dims = 1)
df.tmmnJAN3LAG = standardize(ZScoreTransform, df.tmmnJAN3LAG; dims = 1)
df.tmmxOCT2LAG = standardize(ZScoreTransform, df.tmmxOCT2LAG; dims = 1)
df.vciJUL0LAG = standardize(ZScoreTransform, df.vciJUL0LAG; dims = 1)
=#

# PROPERTY LEVEL
df = CSV.read("D:/UM_Lab/Portugal/data/ScaledTargetVariables.csv", DataFrame; missingstring = "NA")
df = CSV.read("D:/UM_Lab/Portugal/data/scaledCovariatesPropertyLevelNAs.csv", DataFrame; missingstring = "NA") # strongest correlated covariates, standardazed
#=df = df[!, Not(r"vci")] # remove all the columns having vci
df = df[!, Not(r"spdei")]
df = df[!, Not(r"tmmn")]
df = df[!, Not(r"tmmx")]
df = df[!, Not(r"ppt")]
df = df[!, Not(r"PDSI")] =#
df = dropmissing(df, disallowmissing=true)
## MODEL having the scalar

@model logistic_model(y, pptAPR3LAG, N) = begin
  # parameters
  # priors for the scalar parameters
  v_δ ~ Uniform(0, 30)
  m_δ ~ Truncated(Normal(0, 0.01), -1, 1)
#=
  # model
  for i in eachindex(Array(y))
        x ~ Normal(0, 1)
    # deal with missing in covariates
    if ismissing(pptAPR3LAG[i])
        # Initialize x when missing

    pptAPR3LAG[i] = x
end #if
+=#
# Gaussian scalar function
if ismissing(pptAPR3LAG[i]) x3 ~ Normal(0, 1); ppt = exp((-0.5 / v_δ) * (x3 - m_δ)^2) else ppt = exp((-0.5 / v_δ) * (pptAPR3LAG[i] - m_δ)^2) end

    #  δ = exp((-0.5 / v_δ) * (pptAPR3LAG[i] - m_δ)^2)

    # binomial model

    p = 1 * δ
    y[i] ~ Binomial(N[i], p)


end#for
end#begin

  chn = sample(logistic_model(df.CobrilhaInc, Array(df.pptAPR3LAG), df.Ntrials), NUTS(), 3000)
  plot(chn)

  # save MCMC chains as JSON
  write("D:/UM_Lab/Portugal/models/tmmnJAN3LAGS_propertyLevel.json", chn)
  chn = read("D:/UM_Lab/Portugal/models/tmmxOCT2LAGS_propertyLevel.json", Chains) # IMPORTANT: not to replace "Chains"
  # newChains = newChains[1000:2000,:,:]
  plot(newChains, seriestype = :traceplot)

# make predictions
m_lin_reg_test = logistic_model(Vector{Union{Missing, Float64}}(undef, length(df.CobrilhaInc)), df.pptAPR3LAG, df.Ntrials); # IMPORTANT: add df.tmmxOCT2LAG * 0.1 when modeling tmmn or tmmx
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
# IMPORTANT: below when modeling tmmx and tmmx I have to add * 01 throughout df.tmm... * 0.1
stdd = mapslices(std, means.value.data; dims = 1)[sortperm(df.pptAPR3LAG)]
plot(sort(df.pptAPR3LAG), scalavrF.(sort(df.pptAPR3LAG)), seriestype = :line, xlabel = "PPT (mm) April 3 years earlier", label = "", ylabel = "Performance (% max)", linecolor = "black") # yerror = stdd,
# plot!(sort(df.dfEspe9yMean), (ys_pred[sortperm(df.dfEspe9yMean)] ./ maximum(ys_pred)), yerror = stdd[sortperm(df.dfEspe9yMean)]./ df.Ntrials, seriestype = :scatter, label ="two")
plot!(sort(df.pptAPR3LAG), (ys_pred[sortperm(df.pptAPR3LAG)] ./ maximum(ys_pred)), yerror = stdd[sortperm(df.pptAPR3LAG)]./ df.Ntrials,
seriestype = :scatter, label ="", color = "gray", alpha = 0.5)

savefig("pptAPR3LAGS.png")
savefig("posteriorsPptApr3LAGS.png")

##
# AT THE TREE LEVEL
@model logistic_model(y, spdeiJAN, Espe9y, pptAPR3LAG, tmmnJAN3LAG, tmmxOCT2LAG, vciJUL0LAG) = begin #  Espe9yMean, pptAPR3LAG, tmmnJAN3LAG, tmmxOCT2LAG, vciJUL0LAG,
  # parameters
  # priors for the scalar parameters
  v_spei ~ Uniform(0, 30)
  m_spei ~ Truncated(Normal(0, 0.01), -1, 1)
  v_thickness ~ Uniform(0, 30)
  m_thickness ~ Truncated(Normal(0, 0.01), -1, 1)
  v_ppt ~ Uniform(0, 30)
  m_ppt ~ Truncated(Normal(0, 0.01), -1, 1)
  v_tmmn ~ Uniform(0, 30)
  m_tmmn ~ Truncated(Normal(0, 0.01),-1, 1)
  v_tmmx ~ Uniform(0, 30)
  m_tmmx ~ Truncated(Normal(0, 0.01), -1, 1)
  v_vci ~ Uniform(0, 30)
  m_vci ~ Truncated(Normal(0, 0.01), -1, 1)

  # model
  for i in eachindex(Array(y))

    # Gaussian scalar function
          δ = exp((-0.5 / v_δ) * (pptAPR3LAG[i] - m_δ)^2)

        # binomial model

        p = 1 * δ
        y[i] ~ Binomial(N[i], p)

    # Gaussian scalar function (gives a numer between 0 an 1)
      spei = exp((-0.5 / v_spei) * (spdeiJAN[i] - m_spei)^2)
      thickness = exp((-0.5 / v_thickness) * (Espe9y[i] - m_thickness)^2)
      ppt = exp((-0.5 / v_ppt) * (pptAPR3LAG[i] - m_ppt)^2)
      tmmn = exp((-0.5 / v_tmmn) * (tmmnJAN3LAG[i] - m_tmmn)^2)
      tmmx = exp((-0.5 / v_tmmx) * (tmmxOCT2LAG[i] - m_tmmx)^2)
      vci = exp((-0.5 / v_vci) * (vciJUL0LAG[i] - m_vci)^2)

# binomial model

    p = spei * thickness * ppt * tmmn * tmmx * vci
    y[i] ~ Binomial(1, p)

end # for
end # begin

  chn = sample(logistic_model(df.CobrilhaInc, df.spdeiJAN, df.Espe9y, df.pptAPR3LAG, df.tmmnJAN3LAG, df.tmmxOCT2LAG, df.vciJUL0LAG), NUTS(), 3000)
  # chn2 = chn[500:3000,:,:] # burn-in
  plot(chn)
  chn = read("D:/UM_Lab/Portugal/models/allCovariatesTreeLevel.json", Chains) # IMPORTANT: not to replace "Chains"

  # make predictions
  ind = rand(1:length(df.CobrilhaInc), 500)
  m_lin_reg_test = logistic_model(Vector{Union{Missing, Float64}}(undef, length(ind)), df.spdeiJAN[ind], df.Espe9y[ind], df.pptAPR3LAG[ind], df.tmmnJAN3LAG[ind], df.tmmxOCT2LAG[ind], df.vciJUL0LAG[ind])
  pp = predict(m_lin_reg_test, chn)
  # Get the mean predicted values.
  ys_pred = collect(vec(mean(pp.value; dims = 1)))
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
##

# PROPERTY LEVEL; ALL COVARIATES TOGETHER
@model logistic_model(y, spdeiJAN, Espe9yMean, pptAPR3LAG, tmmnJAN3LAG, tmmxOCT2LAG, vciJUL0LAG, N) = begin
  # parameters
  # priors for the scalar parameters
  v_spei ~ Uniform(0, 30)
  m_spei ~ Truncated(Normal(0, 0.01), -1, 1)
  v_thickness ~ Uniform(0, 30)
  m_thickness ~ Truncated(Normal(0, 0.01), -1, 1)
  v_ppt ~ Uniform(0, 30)
  m_ppt ~ Truncated(Normal(0, 0.01), -1, 1)
  v_tmmn ~ Uniform(0, 30)
  m_tmmn ~ Truncated(Normal(0, 0.01),-1, 1)
  v_tmmx ~ Uniform(0, 30)
  m_tmmx ~ Truncated(Normal(0, 0.01), -1, 1)
  v_vci ~ Uniform(0, 30)
  m_vci ~ Truncated(Normal(0, 0.01), -1, 1)
#  prob ~ Beta(1, 1)
#  p = prob * 1

  # model
  for i in eachindex(Array(y))

    # Gaussian scalar function (gives a numer between 0 an 1)
      spei = exp((-0.5 / v_spei) * (spdeiJAN[i] - m_spei)^2)
      thickness = exp((-0.5 / v_thickness) * (Espe9yMean[i] - m_thickness)^2)
      ppt = exp((-0.5 / v_ppt) * (pptAPR3LAG[i] - m_ppt)^2)
      tmmn = exp((-0.5 / v_tmmn) * (tmmnJAN3LAG[i] - m_tmmn)^2)
      tmmx = exp((-0.5 / v_tmmx) * (tmmxOCT2LAG[i] - m_tmmx)^2)
      vci = exp((-0.5 / v_vci) * (vciJUL0LAG[i] - m_vci)^2)

# binomial model

    p = spei * thickness * ppt * tmmn * tmmx * vci
    y[i] ~ Binomial(N[i], p)
  end
end

  chn = sample(logistic_model(df.CobrilhaInc, df.spdeiJAN, df.Espe9yMean, df.pptAPR3LAG, df.tmmnJAN3LAG, df.tmmxOCT2LAG, df.vciJUL0LAG, df.Ntrials), NUTS(), 3000)
  # chn2 = chn[500:3000,:,:] # burn-in
  plot(chn)

  # generate quantities (p)
  mod = logistic_model(df.CobrilhaInc, df.spdeiJAN, df.Espe9yMean, df.pptAPR3LAG, df.tmmnJAN3LAG, df.tmmxOCT2LAG, df.vciJUL0LAG, df.Ntrials)
  gen = generated_quantities(mod, chn)

  # save MCMC chains as JSON
  write("D:/UM_Lab/Portugal/models/allCovariatesTreeLevelNoeachindex.json", chn)
  chn = read("D:/UM_Lab/Portugal/models/allCovariatesPropertyLevel.json", Chains) # IMPORTANT: not to replace "Chains"

  # newChains = newChains[1000:2000,:,:]
  plot(newChains, seriestype = :traceplot)

## make predictions
#=
# all covariates together
m_lin_reg_test = logistic_model(Vector{Union{Missing, Float64}}(undef, length(df.CobrilhaInc)),df.spdeiJAN, df.Espe9yMean, df.pptAPR3LAG, df.tmmnJAN3LAG, df.tmmxOCT2LAG, df.vciJUL0LAG, df.Ntrials)
pp = predict(m_lin_reg_test, chn)
# Get the mean predicted values.
ys_pred = collect(vec(mean(pp.value; dims = 1))) ./ df.Ntrials
=#

# marginal predictions
m_lin_reg_test = logistic_model(Vector{Union{Missing, Float64}}(undef, length(df.CobrilhaInc)), # response
SentinelArray(zeros(Float64, length(df.CobrilhaInc))), # spei
df.Espe9yMean, # thickness
SentinelArray(zeros(Float64, length(df.CobrilhaInc))), # ppt
SentinelArray(zeros(Float64, length(df.CobrilhaInc))), # tmmn
SentinelArray(zeros(Float64, length(df.CobrilhaInc))), # tmmx
SentinelArray(zeros(Float64, length(df.CobrilhaInc))), # vci
df.Ntrials)
pp = predict(m_lin_reg_test, chn)
# Get the mean predicted values.
ys_pred = collect(vec(mean(pp.value; dims = 1))) ./ df.Ntrials
# Get the prediction error:
# errors = df.CobrilhaInc - ys_pred

# plot the scalar
m = mean(get(chn, :m_thickness).m_thickness)
v = mean(get(chn, :v_thickness).v_thickness)
function scalavrF(x)
  exp((-0.5 / v) * (x - m)^2)
end

means = group(pp, :y)
# IMPORTANT: below when modeling tmmx and tmmx I have to add * 01 throughout df.tmm... * 0.1
stdd = mapslices(std, means.value.data; dims = 1)[sortperm(df.Espe9yMean)]
plot(sort(df.Espe9yMean), scalavrF.(sort(df.Espe9yMean)), seriestype = :line, xlabel = "PPT (mm) April 3 years earlier", label = "", ylabel = "Performance (% max)", linecolor = "black", ylims = (0, 1.1)) # yerror = stdd,
# plot!(sort(df.dfEspe9yMean), (ys_pred[sortperm(df.dfEspe9yMean)] ./ maximum(ys_pred)), yerror = stdd[sortperm(df.dfEspe9yMean)]./ df.Ntrials, seriestype = :scatter, label ="two")
plot!(sort(df.Espe9yMean), (ys_pred[sortperm(df.Espe9yMean)] ./ maximum(ys_pred)), yerror = stdd[sortperm(df.Espe9yMean)]./ df.Ntrials,
seriestype = :scatter, label ="", color = "gray", alpha = 0.5)

savefig("pptAPR3LAGS.png")
savefig("posteriorsPptApr3LAGS.png")

# R²
observed = df.CobrilhaInc ./ df.Ntrials
ss_total = sum((observed .- mean(ys_pred)).^2)
ss_res = sum((observed .- ys_pred).^2)
R² = 1 - (ss_res / ss_total)

# R² in Gelman et al 2019 R-squared for Bayesian Regression Models
var(ys_pred) / (var(ys_pred) + var(observed - ys_pred))


# binomial log likelihood function
l(x, n, p) = x * log(p) + (n - x) * log(1 - p) # I did not the cte because is statistically irrelevant log(n! / x!(n - x)!)
logk = zeros(Float64, length(df.CobrilhaInc))

for i in eachindex(df.CobrilhaInc)
  logk[i] = l(df.CobrilhaInc[i], df.Ntrials[i], gen)
end
logk_full = sum(logk)

pseudo_R² = 1 - (logk_full / logk_null)
