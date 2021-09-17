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
using JLSO


## property levels, account for NAs

@model logistic_model(y, spdeiJAN, Espe9yMean, pptAPR3LAG, tmmnJAN3LAG, tmmxOCT2LAG, vciJUL0LAG, N) = begin

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

    # Gaussian scalar function (gives a numer between 0 an 1)
    # account for missing and zero values (just in thickness) by sampling from Normal(0,1)
    if ismissing(spdeiJAN[i]) x1 ~ Normal(0, 1); spei = exp((-0.5 / v_spei) * (x1 - m_spei)^2) else spei = exp((-0.5 / v_spei) * (spdeiJAN[i] - m_spei)^2) end
    if ismissing(Espe9yMean[i]) x2 ~ Normal(0, 1); thickness = exp((-0.5 / v_thickness) * (x2 - m_thickness)^2) else thickness = exp((-0.5 / v_thickness) * (Espe9yMean[i] - m_thickness)^2) end # || df.Espe9yMean[i] == 0
    if ismissing(pptAPR3LAG[i]) x3 ~ Normal(0, 1); ppt = exp((-0.5 / v_ppt) * (x3 - m_ppt)^2) else ppt = exp((-0.5 / v_ppt) * (pptAPR3LAG[i] - m_ppt)^2) end
    if ismissing(tmmnJAN3LAG[i]) x4 ~ Normal(0, 1); tmmn = exp((-0.5 / v_tmmn) * (x4 - m_tmmn)^2) else tmmn = exp((-0.5 / v_tmmn) * (tmmnJAN3LAG[i] - m_tmmn)^2) end
    if ismissing(tmmxOCT2LAG[i]) x5 ~ Normal(0, 1); tmmx = exp((-0.5 / v_tmmx) * (x5 - m_tmmx)^2) else tmmx = exp((-0.5 / v_tmmx) * (tmmxOCT2LAG[i] - m_tmmx)^2) end
    if ismissing(vciJUL0LAG[i]) x6 ~ Normal(0, 1); vci = exp((-0.5 / v_vci) * (x6 - m_vci)^2) else vci = exp((-0.5 / v_vci) * (vciJUL0LAG[i] - m_vci)^2) end

# binomial model

    p = spei * thickness * ppt * tmmn * tmmx * vci
    y[i] ~ Binomial(N[i], p)

end #for
end #begin

  chn = sample(logistic_model(df.CobrilhaInc, df.spdeiJAN, df.Espe9yMean, df.pptAPR3LAG, df.tmmnJAN3LAG, df.tmmxOCT2LAG, df.vciJUL0LAG, df.Ntrials), NUTS(), 3000)
  # chn2 = chn[500:3000,:,:] # burn-in
  plot(chn)


## tree level

df = CSV.read("U:/juanmi/cobrilha_project/data/dfFULLCobrilha_treeScaleDEFscaled.csv", DataFrame; missingstring = "NA")
df = CSV.read("U:/juanmi/cobrilha_project/data/dfFULLCobrilha_propertyLevelNEWscaled.csv", DataFrame; missingstring = "NA")

@model logistic_model(y, y2, property, yearIndx, pptAPR3LAG, pptAPR2LAG, pptAPR1LAG, N) = begin # spdeiJAN, Espe9yMean, pptAPR3LAG, tmmnJAN3LAG, tmmxOCT2LAG, vciJUL0LAG

  # priors
  v_ppt ~ Uniform(0, 30)
  m_ppt ~ Truncated(Normal(0, 0.01), -1, 1)
  #=
  v_spei ~ Uniform(0, 30)
  m_spei ~ Truncated(Normal(0, 0.01), -1, 1)
  v_thickness ~ Uniform(0, 30)
  m_thickness ~ Truncated(Normal(0, 0.01), -1, 1)
  v_tmmn ~ Uniform(0, 30)
  m_tmmn ~ Truncated(Normal(0, 0.01),-1, 1)
  v_tmmx ~ Uniform(0, 30)
  m_tmmx ~ Truncated(Normal(0, 0.01), -1, 1)
  v_vci ~ Uniform(0, 30)
  m_vci ~ Truncated(Normal(0, 0.01), -1, 1)=#
  φ ~ Uniform(-1, 1)
  ω ~ Dirichlet(3, 1.0)


  # model
  for i in eachindex(Array(y))

    if yearIndx[i] == 1 continue end # this line is a work-around to avoid including the first year for each property in the model (as rquired by ar1) since I cannot just used 2:N as usually because there are multiple observations per year

    # Gaussian scalar function (gives a numer between 0 an 1)
    # account for missing and zero values (just in thickness) by sampling from Normal(0,1)
    if ismissing(pptAPR3LAG[i]) || ismissing(pptAPR2LAG[i]) || ismissing(pptAPR1LAG[i]) x ~ Normal(0, 1); ppt = exp((-0.5 / v_ppt) * (x - m_ppt)^2) else ppt = exp((-0.5 / v_ppt) * ((ω[1] * pptAPR3LAG[i] + ω[2] * pptAPR2LAG[i] + ω[3] * pptAPR1LAG[i]) - m_ppt)^2) end
#=    if ismissing(spdeiJAN[i]) x1 ~ Normal(0, 1); spei = exp((-0.5 / v_spei) * (x1 - m_spei)^2) else spei = exp((-0.5 / v_spei) * (spdeiJAN[i] - m_spei)^2) end
    if ismissing(Espe9yMean[i]) x2 ~ Normal(0, 1); thickness = exp((-0.5 / v_thickness) * (x2 - m_thickness)^2) else thickness = exp((-0.5 / v_thickness) * (Espe9yMean[i] - m_thickness)^2) end # || df.Espe9yMean[i] == 0
    if ismissing(tmmnJAN3LAG[i]) x4 ~ Normal(0, 1); tmmn = exp((-0.5 / v_tmmn) * (x4 - m_tmmn)^2) else tmmn = exp((-0.5 / v_tmmn) * (tmmnJAN3LAG[i] - m_tmmn)^2) end
    if ismissing(tmmxOCT2LAG[i]) x5 ~ Normal(0, 1); tmmx = exp((-0.5 / v_tmmx) * (x5 - m_tmmx)^2) else tmmx = exp((-0.5 / v_tmmx) * (tmmxOCT2LAG[i] - m_tmmx)^2) end
    if ismissing(vciJUL0LAG[i]) x6 ~ Normal(0, 1); vci = exp((-0.5 / v_vci) * (x6 - m_vci)^2) else vci = exp((-0.5 / v_vci) * (vciJUL0LAG[i] - m_vci)^2) end =#

# binomial model
# sum(...) is a work-around to avoid using i - 1 for the ar1 term

    p = logistic(φ * sum(y2[(property .== property[i]) .& (yearIndx .== (yearIndx[i] - 1))] )) * ppt  #spei * thickness *  * tmmn * tmmx * vci
    y[i] ~ Binomial(N[i], p)

end #for
end #begin

  chn = sample(logistic_model(df.CobrilhaInc, df.CobrilhaInc, df.property, df.yearIndx, df.pptAPR3LAG, df.pptAPR2LAGLS, df.pptAPR1LAGLS, df.Ntrials), NUTS(), 3000) # df.spdeiJAN, df.Espe9yMean, df.pptAPR3LAG, df.tmmnJAN3LAG, df.tmmxOCT2LAG, df.vciJUL0LAG
  plot(chn)

  write("U:/juanmi/cobrilha_project/models/pptAPR321LAGS.json", chn)
  # chn = read("U:/juanmi/cobrilha_project/models/chn.json", Chains)
  chn = Serialization.deserialize("U:/juanmi/cobrilha_project/models/pptAPR321LAGS.json")

  # make predictions
  m_lin_reg_test = logistic_model(Vector{Union{Missing, Float64}}(undef, length(df.CobrilhaInc)), # response
  df.CobrilhaInc,
  df.property,
  df.yearIndx,
  df.pptAPR3LAG,
  df.pptAPR2LAGLS, # thickness
  df.pptAPR1LAGLS,
  df.Ntrials)
  pp = predict(m_lin_reg_test, chn)
  # Get the mean predicted values.
  ys_pred = collect(vec(mean(pp.value; dims = 1))) ./ df.Ntrials

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

  savefig("pptAPR3LAGS.png") # save it
