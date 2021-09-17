# NULL-model

mf = ModelFrame(@formula(CobrilhaInc ~ 1), df)
X = ModelMatrix(mf)


@model logistic_model2(y, X, N) = begin
  # parameters
  # priors for the scalar parameters
  β0 ~ filldist(Normal(0, 10), 1)

# binomial model

p = logistic.(X * β0) # + Z₂ * u₂
y .~ Binomial.(N, p)
return p
end

  chn2 = sample(logistic_model2(df.CobrilhaInc, X.m, df.Ntrials), NUTS(), 3000)
  # chn2 = chn[500:3000,:,:] # burn-in
  plot(chn2)

  mean(df.CobrilhaInc ./ df.Ntrials)

  # generate quantities (p)
  mod2 = logistic_model2(df.CobrilhaInc, X.m, df.Ntrials)
  gen2 = generated_quantities(mod2, chn2)
mean(gen2)[1]

# binomial log likelihood function
l(x, n, p) = x * log(p) + (n - x) * log(1 - p) # I did not the cte because is statistically irrelevant log(n! / x!(n - x)!)
logk2 = zeros(Float64, length(df.CobrilhaInc))

for i in eachindex(df.CobrilhaInc)
  logk2[i] = l(df.CobrilhaInc[i], df.Ntrials[i], mean(gen2)[1])
end
logk_null = sum(logk2)
