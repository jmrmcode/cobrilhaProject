# loglikelihhod function
l(x, n, p) = x * log(p) + (n - x) * log(1 - p)# + log(factorial(n) / (factorial(x)*(factorial(n - x)))) # I did not the cte because is statistically irrelevant log(n! / x!(n - x)!)

# loglikelihood across iterations
lp = zeros(3000) # resulting logpdf values
for i = 1:3000
    # lp[i] = sum(map((x, n, p) -> l(x, n, p), df.CobrilhaInc[df.yearIndx .> 1], df.Ntrials[df.yearIndx .> 1], mean(qq[i][df.yearIndx .> 1]))) # Compute the sum of log-densities at each iteration of MCMC output for stochastic nodes.
      lp[i] = sum(l(df.CobrilhaInc[df.yearIndx .> 1], df.Ntrials[df.yearIndx .> 1], mean(qq[i][df.yearIndx .> 1])))
end


# D = Deviance = -2 * loglikehood
D = -2 * lp # deviance
Dbar = mean(D)

# BINOMIAL effective number of parameters pᵩ = D̄ + D_θ_hat = mean deviance - deviance of the parameters mean
# discard probs when year = 1
mm = zeros(3000)
for i in 1:3000 mm[i] = mean(qq[i][df.yearIndx .> 1]) end
Dhat = -2*sum(l(df.CobrilhaInc[df.yearIndx .> 1], df.Ntrials[df.yearIndx .> 1], mean(mm)))
Pd = Dbar - Dhat # effective number of parameters
# deviance information criterion (DIC)
DIC = Dhat + 2*Pd
DIC = Dbar + Pd
