##############################################################################
# Gibbs sampling vs. Metropolis-Hastings, on the identical unfolded data,
# priors, and fit range (65-99.5 MeV, no background -- the settled
# production configuration).
#
# The model is linear in (C0, C1): combined_model(E, C0, C1) = C0*n0(E) +
# C1*n1(E). With a Gaussian likelihood (full covariance Sigma) and
# independent Gaussian priors on C0, C1, the *untruncated* joint posterior
# is exactly multivariate normal (standard Bayesian linear regression /
# conjugate-Gaussian update). Truncating each prior to C >= 0 breaks that
# closed form for the joint, but each parameter's FULL CONDITIONAL
# (C_i | C_j, data) is still a truncated univariate normal -- so a Gibbs
# sampler here just means alternately drawing each parameter from its own
# truncated-normal conditional, using the standard multivariate-normal
# conditioning formula applied to the untruncated joint's precision matrix.
##############################################################################

using Statistics, Distributions, StatsBase, BAT, DensityInterface, IntervalSets
using LinearAlgebra, CSV, DataFrames, Printf, Plots, Random
using Plots.PlotMeasures

include(joinpath(@__DIR__, "..", "src", "DetectorResponse.jl"))
include(joinpath(@__DIR__, "..", "src", "Unfolding.jl"))
include(joinpath(@__DIR__, "..", "src", "PhysicsModel.jl"))
using .DetectorResponse, .Unfolding, .PhysicsModel

const DATA_LO, DATA_HI = 57.0, 105.0
const FIT_LO, FIT_HI = 65.0, 99.5
const BUFFER_LO = 50.5
const RIDGE_LAMBDA, RIDGE_FLOOR = 0.1, 1e-6
const DATA_PATH = joinpath(@__DIR__, "..", "data", "1992_Al_full.csv")
const FIGURE_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURE_DIR)

const PRIOR_MEAN = [18_000.0, 600_000.0]
const PRIOR_SIGMA = [10_000.0, 100_000.0]

default(fontfamily="Computer Modern", framestyle=:box, grid=false,
        tick_direction=:in, minorticks=true, dpi=300, size=(750, 650),
        guidefontsize=13, tickfontsize=11, legendfontsize=10, titlefontsize=13, margin=5mm)

# --- Unfold once, shared by both samplers -------------------------------------
data = CSV.read(DATA_PATH, DataFrame)
data_mask = (data.energy .>= DATA_LO) .& (data.energy .<= DATA_HI)
reco_energy = data.energy[data_mask]
counts_raw = data.counts[data_mask]
true_energy = vcat(collect(BUFFER_LO:1.0:(reco_energy[1] - 1.0)), reco_energy)
M = DetectorResponse.response_matrix(true_energy, reco_energy)
I0, I1 = channel_integrals()

println("Unfolding data (10000 covariance toys)...")
cov_unfolded, counts, _ = unfold_with_covariance(counts_raw, M; n_toys=10_000, iterations=10, regularization=true)
full_mask = (true_energy .>= DATA_LO) .& (true_energy .<= DATA_HI)
energy_full = true_energy[full_mask]
cov_full = cov_unfolded[full_mask, full_mask]
counts_full = counts[full_mask]

fit_mask = (energy_full .> FIT_LO) .& (energy_full .<= FIT_HI)
en = energy_full[fit_mask]
ct = counts_full[fit_mask]
bw = median(diff(en))
cov_fit = cov_full[fit_mask, fit_mask]
Sigma = Symmetric(cov_fit) + Diagonal(diag(cov_fit) .* RIDGE_LAMBDA .+ RIDGE_FLOOR)

# ============================================================ 1. BAT / Metropolis-Hastings
prior = distprod(C_0 = Truncated(Normal(PRIOR_MEAN[1], PRIOR_SIGMA[1]), 0.0, Inf),
                  C_1 = Truncated(Normal(PRIOR_MEAN[2], PRIOR_SIGMA[2]), 0.0, Inf))
likelihood = let observed = ct, Sigma = Sigma, bc = en, bw = bw
    logfuncdensity(p -> logpdf(MvNormal(combined_model(bc, p.C_0, p.C_1) .* bw, Sigma), observed))
end
println("Running BAT MCMC (Metropolis-Hastings)...")
mh_samples = bat_sample(PosteriorMeasure(likelihood, prior),
    MCMCSampling(mcalg=MetropolisHastings(), nsteps=5*10^4, nchains=4)).result
mh_C0 = getproperty.(mh_samples.v, :C_0)
mh_C1 = getproperty.(mh_samples.v, :C_1)
mh_w = FrequencyWeights(mh_samples.weight)

# ============================================================ 2. Hand-rolled Gibbs sampler
A = hcat(no_knockout.(en) .* bw, first_knockout.(en) .* bw)
SigInvA = Sigma \ A
Lambda_L = A' * SigInvA               # likelihood precision contribution
b_L = A' * (Sigma \ ct)               # likelihood linear term
Lambda_0 = Diagonal(1.0 ./ PRIOR_SIGMA .^ 2)
b_0 = PRIOR_MEAN ./ PRIOR_SIGMA .^ 2
Lambda_post = Lambda_L + Lambda_0     # untruncated joint posterior precision
mean_post = Lambda_post \ (b_L + b_0) # untruncated joint posterior mean

function gibbs_sample(Lambda, mu; n_iter=60_000, burn_in=5_000, seed=42)
    Random.seed!(seed)
    n = length(mu)
    theta = max.(mu, 0.0)             # start near the untruncated mean
    chain = zeros(n, n_iter)
    for it in 1:n_iter
        for i in 1:n
            others = setdiff(1:n, i)
            cond_var = 1.0 / Lambda[i, i]
            cond_mean = mu[i] - cond_var * sum(Lambda[i, j] * (theta[j] - mu[j]) for j in others)
            theta[i] = rand(Truncated(Normal(cond_mean, sqrt(cond_var)), 0.0, Inf))
        end
        chain[:, it] = theta
    end
    return chain[:, (burn_in+1):end]
end

println("Running hand-rolled Gibbs sampler...")
gibbs_chain = gibbs_sample(Lambda_post, mean_post)
gibbs_C0 = gibbs_chain[1, :]
gibbs_C1 = gibbs_chain[2, :]

# ============================================================ Compare
function br0_stats(C0s, C1s, w=nothing)
    norm_vec = C0s .* I0 .+ C1s .* I1
    BR0 = (C0s .* I0) ./ norm_vec
    if w === nothing
        return mean(BR0), std(BR0)
    else
        return mean(BR0, w), std(BR0, w)
    end
end

mh_BR0_mean, mh_BR0_std = br0_stats(mh_C0, mh_C1, mh_w)
gibbs_BR0_mean, gibbs_BR0_std = br0_stats(gibbs_C0, gibbs_C1)

println("\n=== Metropolis-Hastings (BAT.jl) ===")
@printf("C0 = %.1f +/- %.1f   C1 = %.1f +/- %.1f\n", mean(mh_C0, mh_w), std(mh_C0, mh_w), mean(mh_C1, mh_w), std(mh_C1, mh_w))
@printf("BR0 = %.4f +/- %.4f\n", mh_BR0_mean, mh_BR0_std)

println("\n=== Gibbs (hand-rolled, truncated-normal conditionals) ===")
@printf("C0 = %.1f +/- %.1f   C1 = %.1f +/- %.1f\n", mean(gibbs_C0), std(gibbs_C0), mean(gibbs_C1), std(gibbs_C1))
@printf("BR0 = %.4f +/- %.4f\n", gibbs_BR0_mean, gibbs_BR0_std)

@printf("\nDifference in BR0 mean: %.5f (%.2f%% of MH's std)\n", gibbs_BR0_mean - mh_BR0_mean, 100*(gibbs_BR0_mean - mh_BR0_mean)/mh_BR0_std)
@printf("Ratio of BR0 std (Gibbs/MH): %.3f\n", gibbs_BR0_std / mh_BR0_std)

results = DataFrame(
    sampler=["Metropolis-Hastings", "Gibbs"],
    C0_mean=[mean(mh_C0, mh_w), mean(gibbs_C0)],
    C0_std=[std(mh_C0, mh_w), std(gibbs_C0)],
    C1_mean=[mean(mh_C1, mh_w), mean(gibbs_C1)],
    C1_std=[std(mh_C1, mh_w), std(gibbs_C1)],
    BR0_mean=[mh_BR0_mean, gibbs_BR0_mean],
    BR0_std=[mh_BR0_std, gibbs_BR0_std],
)
CSV.write(joinpath(@__DIR__, "..", "gibbs_vs_mh_results.csv"), results)

# --- Comparison plot: 2D posterior scatter -------------------------------------
p = scatter(mh_C0[1:20:end], mh_C1[1:20:end], markersize=2, markeralpha=0.35,
    color=:steelblue, label="Metropolis-Hastings",
    xlabel="C0", ylabel="C1", title="Gibbs vs. Metropolis-Hastings Posterior")
scatter!(p, gibbs_C0[1:20:end], gibbs_C1[1:20:end], markersize=2, markeralpha=0.35,
    color=:orangered, label="Gibbs")
savefig(p, joinpath(FIGURE_DIR, "gibbs_vs_mh_posterior.pdf"))

println("\nSaved: gibbs_vs_mh_results.csv, figures/gibbs_vs_mh_posterior.pdf")
