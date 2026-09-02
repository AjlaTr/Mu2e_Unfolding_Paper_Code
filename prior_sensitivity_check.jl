##############################################################################
# Prior-informativeness check: fix the Gaussian prior means for C0, C1 and
# progressively relax (widen) their sigmas, re-running the BAT fit at each
# width. For each level, compare the posterior's own std dev to the prior's
# std dev (ratio = posterior_std / prior_std):
#   ratio << 1  -> the data is constraining that parameter (posterior is
#                  much tighter than what the prior alone would give)
#   ratio ~ 1   -> the posterior just reproduces the prior; the data isn't
#                  adding information beyond what was assumed
#
# The unfolding step is independent of the prior, so it's done once and
# reused across all prior widths (unlike the BAT fit, which is rerun at
# each level).
##############################################################################

using Statistics, Distributions, StatsBase, BAT, DensityInterface, IntervalSets
using LinearAlgebra, CSV, DataFrames, Printf, Plots
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

const PRIOR_MEAN_C0, PRIOR_MEAN_C1 = 18_000.0, 600_000.0
const RELATIVE_WIDTHS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]  # fraction of the mean

default(fontfamily="Computer Modern", framestyle=:box, grid=false,
        tick_direction=:in, minorticks=true, dpi=300, size=(800, 550),
        guidefontsize=13, tickfontsize=11, legendfontsize=10, titlefontsize=13, margin=5mm)

# --- Unfold once (independent of the prior) ----------------------------------
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
cov_stable = Symmetric(cov_fit) + Diagonal(diag(cov_fit) .* RIDGE_LAMBDA .+ RIDGE_FLOOR)

# --- Re-fit at each prior width -----------------------------------------------
results = DataFrame(rel_width=Float64[], sigma_C0_prior=Float64[], sigma_C1_prior=Float64[],
                     sigma_C0_post=Float64[], sigma_C1_post=Float64[],
                     ratio_C0=Float64[], ratio_C1=Float64[], BR0=Float64[])

println("\n$(rpad("rel_width", 10)) $(rpad("ratio_C0", 10)) $(rpad("ratio_C1", 10)) BR0")
for w in RELATIVE_WIDTHS
    sigma_C0 = w * PRIOR_MEAN_C0
    sigma_C1 = w * PRIOR_MEAN_C1
    prior = distprod(C_0 = Truncated(Normal(PRIOR_MEAN_C0, sigma_C0), 0.0, Inf),
                      C_1 = Truncated(Normal(PRIOR_MEAN_C1, sigma_C1), 0.0, Inf))
    likelihood = let observed = ct, Sigma = cov_stable, bc = en, bw = bw
        logfuncdensity(p -> logpdf(MvNormal(combined_model(bc, p.C_0, p.C_1) .* bw, Sigma), observed))
    end
    samples = bat_sample(PosteriorMeasure(likelihood, prior),
        MCMCSampling(mcalg=MetropolisHastings(), nsteps=5*10^4, nchains=4)).result

    C0s = getproperty.(samples.v, :C_0)
    C1s = getproperty.(samples.v, :C_1)
    sw = FrequencyWeights(samples.weight)
    post_std_C0 = std(C0s, sw)
    post_std_C1 = std(C1s, sw)
    ratio_C0 = post_std_C0 / sigma_C0
    ratio_C1 = post_std_C1 / sigma_C1

    C0_mode, C1_mode = mean(C0s, sw), mean(C1s, sw)
    BR0, _ = branching_ratios(C0_mode, C1_mode, (I0, I1))

    push!(results, (w, sigma_C0, sigma_C1, post_std_C0, post_std_C1, ratio_C0, ratio_C1, BR0))
    @printf("%-10.2f %-10.3f %-10.3f %.4f\n", w, ratio_C0, ratio_C1, BR0)
end

CSV.write(joinpath(@__DIR__, "..", "prior_sensitivity_results.csv"), results)

p = plot(results.rel_width, results.ratio_C0, marker=:circle, lw=2, label="C0",
    xscale=:log10, xlabel="Prior width (fraction of mean)", ylabel="Posterior std / Prior std",
    title="Prior Informativeness Check", legend=:topleft, ylims=(0, 1.15))
plot!(p, results.rel_width, results.ratio_C1, marker=:square, lw=2, label="C1")
hline!(p, [0.8], color=:gray, linestyle=:dash, label="ratio = 0.8 (data-constrained threshold)")
hline!(p, [1.0], color=:black, linestyle=:dot, label="ratio = 1 (prior-dominated)")
savefig(p, joinpath(FIGURE_DIR, "prior_sensitivity_check.pdf"))
println("\nSaved: prior_sensitivity_results.csv, figures/prior_sensitivity_check.pdf")
