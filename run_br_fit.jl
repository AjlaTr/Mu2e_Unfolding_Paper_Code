##############################################################################
# Unfold the 1992 Al RMC spectrum and fit BR0/BR1 via BAT over a configurable
# energy window, prior type, and background treatment.
##############################################################################

using Statistics, Distributions, StatsBase, BAT, DensityInterface, IntervalSets
using LinearAlgebra, CSV, DataFrames, Printf, Plots
using Plots.PlotMeasures

include(joinpath(@__DIR__, "..", "src", "DetectorResponse.jl"))
include(joinpath(@__DIR__, "..", "src", "Unfolding.jl"))
include(joinpath(@__DIR__, "..", "src", "PhysicsModel.jl"))
using .DetectorResponse, .Unfolding, .PhysicsModel

# --- Configuration ----------------------------------------------------------
const FIT_LO, FIT_HI = 65.0, 105.0
const PRIOR_TYPE = :gaussian        # :gaussian or :uniform
const DATA_LO, DATA_HI = 57.0, 105.0
const BUFFER_LO = 50.5              # true-energy buffer below the lowest measured bin
const EM_ITERATIONS = 10
const RIDGE_LAMBDA, RIDGE_FLOOR = 0.1, 1e-6
const N_COV_TOYS = 10_000
const DATA_PATH = joinpath(@__DIR__, "..", "data", "1992_Al_full.csv")
const FIGURE_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURE_DIR)

default(fontfamily="Computer Modern", framestyle=:box, grid=false,
        tick_direction=:in, minorticks=true, dpi=300, size=(750, 550),
        guidefontsize=13, tickfontsize=11, legendfontsize=10, titlefontsize=13,
        margin=5mm)

# --- Load data and build the response matrix --------------------------------
data = CSV.read(DATA_PATH, DataFrame)
data_mask = (data.energy .>= DATA_LO) .& (data.energy .<= DATA_HI)
reco_energy = data.energy[data_mask]
counts_raw = data.counts[data_mask]

low_true_buffer = collect(BUFFER_LO:1.0:(reco_energy[1] - 1.0))
true_energy = vcat(low_true_buffer, reco_energy)
M = DetectorResponse.response_matrix(true_energy, reco_energy)
I0, I1 = channel_integrals()

# --- Unfold and restrict to the fit window -----------------------------------
println("Unfolding data ($N_COV_TOYS covariance toys)...")
cov_unfolded, counts, errors = unfold_with_covariance(counts_raw, M;
    n_toys=N_COV_TOYS, iterations=EM_ITERATIONS, regularization=true)

full_mask = (true_energy .>= DATA_LO) .& (true_energy .<= DATA_HI)
energy_full = true_energy[full_mask]
counts_full = counts[full_mask]
errors_full = errors[full_mask]
cov_full = cov_unfolded[full_mask, full_mask]

fit_mask = (energy_full .> FIT_LO) .& (energy_full .<= FIT_HI)
en = energy_full[fit_mask]
ct = counts_full[fit_mask]
bw = median(diff(en))
cov_stable = Symmetric(cov_full[fit_mask, fit_mask]) + Diagonal(diag(cov_full[fit_mask, fit_mask]) .* RIDGE_LAMBDA .+ RIDGE_FLOOR)

# --- Bayesian fit (C0, C1, C_bkg) --------------------------------------------
prior = PRIOR_TYPE == :uniform ?
    distprod(C_0 = 0.0..50_000.0, C_1 = 0.0..2_000_000.0, C_bkg = 0.0..5.0) :
    distprod(C_0 = Truncated(Normal(18_000, 10_000), 0.0, Inf),
              C_1 = Truncated(Normal(600_000, 100_000), 0.0, Inf),
              C_bkg = Truncated(Normal(0.1, 1.0), 0.0, Inf))

likelihood = let observed = ct, Sigma = cov_stable, bc = en, bw = bw
    logfuncdensity(p -> logpdf(MvNormal(combined_model(bc, p.C_0, p.C_1, p.C_bkg) .* bw, Sigma), observed))
end
posterior = PosteriorMeasure(likelihood, prior)

println("Running BAT MCMC...")
samples = bat_sample(posterior, MCMCSampling(mcalg=MetropolisHastings(), nsteps=5*10^4, nchains=4)).result

mode_s = mode(samples)
C0, C1, C_bkg = mode_s.C_0, mode_s.C_1, mode_s.C_bkg
BR0, BR1 = branching_ratios(C0, C1, (I0, I1))

model_curve = combined_model(en, C0, C1, C_bkg)
resid = model_curve .- ct
chi2 = dot(resid, cov_stable \ resid)
ndof = length(en) - 3
pulls = -resid ./ sqrt.(diag(cov_stable))

@printf("\nFit range %.1f-%.1f MeV, %s priors\n", FIT_LO, FIT_HI, PRIOR_TYPE)
@printf("C0 = %.1f   C1 = %.1f   C_bkg = %.4f counts/MeV\n", C0, C1, C_bkg)
@printf("BR0 = %.2f%%   BR1 = %.2f%%   reduced_chi2 = %.2f\n", 100BR0, 100BR1, chi2 / ndof)

# --- Plots --------------------------------------------------------------------
tag = "$(FIT_LO)to$(FIT_HI)_$(PRIOR_TYPE)"

p_spec = plot(energy_full, counts_full, yerr=errors_full, seriestype=:scatter,
    label="Unfolded data (full range)", markercolor=:gray, markersize=4,
    xlabel="True Energy (MeV)", ylabel="Counts",
    title="Fit $FIT_LO-$FIT_HI MeV, $(PRIOR_TYPE) priors")
plot!(p_spec, en, ct, yerr=errors_full[fit_mask], seriestype=:scatter,
    label="Unfolded data (fit range)", markercolor=:black, markersize=4)
plot!(p_spec, en, model_curve, lw=2, color=:blue,
    label=@sprintf("Fit (BR0=%.2f%%, C_bkg=%.4f)", 100BR0, C_bkg))
vspan!(p_spec, [FIT_LO, FIT_HI], alpha=0.06, color=:green, label="Fit range")
savefig(p_spec, joinpath(FIGURE_DIR, "fit_spectrum_$tag.pdf"))

p_pull = plot(en, pulls, seriestype=:scatter, markercolor=:blue, markersize=4,
    xlabel="True Energy (MeV)", ylabel="Pull (sigma)",
    title="Per-Bin Pulls, $FIT_LO-$FIT_HI MeV", legend=false)
hline!(p_pull, [0], color=:black, lw=1)
hline!(p_pull, [-1, 1], color=:gray, linestyle=:dash, lw=1)
hline!(p_pull, [-3, 3], color=:red, linestyle=:dot, lw=1)
savefig(p_pull, joinpath(FIGURE_DIR, "fit_pulls_$tag.pdf"))

println("Figures saved to $FIGURE_DIR (tag: $tag)")
