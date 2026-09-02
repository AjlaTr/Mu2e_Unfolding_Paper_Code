##############################################################################
# Re-fit with the background modeled as C_bkg * (1/eps(E_true)) instead of a
# flat constant in true-energy space.
#
# Motivation: EM/RL unfolding's efficiency-correction step divides by
# eps(E_true) at every iteration. A background that's actually flat where it
# physically originates (measured/reco space -- most instrumental/accidental
# backgrounds are) would NOT come out flat in true-energy space after
# unfolding; to first order it picks up the same 1/eps(E_true) shape the
# signal's efficiency correction imposes. A flat CONSTANT in true-energy
# space implicitly assumes the background was already flat in true energy
# to begin with, which is a much less physically motivated assumption.
#
# This uses the same settled configuration (matrix_DR_absolute /
# response_matrix, buffer to 50.5 MeV, EM iter=10/reg=true, ridge=0.1,
# 65-105 MeV fit range including the ~100 MeV tail) as the other floating-
# background checks this session, swapping only the background's shape.
##############################################################################

using Statistics, Distributions, StatsBase, BAT, DensityInterface, IntervalSets
using LinearAlgebra, CSV, DataFrames, Printf, Plots
using Plots.PlotMeasures

include(joinpath(@__DIR__, "..", "src", "DetectorResponse.jl"))
include(joinpath(@__DIR__, "..", "src", "Unfolding.jl"))
include(joinpath(@__DIR__, "..", "src", "PhysicsModel.jl"))
using .DetectorResponse, .Unfolding, .PhysicsModel

const DATA_LO, DATA_HI = 57.0, 105.0
const FIT_LO, FIT_HI = 65.0, 105.0
const BUFFER_LO = 50.5
const RIDGE_LAMBDA, RIDGE_FLOOR = 0.1, 1e-6
const DATA_PATH = joinpath(@__DIR__, "..", "data", "1992_Al_full.csv")
const FIGURE_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURE_DIR)

default(fontfamily="Computer Modern", framestyle=:box, grid=false,
        tick_direction=:in, minorticks=true, dpi=300, size=(900, 620),
        guidefontsize=13, tickfontsize=11, legendfontsize=10, titlefontsize=13, margin=5mm)

# --- Unfold ---------------------------------------------------------------
data = CSV.read(DATA_PATH, DataFrame)
data_mask = (data.energy .>= DATA_LO) .& (data.energy .<= DATA_HI)
reco_energy = data.energy[data_mask]
counts_raw = data.counts[data_mask]
true_energy = vcat(collect(BUFFER_LO:1.0:(reco_energy[1] - 1.0)), reco_energy)
M = DetectorResponse.response_matrix(true_energy, reco_energy)
I0, I1 = channel_integrals()

println("Unfolding data (10000 covariance toys)...")
cov_unfolded, counts, errors = unfold_with_covariance(counts_raw, M; n_toys=10_000, iterations=10, regularization=true)
full_mask = (true_energy .>= DATA_LO) .& (true_energy .<= DATA_HI)
energy_full = true_energy[full_mask]
counts_full = counts[full_mask]
errors_full = errors[full_mask]
cov_full = cov_unfolded[full_mask, full_mask]

# --- Efficiency-shaped background template ---------------------------------
eps_full = vec(sum(M, dims=1))[full_mask]     # eps(E_true), aligned with energy_full
bkg_shape(E) = 1.0 / eps_full[findfirst(x -> x == E, energy_full)]

fit_mask = (energy_full .> FIT_LO) .& (energy_full .<= FIT_HI)
en = energy_full[fit_mask]
ct = counts_full[fit_mask]
bw = median(diff(en))
cov_fit = cov_full[fit_mask, fit_mask]
cov_stable = Symmetric(cov_fit) + Diagonal(diag(cov_fit) .* RIDGE_LAMBDA .+ RIDGE_FLOOR)
bkg_shape_fit = bkg_shape.(en)

println("\n1/eps(E_true) background template over the fit range:")
@printf("  min = %.2f (at E=%.1f)   max = %.2f (at E=%.1f)\n",
    minimum(bkg_shape_fit), en[argmin(bkg_shape_fit)], maximum(bkg_shape_fit), en[argmax(bkg_shape_fit)])

model_efftemplate(E, C0, C1, C_bkg, shape) = combined_model(E, C0, C1) .+ C_bkg .* shape

prior = distprod(C_0 = Truncated(Normal(18_000, 10_000), 0.0, Inf),
                  C_1 = Truncated(Normal(600_000, 100_000), 0.0, Inf),
                  C_bkg = 0.0..500.0)
likelihood = let observed = ct, Sigma = cov_stable, bc = en, bw = bw, shape = bkg_shape_fit
    logfuncdensity(p -> logpdf(MvNormal(model_efftemplate(bc, p.C_0, p.C_1, p.C_bkg, shape) .* bw, Sigma), observed))
end
println("\nRunning BAT MCMC (efficiency-shaped background)...")
samples = bat_sample(PosteriorMeasure(likelihood, prior),
    MCMCSampling(mcalg=MetropolisHastings(), nsteps=5*10^4, nchains=4)).result

mode_s = mode(samples)
C0, C1, C_bkg = mode_s.C_0, mode_s.C_1, mode_s.C_bkg
BR0, BR1 = branching_ratios(C0, C1, (I0, I1))

model_curve = model_efftemplate(en, C0, C1, C_bkg, bkg_shape_fit)
resid = model_curve .- ct
chi2 = dot(resid, cov_stable \ resid)
ndof = length(en) - 3

@printf("\n=== Efficiency-shaped background (1/eps template), 65-105 MeV ===\n")
@printf("C0 = %.1f   C1 = %.1f   C_bkg (scale) = %.4f\n", C0, C1, C_bkg)
@printf("BR0 = %.2f%%   BR1 = %.2f%%   reduced_chi2 = %.2f\n", 100BR0, 100BR1, chi2 / ndof)
@printf("Background contribution at E=65: %.4f counts/MeV   at E=100: %.4f counts/MeV   at E=105: %.4f counts/MeV\n",
    C_bkg * bkg_shape(65.5), C_bkg * bkg_shape(100.5), C_bkg * bkg_shape(104.5))

# --- Compare against the flat-constant background result for reference -----
prior_flat = distprod(C_0 = Truncated(Normal(18_000, 10_000), 0.0, Inf),
                       C_1 = Truncated(Normal(600_000, 100_000), 0.0, Inf),
                       C_bkg = Truncated(Normal(0.1, 1.0), 0.0, Inf))
likelihood_flat = let observed = ct, Sigma = cov_stable, bc = en, bw = bw
    logfuncdensity(p -> logpdf(MvNormal(combined_model(bc, p.C_0, p.C_1, p.C_bkg) .* bw, Sigma), observed))
end
println("\nRunning BAT MCMC (flat-constant background, for comparison)...")
samples_flat = bat_sample(PosteriorMeasure(likelihood_flat, prior_flat),
    MCMCSampling(mcalg=MetropolisHastings(), nsteps=5*10^4, nchains=4)).result
mode_flat = mode(samples_flat)
BR0_flat, BR1_flat = branching_ratios(mode_flat.C_0, mode_flat.C_1, (I0, I1))
resid_flat = combined_model(en, mode_flat.C_0, mode_flat.C_1, mode_flat.C_bkg) .- ct
chi2_flat = dot(resid_flat, cov_stable \ resid_flat)
@printf("\n=== Flat-constant background, 65-105 MeV (for comparison) ===\n")
@printf("C0 = %.1f   C1 = %.1f   C_bkg = %.4f\n", mode_flat.C_0, mode_flat.C_1, mode_flat.C_bkg)
@printf("BR0 = %.2f%%   BR1 = %.2f%%   reduced_chi2 = %.2f\n", 100BR0_flat, 100BR1_flat, chi2_flat / ndof)

# --- Plot -------------------------------------------------------------------
p = plot(energy_full, counts_full, yerr=errors_full, seriestype=:scatter,
    label="Unfolded 1992 Al data", markercolor=:black, markersize=4,
    xlabel="True Energy (MeV)", ylabel="Counts",
    title="Efficiency-Shaped vs. Flat Background")
plot!(p, en, model_curve, lw=2.5, color=:magenta,
    label=@sprintf("1/eps(E) background (BR0=%.2f%%)", 100BR0))
plot!(p, en, combined_model(en, mode_flat.C_0, mode_flat.C_1, mode_flat.C_bkg), lw=2.5, color=:gray, linestyle=:dash,
    label=@sprintf("Flat background (BR0=%.2f%%)", 100BR0_flat))
vspan!(p, [FIT_LO, FIT_HI], alpha=0.06, color=:green, label="Fit range")
savefig(p, joinpath(FIGURE_DIR, "efficiency_shaped_vs_flat_background.pdf"))
println("\nFigure saved to $(joinpath(FIGURE_DIR, "efficiency_shaped_vs_flat_background.pdf"))")
