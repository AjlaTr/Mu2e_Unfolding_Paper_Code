##############################################################################
# Full unfolded spectrum table (57-105 MeV) plus a background-overlay plot:
# the fitted phase-space model (65-99.5 MeV) with a fixed flat background,
# compared against the full unfolded spectrum.
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
const BKG_RATE = 0.196   # counts/MeV, fixed overlay for the background-check plot
const DATA_PATH = joinpath(@__DIR__, "..", "data", "1992_Al_full.csv")
const FIGURE_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURE_DIR)

default(fontfamily="Computer Modern", framestyle=:box, grid=false,
        tick_direction=:in, minorticks=true, dpi=300, size=(750, 550),
        guidefontsize=13, tickfontsize=11, legendfontsize=10, titlefontsize=13, margin=5mm)

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

# --- Table --------------------------------------------------------------------
table_df = DataFrame(E_true_MeV=energy_full, unfolded_counts=round.(counts_full, digits=2),
                      unfolded_error=round.(errors_full, digits=2))
CSV.write(joinpath(@__DIR__, "..", "unfolded_spectrum_table.csv"), table_df)
println("Unfolded spectrum table saved ($(Int(DATA_LO))-$(Int(DATA_HI)) MeV, $(nrow(table_df)) bins)")

# --- Fit (no floating background) --------------------------------------------
fit_mask = (energy_full .> FIT_LO) .& (energy_full .<= FIT_HI)
en = energy_full[fit_mask]
ct = counts_full[fit_mask]
bw = median(diff(en))
cov_fit = cov_full[fit_mask, fit_mask]
cov_stable = Symmetric(cov_fit) + Diagonal(diag(cov_fit) .* RIDGE_LAMBDA .+ RIDGE_FLOOR)

prior = distprod(C_0 = Truncated(Normal(18_000, 10_000), 0.0, Inf),
                  C_1 = Truncated(Normal(600_000, 100_000), 0.0, Inf))
likelihood = let observed = ct, Sigma = cov_stable, bc = en, bw = bw
    logfuncdensity(p -> logpdf(MvNormal(combined_model(bc, p.C_0, p.C_1) .* bw, Sigma), observed))
end
println("Running BAT MCMC ($FIT_LO-$FIT_HI MeV)...")
samples = bat_sample(PosteriorMeasure(likelihood, prior), MCMCSampling(mcalg=MetropolisHastings(), nsteps=5*10^4, nchains=4)).result

mode_s = mode(samples)
C0, C1 = mode_s.C_0, mode_s.C_1
BR0, BR1 = branching_ratios(C0, C1, (I0, I1))
resid = combined_model(en, C0, C1) .- ct
chi2 = dot(resid, cov_stable \ resid)
ndof = length(en) - 2
@printf("\nC0 = %.1f   C1 = %.1f\n", C0, C1)
@printf("BR0 = %.2f%%   BR1 = %.2f%%   reduced_chi2 = %.2f   p = %.4f\n", 100BR0, 100BR1, chi2 / ndof, ccdf(Chisq(ndof), chi2))

# --- Background-overlay plot ---------------------------------------------------
model_with_bkg = combined_model(energy_full, C0, C1) .+ BKG_RATE
resid_bkg = model_with_bkg[fit_mask] .- ct
chi2_bkg = dot(resid_bkg, cov_stable \ resid_bkg)
reduced_chi2_bkg = chi2_bkg / ndof
@printf("With fixed background=%.3f counts/MeV overlay: reduced_chi2 = %.2f   p = %.4f\n",
        BKG_RATE, reduced_chi2_bkg, ccdf(Chisq(ndof), chi2_bkg))

p_bkg = plot(energy_full, counts_full, yerr=errors_full, seriestype=:scatter,
    label="Unfolded data (1992 Al)", markercolor=:black, markersize=4,
    xlabel="True Energy (MeV)", ylabel="Counts",
    title="Phase-Space Model + Background vs. Unfolded 1992 Data")
plot!(p_bkg, energy_full, model_with_bkg, lw=2, color=:blue,
    label=@sprintf("Fit + bkg=%.3f counts/MeV (chi2_nu=%.2f)", BKG_RATE, reduced_chi2_bkg))
hline!(p_bkg, [BKG_RATE], color=:gray, linestyle=:dot, lw=1, label="Flat background")
vspan!(p_bkg, [FIT_LO, FIT_HI], alpha=0.08, color=:green, label="Fit range")
savefig(p_bkg, joinpath(FIGURE_DIR, "spectrum_with_background_overlay.pdf"))
println("Figure saved to $(joinpath(FIGURE_DIR, "spectrum_with_background_overlay.pdf"))")
