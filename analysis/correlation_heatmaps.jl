##############################################################################
# Two correlation-matrix diagnostics:
#   1. Bin-to-bin correlation of the unfolded spectrum itself, from the
#      toy-based covariance estimate -- shows the collinearity structure
#      induced by the EM unfolding / near-singular response matrix.
#   2. Posterior correlation of the BAT fit parameters (C0, C1, C_bkg) for
#      the settled 65-105 MeV, Gaussian-prior configuration.
##############################################################################

using Statistics, StatsBase, Distributions, LinearAlgebra, DensityInterface, IntervalSets, BAT
using CSV, DataFrames, Printf, Plots
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
        tick_direction=:in, minorticks=true, dpi=300,
        guidefontsize=13, tickfontsize=10, legendfontsize=10, titlefontsize=13, margin=5mm)

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

# --- 1. Unfolded-spectrum bin-to-bin correlation matrix ---------------------
d = sqrt.(diag(cov_full))
spectrum_cormat = cov_full ./ (d * d')

p_spec_corr = heatmap(energy_full, energy_full, spectrum_cormat, c=:RdBu, clims=(-1, 1), aspect_ratio=1,
    xlabel="True Energy (MeV)", ylabel="True Energy (MeV)",
    title="Unfolded Spectrum Bin-to-Bin Correlation ($(Int(DATA_LO))-$(Int(DATA_HI)) MeV)",
    colorbar_title="Correlation",
    xlims=(energy_full[1] - 0.5, energy_full[end] + 0.5), ylims=(energy_full[1] - 0.5, energy_full[end] + 0.5))
savefig(p_spec_corr, joinpath(FIGURE_DIR, "correlation_heatmap_unfolded_spectrum.pdf"))
CSV.write(joinpath(@__DIR__, "..", "correlation_matrix_unfolded_spectrum.csv"),
    insertcols!(DataFrame(spectrum_cormat, Symbol.(string.(energy_full))), 1, :E_true_MeV => energy_full))

# --- 2. BAT posterior correlation matrix (C0, C1, C_bkg) --------------------
fit_mask = (energy_full .> FIT_LO) .& (energy_full .<= FIT_HI)
en = energy_full[fit_mask]
ct = counts[full_mask][fit_mask]
bw = median(diff(en))
cov_fit = cov_full[fit_mask, fit_mask]
cov_stable = Symmetric(cov_fit) + Diagonal(diag(cov_fit) .* RIDGE_LAMBDA .+ RIDGE_FLOOR)

prior = distprod(C_0 = Truncated(Normal(18_000, 10_000), 0.0, Inf),
                  C_1 = Truncated(Normal(600_000, 100_000), 0.0, Inf),
                  C_bkg = Truncated(Normal(0.1, 1.0), 0.0, Inf))
likelihood = let observed = ct, Sigma = cov_stable, bc = en, bw = bw
    logfuncdensity(p -> logpdf(MvNormal(combined_model(bc, p.C_0, p.C_1, p.C_bkg) .* bw, Sigma), observed))
end
println("Running BAT MCMC...")
samples = bat_sample(PosteriorMeasure(likelihood, prior), MCMCSampling(mcalg=MetropolisHastings(), nsteps=5*10^4, nchains=4)).result

param_matrix = hcat(getproperty.(samples.v, :C_0), getproperty.(samples.v, :C_1), getproperty.(samples.v, :C_bkg))
w = FrequencyWeights(samples.weight)
param_cov = StatsBase.cov(param_matrix, w; corrected=false)
dp = sqrt.(diag(param_cov))
param_cormat = param_cov ./ (dp * dp')

println("\nPosterior correlation matrix (C0, C1, C_bkg):")
show(stdout, "text/plain", round.(param_cormat, digits=3))
println()

labels = ["C0", "C1", "C_bkg"]
p_param_corr = heatmap(labels, labels, param_cormat, c=:RdBu, clims=(-1, 1), aspect_ratio=1,
    title="Posterior Parameter Correlation\n($FIT_LO-$FIT_HI MeV, floating background)",
    colorbar_title="Correlation", yflip=true, size=(600, 550))
for i in 1:3, j in 1:3
    annotate!(p_param_corr, j, i, Plots.text(@sprintf("%.2f", param_cormat[i, j]), 12,
        abs(param_cormat[i, j]) > 0.6 ? :white : :black))
end
savefig(p_param_corr, joinpath(FIGURE_DIR, "correlation_heatmap_fit_parameters.pdf"))

println("\nFigures saved to $FIGURE_DIR")
