##############################################################################
# Isolate the ~100 MeV "bump" question from the unfolding step entirely.
#
# Forward-fold the two-channel phase-space model through the response
# matrix (predicted_reco = M * x_true), add an optional flat RECO-space
# background, and compare directly to the RAW measured counts via a Poisson
# likelihood -- no EM/RL unfolding involved anywhere. This tests whether
# the raw counts near 96-105 MeV are consistent with pure Poisson noise
# around the model, independent of anything the unfolding's efficiency
# correction does to the unfolded spectrum.
##############################################################################

using Statistics, Distributions, StatsBase, BAT, DensityInterface, IntervalSets
using LinearAlgebra, CSV, DataFrames, Printf, Plots
using Plots.PlotMeasures

include(joinpath(@__DIR__, "..", "src", "DetectorResponse.jl"))
include(joinpath(@__DIR__, "..", "src", "PhysicsModel.jl"))
using .DetectorResponse, .PhysicsModel

const DATA_LO, DATA_HI = 57.0, 105.0
const BUFFER_LO = 50.5
const DATA_PATH = joinpath(@__DIR__, "..", "data", "1992_Al_full.csv")
const FIGURE_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURE_DIR)

default(fontfamily="Computer Modern", framestyle=:box, grid=false,
        tick_direction=:in, minorticks=true, dpi=300, size=(900, 620),
        guidefontsize=13, tickfontsize=11, legendfontsize=10, titlefontsize=13, margin=5mm)

data = CSV.read(DATA_PATH, DataFrame)
data_mask = (data.energy .>= DATA_LO) .& (data.energy .<= DATA_HI)
reco_energy = data.energy[data_mask]
counts_raw = data.counts[data_mask]
true_energy = vcat(collect(BUFFER_LO:1.0:(reco_energy[1] - 1.0)), reco_energy)
M = DetectorResponse.response_matrix(true_energy, reco_energy)

forward_predict(C0, C1, C_bkg) = M * combined_model(true_energy, C0, C1) .+ C_bkg

function fit_poisson(; float_bkg::Bool)
    prior = float_bkg ?
        distprod(C_0 = Truncated(Normal(18_000, 10_000), 0.0, Inf),
                  C_1 = Truncated(Normal(600_000, 100_000), 0.0, Inf),
                  C_bkg = 0.0..10.0) :
        distprod(C_0 = Truncated(Normal(18_000, 10_000), 0.0, Inf),
                  C_1 = Truncated(Normal(600_000, 100_000), 0.0, Inf))
    likelihood = if float_bkg
        let observed = counts_raw
            logfuncdensity(p -> sum(logpdf.(Poisson.(max.(forward_predict(p.C_0, p.C_1, p.C_bkg), 1e-10)), observed)))
        end
    else
        let observed = counts_raw
            logfuncdensity(p -> sum(logpdf.(Poisson.(max.(forward_predict(p.C_0, p.C_1, 0.0), 1e-10)), observed)))
        end
    end
    samples = bat_sample(PosteriorMeasure(likelihood, prior),
        MCMCSampling(mcalg=MetropolisHastings(), nsteps=5*10^4, nchains=4)).result
    mode_s = mode(samples)
    C0, C1 = mode_s.C_0, mode_s.C_1
    C_bkg = float_bkg ? mode_s.C_bkg : 0.0
    return C0, C1, C_bkg, samples
end

println("Fitting raw counts directly (Poisson likelihood, forward-folded, no unfolding)...")
println("\n=== No background ===")
C0_nb, C1_nb, _, _ = fit_poisson(float_bkg=false)
BR0_nb, BR1_nb = branching_ratios(C0_nb, C1_nb, channel_integrals())
pred_nb = forward_predict(C0_nb, C1_nb, 0.0)
@printf("C0 = %.1f   C1 = %.1f   BR0 = %.2f%%\n", C0_nb, C1_nb, 100BR0_nb)

println("\n=== Floating flat reco-space background ===")
C0_b, C1_b, C_bkg_b, _ = fit_poisson(float_bkg=true)
BR0_b, BR1_b = branching_ratios(C0_b, C1_b, channel_integrals())
pred_b = forward_predict(C0_b, C1_b, C_bkg_b)
@printf("C0 = %.1f   C1 = %.1f   C_bkg = %.4f counts/MeV   BR0 = %.2f%%\n", C0_b, C1_b, C_bkg_b, 100BR0_b)

# --- Direct look at the tail: observed vs predicted raw counts, no-bkg model ---
println("\nE (MeV)   raw_observed   predicted(no bkg)   predicted(with bkg)   pull(no bkg) = (obs-pred)/sqrt(pred)")
for e in 95.5:1.0:104.5
    idx = findfirst(x -> x == e, reco_energy)
    idx === nothing && continue
    obs = counts_raw[idx]
    p_nb = pred_nb[idx]
    p_b = pred_b[idx]
    pull = (obs - p_nb) / sqrt(max(p_nb, 1e-10))
    @printf("%6.1f    %-12.0f  %-18.4f  %-20.4f  %.2f\n", e, obs, p_nb, p_b, pull)
end

chi2_tail_nb = sum(((counts_raw[i] - pred_nb[i])^2 / max(pred_nb[i], 1e-10)) for i in eachindex(reco_energy) if reco_energy[i] >= 95.5)
n_tail = count(e -> e >= 95.5, reco_energy)
@printf("\nTail region (95.5-104.5 MeV, %d bins) Pearson chi2 (no-bkg model) = %.2f  (chi2/bin = %.2f)\n",
    n_tail, chi2_tail_nb, chi2_tail_nb / n_tail)

# --- Plot ---------------------------------------------------------------------
p = plot(reco_energy, counts_raw, seriestype=:scatter, yerr=sqrt.(max.(counts_raw, 1.0)),
    label="Raw measured counts", markercolor=:black, markersize=4,
    xlabel="Measured Photon Energy (MeV)", ylabel="Counts",
    title="Forward-Folded Fit vs. Raw Data (no unfolding)")
plot!(p, reco_energy, pred_nb, lw=2.5, color=:blue, label=@sprintf("No background (BR0=%.2f%%)", 100BR0_nb))
plot!(p, reco_energy, pred_b, lw=2.5, color=:red, linestyle=:dash, label=@sprintf("Floating background (BR0=%.2f%%)", 100BR0_b))
savefig(p, joinpath(FIGURE_DIR, "forward_fold_raw_bump_check.pdf"))

p_log = plot(p, yscale=:log10, ylims=(0.3, :auto), title="Forward-Folded Fit vs. Raw Data (log scale)")
savefig(p_log, joinpath(FIGURE_DIR, "forward_fold_raw_bump_check_log.pdf"))

println("\nFigures saved to $FIGURE_DIR")
