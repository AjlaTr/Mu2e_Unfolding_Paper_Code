# Mu2e RMC Unfolding — Al Branching Ratio Extraction

Extracts the 0-neutron-knockout (BR0) and 1-neutron-knockout (BR1) branching
ratios from the 1992 Aluminum radiative muon capture (RMC) photon spectrum,
via Richardson-Lucy/EM unfolding followed by a Bayesian (BAT.jl) fit of the
two-channel phase-space model.

## Pipeline

1. **Response matrix** (`src/DetectorResponse.jl`) — piecewise Gaussian +
   exponential-tail detector response, built into a rectangular
   true-energy -> measured-energy matrix. The true-energy axis extends below
   the lowest measured bin (a "buffer" region) so photons born there can
   still smear up into the visible measured range.
2. **Unfolding** (`src/Unfolding.jl`) — EM/Richardson-Lucy iteration with
   efficiency correction and periodic 3-point smoothing regularization.
   Covariance on the unfolded spectrum is estimated via Poisson-bootstrap
   toys (`unfold_with_covariance`), so bin-to-bin correlations from
   response-matrix collinearity are captured, not assumed diagonal.
3. **Physics model** (`src/PhysicsModel.jl`) — normalized phase-space shapes
   for each knockout channel, and the BR0/BR1 extraction from fitted yields.
4. **Fit** (`analysis/run_br_fit.jl`) — Bayesian fit (BAT.jl, Metropolis-
   Hastings) of `C0`, `C1`, and an optional flat background against the
   unfolded spectrum, using the full unfolded covariance as the likelihood's
   noise model.

## Repository structure

```
src/
  DetectorResponse.jl   response matrix
  Unfolding.jl          EM unfolding + toy-based covariance
  PhysicsModel.jl        phase-space shapes, BR extraction
analysis/
  run_br_fit.jl                        BR0/BR1 fit (configurable range/priors)
  pull_test.jl                         closure/calibration test
  spectrum_table_and_background_check.jl   full spectrum table + bkg overlay
  correlation_heatmaps.jl              unfolded-spectrum and fit-parameter correlations
data/
  1992_Al_full.csv       measured 1992 Al RMC spectrum
```

## Requirements

Julia 1.9+, with:

```julia
using Pkg
Pkg.add(["Statistics", "Distributions", "StatsBase", "LinearAlgebra",
         "CSV", "DataFrames", "QuadGK", "Printf", "Plots", "Random",
         "Optim", "LaTeXStrings", "BAT", "DensityInterface", "IntervalSets"])
```

## Running

From the repo root:

```bash
julia analysis/run_br_fit.jl
julia analysis/pull_test.jl
julia analysis/spectrum_table_and_background_check.jl
julia analysis/correlation_heatmaps.jl
```

`run_br_fit.jl` exposes its configuration (fit range, prior type, EM
iterations, ridge stabilization) as constants at the top of the file.

## Notes

- The response matrix's raw `detector_response` value is already an absolute
  probability density (its integral over measured energy is the detection
  efficiency), so `response_matrix` applies no column renormalization and no
  separate efficiency factor — either would double-count the same scale.
- EM unfolding settings (10 iterations, regularization on) and the ridge
  stabilization (`lambda=0.1`) applied before matrix inversion in the fits
  were tuned via closure testing against known-truth toy spectra.
