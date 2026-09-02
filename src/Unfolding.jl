module Unfolding

export em_unfold, unfold_with_covariance

using Statistics, Distributions, LinearAlgebra

# Richardson-Lucy / EM (D'Agostini-style) iterative unfolding with an
# efficiency correction (eps = column sums of R) and periodic 3-point
# box-smoothing regularization applied every 5th iteration. The interior-only
# smoothing avoids wrapping the huge low-energy peak bin into the empty
# high-energy tail bin (circshift would do that on this open spectrum).
function em_unfold(d, R; prior=nothing, iterations::Int=10, regularization::Bool=true)
    eps = max.(sum(R, dims=1)[:], 1e-10)
    x = prior === nothing ? fill(1.0, size(R, 2)) : copy(prior)

    for n in 1:iterations
        denom = max.(R * x, 1e-10)
        x .= (x ./ eps) .* (transpose(R) * (d ./ denom))
        if regularization && n % 5 == 0
            x[2:end-1] .= (x[1:end-2] .+ x[2:end-1] .+ x[3:end]) ./ 3.0
        end
    end
    return x
end

# Unfolds `data` and estimates its covariance via Poisson-bootstrap toys:
# each toy re-fluctuates the measured spectrum and re-runs the full unfolding,
# so the resulting covariance captures the unfolding's own error propagation
# (including bin-to-bin correlations from response-matrix collinearity).
function unfold_with_covariance(data, R; n_toys::Int=10_000, iterations::Int=10, regularization::Bool=true, prior=nothing)
    n_true = size(R, 2)
    ensemble = zeros(n_true, n_toys)
    for t in 1:n_toys
        toy = rand.(Poisson.(max.(data, 1e-5)))
        ensemble[:, t] = em_unfold(toy, R; prior=prior, iterations=iterations, regularization=regularization)
    end

    cov_unfolded = cov(ensemble, dims=2)
    counts = em_unfold(data, R; prior=prior, iterations=iterations, regularization=regularization)
    errors = sqrt.(diag(cov_unfolded))
    return cov_unfolded, counts, errors
end

end
