module DetectorResponse

export detector_response, response_matrix

using QuadGK

# Piecewise Gaussian-plus-exponential-tail parameterization of the TRIUMF
# photon detector response, fit as a function of true gamma energy E_gamma.
function poly3(E, coeffs)
    P0, P1, P2, P3 = coeffs
    return P0 + P1 * E + P2 * E^2 + P3 * E^3
end

function detector_response(E_gamma, E_measured)
    sigma_0_coeffs    = [-0.5836, 0.0352, 0.0, 0.0]
    sigma_1_coeffs    = [-5.879, 0.1653, -5.149e-4, 0.0]
    sigma_2_coeffs    = [1.596, -0.03859, 3.883e-4, 0.0]
    sigma_3_coeffs    = [-47.80, 1.010, -4.406e-3, 0.0]
    E_3_coeffs        = [1.068, 0.7507, 0.0, 0.0]
    E_0_high_energies = [-1.161, 0.9481, 1.724e-4, 0.0]
    E_0_low_energies  = [22.73, 0.1995, 5.993e-3, 0.0]
    A_coeffs          = [3.259e-4, -4.120e-4, 1.015e-5, -4.05e-8]
    F_over_A_coeffs   = [-0.1337, 2.828e-3, -9.701e-6, 0.0]

    E_0 = poly3(E_gamma, E_gamma > 60 ? E_0_high_energies : E_0_low_energies)
    sigma_0 = poly3(E_gamma, sigma_0_coeffs)
    sigma_1 = poly3(E_gamma, sigma_1_coeffs)
    sigma_2 = poly3(E_gamma, sigma_2_coeffs)
    sigma_3 = poly3(E_gamma, sigma_3_coeffs)
    E_3 = poly3(E_gamma, E_3_coeffs)
    A = poly3(E_gamma, A_coeffs)
    F = E_gamma < 60 ? 0.0 : poly3(E_gamma, F_over_A_coeffs) * A

    B = A * exp(-sigma_0^2 / (2 * sigma_1^2))
    C = A * exp(-sigma_0^2 / (2 * sigma_2^2))
    E_1 = E_0 - (sigma_0^2 / sigma_1)
    E_2 = E_0 + (sigma_0^2 / sigma_2)

    tail(E) = F * exp(-((E - E_3)^2) / (2 * sigma_3^2))

    if E_measured < E_1
        return B * exp(-1 / sigma_1 * (E_1 - E_measured)) + tail(E_measured)
    elseif E_measured <= E_2
        return A * exp(-((E_measured - E_0)^2) / (2 * sigma_0^2)) + tail(E_measured)
    else
        return C * exp(-1 / sigma_2 * (E_measured - E_2)) + tail(E_measured)
    end
end

# Absolute (non-renormalized) true-energy -> measured-energy response matrix.
# true_energy and reco_energy may differ in length/range, so the true axis
# can extend below the lowest measured bin (a "buffer" region), letting a
# true photon born there still smear up into the visible measured range.
#
# detector_response's raw value is already the correct absolute probability
# density, so no column renormalization or separate efficiency factor is
# applied here -- doing either would double-count the same scale factor.
function response_matrix(true_energy::AbstractVector, reco_energy::AbstractVector; reco_bin_width::Float64=1.0)
    M = zeros(length(reco_energy), length(true_energy))
    for (j, E_true) in enumerate(true_energy), (i, E_meas) in enumerate(reco_energy)
        M[i, j] = detector_response(E_true, E_meas) * reco_bin_width
    end
    return M
end

end
