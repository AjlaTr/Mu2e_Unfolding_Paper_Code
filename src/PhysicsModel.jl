module PhysicsModel

export kmax0, kmax1, no_knockout, first_knockout, combined_model, channel_integrals, branching_ratios

using QuadGK

# Endpoint energies (MeV) for the 0-neutron and 1-neutron knockout RMC
# photon spectra on Aluminum.
const kmax0 = 101.86686452707785
const kmax1 = 95.44893883499826

# Normalized (integrate to 1 over [0, kmax]) phase-space shapes for each
# knockout channel.
no_knockout(E, k=kmax0) = (12 / k) * (E / k) * max(1 - E / k, 0.0)^2
first_knockout(E, k=kmax1) = (99 / (4k)) * (E / k) * max(1 - E / k, 0.0)^3.5

combined_model(E, C0, C1) = C0 .* no_knockout.(E) .+ C1 .* first_knockout.(E)
combined_model(E, C0, C1, C_bkg) = combined_model(E, C0, C1) .+ C_bkg

# Windowed integrals of each channel's shape over [E_lo, E_hi]; both shapes
# are exactly zero above their own kmax, so this is the full-channel yield
# whenever E_hi >= kmax0.
function channel_integrals(E_lo=57.0, E_hi=102.0)
    I0 = quadgk(E -> no_knockout(E), E_lo, E_hi)[1]
    I1 = quadgk(E -> first_knockout(E), E_lo, E_hi)[1]
    return I0, I1
end

function branching_ratios(C0, C1, integrals)
    norm = C0 * integrals[1] + C1 * integrals[2]
    return C0 * integrals[1] / norm, C1 * integrals[2] / norm
end

end
