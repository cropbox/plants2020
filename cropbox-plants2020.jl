# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Julia 1.5.2
#     language: julia
#     name: julia-1.5
# ---

using Cropbox

using CSV
using DataFrames
using DataFramesMeta
using Statistics
using GLM

import Gadfly
import Cairo

using Logging
Logging.disable_logging(Logging.Warn)

# ## Dataset

# We have two datasets from an experiment conducted in 2005 with maize plots under multiple levels of nitrogen application.
#
# - `corn2005spad.csv`: overal growth data including SPAD
# - `corn2005ge.csv`: corresponding gas exchange measurements from LI-COR

# ### SPAD

obs_spad = CSV.read("corn2005spad.csv") |> unitfy

plot(obs_spad, :SPAD, :leaf_N, kind=:scatter)

plot(obs_spad, :leaf_N, :A_max, kind=:scatter)

plot(obs_spad, :nitrogen, :leaf_N, kind=:scatter)

# ### Gas Exchange

obs_ge = CSV.read("corn2005ge.csv") |> unitfy

# Recalculate stomatal conductance in units that our model uses for taking account air pressure.

obs_ge.gs = obs_ge.Cond ./ obs_ge.Press .|> u"mol/m^2/s/bar";

# Make a combined dataset.

obs = join(obs_ge, obs_spad, on=[:plot, :subplot], makeunique=true);

# ## Model

# Gas exchange model is mostly derived from MAIZSIM, adapting to Cropbox framework.

# ### SPAD - N

@system Nitrogen begin
    SPAD: SPAD_greenness ~ preserve(parameter)
    SNa: SPAD_N_coeff_a ~ preserve(u"g/m^2", parameter)
    SNb: SPAD_N_coeff_b ~ preserve(u"g/m^2", parameter)
    SNc: SPAD_N_coeff_c ~ preserve(u"g/m^2", parameter)
    N(SPAD, a=SNa, b=SNb, c=SNc): leaf_nitrogen_content => begin
        a*SPAD^2 + b*SPAD + c
    end ~ preserve(u"g/m^2", parameter)
    Np(N, SLA) => N * SLA ~ track(u"percent")
    SLA: specific_leaf_area => 200 ~ preserve(u"cm^2/g")
end

@system NitrogenController(Nitrogen, Controller)

spad_configs = [:Nitrogen => :SPAD => x for x in obs_spad.SPAD]

nitrogen_config = :Nitrogen => (SNa=0.0004, SNb=0.0120, SNc=0)

# We have a quadratic relationship $N = 0.0004S^2 + 0.0120S$. Note that intercept (`SNc`) is close to zero.

est = simulate(NitrogenController, [(index=:SPAD, target=:N)], [(c, nitrogen_config) for c in spad_configs], skipfirst=true)[1]

p = plot(obs_spad, :SPAD, :leaf_N, ylab="Leaf nitrogen", name="Observed", legendpos=(0.1,-0.3), kind=:scatter)
plot!(p, est, :SPAD, :N, name="Fitted (R^2=0.92)", kind=:line)
p' |> Gadfly.PDF("SPAD-N.pdf")
p

lm(@formula(est ~ obs), DataFrame(obs=deunitfy(obs_spad.leaf_N), est=deunitfy(est.N))) |> r2

# ### Environment

@system VaporPressure begin
    a => 0.611 ~ preserve(u"kPa", parameter)
    b => 17.502 ~ preserve(parameter)
    c => 240.97 ~ preserve(parameter)

    es(a, b, c; T(u"°C")): saturation => (t = deunitfy(T); a*exp((b*t)/(c+t))) ~ call(u"kPa")
    ea(es; T(u"°C"), RH(u"percent")): ambient => es(T) * RH ~ call(u"kPa")
    D(es; T(u"°C"), RH(u"percent")): deficit => es(T) * (1 - RH) ~ call(u"kPa")
    RH(es; T(u"°C"), VPD(u"kPa")): relative_humidity => 1 - VPD / es(T) ~ call(u"NoUnits")

    Δ(es, b, c; T(u"°C")): saturation_slope_delta => (e = es(T); t = deunitfy(T); e*(b*c)/(c+t)^2 / u"K") ~ call(u"kPa/K")
    s(Δ; T(u"°C"), P(u"kPa")): saturation_slope => Δ(T) / P ~ call(u"K^-1")
end

@system Weather begin
    vp(context): vapor_pressure ~ ::VaporPressure

    PFD: photon_flux_density ~ preserve(u"μmol/m^2/s", parameter)
    CO2: carbon_dioxide ~ preserve(u"μmol/mol", parameter)
    RH: relative_humidity ~ preserve(u"percent", parameter)
    T_air: air_temperature ~ preserve(u"°C", parameter)
    Tk_air(T_air): absolute_air_temperature ~ track(u"K")
    wind: wind_speed ~ preserve(u"m/s", parameter)
    P_air: air_pressure => 100 ~ preserve(u"kPa", parameter)

    VPD(T_air, RH, D=vp.D): vapor_pressure_deficit => D(T_air, RH) ~ track(u"kPa")
    VPD_Δ(T_air, Δ=vp.Δ): vapor_pressure_saturation_slope_delta => Δ(T_air) ~ track(u"kPa/K")
    VPD_s(T_air, P_air, s=vp.s): vapor_pressure_saturation_slope => s(T_air, P_air) ~ track(u"K^-1")
end

@system Diffusion begin
    Dw: diffusion_coeff_for_water_vapor_in_air_at_20 => 24.2 ~ preserve(u"mm^2/s", parameter)
    Dc: diffusion_coeff_for_co2_in_air_at_20 => 14.7 ~ preserve(u"mm^2/s", parameter)
    Dh: diffusion_coeff_for_heat_in_air_at_20 => 21.5 ~ preserve(u"mm^2/s", parameter)
    Dm: diffusion_coeff_for_momentum_in_air_at_20 => 15.1 ~ preserve(u"mm^2/s", parameter)
end

@system Irradiance begin
    PFD ~ hold
    PPFD(PFD): photosynthetic_photon_flux_density ~ track(u"μmol/m^2/s")

    δ: leaf_scattering => 0.15 ~ preserve(parameter)
    f: leaf_spectral_correction => 0.15 ~ preserve(parameter)

    Ia(PPFD, δ): absorbed_irradiance => begin
        PPFD * (1 - δ)
    end ~ track(u"μmol/m^2/s")

    I2(Ia, f): effective_irradiance => begin
        Ia * (1 - f) / 2
    end ~ track(u"μmol/m^2/s")
end

# ### C4

@system TemperatureDependence begin
    T: leaf_temperature ~ hold
    Tk(T): absolute_leaf_temperature ~ track(u"K")

    Tb: base_temperature => 25 ~ preserve(u"°C", parameter)
    Tbk(Tb): absolute_base_temperature ~ preserve(u"K")

    kT(T, Tk, Tb, Tbk; Ea(u"kJ/mol")): arrhenius_equation => begin
        exp(Ea * (T - Tb) / (u"R" * Tk * Tbk))
    end ~ call

    kTpeak(Tk, Tbk, kT; Ea(u"kJ/mol"), S(u"J/mol/K"), H(u"kJ/mol")): peaked_function => begin
        R = u"R"
        kT(Ea) * (1 + exp((S*Tbk - H) / (R*Tbk))) / (1 + exp((S*Tk - H) / (R*Tk)))
    end ~ call

    Q10 => 2 ~ preserve(parameter)
    kTQ10(T, Tb, Q10): q10_rate => begin
        Q10^((T - Tb) / 10u"K")
    end ~ track
end

@system NitrogenDependence begin
    N: leaf_nitrogen_content ~ hold

    s => 2.9 ~ preserve(u"m^2/g", parameter)
    N0 => 0.25 ~ preserve(u"g/m^2", parameter)

    kN(N, s, N0): nitrogen_limited_rate => begin
        2 / (1 + exp(-s * (max(N0, N) - N0))) - 1
    end ~ track
end

@system CBase(TemperatureDependence, NitrogenDependence) begin
    Ci: intercellular_co2 ~ hold
    I2: effective_irradiance ~ hold
end

@system C4Base(CBase) begin
    Cm(Ci): mesophyll_co2 ~ track(u"μbar")
    gbs: bundle_sheath_conductance => 0.003 ~ preserve(u"mol/m^2/s/bar", parameter)
end

@system C4c(C4Base) begin
    Kp25: pep_carboxylase_constant_for_co2_at_25 => 80 ~ preserve(u"μbar", parameter)
    Kp(Kp25, kTQ10): pep_carboxylase_constant_for_co2 => begin
        Kp25 * kTQ10
    end ~ track(u"μbar")

    Vpm25: maximum_pep_carboxylation_rate_for_co2_at_25 => 70 ~ preserve(u"μmol/m^2/s", parameter)
    EaVp: activation_energy_for_pep_carboxylation => 75.1 ~ preserve(u"kJ/mol", parameter)
    Vpmax(Vpm25, kT, EaVp, kN): maximum_pep_carboxylation_rate => begin
        Vpm25 * kT(EaVp) * kN
    end ~ track(u"μmol/m^2/s")

    Vpr25: regeneration_limited_pep_carboxylation_rate_for_co2_at_25 => 80 ~ preserve(u"μmol/m^2/s", parameter)
    Vpr(Vpr25, kTQ10): regeneration_limited_pep_carboxylation_rate => begin
        Vpr25 * kTQ10
    end ~ track(u"μmol/m^2/s")
    Vp(Vpmax, Vpr, Cm, Kp): pep_carboxylation_rate => begin
        (Cm * Vpmax) / (Cm + Kp)
    end ~ track(u"μmol/m^2/s", max=Vpr)

    Vcm25: maximum_carboxylation_rate_at_25 => 50 ~ preserve(u"μmol/m^2/s", parameter)
    EaVc: activation_energy_for_carboxylation => 55.9 ~ preserve(u"kJ/mol", parameter)
    Vcmax(Vcm25, kT, EaVc, kN): maximum_carboxylation_rate => begin
        Vcm25 * kT(EaVc) * kN
    end ~ track(u"μmol/m^2/s")
end

@system C4j(C4Base) begin
    Jm25: maximum_electron_transport_rate_at_25 => 300 ~ preserve(u"μmol/m^2/s", parameter)
    Eaj: activation_energy_for_electron_transport => 32.8 ~ preserve(u"kJ/mol", parameter)
    Sj: electron_transport_temperature_response => 702.6 ~ preserve(u"J/mol/K", parameter)
    Hj: electron_transport_curvature => 220 ~ preserve(u"kJ/mol", parameter)
    Jmax(Jm25, kTpeak, Eaj, Sj, Hj, kN): maximum_electron_transport_rate => begin
        Jm25 * kTpeak(Eaj, Sj, Hj) * kN
    end ~ track(u"μmol/m^2/s")

    θ: light_transition_sharpness => 0.5 ~ preserve(parameter)
    J(I2, Jmax, θ): electron_transport_rate => begin
        a = θ
        b = -(I2+Jmax)
        c = I2*Jmax
        a*J^2 + b*J + c ⩵ 0
    end ~ solve(lower=0, upper=Jmax, u"μmol/m^2/s")
end

@system C4r(C4Base) begin
    Kc25: rubisco_constant_for_co2_at_25 => 650 ~ preserve(u"μbar", parameter)
    Eac: activation_energy_for_co2 => 59.4 ~ preserve(u"kJ/mol", parameter)
    Kc(kT, Kc25, Eac): rubisco_constant_for_co2 => begin
        Kc25 * kT(Eac)
    end ~ track(u"μbar")

    Ko25: rubisco_constant_for_o2_at_25 => 450 ~ preserve(u"mbar", parameter)
    # Activation energy for Ko, Bernacchi (2001)
    Eao: activation_energy_for_o2 => 36 ~ preserve(u"kJ/mol", parameter)
    Ko(Ko25, kT, Eao): rubisco_constant_for_o2 => begin
        Ko25 * kT(Eao)
    end ~ track(u"mbar")

    Om: mesophyll_o2_partial_pressure => 210 ~ preserve(u"mbar", parameter)
    Km(Kc, Om, Ko): rubisco_constant_for_co2_with_o2 => begin
        Kc * (1 + Om / Ko)
    end ~ track(u"μbar")

    Rd25: dark_respiration_at_25 => 2 ~ preserve(u"μmol/m^2/s", parameter)
    Ear: activation_energy_for_respiration => 39.8 ~ preserve(u"kJ/mol", parameter)
    Rd(Rd25, kT, Ear): dark_respiration => begin
        Rd25 * kT(Ear)
    end ~ track(u"μmol/m^2/s")
    Rm(Rd) => 0.5Rd ~ track(u"μmol/m^2/s")
end

@system C4Rate(C4c, C4j, C4r) begin
    Ac1(Vp, gbs, Cm, Rm) => (Vp + gbs*Cm - Rm) ~ track(u"μmol/m^2/s")
    Ac2(Vcmax, Rd) => (Vcmax - Rd) ~ track(u"μmol/m^2/s")
    Ac(Ac1, Ac2): enzyme_limited_photosynthesis_rate => begin
        min(Ac1, Ac2)
    end ~ track(u"μmol/m^2/s")

    x: electron_transport_partitioning_factor => 0.4 ~ preserve(parameter)
    Aj1(x, J, Rm, gbs, Cm) => (x * J/2 + gbs*Cm - Rm) ~ track(u"μmol/m^2/s")
    Aj0(x, J, Rm, gbs, Cm) => (x * J/2 - gbs*Cm - Rm) ~ track(u"μmol/m^2/s")
    Aj2(x, J, Rd) => (1-x) * J/3 - Rd ~ track(u"μmol/m^2/s")
    Aj(Aj1, Aj2): transport_limited_photosynthesis_rate => begin
        min(Aj1, Aj2)
    end ~ track(u"μmol/m^2/s")

    β: photosynthesis_transition_factor => 0.99 ~ preserve(parameter)
    A_net(Ac, Aj, β): net_photosynthesis => begin
        x = A_net
        a = β
        b = -(Ac+Aj)
        c = Ac*Aj
        a*x^2 + b*x + c ⩵ 0
    end ~ solve(pick=:minimum, u"μmol/m^2/s")

    A_gross(A_net, Rd): gross_photosynthesis => begin
        A_gross = A_net + Rd
    end ~ track(u"μmol/m^2/s")
end

@system C4(C4Rate)

# ### Interface

# #### Boundary Layer

@system BoundaryLayer(Weather, Diffusion) begin
    w: leaf_width => 10 ~ preserve(u"cm", parameter)

    sr: stomatal_ratio => 1.0 ~ preserve(parameter)
    scr(sr): sides_conductance_ratio => ((sr + 1)^2 / (sr^2 + 1)) ~ preserve
    ocr: outdoor_conductance_ratio => 1.4 ~ preserve

    u(u=wind): wind_velocity ~ track(u"m/s", min=0.1)
    d(w): characteristic_dimension => 0.72w ~ track(u"m")
    v(Dm): kinematic_viscosity_of_air ~ preserve(u"m^2/s", parameter)
    κ(Dh): thermal_diffusivity_of_air ~ preserve(u"m^2/s", parameter)
    Re(u, d, v): reynolds_number => u*d/v ~ track
    Nu(Re): nusselt_number => 0.60sqrt(Re) ~ track
    gh(κ, Nu, d, scr, ocr, P_air, Tk_air): boundary_layer_heat_conductance => begin
        g = κ * Nu / d
        g *= scr * ocr
        g * P_air / (u"R" * Tk_air)
    end ~ track(u"mmol/m^2/s")
    rhw(Dw, Dh): ratio_from_heat_to_water_vapor => (Dw / Dh)^(2/3) ~ preserve
    gb(rhw, gh, P_air): boundary_layer_conductance => rhw * gh / P_air ~ track(u"mol/m^2/s/bar")
end

# #### Stomata

@system StomataBase(Weather, Diffusion) begin
    gs: stomatal_conductance ~ hold
    gb: boundary_layer_conductance ~ hold
    A_net: net_photosynthesis ~ hold
    T: leaf_temperature ~ hold

    drb(Dw, Dc): diffusivity_ratio_boundary_layer => (Dw / Dc)^(2/3) ~ preserve(parameter)
    dra(Dw, Dc): diffusivity_ratio_air => (Dw / Dc) ~ preserve(parameter)

    Ca(CO2, P_air): co2_air => (CO2 * P_air) ~ track(u"μbar")
    Cs(Ca, A_net, gbc): co2_at_leaf_surface => begin
        Ca - A_net / gbc
    end ~ track(u"μbar")

    gv(gs, gb): total_conductance_h2o => (gs * gb / (gs + gb)) ~ track(u"mol/m^2/s/bar")

    rbc(gb, drb): boundary_layer_resistance_co2 => (drb / gb) ~ track(u"m^2*s/mol*bar")
    rsc(gs, dra): stomatal_resistance_co2 => (dra / gs) ~ track(u"m^2*s/mol*bar")
    rvc(rbc, rsc): total_resistance_co2 => (rbc + rsc) ~ track(u"m^2*s/mol*bar")

    gbc(rbc): boundary_layer_conductance_co2 => (1 / rbc) ~ track(u"mol/m^2/s/bar")
    gsc(rsc): stomatal_conductance_co2 => (1 / rsc) ~ track(u"mol/m^2/s/bar")
    gvc(rvc): total_conductance_co2 => (1 / rvc) ~ track(u"mol/m^2/s/bar")
end

@system StomataTuzet begin
    WP_leaf: leaf_water_potential => 0 ~ preserve(u"MPa", parameter)
    Ψv(WP_leaf): bulk_leaf_water_potential ~ track(u"MPa")
    Ψf: reference_leaf_water_potential => -1.2 ~ preserve(u"MPa", parameter)
    sf: stomata_sensitivity_param => 2.3 ~ preserve(u"MPa^-1", parameter)
    fΨv(Ψv, Ψf, sf): stomata_sensitivty => begin
        (1 + exp(sf*Ψf)) / (1 + exp(sf*(Ψf-Ψv)))
    end ~ track
end

# ##### Ball-Berry Model

@system StomataBallBerry(StomataBase, StomataTuzet) begin
    g0 => 0.017 ~ preserve(u"mol/m^2/s/bar", parameter)
    g1 => 4.53 ~ preserve(parameter)

    hs(g0, g1, gb, A_net, Cs, fΨv, RH): relative_humidity_at_leaf_surface => begin
        gs = g0 + g1*(A_net*hs/Cs) * fΨv
        (hs - RH)*gb ⩵ (1 - hs)*gs
    end ~ solve(lower=0, upper=1)
    Ds(D=vp.D, T, hs): vapor_pressure_deficit_at_leaf_surface => begin
        D(T, hs)
    end ~ track(u"kPa")

    gs(g0, g1, A_net, hs, Cs, fΨv): stomatal_conductance => begin
        g0 + g1*(A_net*hs/Cs) * fΨv
    end ~ track(u"mol/m^2/s/bar", min=g0)
end

# ##### Medlyn Model

@system StomataMedlyn(StomataBase, StomataTuzet) begin
    g0 => 0.02 ~ preserve(u"mol/m^2/s/bar", parameter)
    g1 => 4.0 ~ preserve(u"√kPa", parameter)

    wa(ea=vp.ea, T_air, RH): vapor_pressure_at_air => ea(T_air, RH) ~ track(u"kPa")
    wi(es=vp.es, T): vapor_pressure_at_intercellular_space => es(T) ~ track(u"kPa")
    ws(Ds, wi): vapor_pressure_at_leaf_surface => (wi - Ds) ~ track(u"kPa")
    Ds¹ᐟ²(g0, g1, gb, A_net, Cs, fΨv, wi, wa) => begin
        gs = g0 + (1 + g1 / Ds¹ᐟ²) * (A_net / Cs) * fΨv
        ws = wi - Ds¹ᐟ²^2
        (ws - wa)*gb ⩵ (wi - ws)*gs
    end ~ solve(lower=0, upper=√wi', u"√kPa")
    Ds(Ds¹ᐟ²): vapor_pressure_deficit_at_leaf_surface => Ds¹ᐟ²^2 ~ track(u"kPa", min=1u"Pa")
    hs(RH=vp.RH, T, Ds): relative_humidity_at_leaf_surface => RH(T, Ds) ~ track

    gs(g0, g1, A_net, Ds, Cs, fΨv): stomatal_conductance => begin
        g0 + (1 + g1/√Ds)*(A_net/Cs) * fΨv
    end ~ track(u"mol/m^2/s/bar", min=g0)
end

# #### Intercellular Space

@system IntercellularSpace(Weather) begin
    A_net ~ hold
    gvc ~ hold

    Ca(CO2, P_air): co2_air => (CO2 * P_air) ~ track(u"μbar")
    Cimax(Ca): intercellular_co2_upper_limit => 2Ca ~ track(u"μbar")
    Cimin: intercellular_co2_lower_limit => 0 ~ preserve(u"μbar")
    Ci(Ca, Ci, A_net, gvc): intercellular_co2 => begin
        Ca - Ci ⩵ A_net / gvc
    end ~ bisect(min=Cimin, upper=Cimax, u"μbar")
end

# ### Energy Balance

@system EnergyBalance(Weather) begin
    gv ~ hold
    gh ~ hold
    PPFD ~ hold

    ϵ: leaf_thermal_emissivity => 0.97 ~ preserve(parameter)
    σ: stefan_boltzmann_constant => u"σ" ~ preserve(u"W/m^2/K^4")
    λ: latent_heat_of_vaporization_at_25 => 44 ~ preserve(u"kJ/mol", parameter)
    Cp: specific_heat_of_air => 29.3 ~ preserve(u"J/mol/K", parameter)

    k: radiation_conversion_factor => (1 / 4.55) ~ preserve(u"J/μmol")
    α_s: absorption_coefficient => 0.79 ~ preserve(parameter)
    PAR(PPFD, k): photosynthetically_active_radiation => (PPFD * k) ~ track(u"W/m^2")
    R_sw(PAR, α_s): shortwave_radiation_absorbed => (α_s * PAR) ~ track(u"W/m^2")

    R_wall(ϵ, σ, Tk_air): thermal_radiation_absorbed_from_wall => 2ϵ*σ*Tk_air^4 ~ track(u"W/m^2")
    R_leaf(ϵ, σ, Tk): thermal_radiation_emitted_by_leaf => 2ϵ*σ*Tk^4 ~ track(u"W/m^2")
    R_thermal(R_wall, R_leaf): thermal_radiation_absorbed => R_wall - R_leaf ~ track(u"W/m^2")
    R_net(R_sw, R_thermal): net_radiation_absorbed => R_sw + R_thermal ~ track(u"W/m^2")

    Δw(T, T_air, RH, ea=vp.ambient, es=vp.saturation): leaf_vapor_pressure_gradient => begin
        es(T) - ea(T_air, RH)
    end ~ track(u"kPa")
    E(gv, Δw): transpiration => gv*Δw ~ track(u"mmol/m^2/s")

    H(Cp, gh, ΔT): sensible_heat_flux => Cp*gh*ΔT ~ track(u"W/m^2")
    λE(λ, E): latent_heat_flux => λ*E ~ track(u"W/m^2")

    ΔT(R_net, H, λE): temperature_adjustment => begin
        R_net ⩵ H + λE
    end ~ bisect(lower=-10, upper=10, u"K", evalunit=u"W/m^2")

    T(T_air, ΔT): leaf_temperature => (T_air + ΔT) ~ track(u"°C")
    Tk(T): absolute_leaf_temperature ~ track(u"K")
end

# ### Coupling

abstract type GasExchange <: System end

@system GasExchangeBallBerry(
    Weather, Nitrogen,
    BoundaryLayer, StomataBallBerry, IntercellularSpace, Irradiance, EnergyBalance,
    C4, Controller
) <: GasExchange

@system GasExchangeMedlyn(
    Weather, Nitrogen,
    BoundaryLayer, StomataMedlyn, IntercellularSpace, Irradiance, EnergyBalance,
    C4, Controller
) <: GasExchange

h = Cropbox.hierarchy(GasExchangeMedlyn; skipcontext=true)
Cropbox.writeimage("GasExchangeMedlyn.pdf", h)
h

d = Cropbox.dependency(GasExchangeMedlyn)

# Default parameters are mostly from MAIZSIM.

parameters(C4)

# For simplicity, environmental inputs are set as parameters, instead of driving variables loaded from external data frame.

base_config = (
    nitrogen_config,
    :Nitrogen => (
        SPAD = 60,
    ),
    :Weather => (
        PFD = 2000,
        CO2 = 400,
        RH = 66,
        T_air = 32,
        wind = 2.0,
        P_air = 99.4,
    ),
);

# ## Calibration

# Since our model generally performed well under nitrogen non-limiting condition, we decided to keep the most of existing parameter set and calibrate only a small set of parameters. We will calibrate two parameters (`s`, `N0`) for nitrogen dependency along `A_net` and two parameters (`g0`, `g1`) for stomatal conductance along `gs`.

obs_configs = [(
    nitrogen_config,
    :Nitrogen => (
        SPAD = r[:SPAD],
    ),
    :Weather => (
        PFD = r[:PARi],
        CO2 = r[:CO2S],
        RH = r[:RH_S],
        T_air = r[:Tair],
        wind = 2.0,
        P_air = r[:Press],
    ),
) for r in eachrow(obs)];

# +
# bb_calib_config = calibrate(GasExchangeBallBerry, obs, obs_configs;
#     index=[:PARi => :PFD, :CO2S => :CO2, :RH_S => :RH, :Tair => :T_air, :Press => :P_air, :SPAD],
#     target=[:Photo => :A_net, :gs],
#     parameters=(
#         :NitrogenDependence => (s=(0, 10), N0=(0, 1)),
#         :StomataBallBerry => (g0=(0, 1), g1=(0, 10)),
#     ),
#     skipfirst=true,
#     optim=(
#         MaxSteps=2000,
#         TraceInterval=10,
#         RandomizeRngSeed=false,
#     ),
#     metric=:prmse,
#     #weight=[1, 100],
#     #pareto=true
# )
# -

bb_calib_config = (
    :NitrogenDependence => (s=4.470, N0=0.371),
    :StomataBallBerry => (g0=0.036, g1=2.792),
)

# +
# med_calib_config = calibrate(GasExchangeMedlyn, obs, obs_configs;
#     index=[:PARi => :PFD, :CO2S => :CO2, :RH_S => :RH, :Tair => :T_air, :Press => :P_air, :SPAD],
#     target=[:Photo => :A_net, :gs],
#     parameters=(
#         :NitrogenDependence => (s=(0, 10), N0=(0, 1)),
#         :StomataMedlyn => (g0=(0, 1), g1=(0, 10)),
#     ),
#     skipfirst=true,
#     optim=(
#         MaxSteps=2000,
#         TraceInterval=10,
#         RandomizeRngSeed=false,
#     ),
#     metric=:prmse,
#     #weight=[1, 100],
#     #pareto=true
# )
# -

med_calib_config = (
    :NitrogenDependence => (s=3.912, N0=0.315),
    :StomataMedlyn => (g0=0.031, g1=1.281),
)

bb_med_calib_config = let b=@config(bb_calib_config)[:NitrogenDependence],
                         m=@config(med_calib_config)[:NitrogenDependence]
    (:NitrogenDependence => (s=(b[:s]+m[:s])/2, N0=(b[:N0]+m[:N0])/2),)
end

bb_obs_configs = @config bb_calib_config + obs_configs;
bb_config = (base_config, bb_calib_config, bb_med_calib_config);

med_obs_configs = @config med_calib_config + obs_configs;
med_config = (base_config, med_calib_config, bb_med_calib_config);

# ### Result

visualize_fit(S::Type{<:GasExchange};
    configs=[],
    obs=obs_ge,
    y=:Photo=>:A_net,
    title="",
    xlab="Observation",
    ylab="Model",
    pdfname=nothing,
    kwargs...
) = begin
    p = visualize(obs, S, y; configs=configs, title=title, xlab=xlab, ylab=ylab, name="", kwargs...)
    !isnothing(pdfname) && p' |> Gadfly.PDF("$pdfname.pdf")
    p' |> Gadfly.SVG()
end

visualize_fit_BB(; configs=bb_obs_configs, kw...) = visualize_fit(GasExchangeBallBerry; configs=configs, kw...)
visualize_fit_MED(; configs=med_obs_configs, kw...) = visualize_fit(GasExchangeMedlyn; configs=configs, kw...)
visualize_fit(; kw...) = visualize_fit_MED(; kw...)

calculate_fit_BB(; configs=bb_obs_configs, kw...) = calculate_fit(GasExchangeBallBerry; configs=configs, kw...)
calculate_fit_MED(; configs=med_obs_configs, kw...) = calculate_fit(GasExchangeMedlyn; configs=configs, kw...)
calculate_fit(S::Type{<:GasExchange};
    configs=[],
    obs=obs_ge,
    y=:Photo=>:A_net,
    metric,
    stop=nothing, skipfirst=true, filter=nothing,
) = begin
    y = y isa Pair ? y : y => y
    yo, ye = y
    est = simulate(S; configs=configs, stop=stop, skipfirst=skipfirst, filter=filter)
    O = obs[yo] |> deunitfy
    E = est[ye] |> deunitfy
    # Nash-Sutcliffe model efficiency coefficient (NSE)
    if metric == :ef
        1 - sum((E - O).^2) / sum((O .- mean(O)).^2)
    # Willmott's refined index of agreement (d_r)
    elseif metric == :dr
        let a = sum(abs.(E .- O)),
            b = 2sum(abs.(O .- mean(O)))
            a <= b ? 1 - a/b : b/a - 1
        end
    end
end

visualize_fits(maps;
    obs=obs_ge,
    y=:Photo=>:A_net,
    title="",
    xlab="Observation",
    ylab="Model",
    names=nothing,
    pdfname=nothing,
    kwargs...
) = begin
    isnothing(names) && (names = [string(Cropbox.namefor(m.system)) for m in maps])
    p = visualize(obs, maps, y; title=title, xlab=xlab, ylab=ylab, names, kwargs...)
    !isnothing(pdfname) && p' |> Gadfly.PDF("$pdfname.pdf")
    p' |> Gadfly.SVG()
end

# #### Photosynthesis

visualize_fit_BB(y=:Photo=>:A_net, lim=(0,60), #=title="Net Photosynthesis Rate (An)",=# pdfname="fit-BB-An")

calculate_fit_BB(y=:Photo=>:A_net, metric=:dr)

calculate_fit_BB(y=:Photo=>:A_net, metric=:ef)

visualize_fit_MED(y=:Photo=>:A_net, lim=(0,60), #=title="Net Photosynthesis Rate (An)",=# pdfname="fit-MED-An")

calculate_fit_MED(y=:Photo=>:A_net, metric=:dr)

calculate_fit_MED(y=:Photo=>:A_net, metric=:ef)

visualize_fits([
    (system=GasExchangeBallBerry, configs=bb_obs_configs),
    (system=GasExchangeMedlyn, configs=med_obs_configs),
];
    names=["BB (dr=0.879, ef=0.941)", "MED (dr=0.881, ef=0.937)"],
    y=:Photo=>:A_net, lim=(0,60),
    #legendpos=(0.1,-0.3),
    legendpos=(0.5,0.3),
    pdfname="fit-BB-MED-An",
)

# #### Stomatal Conductance

visualize_fit_BB(y=:gs, lim=(0,0.65), #=title="Stomatal Conductance (gs)",=# pdfname="fit-BB-gs")

calculate_fit_BB(y=:gs, metric=:dr)

calculate_fit_BB(y=:gs, metric=:ef)

visualize_fit_MED(y=:gs, lim=(0,0.65), #=title="Stomatal Conductance (gs)",=# pdfname="fit-MED-gs")

calculate_fit_MED(y=:gs, metric=:dr)

calculate_fit_MED(y=:gs, metric=:ef)

visualize_fits([
    (system=GasExchangeBallBerry, configs=bb_obs_configs),
    (system=GasExchangeMedlyn, configs=med_obs_configs),
];
    names=["BB (dr=0.804, ef=0.798)", "MED (dr=0.820, ef=0.796)"],
    y=:gs, lim=(0,0.65),
    legendpos=(0.5,0.3),
    pdfname="fit-BB-MED-gs",
)

# #### Temperature

visualize_fit_BB(y=:Tleaf=>:T, lim=(31,36), pdfname="fit-BB-Tl")

calculate_fit_BB(y=:Tleaf=>:T, metric=:dr)

calculate_fit_BB(y=:Tleaf=>:T, metric=:ef)

visualize_fit_MED(y=:Tleaf=>:T, lim=(31,36), pdfname="fit-MED-Tl")

calculate_fit_MED(y=:Tleaf=>:T, metric=:dr)

calculate_fit_MED(y=:Tleaf=>:T, metric=:ef)

visualize_fits([
    (system=GasExchangeBallBerry, configs=bb_obs_configs),
    (system=GasExchangeMedlyn, configs=med_obs_configs),
];
    names=["BB", "MED"],
    y=:Tleaf=>:T, lim=(31,36),
    legendpos=(0.1,-0.3),
    pdfname="fit-BB-MED-T",
)

# ## Analysis

co2_xstep = :Weather => :CO2 => 10:10:1500

visualize_model(S::Type{<:GasExchange};
    config=(),
    configΔ=(),
    x=:Ca,
    y=:A_net,
    group,
    xstep=co2_xstep,
    kind=:line,
    pdfname=nothing,
    kwargs...
) = begin
    p = visualize(S, x, y; config=(config, configΔ), group=group, xstep=xstep, kind=kind, kwargs...)
    !isnothing(pdfname) && p' |> Gadfly.PDF("$pdfname.pdf")
    p' |> Gadfly.SVG()
end

visualize_model_BB(; config=bb_config, kw...) = visualize_model(GasExchangeBallBerry; config=config, kw...)
visualize_model_MED(; config=med_config, kw...) = visualize_model(GasExchangeMedlyn; config=config, kw...)
visualize_model(; kw...) = visualize_model_MED(; kw...)

# ### Ball-Berry vs. Medlyn

# #### Photosynthesis

# ##### Ca

rh_group = :Weather => :RH => 80:-20:20
rh_xstep = :Weather => :RH => 0:1:100

visualize_model_BB(group=rh_group,
    x=:Ca, y=:A_net, ylab="An", xlim=(0,1500), ylim=(-10,60), pdfname="BB-Ca-An-RH")

visualize_model_MED(group=rh_group,
    x=:Ca, y=:A_net, ylab="An", xlim=(0,1500), ylim=(-10,60), pdfname="MED-Ca-An-RH")

# ##### Ci

visualize_model_BB(group=rh_group,
    legendpos=(0.8,0),
    x=:Ci, y=:A_net, ylab="An", xlim=(0,600), ylim=(-10,60), pdfname="BB-Ci-An-RH")

visualize_model_BB(group=:Weather=>:RH=>[70,30],
    legendpos=(0.8,0), colors=Gadfly.Scale.default_discrete_colors(4)[[1,3]],
    x=:Ci, y=:A_net, ylab="An", xlim=(0,600), ylim=(-10,60), pdfname="BB-Ci-An-RH")

visualize_model_MED(group=rh_group,
    legendpos=(0.8,0),
    x=:Ci, y=:A_net, ylab="An", xlim=(0,600), ylim=(-10,60), pdfname="MED-Ci-An-RH")

visualize_model_MED(group=:Weather=>:RH=>[70,30],
    legendpos=(0.8,0), colors=Gadfly.Scale.default_discrete_colors(4)[[1,3]],
    x=:Ci, y=:A_net, ylab="An", xlim=(0,600), ylim=(-10,60), pdfname="MED-Ci-An-RH")

# ##### Ta

ta_xstep = :Weather => :T_air => 0:1:50

visualize_model_BB(group=rh_group, xstep=ta_xstep,
    legendpos=(0.1,-0.2),
    x=:T_air, y=:A_net, xlab="Ta", ylab="An", xlim=(0,50), ylim=(-10,60), pdfname="BB-T-An-RH")

visualize_model_BB(group=rh_group, xstep=ta_xstep,
    x=:T_air, y=:Ci, xlab="Ta", xlim=(0,50), ylim=(0,600), pdfname="BB-T-Ci-RH")

visualize_model_BB(group=rh_group, xstep=ta_xstep,
    x=:T, y=:Ci, xlab="Tl", xlim=(0,50), ylim=(0,600), pdfname="BB-Tl-Ci-RH")

visualize_model_MED(group=rh_group, xstep=ta_xstep,
    legendpos=(0.1,-0.2),
    x=:T_air, y=:A_net, xlab="Ta", ylab="An", xlim=(0,50), ylim=(-10,60), pdfname="MED-T-An-RH")

visualize_model_MED(group=rh_group, xstep=ta_xstep,
    x=:T_air, y=:Ci, xlab="Ta", xlim=(0,50), ylim=(0,600), pdfname="MED-T-Ci-RH")

visualize_model_MED(group=rh_group, xstep=ta_xstep,
    x=:T, y=:Ci, xlab="Tl", xlim=(0,50), ylim=(0,600), pdfname="MED-Tl-Ci-RH")

# ##### RH

visualize_model_BB(group=:Weather=>:CO2=>[1500,800,400,300,200,100], xstep=rh_xstep,
    x=:RH, y=:A_net, ylab="An", xlim=(0,100), ylim=(-10,60), pdfname="BB-RH-An-CO2")

visualize_model_MED(group=:Weather=>:CO2=>[1500,800,400,300,200,100], xstep=rh_xstep,
    x=:RH, y=:A_net, ylab="An", xlim=(0,100), ylim=(-10,60), pdfname="MED-RH-An-CO2")

# #### Stomatal Conductance

# ##### Ca

visualize_model_BB(group=rh_group,
    x=:Ca, y=:gs, xlim=(0,1500), ylim=(0,0.8), pdfname="BB-Ca-gs-RH")

visualize_model_MED(group=rh_group,
    x=:Ca, y=:gs, xlim=(0,1500), ylim=(0,0.8), pdfname="MED-Ca-gs-RH")

# ##### Ci

visualize_model_BB(group=rh_group,
    legendpos=(0.8,0),
    x=:Ci, y=:gs, xlim=(0,600), ylim=(0,0.8), pdfname="BB-Ci-gs-RH")

visualize_model_MED(group=rh_group,
    legendpos=(0.8,0),
    x=:Ci, y=:gs, xlim=(0,600), ylim=(0,0.8), pdfname="MED-Ci-gs-RH")

# ##### Ta

visualize_model_BB(group=rh_group, xstep=ta_xstep,
    x=:T_air, y=:gs, xlim=(0,50), ylim=(0,0.4), pdfname="BB-T-gs-RH")

visualize_model_MED(group=rh_group, xstep=ta_xstep,
    x=:T_air, y=:gs, xlim=(0,50), ylim=(0,0.4), pdfname="MED-T-gs-RH")

# ##### RH

visualize_model_BB(group=:Weather=>:CO2=>[100,200,300,400,800,1500], xstep=rh_xstep,
    x=:RH, y=:gs, xlim=(0,100), ylim=(0,1.5), pdfname="BB-RH-gs-CO2")

visualize_model_MED(group=:Weather=>:CO2=>[100,200,300,400,800,1500], xstep=rh_xstep,
    x=:RH, y=:gs, xlim=(0,100), ylim=(0,1.5), pdfname="MED-RH-gs-CO2")

# #### Surface Humidity

visualize_model_BB(group=rh_group,
    x=:Ca, y=:hs, xlim=(0,1500), ylim=(0,1), pdfname="BB-Ca-hs-RH")

visualize_model_MED(group=rh_group,
    x=:Ca, y=:hs, xlim=(0,1500), ylim=(0,1), pdfname="MED-Ca-hs-RH")

# #### Leaf Temperature

visualize_model_BB(group=rh_group,
    x=:Ca, y=:T, ylab="Tl", xlim=(0,1500), ylim=(31,36), pdfname="BB-Ca-Tl-RH")

visualize_model_MED(group=rh_group,
    x=:Ca, y=:T, ylab="Tl", xlim=(0,1500), ylim=(31,36), pdfname="MED-Ca-Tl-RH")

# ## Stress Response

# ### Nitrogen Stress

nitrogen_group = :Nitrogen => :N => 2:-0.5:0.5
nitrogen_xstep = :Nitrogen => :N => 0.5:0.01:2.0;

# #### Relative Humidity

visualize_model_MED(group=rh_group, xstep=nitrogen_xstep,
    x=:Np, y=:A_net, xlab="N", ylab="An", ylim=(-10,60), pdfname="MED-N-An-RH")

visualize_model_MED(group=nitrogen_group, xstep=rh_xstep, legend="N", names=:Np,
    x=:RH, y=:A_net, ylab="An", xlim=(0,100), ylim=(-10,60), pdfname="MED-N-RH-An")

# #### CO2

visualize_model_MED(group=:Weather=>:CO2=>[800,400,300,200,100], xstep=nitrogen_xstep,
    x=:Np, y=:A_net, xlab="N", ylab="An", legend="Ca", ylim=(-10,60), pdfname="MED-N-An-Ca")

visualize_model_MED(group=nitrogen_group, xstep=co2_xstep, legend="N", names=:Np,
    x=:Ca, y=:A_net, ylab="An", xlim=(0,1500), ylim=(-10,60), pdfname="MED-N-Ca-An")

visualize_model_MED(group=nitrogen_group, xstep=co2_xstep, legend="N", names=:Np,
    x=:Ci, y=:A_net, ylab="An", xlim=(0,800), ylim=(-10,60), pdfname="MED-N-Ci-An")

# #### Air Temperature

visualize_model_MED(group=:Weather=>:T_air=>40:-5:10, xstep=nitrogen_xstep,
    x=:Np, y=:A_net, xlab="N", ylab="An", legend="Ta", ylim=(-10,60), pdfname="MED-N-An-Ta")

visualize_model_MED(group=nitrogen_group, xstep=ta_xstep,
    legend="Np", legendpos=(0.1,-0.1), names=:Np,
    x=:T_air, y=:A_net, xlab="Ta", ylab="An", xlim=(0,50), ylim=(-10,60), pdfname="MED-N-Ta-An")

visualize_model_BB(group=nitrogen_group, xstep=ta_xstep,
    legend="Np", legendpos=(0.1,-0.1), names=:Np,
    x=:T_air, y=:A_net, xlab="Ta", ylab="An", xlim=(0,50), ylim=(-10,60), pdfname="BB-N-Ta-An")

# #### Irradiance

visualize_model_MED(group=:Weather=>:PFD=>1800:-400:600, xstep=nitrogen_xstep,
    legendpos=(0.1,0.25),
    x=:Np, y=:A_net, xlab="Np", ylab="An", legend="I", ylim=(-10,60), pdfname="MED-N-An-I")

visualize_model_BB(group=:Weather=>:PFD=>1800:-400:600, xstep=nitrogen_xstep,
    legendpos=(0.1,0.25),
    x=:Np, y=:A_net, xlab="Np", ylab="An", legend="I", ylim=(-10,60), pdfname="BB-N-An-I")

visualize_model_MED(group=nitrogen_group, xstep=:Weather=>:PFD=>0:10:2000, legend="N", names=:Np,
    x=:PFD, y=:A_net, xlab="I", ylab="An", xlim=(0,2000), ylim=(-10,60), pdfname="MED-N-I-An")

# ### Water Stress

water_group = :StomataTuzet => :WP_leaf => 0:-0.5:-2
water_xstep = :StomataTuzet => :WP_leaf => -2:0.02:0;

# #### Relative Humidity

visualize_model_MED(group=rh_group, xstep=water_xstep,
    legendpos=(0.1,-0.1),
    x=:WP_leaf, y=:A_net, xlab="Ψv", ylab="An", ylim=(-10,60), pdfname="MED-Ψ-An-RH")

visualize_model_BB(group=rh_group, xstep=water_xstep,
    legendpos=(0.1,-0.1),
    x=:WP_leaf, y=:A_net, xlab="Ψv", ylab="An", ylim=(-10,60), pdfname="BB-Ψ-An-RH")

visualize_model_MED(group=water_group, xstep=rh_xstep,
    x=:RH, y=:A_net, ylab="An", legend="Ψv", xlim=(0,100), ylim=(-10,60), pdfname="MED-Ψ-RH-An")

# #### CO2

visualize_model_MED(group=:Weather=>:CO2=>[1500,800,400,300,200,100], xstep=water_xstep,
    x=:WP_leaf, y=:A_net, xlab="Ψv", ylab="An", legend="Ca", ylim=(-10,60), pdfname="MED-Ψ-An-Ca")

visualize_model_MED(group=water_group, xstep=co2_xstep,
    x=:Ca, y=:A_net, ylab="An", legend="Ψv", xlim=(0,1500), ylim=(-10,60), pdfname="MED-Ψ-Ca-An")

visualize_model_MED(group=water_group, xstep=co2_xstep,
    x=:Ci, y=:A_net, ylab="An", legend="Ψv", xlim=(0,800), ylim=(-10,60), pdfname="MED-Ψ-Ci-An")

# #### Air Temperature

visualize_model_MED(group=:Weather=>:T_air=>40:-5:10, xstep=water_xstep,
    x=:WP_leaf, y=:A_net, xlab="Ψv", ylab="An", legend="Ta", ylim=(-10,60), pdfname="MED-Ψ-An-Ta")

visualize_model_MED(group=water_group, xstep=ta_xstep,
    legendpos=(0.1,-0.1),
    x=:T_air, y=:A_net, xlab="Ta", ylab="An", legend="Ψv", xlim=(0,50), ylim=(-10,60), pdfname="MED-Ψ-Ta-An")

visualize_model_BB(group=water_group, xstep=ta_xstep,
    legendpos=(0.1,-0.1),
    x=:T_air, y=:A_net, xlab="Ta", ylab="An", legend="Ψv", xlim=(0,50), ylim=(-10,60), pdfname="BB-Ψ-Ta-An")

# #### Irradiance

visualize_model_MED(group=:Weather=>:PFD=>1800:-400:200, xstep=water_xstep,
    x=:WP_leaf, y=:A_net, xlab="Ψv", ylab="An", legend="I", ylim=(-10,60), pdfname="MED-Ψ-An-I")

visualize_model_MED(group=water_group, xstep=:Weather=>:PFD=>0:10:2000,
    x=:PFD, y=:A_net, xlab="I", ylab="An", legend="Ψv", xlim=(0,2000), ylim=(-10,60), pdfname="MED-Ψ-I-An")

# ## Stress Interaction

visualize_stress(S::Type{<:GasExchange};
    config=(),
    configΔ=(),
    x=:Np,
    y=:WP_leaf,
    z=:A_net,
    xstep=:Nitrogen=>:N=>1:0.02:2,
    ystep=:StomataTuzet=>:WP_leaf=>-2:0.04:0,
    kind=:contour, #:heatmap,
    legend=false,
    aspect=1,
    pdfname=nothing,
    kwargs...
) = begin
    p = visualize(S, x, y, z; config=(config, configΔ), kind, xstep, ystep, legend, aspect, kwargs...)
    !isnothing(pdfname) && p' |> Gadfly.PDF("$pdfname.pdf", 10*Gadfly.cm)
    p' |> Gadfly.SVG()
end

visualize_stress_BB(; config=bb_config, kw...) = visualize_stress(GasExchangeBallBerry; config=config, kw...)
visualize_stress_MED(; config=med_config, kw...) = visualize_stress(GasExchangeMedlyn; config=config, kw...)
visualize_stress(; kw...) = visualize_stress_MED(; kw...)

# ### RH

for x in (30, 50, 70, 90)
    visualize_stress_BB(
        configΔ=:Weather=>:RH=>x,
        xlab="Np", ylab="Ψv", zlab="An", zlim=(-10,60), zgap=1, zlabgap=10,
        pdfname="BB-NxΨ-RH-$x"
    )
    visualize_stress_MED(
        configΔ=:Weather=>:RH=>x,
        xlab="Np", ylab="Ψv", zlab="An", zlim=(-10,60), zgap=1, zlabgap=10,
        pdfname="MED-NxΨ-RH-$x"
    )
end

# ### CO2

for x in (200, 400, 600, 800)
    visualize_stress_BB(
        configΔ=:Weather=>:CO2=>x,
        xlab="Np", ylab="Ψv", zlab="An", zlim=(-10,60), zgap=1, zlabgap=10,
        pdfname="BB-NxΨ-Ca-$x"
    )
    visualize_stress_MED(
        configΔ=:Weather=>:CO2=>x,
        xlab="Np", ylab="Ψv", zlab="An", zlim=(-10,60), zgap=1, zlabgap=10,
        pdfname="MED-NxΨ-Ca-$x"
    )
end

for x in (400, 800)
    visualize_stress_BB(
        configΔ=:Weather=>:CO2=>x,
        z=:gs,
        xlab="Np", ylab="Ψv", zlim=(0,0.4), zgap=0.01, zlabgap=0.1,
        #pdfname="BB-NxΨ-gs-Ca-$x"
    )
    visualize_stress_MED(
        configΔ=:Weather=>:CO2=>x,
        z=:gs,
        xlab="Np", ylab="Ψv", zlim=(0,0.4), zgap=0.01, zlabgap=0.1,
        #pdfname="MED-NxΨ-gs-Ca-$x"
    )
end

# ### PFD

for x in (500, 1000, 1500, 2000)
    visualize_stress_BB(
        configΔ=:Weather=>:PFD=>x,
        xlab="Np", ylab="Ψv", zlab="An", zlim=(-10,60), zgap=1, zlabgap=10,
        pdfname="BB-NxΨ-I-$x"
    )
    visualize_stress_MED(
        configΔ=:Weather=>:PFD=>x,
        xlab="Np", ylab="Ψv", zlab="An", zlim=(-10,60), zgap=1, zlabgap=10,
        pdfname="MED-NxΨ-I-$x"
    )
end

# ## Extras

# ### Nitrogen Parameter Sensitivty

visualize_model(group=:NitrogenDependence=>:s=>5.0:-0.5:3.5,
    legendpos=(0.8,0),
    x=:Ca, y=:A_net, ylab="An", xlim=(0,800), ylim=(0,60), pdfname="sen-N-Ca-An-s")

visualize_model(group=:NitrogenDependence=>:N0=>0.5:-0.1:0.2,
    legendpos=(0.8,0),
    x=:Ca, y=:A_net, ylab="An", xlim=(0,800), ylim=(0,60), pdfname="sen-N-Ca-An-N0")

# ### Kubien Replicates

@system WeatherVPD(Weather) begin
    VPD: vapor_pressure_deficit ~ preserve(u"kPa", parameter)
    RH(T_air, VPD, RH=vp.RH): relative_humidity => RH(T_air, VPD) * 100 ~ track(u"percent")
end

@system GasExchangeBallBerry2(GasExchangeBallBerry, WeatherVPD, Controller) <: GasExchange

@system GasExchangeMedlyn2(GasExchangeMedlyn, WeatherVPD, Controller) <: GasExchange

Kubien = (
    :C4 => (
        :Om => 200,
    ),
    :Weather => (
        :PFD => 1500,
        :CO2 => 370,
        :VPD => 12u"mbar",
    )
);

KubienWT = (Kubien, (
    :C4 => (
        :Vcm25 => 13.2 * 3.9, # 51.48
        :EaVc => 56.1,
        :Vpm25 => 159.9,
        :EaVp => 71.6,
    )
));

KubienAR1 = (Kubien, (
    :C4 => (
        :Vcm25 => 6.5 * 3.8, # 24.7
        :EaVc => 57.0,
        :Vpm25 => 186.3,
        :EaVp => 69.4,
    )
));

KubienAR2 = (Kubien, (
    :C4 => (
        :Vcm25 => 4.2 * 3.7, # 15.54
        :EaVc => 59.3,
        :Vpm25 => 146.9,
        :EaVp => 74.0,
    )
));

KubienPatch = @config(
    (:StomataBallBerry => :g0 => 0.138) +
    (:StomataMedlyn => :g0 => 0.138) +
    !(:C4 => :Jm25 => [300, 240, 120])
);

# #### Initial

# ##### Ball-Berry

visualize(GasExchangeBallBerry2, :T, :A_net, config=bb_config,
    group=[KubienWT, KubienAR1, KubienAR2],
    xstep=:Weather=>:T_air=>0:0.2:45,
    xlab="Tl", ylim=(0,60), kind=:line, names=["WT", "AR1", "AR2"])

ans' |> Gadfly.PDF("BB-Tl-An-Kubien.pdf")

visualize(GasExchangeBallBerry2, :T, :gs, config=bb_config,
    group=[KubienWT, KubienAR1, KubienAR2],
    xstep=:Weather=>:T_air=>0:0.2:45,
    xlab="Tl", ylim=(0,0.8), kind=:line, names=["WT", "AR1", "AR2"])

ans' |> Gadfly.PDF("BB-Tl-gs-Kubien.pdf")

visualize(GasExchangeBallBerry2, :T, :Ci, config=bb_config,
    group=[KubienWT, KubienAR1, KubienAR2],
    xstep=:Weather=>:T_air=>0:0.2:45,
    xlab="Tl", ylim=(0,350), kind=:line, names=["WT", "AR1", "AR2"])

ans' |> Gadfly.PDF("BB-Tl-Ci-Kubien.pdf")

# ##### Medlyn

visualize(GasExchangeMedlyn2, :T, :A_net, config=med_config,
    group=[KubienWT, KubienAR1, KubienAR2],
    xstep=:Weather=>:T_air=>0:0.2:45,
    xlab="Tl", ylim=(0,60), kind=:line, names=["WT", "AR1", "AR2"])

ans' |> Gadfly.PDF("MED-Tl-An-Kubien.pdf")

visualize(GasExchangeMedlyn2, :T, :gs, config=med_config,
    group=[KubienWT, KubienAR1, KubienAR2],
    xstep=:Weather=>:T_air=>0:0.2:45,
    xlab="Tl", ylim=(0,0.8), kind=:line, names=["WT", "AR1", "AR2"])

ans' |> Gadfly.PDF("MED-Tl-gs-Kubien.pdf")

visualize(GasExchangeMedlyn2, :T, :Ci, config=med_config,
    group=[KubienWT, KubienAR1, KubienAR2],
    xstep=:Weather=>:T_air=>0:0.2:45,
    xlab="Tl", ylim=(0,350), kind=:line, names=["WT", "AR1", "AR2"])

ans' |> Gadfly.PDF("MED-Tl-Ci-Kubien.pdf")

# #### Final

obs_kubien = CSV.read("obs-kubien.csv") |> unitfy

visualize_kubien(S::Type{<:GasExchange}, x, y, df; pdfname=nothing, kwargs...) = begin
    p = visualize(S, x, y; kwargs...)
    plot!(p, @where(df, :treatment .== "WT"), :Tl, y, name="", color=1)
    plot!(p, @where(df, :treatment .== "AR1"), :Tl, y, name="", color=2)
    plot!(p, @where(df, :treatment .== "AR2"), :Tl, y, name="", color=3)
    !isnothing(pdfname) && p' |> Gadfly.PDF("$pdfname.pdf")
    p' |> Gadfly.SVG()
end

# ##### Ball-Berry

visualize_kubien(GasExchangeBallBerry2, :T, :A_net, obs_kubien; config=bb_config,
    group=@config([KubienWT, KubienAR1, KubienAR2] + KubienPatch),
    xstep=:Weather=>:T_air=>0:0.2:45,
    xlab="Tl", ylab="An", ylim=(0,55), kind=:line, names=["WT", "AR1", "AR2"],
    legendpos=(0.1,-0.1),
    pdfname="BB-Tl-An-Kubien-patch")

visualize_kubien(GasExchangeBallBerry2, :T, :gs, obs_kubien; config=bb_config,
    group=@config([KubienWT, KubienAR1, KubienAR2] + KubienPatch),
    xstep=:Weather=>:T_air=>0:0.2:45,
    xlab="Tl", ylim=(0,0.8), kind=:line, names=["WT", "AR1", "AR2"],
    legendpos=(0.1,-0.1),
    pdfname="BB-Tl-gs-Kubien-patch")

visualize_kubien(GasExchangeBallBerry2, :T, :Ci, obs_kubien; config=bb_config,
    group=@config([KubienWT, KubienAR1, KubienAR2] + KubienPatch),
    xstep=:Weather=>:T_air=>0:0.2:45,
    xlab="Tl", ylim=(0,370), kind=:line, names=["WT", "AR1", "AR2"],
    legendpos=(0.7,-0.4),
    pdfname="BB-Tl-Ci-Kubien-patch")

# ##### Medlyn

visualize_kubien(GasExchangeMedlyn2, :T, :A_net, obs_kubien; config=med_config,
    group=@config([KubienWT, KubienAR1, KubienAR2] + KubienPatch),
    xstep=:Weather=>:T_air=>0:0.2:45,
    xlab="Tl", ylab="An", ylim=(0,55), kind=:line, names=["WT", "AR1", "AR2"],
    legendpos=(0.1,-0.1),
    pdfname="MED-Tl-An-Kubien-patch")

visualize_kubien(GasExchangeMedlyn2, :T, :gs, obs_kubien; config=med_config,
    group=@config([KubienWT, KubienAR1, KubienAR2] + KubienPatch),
    xstep=:Weather=>:T_air=>0:0.2:45,
    xlab="Tl", ylim=(0,0.8), kind=:line, names=["WT", "AR1", "AR2"],
    legendpos=(0.1,-0.1),
    pdfname="MED-Tl-gs-Kubien-patch")

visualize_kubien(GasExchangeMedlyn2, :T, :Ci, obs_kubien; config=med_config,
    group=@config([KubienWT, KubienAR1, KubienAR2] + KubienPatch),
    xstep=:Weather=>:T_air=>0:0.2:45,
    xlab="Tl", ylim=(120,370), kind=:line, names=["WT", "AR1", "AR2"],
    legendpos=(0.1,0.25),
    pdfname="MED-Tl-Ci-Kubien-patch")

# ### Sample Code

Cropbox.writecodehtml("energy-balance.html", """
@system EnergyBalance(Weather) begin
  ..
  ϵ: leaf_thermal_emissivity => 0.97 ~ preserve(parameter)
  σ: stefan_boltzmann_constant => u"σ" ~ preserve(u"W/m^2/K^4")
  λ: latent_heat_of_vaporization_at_25 => 44 ~ preserve(u"kJ/mol", parameter)
  Cp: specific_heat_of_air => 29.3 ~ preserve(u"J/mol/K", parameter)

  Δw(T, T_air, RH, ea=vp.ambient, es=vp.saturation): leaf_vapor_pressure_gradient => begin
      es(T) - ea(T_air, RH)
  end ~ track(u"kPa")
  E(gv, Δw): transpiration => gv*Δw ~ track(u"mmol/m^2/s")

  H(Cp, gh, ΔT): sensible_heat_flux => Cp*gh*ΔT ~ track(u"W/m^2")
  λE(λ, E): latent_heat_flux => λ*E ~ track(u"W/m^2")

  ΔT(R_net, H, λE): temperature_adjustment => begin
      R_net ⩵ H + λE
  end ~ bisect(lower=-10, upper=10, u"K", evalunit=u"W/m^2")

  T(T_air, ΔT): leaf_temperature => (T_air + ΔT) ~ track(u"°C")
  Tk(T): absolute_leaf_temperature ~ track(u"K")
end""")

Cropbox.writecodehtml("simulate.html", """
c0 = (
    :StomataMedlyn => (g0 = 0.031, g1 = 1.281),
    :NitrogenDependence => (s = 4.191, N0 = 0.343),
    .. # default configuration
)
r = simulate(GasExchangeMedlyn; config=c0)""")

Cropbox.writecodehtml("calibrate.html", """
obs_df = .. # data frame contains gas exchange measurements
obs_C = .. # list of configurations for each measurement
c1 = calibrate(GasExchangeMedlyn, obs_df, obs_C;
    index = [:PARi => :PFD, :CO2S => :CO2, :RH_S => :RH, :Tair => :T_air, :Press => :P_air, :SPAD],
    target = [:Photo => :A_net, :gs],
    parameters = (
        :NitrogenDependence => (s=(0, 10), N0=(0, 1)),
        :StomataMedlyn => (g0=(0, 1), g1=(0, 10)),
    ),
    metric = :prmse,
    .. # other options
)""")

Cropbox.writecodehtml("visualize.html", """
visualize(GasExchangeMedlyn; config=(c0, c1),
    x = :Ci, y = :gs,
    xstep = :Weather => :CO2 => 10:10:1500,
    group = :Weather => :RH => [80, 60, 40, 20],
    xlim = (0, 600), ylim = (0, 1), legendpos = (0.8, 0),
    kind = :line
)""")
