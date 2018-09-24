using RRTM
using Base.Test

# write your own tests here
@test radiation() ≈ 

radiation(
  philat,
  laland,
  laglac,
  ktype,
  pp_fl,
  pp_hl,
  pp_sfc,
  tk_fl,
  tk_hl,
  tk_sfc,
  xm_vap,
  xm_liq,
  xm_ice,
  cdnc,
  cld_frc,
  xm_o3,
  xm_co2,
  xm_ch4,
  xm_n2o,
  solar_constant,
  cos_mu0,
  cos_mu0m,
  alb,
  hyai,
  hybi
)



# pressure = [pp_fl, pp_hl, pp_sfc]
# temperature = [tk_fl, tk_hl, tk_sfc]
# water_vapor = [q_vap]
# clouds = [q_liq, q_ice, cdnc, cld_frc]
# albedo = [alb]
# solar = [psctm, cos_mu0, cos_mu0m]
# other_ghgs = [m_co2, m_o3, m_ch4, m_n2o]
# convection = [ktype]
# ??? = [hyai, hybi]
# number of levels/layers/timesteps = mlev, ilev, time