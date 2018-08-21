using RRTM
using Base.Test

# write your own tests here
@test radiation() ≈ 



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