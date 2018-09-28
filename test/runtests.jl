using RRTM
using Base.Test

using PyCall
@pyimport xarray as xr

dset = xr.open_dataset("$(@__DIR__)/true.nc",decode_times=false)

# # write your own tests here
# @test radiation() ≈
#

mask_dset = xr.open_dataset("$(@__DIR__)/../netcdfs/unit.24.T63GR15.nc",decode_times=false)
time_i = 1
lat_i = 91
lon_i = 91
CO2_multiple = 1
flx_lw_up_toa,flx_lw_up_clr_toa,flx_sw_up_toa,flx_sw_dn_toa,flx_sw_up_clr_toa,flx_lw_up_surf,flx_lw_dn_surf,flx_lw_up_clr_surf,flx_lw_dn_clr_surf,flx_sw_up_surf,flx_sw_dn_surf,flx_sw_up_clr_surf,flx_sw_dn_clr_surf,flx_lw_up_trop,flx_lw_dn_trop,flx_lw_up_clr_trop,flx_lw_dn_clr_trop,flx_sw_up_trop,flx_sw_dn_trop,flx_sw_up_clr_trop,flx_sw_dn_clr_trop,flx_lw_up_vr,flx_lw_dn_vr,flx_lw_up_clr_vr,flx_lw_dn_clr_vr,flx_sw_up,flx_sw_dn,flx_sw_up_clr,flx_sw_dn_clr = radiation(
  mask_dset["lat"][:values][[lat_i]],
  mask_dset["SLM"][:values][[lat_i],[lon_i]],
  mask_dset["GLAC"][:values][[lat_i],[lon_i]],
  dset["ktype"][:values][[time_i],[lat_i],[lon_i]],
  dset["pp_fl"][:values][[time_i],:,[lat_i],[lon_i]],
  dset["pp_hl"][:values][[time_i],:,[lat_i],[lon_i]],
  dset["pp_sfc"][:values][[time_i],[lat_i],[lon_i]],
  dset["tk_fl"][:values][[time_i],:,[lat_i],[lon_i]],
  dset["tk_hl"][:values][[time_i],:,[lat_i],[lon_i]],
  dset["tk_sfc"][:values][[time_i],[lat_i],[lon_i]],
  dset["q_vap"][:values][[time_i],:,[lat_i],[lon_i]],
  dset["q_liq"][:values][[time_i],:,[lat_i],[lon_i]],
  dset["q_ice"][:values][[time_i],:,[lat_i],[lon_i]],
  dset["cdnc"][:values][[time_i],:,[lat_i],[lon_i]],
  dset["cld_frc"][:values][[time_i],:,[lat_i],[lon_i]],
  dset["m_o3"][:values][[time_i],:,[lat_i],[lon_i]],
  CO2_multiple * fill(0.000284725 * 44.011 / 28.970, size(dset["m_o3"][:values][[time_i],:,[lat_i],[lon_i]])),
  dset["m_ch4"][:values][[time_i],:,[lat_i],[lon_i]],
  dset["m_n2o"][:values][[time_i],:,[lat_i],[lon_i]],
  dset["psctm"][:values][[time_i],1,1],
  dset["cos_mu0"][:values][[time_i],[lat_i],[lon_i]],
  dset["cos_mu0m"][:values][[time_i],[lat_i],[lon_i]],
  dset["alb"][:values][[time_i],[lat_i],[lon_i]],
  dset["hyai"][:values],
  dset["hybi"][:values]
)

# radiation(
#   philat,
#   laland,
#   laglac,
#   ktype,
#   pp_fl,
#   pp_hl,
#   pp_sfc,
#   tk_fl,
#   tk_hl,
#   tk_sfc,
#   xm_vap,
#   xm_liq,
#   xm_ice,
#   cdnc,
#   cld_frc,
#   xm_o3,
#   xm_co2,
#   xm_ch4,
#   xm_n2o,
#   solar_constant,
#   cos_mu0,
#   cos_mu0m,
#   alb,
#   hyai,
#   hybi
# )
# #
# #
# #
# # # pressure = [pp_fl, pp_hl, pp_sfc]
# # # temperature = [tk_fl, tk_hl, tk_sfc]
# # # water_vapor = [q_vap]
# # # clouds = [q_liq, q_ice, cdnc, cld_frc]
# # # albedo = [alb]
# # # solar = [psctm, cos_mu0, cos_mu0m]
# # # other_ghgs = [m_co2, m_o3, m_ch4, m_n2o]
# # # convection = [ktype]
# # # ??? = [hyai, hybi]
# # # number of levels/layers/timesteps = mlev, ilev, time