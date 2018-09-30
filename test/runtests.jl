using RRTM
using Base.Test

using PyCall
@pyimport xarray as xr

dset = xr.open_dataset("$(@__DIR__)/true.nc",decode_times=false)
# dset = xr.open_dataset("true.nc",decode_times=false)

mask_dset = xr.open_dataset("$(@__DIR__)/../netcdfs/unit.24.T63GR15.nc",decode_times=false)
# mask_dset = xr.open_dataset("../netcdfs/unit.24.T63GR15.nc",decode_times=false)

time_i = 1
lat_is = [1,8,16,24,32,40,48,57,65,73,81,89,96] #90ºN:15º:90ºS
lon_i = 1 # 0º
CO2_multiple = 1

CO2s = [0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552, 0.000432552]
psctm = 1407.5099

# 90ºN - north pole
atmosphere_90N = Dict(
  :philat => 88.57216851400727,
  :laland => 0.0,
  :laglac => 0.0,
  :ktype => 2.0f0,
  :pp_fl => Float32[0.994593, 4.28064, 11.123, 23.1491, 42.585, 73.5523, 121.532, 192.916, 295.714, 440.01, 637.481, 901.09, 1245.06, 1684.58, 2235.44, 2913.63, 3743.06, 4751.77, 5964.33, 7405.94, 9102.23, 11068.1, 13248.3, 15601.2, 18159.9, 20941.2, 23947.1, 27177.2, 30620.8, 34269.6, 38109.6, 42118.6, 46284.7, 50598.3, 55039.9, 59588.6, 64224.7, 68923.0, 73646.4, 78348.7, 82973.6, 87438.1, 91630.7, 95421.1, 98643.9, 1.01082f5, 1.02471f5],
  :pp_hl => Float32[0.0, 1.98919, 6.57209, 15.6739, 30.6243, 54.5457, 92.5588, 150.505, 235.327, 356.1, 523.919, 751.043, 1051.14, 1438.99, 1930.18, 2540.7, 3286.55, 4199.57, 5303.96, 6624.7, 8187.19, 10017.3, 12118.9, 14377.8, 16824.6, 19495.3, 22387.2, 25507.0, 28847.3, 32394.3, 36144.9, 40074.2, 44163.0, 48406.3, 52790.2, 57289.7, 61887.5, 66561.8, 71284.3, 76008.5, 80688.9, 85258.4, 89617.7, 93643.6, 97198.7, 1.00089f5, 1.02074f5, 1.02867f5],
  :tk_fl => Float32[223.884, 248.48, 255.449, 266.826, 270.408, 259.021, 246.1, 235.706, 233.366, 230.77, 226.04, 223.52, 219.901, 216.931, 215.236, 214.075, 212.593, 210.68, 209.366, 208.503, 209.923, 209.844, 209.16, 207.97, 206.084, 206.361, 207.787, 209.201, 212.003, 215.646, 219.905, 224.229, 228.885, 232.647, 236.42, 239.503, 242.464, 243.299, 247.399, 250.615, 253.884, 257.04, 259.985, 262.666, 264.844, 266.385, 267.419],
  :tk_hl => Float32[207.864, 239.904, 252.43, 261.807, 268.742, 264.478, 252.3, 240.692, 234.493, 232.023, 228.324, 224.737, 221.65, 218.367, 216.056, 214.637, 213.317, 211.614, 210.007, 208.924, 209.23, 209.882, 209.484, 208.54, 206.997, 206.227, 207.095, 208.515, 210.643, 213.876, 217.839, 222.132, 226.624, 230.819, 234.588, 238.006, 241.027, 242.894, 245.416, 249.064, 252.316, 255.541, 258.605, 261.435, 263.885, 265.767, 267.127, 269.135],
  :xm_vap => Float32[2.39237f-6, 2.39403f-6, 2.39567f-6, 2.40148f-6, 2.40799f-6, 2.40958f-6, 2.41401f-6, 2.4205f-6, 2.42498f-6, 2.42571f-6, 2.42543f-6, 2.42371f-6, 2.41548f-6, 2.40475f-6, 2.395f-6, 2.39481f-6, 2.4099f-6, 2.42986f-6, 2.44496f-6, 2.48615f-6, 2.57795f-6, 2.64616f-6, 2.8534f-6, 3.92202f-6, 7.71968f-6, 1.03065f-5, 9.37027f-6, 9.46506f-6, 1.15951f-5, 1.67026f-5, 2.45891f-5, 3.92644f-5, 5.00154f-5, 7.7754f-5, 0.000108968, 0.000168396, 0.000213419, 0.000328351, 0.000519952, 0.000605298, 0.000809924, 0.0010138, 0.0011645, 0.00139964, 0.00166935, 0.00179955, 0.001852],
  :xm_liq => Float32[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.42299f-26, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  :xm_ice => Float32[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.27147f-5, 0.0, 0.0, 0.0, 0.0, 1.89825f-6, 0.0, 0.0],
  :cdnc => Float32[5.0f7, 5.0f7, 5.0f7, 5.0f7, 5.0f7, 5.0f7, 5.0f7, 5.0f7, 5.0f7, 5.0f7, 5.0f7, 5.0f7, 5.0f7, 5.0f7, 5.0f7, 5.0f7, 5.0f7, 5.0f7, 5.0f7, 5.0f7, 5.0f7, 5.0f7, 5.0f7, 5.0f7, 5.0f7, 5.0f7, 5.00009f7, 5.00113f7, 5.0072f7, 5.02895f7, 5.08351f7, 5.18876f7, 5.35658f7, 5.58934f7, 5.87981f7, 6.21451f7, 6.57777f7, 6.95414f7, 7.32951f7, 7.6923f7, 8.0f7, 8.0f7, 8.0f7, 8.0f7, 8.0f7, 8.0f7, 8.0f7],
  :cld_frc => Float32[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.546269, 0.0, 0.0, 0.0, 0.0, 0.0369192, 0.0, 0.0],
  :xm_o3 => Float32[1.4375f-8, 2.43107f-7, 7.17661f-7, 1.52752f-6, 2.6913f-6, 4.06779f-6, 5.70307f-6, 6.96006f-6, 7.71852f-6, 7.82512f-6, 7.64679f-6, 7.30457f-6, 6.92151f-6, 6.49013f-6, 6.00309f-6, 5.50232f-6, 4.95128f-6, 4.30694f-6, 3.59893f-6, 2.88255f-6, 2.272f-6, 1.85168f-6, 1.53636f-6, 1.12178f-6, 8.4771f-7, 6.23823f-7, 4.28845f-7, 2.95365f-7, 2.11821f-7, 1.60993f-7, 1.27915f-7, 1.04783f-7, 8.61367f-8, 7.15079f-8, 6.73225f-8, 6.371f-8, 6.05139f-8, 5.70196f-8, 5.33389f-8, 4.97705f-8, 4.63004f-8, 4.3429f-8, 4.10897f-8, 3.89749f-8, 3.74552f-8, 3.66676f-8, 3.66676f-8],
  :xm_co2,
  :xm_ch4 => Float32[5.47948f-8, 5.50713f-8, 5.5959f-8, 5.80949f-8, 6.24999f-8, 7.10122f-8, 8.62122f-8, 1.10624f-7, 1.4548f-7, 1.89255f-7, 2.37155f-7, 2.83069f-7, 3.2244f-7, 3.53508f-7, 3.76709f-7, 3.93491f-7, 4.05539f-7, 4.14184f-7, 4.20378f-7, 4.2483f-7, 4.28051f-7, 4.30391f-7, 4.32062f-7, 4.33267f-7, 4.34169f-7, 4.3486f-7, 4.35398f-7, 4.35822f-7, 4.3616f-7, 4.36431f-7, 4.36651f-7, 4.3683f-7, 4.36978f-7, 4.371f-7, 4.37203f-7, 4.3729f-7, 4.37363f-7, 4.37425f-7, 4.37479f-7, 4.37524f-7, 4.37563f-7, 4.37596f-7, 4.37623f-7, 4.37646f-7, 4.37663f-7, 4.37675f-7, 4.37682f-7],
  :xm_n2o => Float32[5.03772f-9, 5.14763f-9, 5.50119f-9, 6.35613f-9, 8.13862f-9, 1.16585f-8, 1.82028f-8, 2.94688f-8, 4.74031f-8, 7.36759f-8, 1.08635f-7, 1.50452f-7, 1.9533f-7, 2.38841f-7, 2.77527f-7, 3.09684f-7, 3.3537f-7, 3.55344f-7, 3.70535f-7, 3.81945f-7, 3.90472f-7, 3.96814f-7, 4.01425f-7, 4.04789f-7, 4.07331f-7, 4.09295f-7, 4.10831f-7, 4.12046f-7, 4.13016f-7, 4.13798f-7, 4.14433f-7, 4.14952f-7, 4.1538f-7, 4.15737f-7, 4.16035f-7, 4.16287f-7, 4.165f-7, 4.16682f-7, 4.16837f-7, 4.1697f-7, 4.17083f-7, 4.17179f-7, 4.17259f-7, 4.17324f-7, 4.17375f-7, 4.1741f-7, 4.1743f-7],
  :solar_constant => psctm,
  :cos_mu0 => 0.0,
  :cos_mu0m => 0.1,
  :alb => 0.07,
  :aer_tau_lw_vr => ,
  :aer_tau_sw_vr => ,
  :aer_piz_sw_vr => ,
  :aer_cg_sw_vr => 
)

output = radiation(atmosphere_90N, SW_correction=false, output=:profile)

@test output[:LW_up] ≈ Float32[297.18, 294.719, 291.857, 287.862, 283.018, 277.469, 271.527, 265.593, 246.598, 241.478, 237.095, 234.587, 231.118, 227.753, 224.316, 221.106, 217.791, 214.953, 212.45, 210.359, 208.633, 207.217, 205.912, 205.028, 204.607, 204.216, 203.847, 203.476, 203.039, 202.838, 202.773, 202.812, 202.869, 202.931, 203.032, 203.188, 203.36, 203.503, 203.66, 203.762, 203.849, 203.996, 204.154, 204.271, 204.284, 204.259, 204.238, 204.21] atol=.001
@test output[:LW_dn] ≈ Float32[217.142, 214.046, 208.899, 200.197, 191.167, 181.478, 171.24, 161.299, 115.353, 102.602, 93.1167, 83.8367, 73.7743, 64.5109, 55.8025, 48.1043, 40.881, 35.0809, 30.4824, 27.0386, 24.373, 21.869, 19.3198, 17.2725, 15.8369, 14.4617, 12.9725, 11.4358, 10.1641, 8.97049, 7.81455, 6.67484, 5.61965, 4.69831, 3.91432, 3.21385, 2.58376, 2.11297, 1.69107, 1.33019, 1.08748, 0.87937, 0.675934, 0.445198, 0.25868, 0.13656, 0.0524619, 0.0] atol=.001
@test output[:SW_up] ≈ Float32[50.3689, 50.3659, 50.3589, 50.3452, 50.3232, 50.289, 50.2368, 50.1608, 50.0537, 49.9057, 49.7041, 49.4359, 49.0868, 48.6431, 48.0916, 47.4206, 46.6205, 45.6677, 44.5503, 43.2597, 41.8072, 40.2323, 38.5523, 36.8422, 35.0879, 33.2812, 31.4936, 29.8098, 28.281, 26.9394, 25.7654, 24.7133, 23.741, 22.8155, 21.9194, 21.0428, 20.1772, 19.3095, 18.4305, 17.5327, 9.17453, 8.38095, 7.54415, 6.6829, 5.84048, 4.92618, 4.38367, 4.15952] atol=.001
@test output[:SW_dn] ≈ Float32[140.751, 140.739, 140.713, 140.639, 140.458, 140.107, 139.622, 139.088, 138.521, 137.897, 137.194, 136.386, 135.451, 134.365, 133.115, 131.695, 130.102, 128.313, 126.345, 124.224, 121.993, 119.695, 117.304, 114.955, 112.637, 110.243, 107.882, 105.695, 103.688, 101.841, 100.085, 98.3609, 96.5777, 94.8004, 92.8888, 90.9038, 88.7595, 86.6197, 84.279, 81.7042, 70.9631, 68.9652, 66.9218, 64.9181, 62.9907, 61.113, 59.911, 59.4217] atol=.001
@test output[:LW_up_clr] ≈ Float32[297.081, 294.621, 291.762, 287.929, 283.074, 277.518, 271.572, 265.635, 260.017, 254.324, 249.641, 246.936, 243.319, 239.847, 236.327, 233.05, 229.676, 226.786, 224.232, 222.089, 220.307, 218.827, 217.451, 216.494, 216.003, 215.542, 215.11, 214.69, 214.21, 213.973, 213.88, 213.897, 213.936, 213.986, 214.078, 214.228, 214.395, 214.535, 214.69, 214.791, 214.877, 215.023, 215.182, 215.299, 215.312, 215.287, 215.266, 215.237] atol = .001
@test output[:LW_dn_clr] ≈ Float32[192.341, 188.937, 182.973, 174.189, 163.623, 152.111, 139.642, 127.095, 115.353, 102.602, 93.1167, 83.8367, 73.7743, 64.5109, 55.8025, 48.1043, 40.881, 35.0809, 30.4824, 27.0386, 24.373, 21.869, 19.3198, 17.2725, 15.8369, 14.4617, 12.9725, 11.4358, 10.1641, 8.97049, 7.81455, 6.67484, 5.61965, 4.69831, 3.91432, 3.21385, 2.58376, 2.11297, 1.69107, 1.33019, 1.08748, 0.87937, 0.675934, 0.445198, 0.25868, 0.13656, 0.0524619, 0.0] atol = .001
@test output[:SW_up_clr] ≈ Float32[45.1486, 45.1455, 45.1386, 45.1248, 45.1027, 45.0683, 45.0159, 44.9393, 44.8313, 44.682, 44.4786, 44.2079, 43.8556, 43.4079, 42.8516, 42.1747, 41.3678, 40.4071, 39.2807, 37.9802, 36.5169, 34.9306, 33.238, 31.5151, 29.7478, 27.9277, 26.1264, 24.429, 22.8868, 21.5323, 20.3456, 19.2807, 18.295, 17.3555, 16.444, 15.5505, 14.665, 13.7743, 12.8662, 11.9285, 10.9399, 9.88911, 8.77861, 7.63398, 6.51355, 5.52083, 4.79489, 4.49525] atol = .001
@test output[:SW_dn_clr] ≈ Float32[140.751, 140.739, 140.713, 140.639, 140.458, 140.107, 139.622, 139.087, 138.52, 137.895, 137.192, 136.383, 135.446, 134.359, 133.107, 131.685, 130.089, 128.296, 126.323, 124.196, 121.959, 119.653, 117.254, 114.896, 112.569, 110.164, 107.792, 105.594, 103.577, 101.719, 99.953, 98.2192, 96.4263, 94.6392, 92.7176, 90.7223, 88.5674, 86.4165, 84.0642, 81.477, 78.9771, 76.357, 73.6898, 71.0844, 68.5886, 66.4062, 64.85, 64.2179] atol = .001

# radiation(atmosphere_90N, SW_correction = true, output = :fluxes)

# 75ºN,0º - greenland sea

# 60ºN,0º - north sea, near shetlands

# 45ºN,0º - france, near bordeaux

# 30ºN,0º - algeria

# 15ºN,0º - near meeting point of mali, burkina faso, and niger

# 0º,0º - atlantic ocean, gulf of guinea

# 15ºS,0º - atlantic ocean

# 30ºS,0º - atlantic ocean

# 45ºS,0º - atlantic ocean

# 60ºS,0º - atlantic/southern ocean

# 75ºS,0º - atlantic ocean

# 90ºS - south pole


# flx_lw_up_toa,flx_lw_up_clr_toa,flx_sw_up_toa,flx_sw_dn_toa,flx_sw_up_clr_toa,flx_lw_up_surf,flx_lw_dn_surf,flx_lw_up_clr_surf,flx_lw_dn_clr_surf,flx_sw_up_surf,flx_sw_dn_surf,flx_sw_up_clr_surf,flx_sw_dn_clr_surf,flx_lw_up_trop,flx_lw_dn_trop,flx_lw_up_clr_trop,flx_lw_dn_clr_trop,flx_sw_up_trop,flx_sw_dn_trop,flx_sw_up_clr_trop,flx_sw_dn_clr_trop,flx_lw_up_vr,flx_lw_dn_vr,flx_lw_up_clr_vr,flx_lw_dn_clr_vr,flx_sw_up,flx_sw_dn,flx_sw_up_clr,flx_sw_dn_clr = radiation(
#   mask_dset["lat"][:values][[lat_i]],
#   mask_dset["SLM"][:values][[lat_i],[lon_i]],
#   mask_dset["GLAC"][:values][[lat_i],[lon_i]],
#   dset["ktype"][:values][[time_i],[lat_i],[lon_i]],
#   dset["pp_fl"][:values][[time_i],:,[lat_i],[lon_i]],
#   dset["pp_hl"][:values][[time_i],:,[lat_i],[lon_i]],
#   dset["tk_fl"][:values][[time_i],:,[lat_i],[lon_i]],
#   dset["tk_hl"][:values][[time_i],:,[lat_i],[lon_i]],
#   dset["q_vap"][:values][[time_i],:,[lat_i],[lon_i]],
#   dset["q_liq"][:values][[time_i],:,[lat_i],[lon_i]],
#   dset["q_ice"][:values][[time_i],:,[lat_i],[lon_i]],
#   dset["cdnc"][:values][[time_i],:,[lat_i],[lon_i]],
#   dset["cld_frc"][:values][[time_i],:,[lat_i],[lon_i]],
#   dset["m_o3"][:values][[time_i],:,[lat_i],[lon_i]],
#   CO2_multiple * fill(0.000284725 * 44.011 / 28.970, size(dset["m_o3"][:values][[time_i],:,[lat_i],[lon_i]])),
#   dset["m_ch4"][:values][[time_i],:,[lat_i],[lon_i]],
#   dset["m_n2o"][:values][[time_i],:,[lat_i],[lon_i]],
#   dset["psctm"][:values][[time_i],1,1],
#   dset["cos_mu0"][:values][[time_i],[lat_i],[lon_i]],
#   dset["cos_mu0m"][:values][[time_i],[lat_i],[lon_i]],
#   dset["alb"][:values][[time_i],[lat_i],[lon_i]],
#   dset["hyai"][:values],
#   dset["hybi"][:values]
# )
#
# println(mask_dset["lat"][:values][[lat_i]])
# println(mask_dset["SLM"][:values][[lat_i],[lon_i]])
# println(mask_dset["GLAC"][:values][[lat_i],[lon_i]])
# println(dset["ktype"][:values][[time_i],[lat_i],[lon_i]])
# println(dset["pp_fl"][:values][[time_i],:,[lat_i],[lon_i]])
# println(dset["pp_hl"][:values][[time_i],:,[lat_i],[lon_i]])
# println(dset["pp_sfc"][:values][[time_i],[lat_i],[lon_i]])
# println(dset["tk_fl"][:values][[time_i],:,[lat_i],[lon_i]])
# println(dset["tk_hl"][:values][[time_i],:,[lat_i],[lon_i]])
# println(dset["tk_sfc"][:values][[time_i],[lat_i],[lon_i]])
# println(dset["q_vap"][:values][[time_i],:,[lat_i],[lon_i]])
# println(dset["q_liq"][:values][[time_i],:,[lat_i],[lon_i]])
# println(dset["q_ice"][:values][[time_i],:,[lat_i],[lon_i]])
# println(dset["cdnc"][:values][[time_i],:,[lat_i],[lon_i]])
# println(dset["cld_frc"][:values][[time_i],:,[lat_i],[lon_i]])
# println(dset["m_o3"][:values][[time_i],:,[lat_i],[lon_i]])
# println(CO2_multiple * fill(0.000284725 * 44.011 / 28.970, size(dset["m_o3"][:values][[time_i],:,[lat_i],[lon_i]])))
# println(dset["m_ch4"][:values][[time_i],:,[lat_i],[lon_i]])
# println(dset["m_n2o"][:values][[time_i],:,[lat_i],[lon_i]])
# println(dset["psctm"][:values][[time_i],1,1])
# println(dset["cos_mu0"][:values][[time_i],[lat_i],[lon_i]])
# println(dset["cos_mu0m"][:values][[time_i],[lat_i],[lon_i]])
# println(dset["alb"][:values][[time_i],[lat_i],[lon_i]])
# println(dset["hyai"][:values])
# println(dset["hybi"][:values])
#
#
# # radiation(
# #   philat,
# #   laland,
# #   laglac,
# #   ktype,
# #   pp_fl,
# #   pp_hl,
# #   pp_sfc,
# #   tk_fl,
# #   tk_hl,
# #   tk_sfc,
# #   xm_vap,
# #   xm_liq,
# #   xm_ice,
# #   cdnc,
# #   cld_frc,
# #   xm_o3,
# #   xm_co2,
# #   xm_ch4,
# #   xm_n2o,
# #   solar_constant,
# #   cos_mu0,
# #   cos_mu0m,
# #   alb,
# #   hyai,
# #   hybi
# # )
# # #
# # #
# # #
# # # # pressure = [pp_fl, pp_hl, pp_sfc]
# # # # temperature = [tk_fl, tk_hl, tk_sfc]
# # # # water_vapor = [q_vap]
# # # # clouds = [q_liq, q_ice, cdnc, cld_frc]
# # # # albedo = [alb]
# # # # solar = [psctm, cos_mu0, cos_mu0m]
# # # # other_ghgs = [m_co2, m_o3, m_ch4, m_n2o]
# # # # convection = [ktype]
# # # # ??? = [hyai, hybi]
# # # # number of levels/layers/timesteps = mlev, ilev, time
# pp_hl_col = zeros(dset["pp_hl"][:values])
# pp_fl_col = zeros(dset["pp_fl"][:values])
# tk_hl_col = zeros(dset["tk_hl"][:values])
# pp_hl_col[time_i,:,lat_i,lon_i] = dset["pp_hl"][:values][time_i,:,lat_i,lon_i]
# pp_fl_col[time_i,:,lat_i,lon_i] = dset["pp_fl"][:values][time_i,:,lat_i,lon_i]
# tk_hl_col[time_i,:,lat_i,lon_i] = dset["tk_hl"][:values][time_i,:,lat_i,lon_i]
#
# cld_tau_lw_vr, cld_tau_sw_vr, cld_piz_sw_vr, cld_cg_sw_vr = clouds(ntime,nlay,nlev,nlat,nlon,laland,laglac,ktype,zlwp_vr,ziwp_vr,zlwc_vr,ziwc_vr,cdnc_vr,icldlyr)
# cld_tau_lw_vr, cld_tau_sw_vr, cld_piz_sw_vr, cld_cg_sw_vr = clouds(ntime,nlay,nlev,nlat,nlon,laland,laglac,ktype,zlwp_vr,ziwp_vr,zlwc_vr,ziwc_vr,cdnc_vr,icldlyr)
#
#
# aer_tau_lw_vr_col, aer_tau_sw_vr_col, aer_piz_sw_vr_col, aer_cg_sw_vr_col = aerosols(ntime,nlay,nlev,nlat,nlon,dset["philat"][:values],dset["hyai"][:values],dset["hybi"][:values],pp_hl_col,pp_fl_col,tk_hl_col)
# aer_tau_lw_vr_col_col, aer_tau_sw_vr_col_col, aer_piz_sw_vr_col_col, aer_cg_sw_vr_col_col = aerosols(1,nlay,nlev,1,1,mask_dset["lat"][:values][[lat_i]],dset["hyai"][:values],dset["hybi"][:values],dset["pp_hl"][:values][[time_i],:,[lat_i],[lon_i]],dset["pp_fl"][:values][[time_i],:,[lat_i],[lon_i]],dset["tk_hl"][:values][[time_i],:,[lat_i],[lon_i]])
#
# philat = mask_dset["lat"][:values];
# laland = mask_dset["SLM"][:values];
# laglac = mask_dset["GLAC"][:values];
# cld_frc = dset["cld_frc"][:values];
# xm_ice = dset["q_ice"][:values];
# xm_liq = dset["q_liq"][:values];
# pp_hl = dset["pp_hl"][:values];
# pp_fl = dset["pp_fl"][:values];
# tk_fl = dset["tk_fl"][:values];
# cdnc = dset["cdnc"][:values];
# ktype = dset["ktype"][:values];
#
#
#
# ntime, nlay, nlat, nlon = size(tk_fl)
# nlev = nlay + 1
#
#
# # println("Preparing input data for RRTM...")
# cld_frc_vr = zeros(ntime,nlay,nlat,nlon);
# ziwgkg_vr = zeros(ntime,nlay,nlat,nlon);
# zlwgkg_vr = zeros(ntime,nlay,nlat,nlon);
# icldlyr = zeros(Int32,(ntime,nlay,nlat,nlon));
#
# function level_function_1(jk)
#   jkb = nlay+1-jk
#   for jl = 1:nlon, t=1:ntime, lat=1:nlat
#     cld_frc_vr[t,jk,lat,jl] = max(eps(Float32),cld_frc[t,jkb,lat,jl])
#     ziwgkg_vr[t,jk,lat,jl]  = xm_ice[t,jkb,lat,jl]*1000.0./cld_frc_vr[t,jk,lat,jl]
#     zlwgkg_vr[t,jk,lat,jl]  = xm_liq[t,jkb,lat,jl]*1000.0./cld_frc_vr[t,jk,lat,jl]
#   end
# end
#
# for jk = 1:nlay
#   level_function_1(jk)
# end
#
# # --- control for zero:infintesimal or negative cloud fractions
#
# function level_function_2(t,jk,lat,jl)
#   if cld_frc_vr[t,jk,lat,jl] > 2.0*eps(Float32)
#     icldlyr[t,jk,lat,jl] = 1
#   else
#     icldlyr[t,jk,lat,jl] = 0
#     ziwgkg_vr[t,jk,lat,jl] = 0.0
#     zlwgkg_vr[t,jk,lat,jl] = 0.0
#   end
# end
#
# for jk = 1:nlay, jl = 1:nlon, t=1:ntime, lat=1:nlat
#   level_function_2(t,jk,lat,jl)
# end
#
# # pm_hl_vr = zeros(ntime,nlev,nlat,nlon)
# ziwc_vr = zeros(ntime,nlay,nlat,nlon);
# ziwp_vr = zeros(ntime,nlay,nlat,nlon);
# zlwc_vr = zeros(ntime,nlay,nlat,nlon);
# zlwp_vr = zeros(ntime,nlay,nlat,nlon);
# cdnc_vr = zeros(ntime,nlay,nlat,nlon);
# const rd    = 287.05
# const g     = 9.80665
#
# function level_function_3(jk)
#   jkb = nlay+1-jk
#   delta = pp_hl[:,jkb+1,:,:]-pp_hl[:,jkb,:,:]
#   #
#   # --- cloud properties
#   #
#   zscratch          = @. pp_fl[:,jkb,:,:]/tk_fl[:,jkb,:,:]
#   @. ziwc_vr[:,jk,:,:] = ziwgkg_vr[:,jk,:,:]*zscratch/rd
#   @. ziwp_vr[:,jk,:,:] = ziwgkg_vr[:,jk,:,:]*delta/g
#   @. zlwc_vr[:,jk,:,:] = zlwgkg_vr[:,jk,:,:]*zscratch/rd
#   @. zlwp_vr[:,jk,:,:] = zlwgkg_vr[:,jk,:,:]*delta/g
#   @. cdnc_vr[:,jk,:,:] = cdnc[:,jkb,:,:]*1.e-6
# end
# for jk = 1:nlay
#   level_function_3(jk)
# end
#
#
# # 3.0 Particulate Optical Properties
# # --------------------------------
#
# cld_tau_lw_vr, cld_tau_sw_vr, cld_piz_sw_vr, cld_cg_sw_vr = clouds(ntime,nlay,nlev,nlat,nlon,laland,laglac,ktype,zlwp_vr,ziwp_vr,zlwc_vr,ziwc_vr,cdnc_vr,icldlyr)
#
#
# philat = mask_dset["lat"][:values][[lat_i]];
# laland = mask_dset["SLM"][:values][[lat_i],[lon_i]];
# laglac = mask_dset["GLAC"][:values][[lat_i],[lon_i]];
# cld_frc = dset["cld_frc"][:values][[time_i],:,[lat_i],[lon_i]];
# xm_ice = dset["q_ice"][:values][[time_i],:,[lat_i],[lon_i]];
# xm_liq = dset["q_liq"][:values][[time_i],:,[lat_i],[lon_i]];
# pp_hl = dset["pp_hl"][:values][[time_i],:,[lat_i],[lon_i]];
# pp_fl = dset["pp_fl"][:values][[time_i],:,[lat_i],[lon_i]];
# tk_fl = dset["tk_fl"][:values][[time_i],:,[lat_i],[lon_i]];
# cdnc = dset["cdnc"][:values][[time_i],:,[lat_i],[lon_i]];
# ktype = dset["ktype"][:values][[time_i],[lat_i],[lon_i]];
#
# ntime, nlay, nlat, nlon = size(tk_fl)
# nlev = nlay + 1
#
#
# # println("Preparing input data for RRTM...")
# cld_frc_vr = zeros(ntime,nlay,nlat,nlon);
# ziwgkg_vr = zeros(ntime,nlay,nlat,nlon);
# zlwgkg_vr = zeros(ntime,nlay,nlat,nlon);
# icldlyr = zeros(Int32,(ntime,nlay,nlat,nlon));
#
# function level_function_1(jk)
#   jkb = nlay+1-jk
#   for jl = 1:nlon, t=1:ntime, lat=1:nlat
#     cld_frc_vr[t,jk,lat,jl] = max(eps(Float32),cld_frc[t,jkb,lat,jl])
#     ziwgkg_vr[t,jk,lat,jl]  = xm_ice[t,jkb,lat,jl]*1000.0./cld_frc_vr[t,jk,lat,jl]
#     zlwgkg_vr[t,jk,lat,jl]  = xm_liq[t,jkb,lat,jl]*1000.0./cld_frc_vr[t,jk,lat,jl]
#   end
# end
#
# for jk = 1:nlay
#   level_function_1(jk)
# end
#
# # --- control for zero:infintesimal or negative cloud fractions
#
# function level_function_2(t,jk,lat,jl)
#   if cld_frc_vr[t,jk,lat,jl] > 2.0*eps(Float32)
#     icldlyr[t,jk,lat,jl] = 1
#   else
#     icldlyr[t,jk,lat,jl] = 0
#     ziwgkg_vr[t,jk,lat,jl] = 0.0
#     zlwgkg_vr[t,jk,lat,jl] = 0.0
#   end
# end
#
# for jk = 1:nlay, jl = 1:nlon, t=1:ntime, lat=1:nlat
#   level_function_2(t,jk,lat,jl)
# end
#
# # pm_hl_vr = zeros(ntime,nlev,nlat,nlon)
# ziwc_vr = zeros(ntime,nlay,nlat,nlon);
# ziwp_vr = zeros(ntime,nlay,nlat,nlon);
# zlwc_vr = zeros(ntime,nlay,nlat,nlon);
# zlwp_vr = zeros(ntime,nlay,nlat,nlon);
# cdnc_vr = zeros(ntime,nlay,nlat,nlon);
# const rd    = 287.05
# const g     = 9.80665
#
# function level_function_3(jk)
#   jkb = nlay+1-jk
#   delta = pp_hl[:,jkb+1,:,:]-pp_hl[:,jkb,:,:]
#   #
#   # --- cloud properties
#   #
#   zscratch          = @. pp_fl[:,jkb,:,:]/tk_fl[:,jkb,:,:]
#   @. ziwc_vr[:,jk,:,:] = ziwgkg_vr[:,jk,:,:]*zscratch/rd
#   @. ziwp_vr[:,jk,:,:] = ziwgkg_vr[:,jk,:,:]*delta/g
#   @. zlwc_vr[:,jk,:,:] = zlwgkg_vr[:,jk,:,:]*zscratch/rd
#   @. zlwp_vr[:,jk,:,:] = zlwgkg_vr[:,jk,:,:]*delta/g
#   @. cdnc_vr[:,jk,:,:] = cdnc[:,jkb,:,:]*1.e-6
# end
# for jk = 1:nlay
#   level_function_3(jk)
# end
#
#
# # 3.0 Particulate Optical Properties
# # --------------------------------
# cld_tau_lw_vr_col_col, cld_tau_sw_vr_col_col, cld_piz_sw_vr_col_col, cld_cg_sw_vr_col_col = clouds(1,nlay,nlev,1,1,laland,laglac,ktype,zlwp_vr,ziwp_vr,zlwc_vr,ziwc_vr,cdnc_vr,icldlyr)
#
# cld_tau_lw_vr[], cld_tau_sw_vr, cld_piz_sw_vr, cld_cg_sw_vr
#
# cld_tau_lw_vr_col_col[1,:,1,1,:]
#
# cld_tau_lw_vr[time_i,:,lat_i,lon_i,:]