module RRTM
using PyCall
using JLD
@pyimport xarray as xr

export radiation, radiation_pert, clouds, aerosols #,setup_RRTM
include("setup_RRTM.jl")

# package code goes here
# RRTM from ECHAM6.1 rewritten in Julia 0.6!

# if you need to debug, use:
# using ASTInterpreter2


# parallel = false

# just_these_lats = 96 # 3
# just_these_lons = 192 # 3

# the following data files are needed to run radiation()
#   T63 land mask dataset, "netcdfs/unit.24.T63GR15.nc"
#   aerosol params, "jlds/aerosol_parameters.jld" (run setup_RRTM.jl to create these)
#   LW params, "jlds/LW_parameters.jld" (run setup_RRTM.jl to create these)
#   LW params, "jlds/SW_parameters.jld" (run setup_RRTM.jl to create these)
#   cloud params, "netcdfs/ECHAM6_CldOptProps.nc"



# The ECHAM radiation transfer code assumes a maximum zenith angle of 84º degrees,
# e.g. THE SUN IS ALWAYS SHINING EVERYWHERE A LITTLE BIT. See:
#
#       For the calculation of radiative transfer, a maximum zenith angle
#       of about 84 degrees is applied in order to avoid to much overshooting
#       when the extrapolation of the radiative fluxes from night time
#       regions to daytime regions is done for time steps at which no
#       radiation calculation is performed. This translates into cosines
#       of the zenith angle > 0.1.
#
#       This approach limits the calculation of the curvature effect
#       above, and may have to be reconsidered when radiation is cleaned
#       up.
#
# So for now, I'm going to leave out the part of the SW code such that only sunlit parts of Earth are being considered, since
# right now all parts are sunlit.

# constants
const nb_lw = 16
const nb_sw = 14 # nb_sw number of shortwave bands
const api = 3.14159265358979323846
# const jpinpx = 7
const rd    = 287.05
const g     = 9.80665
const amco2 = 44.011
const amch4 = 16.043
const amo3  = 47.9982
const amn2o = 44.013
# const amc11 =137.3686
# const amc12 =120.9140
const amo2  = 31.9988
const amw   = 18.0154
const amd   = 28.970
const avo   = 6.02214e23
# const omega = .7292E-4
# const a     = 6371000.0
const cemiss = 0.996 # cemiss surface emissivity
const preflog = [6.9600e+00, 6.7600e+00, 6.5600e+00, 6.3600e+00, 6.1600e+00,5.9600e+00, 5.7600e+00, 5.5600e+00, 5.3600e+00, 5.1600e+00,4.9600e+00, 4.7600e+00, 4.5600e+00, 4.3600e+00, 4.1600e+00,3.9600e+00, 3.7600e+00, 3.5600e+00, 3.3600e+00, 3.1600e+00,2.9600e+00, 2.7600e+00, 2.5600e+00, 2.3600e+00, 2.1600e+00,1.9600e+00, 1.7600e+00, 1.5600e+00, 1.3600e+00, 1.1600e+00,9.6000e-01, 7.6000e-01, 5.6000e-01, 3.6000e-01, 1.6000e-01,-4.0000e-02,-2.4000e-01,-4.4000e-01,-6.4000e-01,-8.4000e-01,-1.0400e+00,-1.2400e+00,-1.4400e+00,-1.6400e+00,-1.8400e+00,-2.0400e+00,-2.2400e+00,-2.4400e+00,-2.6400e+00,-2.8400e+00,-3.0400e+00,-3.2400e+00,-3.4400e+00,-3.6400e+00,-3.8400e+00,-4.0400e+00,-4.2400e+00,-4.4400e+00,-4.6400e+00]
const tref = [2.9420e+02, 2.8799e+02, 2.7894e+02, 2.6925e+02, 2.5983e+02,2.5017e+02, 2.4077e+02, 2.3179e+02, 2.2306e+02, 2.1578e+02,2.1570e+02, 2.1570e+02, 2.1570e+02, 2.1706e+02, 2.1858e+02,2.2018e+02, 2.2174e+02, 2.2328e+02, 2.2479e+02, 2.2655e+02,2.2834e+02, 2.3113e+02, 2.3401e+02, 2.3703e+02, 2.4022e+02,2.4371e+02, 2.4726e+02, 2.5085e+02, 2.5457e+02, 2.5832e+02,2.6216e+02, 2.6606e+02, 2.6999e+02, 2.7340e+02, 2.7536e+02,2.7568e+02, 2.7372e+02, 2.7163e+02, 2.6955e+02, 2.6593e+02,2.6211e+02, 2.5828e+02, 2.5360e+02, 2.4854e+02, 2.4348e+02,2.3809e+02, 2.3206e+02, 2.2603e+02, 2.2000e+02, 2.1435e+02,2.0887e+02, 2.0340e+02, 1.9792e+02, 1.9290e+02, 1.8809e+02,1.8329e+02, 1.7849e+02, 1.7394e+02, 1.7212e+02]
const oneminus = 1.0 - 1.0e-06
const wavenum1 = [ 10., 350., 500., 630., 700., 820., 980.,1080.,1180.,1390.,1480.,1800.,2080.,2250.,2380.,2600., 3250., 4000., 4650., 5150., 6150., 7700.,  8050.,12850.,16000.,22650.,29000.,38000., 820.]
const wavenum2 = [350., 500., 630., 700., 820., 980.,1080.,1180.,1390.,1480.,1800.,2080.,2250.,2380.,2600.,3250., 4000., 4650., 5150., 6150., 7700., 8050., 12850.,16000.,22650.,29000.,38000.,50000.,2600.]
const delwave = wavenum2 - wavenum1
const LW_parameters = load("$(@__DIR__)/../jlds/LW_parameters.jld")
const SW_parameters = load("$(@__DIR__)/../jlds/SW_parameters.jld")
# const aer_tau_lw_vr_full = load("jlds/aerosols.jld")["aer_tau_lw_vr"]
# const cld_tau_lw_vr_full = load("jlds/clouds.jld")["cld_tau_lw_vr"]

  # pref = [1.05363e+03,8.62642e+02,7.06272e+02,5.78246e+02,4.73428e+02,3.87610e+02,3.17348e+02,2.59823e+02,2.12725e+02,1.74164e+02,1.42594e+02,1.16746e+02,9.55835e+01,7.82571e+01,6.40715e+01,5.24573e+01,4.29484e+01,3.51632e+01,2.87892e+01,2.35706e+01,1.92980e+01,1.57998e+01,1.29358e+01,1.05910e+01,8.67114e+00,7.09933e+00,5.81244e+00,4.75882e+00,3.89619e+00,3.18993e+00,2.61170e+00,2.13828e+00,1.75067e+00,1.43333e+00,1.17351e+00,9.60789e-01,7.86628e-01,6.44036e-01,5.27292e-01,4.31710e-01,3.53455e-01,2.89384e-01,2.36928e-01,1.93980e-01,1.58817e-01,1.30029e-01,1.06458e-01,8.71608e-02,7.13612e-02,5.84256e-02,4.78349e-02,3.91639e-02,3.20647e-02,2.62523e-02,2.14936e-02,1.75975e-02,1.44076e-02,1.17959e-02,9.65769e-03]
  
# helper functions
fortran_int(x) = @. Int(sign(x)*floor(abs(x)))

add_layer_dim(array) = reshape(array,(size(array)[1],1,size(array)[2:end]...))

function legtri(zsin,icp)
  # Legendre functions for a triangular truncation.
  ic = icp - 1
  zcos = sqrt(1. - zsin^2)
  jj = 2
  zf1m = sqrt(3.)
  palp = Float32[1.,zf1m*zsin]
  for jm1 = 1:icp
    jm = jm1 - 1
    zm = jm
    z2m = zm + zm
    zre1 = sqrt(z2m+3.)
    ze1 = 1./zre1
    if jm != 0
      zf2m = zf1m*zcos/sqrt(z2m)
      zf1m = zf2m*zre1

      jj += 1
      push!(palp,zf2m)
      if (jm==ic)
        continue
      end

      jj += 1
      push!(palp,zf1m*zsin)
      if (jm1==ic)
        continue
      end
    end
    jm2 = jm + 2
    for jn = jm2:ic
      zn = jn
      zn2 = zn^2
      ze2 = sqrt((4. * zn2-1.)/(zn2-zm^2))
      jj += 1
      push!(palp,ze2*(zsin*palp[jj-1]-ze1*palp[jj-2]))
      ze1 = 1./ze2
    end
  end
  palp
end

function to_tropo_index(ps,tropo,index_type)
  i = 1
  for i in 1:length(ps)
    if ps[i] > tropo
      break
    end
  end

  index_type == :trop ? i : (index_type == :strat ? i-1 : nothing)
end

# big functions

# radiation called on a full file
function radiation(input_fn::String,CO2_multiple,time_i,SW_correction=true,output_type=:flux,output_path::String = "",mask_fn::String = "$(@__DIR__)/../netcdfs/unit.24.T63GR15.nc")
  # println("Calculating offline radiation for ", input_fn, " time step ", time_i)
  #
  mask_dset = xr.open_dataset(mask_fn,decode_times=false) # (lat=collect(0:(just_these_lats-1)),lon=collect(0:(just_these_lons-1)))
  dset = xr.open_dataset(input_fn,decode_times=false)[:isel](time=collect((time_i:time_i)-1)) # (lat=collect(0:(just_these_lats-1)),lon=collect(0:(just_these_lons-1)))
  
  pp_hl = dset["pp_hl"][:values]
  pp_fl = dset["pp_fl"][:values]
  tk_hl = dset["tk_hl"][:values]
  
  aer_tau_lw_vr, aer_tau_sw_vr, aer_piz_sw_vr, aer_cg_sw_vr = aerosols(mask_dset["lat"][:values],dset["hyai"][:values],dset["hybi"][:values],pp_hl,pp_fl,tk_hl)

  #  Grid area stored from N->SLM
  output = radiation(Dict(
    :laland => mask_dset["SLM"][:values], # land
    :laglac => mask_dset["GLAC"][:values], # glacier
    :ktype => dset["ktype"][:values], # type of convection
    :pp_fl => pp_fl,
    :pp_hl => pp_hl,
    :tk_fl => dset["tk_fl"][:values],
    :tk_hl => tk_hl,
    :xm_vap => dset["q_vap"][:values],
    :xm_liq => dset["q_liq"][:values],
    :xm_ice => dset["q_ice"][:values],
    :cdnc => dset["cdnc"][:values],
    :cld_frc => dset["cld_frc"][:values],
    :xm_o3 => dset["m_o3"][:values],
    :xm_ch4 => dset["m_ch4"][:values],
    :xm_n2o => dset["m_n2o"][:values],
    :solar_constant => dset["psctm"][:values][:,1,1],
    # :cos_mu0 => dset["cos_mu0"][:values],
    :cos_mu0m => dset["cos_mu0m"][:values],
    :alb => dset["alb"][:values],
    :zi0 => dset["zi0"][:values],
    :tropo => dset["tropo"][:values],
    :aer_tau_lw_vr => aer_tau_lw_vr,
    :aer_tau_sw_vr => aer_tau_sw_vr,
    :aer_piz_sw_vr => aer_piz_sw_vr,
    :aer_cg_sw_vr => aer_cg_sw_vr
  ),CO2_multiple,SW_correction = SW_correction,output_type = output_type)
    
  # rae   = 0.1277E-2     # ratio of atmosphere to earth radius
  # zrae = rae*(rae+2)
  # cos_mu0m = rae./(sqrt.(cos_mu0.^2+zrae)-cos_mu0)
  # cos_mu0m = max.(cos_mu0m,0.1)
  # xm_sat = dset["q_sat"][:values]
  # cld_cvr = dset["cld_cvr"][:values]
  # xm_co2 = dset["m_co2"][:values]
  # rdayl = cos_mu0 .!= 0.0
  # pgeom1 = dset["pgeom1"][:values] # export?
  # alb_vis = dset["alb_vis"][:values] # alb_vis
  # alb_nir = dset["alb_nir"][:values] # alb_nir
  # alb_vis_dir = dset["alb_vis_dir"][:values] # alb_vis_dir
  # alb_nir_dir = dset["alb_nir_dir"][:values] # alb_nir_dir
  # alb_vis_dif = dset["alb_vis_dif"][:values] # alb_vis_dif
  # alb_nir_dif = dset["alb_nir_dif"][:values] # alb_nir_dif  
  dset = dset[:drop](intersect(["hybi","hyai","hybm","hyam","pp_sfc","psctm","alb","cos_mu0","cos_mu0m","ktype","tod","tk_sfc","dom","pp_hl","tk_hl","q_vap","tk_fl","cld_frc","cdnc","m_o3","m_ch4","pp_fl","q_liq","m_n2o","q_ice","mlev","ilev","flx_lw_dn_surf","flx_lw_dn_clr_surf","flx_lw_up_toa","flx_lw_up_clr_toa","flx_lw_up_surf","flx_lw_up_clr_surf","flx_sw_dn_toa","flx_sw_dn_surf","flx_sw_dn_clr_surf","flx_sw_up_toa","flx_sw_up_clr_toa","flx_sw_up_surf","flx_sw_up_clr_surf","zi0","tropo"],dset[:keys]()))
  dset = dset[:assign](tropopause = (("time","lat","lon"),output[:tropopause]))
  if output_type == :profile
    dset = dset[:assign](LW_up = (("time","ilev","lat","lon"),output[:LW_up]))
    dset = dset[:assign](LW_dn = (("time","ilev","lat","lon"),output[:LW_dn]))
    dset = dset[:assign](SW_up = (("time","ilev","lat","lon"),output[:SW_up]))
    dset = dset[:assign](SW_dn = (("time","ilev","lat","lon"),output[:SW_dn]))
    dset = dset[:assign](LW_up_clr = (("time","ilev","lat","lon"),output[:LW_up_clr]))
    dset = dset[:assign](LW_dn_clr = (("time","ilev","lat","lon"),output[:LW_dn_clr]))
    dset = dset[:assign](SW_up_clr = (("time","ilev","lat","lon"),output[:SW_up_clr]))
    dset = dset[:assign](SW_dn_clr = (("time","ilev","lat","lon"),output[:SW_dn_clr]))
  elseif output_type == :flux
    dset = dset[:assign](LW_up_toa = (("time","lat","lon"),output[:LW_up_toa]))
    dset = dset[:assign](LW_up_clr_toa = (("time","lat","lon"),output[:LW_up_clr_toa]))
    dset = dset[:assign](SW_up_toa = (("time","lat","lon"),output[:SW_up_toa]))
    dset = dset[:assign](SW_dn_toa = (("time","lat","lon"),output[:SW_dn_toa]))
    dset = dset[:assign](SW_up_clr_toa = (("time","lat","lon"),output[:SW_up_clr_toa]))
    # dset = dset[:assign](flx_lw_up_surf = (("time","lat","lon"),flx_lw_up_surf))
    # dset = dset[:assign](flx_lw_dn_surf = (("time","lat","lon"),flx_lw_dn_surf))
    # dset = dset[:assign](flx_lw_up_clr_surf = (("time","lat","lon"),flx_lw_up_clr_surf))
    # dset = dset[:assign](flx_lw_dn_clr_surf = (("time","lat","lon"),flx_lw_dn_clr_surf))
    # dset = dset[:assign](flx_sw_up_surf = (("time","lat","lon"),flx_sw_up_surf))
    # dset = dset[:assign](flx_sw_dn_surf = (("time","lat","lon"),flx_sw_dn_surf))
    # dset = dset[:assign](flx_sw_up_clr_surf = (("time","lat","lon"),flx_sw_up_clr_surf))
    # dset = dset[:assign](flx_sw_dn_clr_surf = (("time","lat","lon"),flx_sw_dn_clr_surf))
    dset = dset[:assign](LW_up_trop = (("time","lat","lon"),output[:LW_up_trop]))
    dset = dset[:assign](LW_up_clr_trop = (("time","lat","lon"),output[:LW_up_clr_trop]))
    dset = dset[:assign](LW_dn_trop = (("time","lat","lon"),output[:LW_dn_trop]))
    dset = dset[:assign](LW_dn_clr_trop = (("time","lat","lon"),output[:LW_dn_clr_trop]))
    dset = dset[:assign](SW_up_trop = (("time","lat","lon"),output[:SW_up_trop]))
    dset = dset[:assign](SW_up_clr_trop = (("time","lat","lon"),output[:SW_up_clr_trop]))
    dset = dset[:assign](SW_dn_trop = (("time","lat","lon"),output[:SW_dn_trop]))
    dset = dset[:assign](SW_dn_clr_trop = (("time","lat","lon"),output[:SW_dn_clr_trop]))
  end

  if isempty(output_path)
    output_fn = replace(input_fn,".nc","_offline_radiation_$(lpad(time_i,3,0))_$(CO2_multiple)x_$(output_type).nc")
  else
    output_fn = output_path * "/" * replace(split(input_fn,"/")[end],".nc","_offline_radiation_$(lpad(time_i,3,0))_$(CO2_multiple)x_$(output_type).nc")
  end
  # dset[:to_netcdf]("netcdfs/tmp/" * output_fn)
  dset[:to_netcdf](output_fn)
  println("Offline radiation completed and written to ", output_fn)
  nothing
end

function radiation_pert(input_fn1::String,input_fn2::String,pert::Symbol,CO2_multiple,time_i,output_fn,output_type,mask_fn::String = "$(@__DIR__)/../netcdfs/unit.24.T63GR15.nc")
  SW_correction = true
  # println("Calculating offline radiation for ", input_fn, " time step ", time_i)
  #
  mask_dset = xr.open_dataset(mask_fn,decode_times=false) # (lat=collect(0:(just_these_lats-1)),lon=collect(0:(just_these_lons-1)))
  dset1 = xr.open_dataset(input_fn1,decode_times=false)[:isel](time=collect((time_i:time_i)-1)) #
  dset2 = xr.open_dataset(input_fn2,decode_times=false)[:isel](time=collect((time_i:time_i)-1)) # (lat=collect(0:(just_these_lats-1)),lon=collect(0:(just_these_lons-1)))
  
  aersols(mask_dset,dset) = aerosols(mask_dset["lat"][:values],dset["hyai"][:values],dset["hybi"][:values],dset["pp_hl"][:values],dset["pp_fl"][:values],dset["tk_hl"][:values])
  aer_tau_lw_vr, aer_tau_sw_vr, aer_piz_sw_vr, aer_cg_sw_vr = aersols(mask_dset,(pert == :aer) ? dset2 : dset1)

  input = Dict(
    :laland => mask_dset["SLM"][:values], # land
    :laglac => mask_dset["GLAC"][:values], # glacier
    :ktype => (pert == :cl ? dset2 : dset1)["ktype"][:values], # type of convection
    :cdnc => (pert == :cl ? dset2 : dset1)["cdnc"][:values],
    :cld_frc => (pert == :cl ? dset2 : dset1)["cld_frc"][:values],
    :xm_liq => (pert == :cl ? dset2 : dset1)["q_liq"][:values],
    :xm_ice => (pert == :cl ? dset2 : dset1)["q_ice"][:values],
    :alb => (pert == :alb ? dset2 : dset1)["alb"][:values],
    :xm_vap => (pert == :q ? dset2 : dset1)["q_vap"][:values],
    :pp_fl => (pert == :p ? dset2 : dset1)["pp_fl"][:values],
    :pp_hl => (pert == :p ? dset2 : dset1)["pp_hl"][:values],
    :tk_fl => dset1["tk_fl"][:values],
    :tk_hl => dset1["tk_hl"][:values],
    :xm_o3 => dset1["m_o3"][:values],
    :xm_ch4 => dset1["m_ch4"][:values],
    :xm_n2o => dset1["m_n2o"][:values],
    :solar_constant => dset1["psctm"][:values][:,1,1],
    # :cos_mu0 => dset["cos_mu0"][:values],
    :cos_mu0m => dset1["cos_mu0m"][:values],
    :zi0 => dset1["zi0"][:values],
    :tropo => dset1["tropo"][:values],
    :aer_tau_lw_vr => aer_tau_lw_vr,
    :aer_tau_sw_vr => aer_tau_sw_vr,
    :aer_piz_sw_vr => aer_piz_sw_vr,
    :aer_cg_sw_vr => aer_cg_sw_vr
  )
  
  if pert in [:T_strat,:Tcol_trop,:TLR_trop,:q_strat]    
    ntime,nlat,nlon = size(input[:alb])
    
    if pert in [:T_strat,:q_strat]
      for t in 1:ntime, lat in 1:nlat, lon in 1:nlon
        tropo = dset1[:tropo][:values][t,lat,lon]
      
        pp_fl = dset1[:pp_fl][:values][t,:,lat,lon]
        rng_fl = 1:(to_tropo_index(pp_fl,tropo,:strat))
      
        if pert == :T_strat
          pp_hl = dset1[:pp_hl][:values][t,:,lat,lon]
          rng_hl = 1:(to_tropo_index(pp_hl,tropo,:strat))
          
          input[:tk_hl][t,rng_hl,lat,lon] = dset2["tk_hl"][:values][t,rng_hl,lat,lon]
          input[:tk_fl][t,rng_fl,lat,lon] = dset2["tk_fl"][:values][t,rng_fl,lat,lon]
        elseif pert == :q_strat
          input[:xm_vap][t,rng_fl,lat,lon] = dset2["q_vap"][:values][t,rng_fl,lat,lon]
        end
      end
    else
      nhl = length(dset1[:pp_hl][:values][1,:,1,1])
      nfl = length(dset1[:pp_fl][:values][1,:,1,1])
      for t in 1:ntime, lat in 1:nlat, lon in 1:nlon
        tropo = dset1[:tropo][:values][t,lat,lon]
        
        pp_hl = dset1[:pp_hl][:values][t,:,lat,lon]
        rng_hl = to_tropo_index(pp_hl,tropo,:trop):nhl
      
        pp_fl = dset1[:pp_fl][:values][t,:,lat,lon]
        rng_fl = to_tropo_index(pp_fl,tropo,:trop):nfl
        
        ΔTs = dset2[:tk_hl][:values][t,end,lat,lon] - dset1[:tk_hl][:values][t,end,lat,lon]

        if pert == :Tcol_trop
          input[:tk_hl][t,rng_hl,lat,lon] += ΔTs
          input[:tk_fl][t,rng_fl,lat,lon] += ΔTs
        elseif pert == :TRH_trop
          input[:tk_hl][t,rng_hl,lat,lon] += (dset2["tk_hl"][:values][t,rng_hl,lat,lon] - (input[:tk_hl][t,rng_hl,lat,lon] + ΔTs))
          input[:tk_fl][t,rng_fl,lat,lon] += (dset2["tk_fl"][:values][t,rng_fl,lat,lon] - (input[:tk_fl][t,rng_fl,lat,lon] + ΔTs))
        end
      end
    end
  end
  
  #  Grid area stored from N->SLM
  output = radiation(input,CO2_multiple,SW_correction = SW_correction,output_type = output_type)
    
  # rae   = 0.1277E-2     # ratio of atmosphere to earth radius
  # zrae = rae*(rae+2)
  # cos_mu0m = rae./(sqrt.(cos_mu0.^2+zrae)-cos_mu0)
  # cos_mu0m = max.(cos_mu0m,0.1)
  # xm_sat = dset["q_sat"][:values]
  # cld_cvr = dset["cld_cvr"][:values]
  # xm_co2 = dset["m_co2"][:values]
  # rdayl = cos_mu0 .!= 0.0
  # pgeom1 = dset["pgeom1"][:values] # export?
  # alb_vis = dset["alb_vis"][:values] # alb_vis
  # alb_nir = dset["alb_nir"][:values] # alb_nir
  # alb_vis_dir = dset["alb_vis_dir"][:values] # alb_vis_dir
  # alb_nir_dir = dset["alb_nir_dir"][:values] # alb_nir_dir
  # alb_vis_dif = dset["alb_vis_dif"][:values] # alb_vis_dif
  # alb_nir_dif = dset["alb_nir_dif"][:values] # alb_nir_dif  
  dset1 = dset1[:drop](intersect(["hybi","hyai","hybm","hyam","pp_sfc","psctm","alb","cos_mu0","cos_mu0m","ktype","tod","tk_sfc","dom","pp_hl","tk_hl","q_vap","tk_fl","cld_frc","cdnc","m_o3","m_ch4","pp_fl","q_liq","m_n2o","q_ice","mlev","ilev","flx_lw_dn_surf","flx_lw_dn_clr_surf","flx_lw_up_toa","flx_lw_up_clr_toa","flx_lw_up_surf","flx_lw_up_clr_surf","flx_sw_dn_toa","flx_sw_dn_surf","flx_sw_dn_clr_surf","flx_sw_up_toa","flx_sw_up_clr_toa","flx_sw_up_surf","flx_sw_up_clr_surf","zi0","tropo"],dset1[:keys]()))
  dset1 = dset1[:assign](tropopause = (("time","lat","lon"),output[:tropopause]))
  if output_type == :profile
    dset1 = dset1[:assign](LW_up = (("time","ilev","lat","lon"),output[:LW_up]))
    dset1 = dset1[:assign](LW_dn = (("time","ilev","lat","lon"),output[:LW_dn]))
    dset1 = dset1[:assign](SW_up = (("time","ilev","lat","lon"),output[:SW_up]))
    dset1 = dset1[:assign](SW_dn = (("time","ilev","lat","lon"),output[:SW_dn]))
    dset1 = dset1[:assign](LW_up_clr = (("time","ilev","lat","lon"),output[:LW_up_clr]))
    dset1 = dset1[:assign](LW_dn_clr = (("time","ilev","lat","lon"),output[:LW_dn_clr]))
    dset1 = dset1[:assign](SW_up_clr = (("time","ilev","lat","lon"),output[:SW_up_clr]))
    dset1 = dset1[:assign](SW_dn_clr = (("time","ilev","lat","lon"),output[:SW_dn_clr]))
  elseif output_type == :flux
    dset1 = dset1[:assign](LW_up_toa = (("time","lat","lon"),output[:LW_up_toa]))
    dset1 = dset1[:assign](LW_up_clr_toa = (("time","lat","lon"),output[:LW_up_clr_toa]))
    dset1 = dset1[:assign](SW_up_toa = (("time","lat","lon"),output[:SW_up_toa]))
    dset1 = dset1[:assign](SW_dn_toa = (("time","lat","lon"),output[:SW_dn_toa]))
    dset1 = dset1[:assign](SW_up_clr_toa = (("time","lat","lon"),output[:SW_up_clr_toa]))
    # dset1 = dset1[:assign](flx_lw_up_surf = (("time","lat","lon"),flx_lw_up_surf))
    # dset1 = dset1[:assign](flx_lw_dn_surf = (("time","lat","lon"),flx_lw_dn_surf))
    # dset1 = dset1[:assign](flx_lw_up_clr_surf = (("time","lat","lon"),flx_lw_up_clr_surf))
    # dset1 = dset1[:assign](flx_lw_dn_clr_surf = (("time","lat","lon"),flx_lw_dn_clr_surf))
    # dset1 = dset1[:assign](flx_sw_up_surf = (("time","lat","lon"),flx_sw_up_surf))
    # dset1 = dset1[:assign](flx_sw_dn_surf = (("time","lat","lon"),flx_sw_dn_surf))
    # dset1 = dset1[:assign](flx_sw_up_clr_surf = (("time","lat","lon"),flx_sw_up_clr_surf))
    # dset1 = dset1[:assign](flx_sw_dn_clr_surf = (("time","lat","lon"),flx_sw_dn_clr_surf))
    dset1 = dset1[:assign](LW_up_trop = (("time","lat","lon"),output[:LW_up_trop]))
    dset1 = dset1[:assign](LW_up_clr_trop = (("time","lat","lon"),output[:LW_up_clr_trop]))
    dset1 = dset1[:assign](LW_dn_trop = (("time","lat","lon"),output[:LW_dn_trop]))
    dset1 = dset1[:assign](LW_dn_clr_trop = (("time","lat","lon"),output[:LW_dn_clr_trop]))
    dset1 = dset1[:assign](SW_up_trop = (("time","lat","lon"),output[:SW_up_trop]))
    dset1 = dset1[:assign](SW_up_clr_trop = (("time","lat","lon"),output[:SW_up_clr_trop]))
    dset1 = dset1[:assign](SW_dn_trop = (("time","lat","lon"),output[:SW_dn_trop]))
    dset1 = dset1[:assign](SW_dn_clr_trop = (("time","lat","lon"),output[:SW_dn_clr_trop]))
  end

  # if isempty(output_path)
  #   output_fn = replace(input_fn,".nc","_offline_radiation_$(lpad(time_i,3,0))_$(CO2_multiple)x_$(output_type).nc")
  # else
  #   output_fn = output_path * "/" * replace(split(input_fn,"/")[end],".nc","_offline_radiation_$(lpad(time_i,3,0))_$(CO2_multiple)x_$(output_type).nc")
  # end
  # dset1[:to_netcdf]("netcdfs/tmp/" * output_fn)
  dset1[:to_netcdf](output_fn)
  println("Offline radiation pert completed and written to ", output_fn)
  nothing
end

# radiation called directly on inputs
function radiation(input,CO2_multiple;SW_correction=true,output_type=:flux)
  input = copy(input)
  input[:xm_co2] = CO2_multiple * fill(0.000284725 * amco2 / amd, size(input[:xm_o3])) # pco2 = 0.000284725
  
  # println(input[:aer_tau_lw_vr][:,1])
  dims = size(input[:tk_fl])
  if length(dims) == 1
    ntime = nlat = nlon = 1
    nlay = dims[1]
    for (k,v) in input
      v_dims = size(v)
      if length(v_dims) == 2 # aerosols
        input[k] = reshape(isa(v,Array) ? v : [v],(1,size(v)[1],1,1,size(v)[2]))        
      else
        if k in [:laland,:laglac]
          input[k] = reshape(isa(v,Array) ? v : [v],(size(v)...,1,1))
        else
          input[k] = reshape(isa(v,Array) ? v : [v],(1,size(v)...,1,1))
        end
      end
    end
  else
    ntime, nlay, nlat, nlon = dims
  end
  nlev = nlay + 1

  zsemiss = cemiss # fill(cemiss,ntime,nlat,nlon,nb_lw)
  xm_cfc = zeros(ntime,nlay,nlat,nlon,2) # export? # icfc == 0 cfcs are 0
  # xm_o2 = dset["m_o2"][:values] # export?
  xm_o2 = fill(0.23135895f0, (ntime, nlay, nlat, nlon)) # 0.23135894536972046 (instead of 0.23135895f0)
  # println("Input datasets loaded.")
  

  # println("Preparing input data for RRTM...")
  cld_frc_vr = zeros(ntime,nlay,nlat,nlon)
  ziwgkg_vr = zeros(ntime,nlay,nlat,nlon)
  zlwgkg_vr = zeros(ntime,nlay,nlat,nlon)
  icldlyr = zeros(Int32,(ntime,nlay,nlat,nlon))

  function level_function_1(jk)
    jkb = nlay+1-jk
    for jl = 1:nlon, t=1:ntime, lat=1:nlat
      cld_frc_vr[t,jk,lat,jl] = max(eps(Float32),input[:cld_frc][t,jkb,lat,jl])
      ziwgkg_vr[t,jk,lat,jl]  = input[:xm_ice][t,jkb,lat,jl]*1000.0./cld_frc_vr[t,jk,lat,jl]
      zlwgkg_vr[t,jk,lat,jl]  = input[:xm_liq][t,jkb,lat,jl]*1000.0./cld_frc_vr[t,jk,lat,jl]
    end
  end

  for jk = 1:nlay
    level_function_1(jk)
  end

  # --- control for zero:infintesimal or negative cloud fractions

  function level_function_2(t,jk,lat,jl)
    if cld_frc_vr[t,jk,lat,jl] > 2.0*eps(Float32)
      icldlyr[t,jk,lat,jl] = 1
    else
      icldlyr[t,jk,lat,jl] = 0
      ziwgkg_vr[t,jk,lat,jl] = 0.0
      zlwgkg_vr[t,jk,lat,jl] = 0.0
    end
  end

  for jk = 1:nlay, jl = 1:nlon, t=1:ntime, lat=1:nlat
    level_function_2(t,jk,lat,jl)
  end

  # pm_hl_vr = zeros(ntime,nlev,nlat,nlon)
  pm_fl_vr = zeros(ntime,nlay,nlat,nlon)
  tk_hl_vr = zeros(ntime,nlev,nlat,nlon)
  tk_fl_vr = zeros(ntime,nlay,nlat,nlon)
  pm_sfc = zeros(ntime,nlat,nlon)
  ziwc_vr = zeros(ntime,nlay,nlat,nlon)
  ziwp_vr = zeros(ntime,nlay,nlat,nlon)
  zlwc_vr = zeros(ntime,nlay,nlat,nlon)
  zlwp_vr = zeros(ntime,nlay,nlat,nlon)
  cdnc_vr = zeros(ntime,nlay,nlat,nlon)
  col_dry_vr = zeros(ntime,nlay,nlat,nlon)
  zoz_vr = zeros(ntime,nlay,nlat,nlon)

  # pm_hl_vr[:,nlay+1,:,:] = 0.01*pp_hl[:,1,:,:]
  tk_hl_vr[:,nlay+1,:,:] = input[:tk_hl][:,1,:,:]
  pm_sfc = 0.01*input[:pp_hl][:,end,:,:]

  wkl_vr = zeros(7,ntime,nlay,nlat,nlon)
  wx_vr = zeros(4,ntime,nlay,nlat,nlon)

  function level_function_3(jk)
    jkb = nlay+1-jk
    delta = input[:pp_hl][:,jkb+1,:,:]-input[:pp_hl][:,jkb,:,:]
    #
    # --- thermodynamic arrays
    #
    # @. pm_hl_vr[:,jk,:,:] = 0.01*pp_hl[:,jkb+1,:,:]
    pm_fl_vr[:,jk,:,:] = 0.01*input[:pp_fl][:,jkb,:,:]
    tk_hl_vr[:,jk,:,:] = input[:tk_hl][:,jkb+1,:,:]
    tk_fl_vr[:,jk,:,:] = input[:tk_fl][:,jkb,:,:]
    #
    # --- cloud properties
    #
    zscratch          = @. input[:pp_fl][:,jkb,:,:]/input[:tk_fl][:,jkb,:,:]
    @. ziwc_vr[:,jk,:,:] = ziwgkg_vr[:,jk,:,:]*zscratch/rd
    @. ziwp_vr[:,jk,:,:] = ziwgkg_vr[:,jk,:,:]*delta/g
    @. zlwc_vr[:,jk,:,:] = zlwgkg_vr[:,jk,:,:]*zscratch/rd
    @. zlwp_vr[:,jk,:,:] = zlwgkg_vr[:,jk,:,:]*delta/g
    @. cdnc_vr[:,jk,:,:] = input[:cdnc][:,jkb,:,:]*1.e-6
    #
    # --- radiatively active gases
    #
    @. zoz_vr[:,jk,:,:]     = input[:xm_o3][:,jkb,:,:]*delta*46.6968/g
    @. wkl_vr[1,:,jk,:,:]   = input[:xm_vap][:,jkb,:,:]*amd/amw
    @. wkl_vr[2,:,jk,:,:]   = input[:xm_co2][:,jkb,:,:]*amd/amco2
    @. wkl_vr[3,:,jk,:,:]   = input[:xm_o3][:,jkb,:,:] *amd/amo3
    @. wkl_vr[4,:,jk,:,:]   = input[:xm_n2o][:,jkb,:,:]*amd/amn2o
    @. wkl_vr[6,:,jk,:,:]   = input[:xm_ch4][:,jkb,:,:]*amd/amch4
    @. wkl_vr[7,:,jk,:,:]   = xm_o2[:,jkb,:,:]*amd/amo2
    amm                  = (1.0-wkl_vr[1,:,jk,:,:])*amd + wkl_vr[1,:,jk,:,:]*amw
    col_dry_vr[:,jk,:,:] .= (0.01*delta)*10.0*avo./g./amm./ (1.0+wkl_vr[1,:,jk,:,:])
    #
    # --- alternate treatment for cfcs
    #
    # wx_vr[jl,:,jk] = 0.0
    # wx_vr[jl,2,jk] = col_dry_vr[jl,jk]*xm_cfc[jl,jkb,1]*1.e-20
    # wx_vr[jl,3,jk] = col_dry_vr[jl,jk]*xm_cfc[jl,jkb,2]*1.e-20
  end
  for jk = 1:nlay
    level_function_3(jk)
  end


  for jp = 1:7
     @. wkl_vr[jp,:,:,:,:] = col_dry_vr*wkl_vr[jp,:,:,:,:]
  end

  # 3.0 Particulate Optical Properties
  # --------------------------------
  cld_tau_lw_vr, cld_tau_sw_vr, cld_piz_sw_vr, cld_cg_sw_vr = clouds(ntime,nlay,nlev,nlat,nlon,input[:laland],input[:laglac],input[:ktype],zlwp_vr,ziwp_vr,zlwc_vr,ziwc_vr,cdnc_vr,icldlyr)

  flx_lw_up_vr = Array{Float64}(ntime,nlev,nlat,nlon)
  flx_lw_dn_vr = Array{Float64}(ntime,nlev,nlat,nlon)
  flx_lw_up_clr_vr = Array{Float64}(ntime,nlev,nlat,nlon)
  flx_lw_dn_clr_vr = Array{Float64}(ntime,nlev,nlat,nlon)
  flx_sw_up = Array{Float64}(ntime,nlev,nlat,nlon)
  flx_sw_dn = Array{Float64}(ntime,nlev,nlat,nlon)
  flx_sw_up_clr = Array{Float64}(ntime,nlev,nlat,nlon)
  flx_sw_dn_clr = Array{Float64}(ntime,nlev,nlat,nlon)
  tropopause = Array{Float64}(ntime,nlat,nlon)
  
  if output_type == :flux
    flx_lw_up_trop = Array{Float64}(ntime,nlat,nlon)
    flx_lw_dn_trop = Array{Float64}(ntime,nlat,nlon)
    flx_lw_up_clr_trop = Array{Float64}(ntime,nlat,nlon)
    flx_lw_dn_clr_trop = Array{Float64}(ntime,nlat,nlon)
    flx_sw_up_trop = Array{Float64}(ntime,nlat,nlon)
    flx_sw_dn_trop = Array{Float64}(ntime,nlat,nlon)
    flx_sw_up_clr_trop = Array{Float64}(ntime,nlat,nlon)
    flx_sw_dn_clr_trop = Array{Float64}(ntime,nlat,nlon)
  end
  
  for jlat = 1:nlat, t in 1:ntime, jlon in 1:nlon
    flx_uplw_vr_col,flx_dnlw_vr_col,flx_uplw_clr_vr_col,flx_dnlw_clr_vr_col,flxd_sw_col,flxu_sw_col,flxd_sw_clr_col,flxu_sw_clr_col = radiation_by_column(nlay,nlev,col_dry_vr[t,:,jlat,jlon],wkl_vr[:,t,:,jlat,jlon],wx_vr[:,t,:,jlat,jlon],cld_frc_vr[t,:,jlat,jlon],input[:tk_hl][t,end,jlat,jlon],tk_hl_vr[t,:,jlat,jlon],tk_fl_vr[t,:,jlat,jlon],pm_sfc[t,jlat,jlon],pm_fl_vr[t,:,jlat,jlon],zsemiss,input[:solar_constant][t],input[:zi0][t,jlat,jlon],input[:cos_mu0m][t,jlat,jlon],input[:alb][t,jlat,jlon],input[:aer_tau_lw_vr][t,:,jlat,jlon,:],input[:aer_tau_sw_vr][t,:,jlat,jlon,:],input[:aer_piz_sw_vr][t,:,jlat,jlon,:],input[:aer_cg_sw_vr][t,:,jlat,jlon,:],cld_tau_lw_vr[t,:,jlat,jlon,:],cld_tau_sw_vr[t,:,jlat,jlon,:],cld_piz_sw_vr[t,:,jlat,jlon,:],cld_cg_sw_vr[t,:,jlat,jlon,:])
    
    flx_lw_up_vr[t,:,jlat,jlon] = flx_uplw_vr_col
    flx_lw_dn_vr[t,:,jlat,jlon] = flx_dnlw_vr_col
    flx_lw_up_clr_vr[t,:,jlat,jlon] = flx_uplw_clr_vr_col
    flx_lw_dn_clr_vr[t,:,jlat,jlon] = flx_dnlw_clr_vr_col
    flx_sw_up[t,:,jlat,jlon] = flxu_sw_col
    flx_sw_dn[t,:,jlat,jlon] = flxd_sw_col
    flx_sw_up_clr[t,:,jlat,jlon] = flxu_sw_clr_col
    flx_sw_dn_clr[t,:,jlat,jlon] = flxd_sw_clr_col
    
    for i in 1:length(input[:pp_hl])
      if input[:pp_hl][t,i,jlat,jlon] > input[:tropo][t,jlat,jlon]
        break
      end
    end
    
    tropo_i_frac = (input[:tropo][t,jlat,jlon]-input[:pp_hl][t,i-1,jlat,jlon])/(input[:pp_hl][t,i,jlat,jlon]-input[:pp_hl][t,i-1,jlat,jlon])
    tropopause[t,jlat,jlon] = (i-1) + tropo_i_frac
    
    if output_type == :flux
      flx_lw_up_trop[t,jlat,jlon] = flx_uplw_vr_col[end:-1:1][i-1] + (flx_uplw_vr_col[end:-1:1][i]-flx_uplw_vr_col[end:-1:1][i-1])*tropo_i_frac
      flx_lw_dn_trop[t,jlat,jlon] = flx_dnlw_vr_col[end:-1:1][i-1] + (flx_dnlw_vr_col[end:-1:1][i]-flx_dnlw_vr_col[end:-1:1][i-1])*tropo_i_frac
      flx_lw_up_clr_trop[t,jlat,jlon] = flx_uplw_clr_vr_col[end:-1:1][i-1] + (flx_uplw_clr_vr_col[end:-1:1][i]-flx_uplw_clr_vr_col[end:-1:1][i-1])*tropo_i_frac
      flx_lw_dn_clr_trop[t,jlat,jlon] = flx_dnlw_clr_vr_col[end:-1:1][i-1] + (flx_dnlw_clr_vr_col[end:-1:1][i]-flx_dnlw_clr_vr_col[end:-1:1][i-1])*tropo_i_frac
      flx_sw_up_trop[t,jlat,jlon] = flxu_sw_col[i-1] + (flxu_sw_col[i]-flxu_sw_col[i-1])*tropo_i_frac
      flx_sw_dn_trop[t,jlat,jlon] = flxd_sw_col[i-1] + (flxd_sw_col[i]-flxd_sw_col[i-1])*tropo_i_frac
      flx_sw_up_clr_trop[t,jlat,jlon] = flxu_sw_clr_col[i-1] + (flxu_sw_clr_col[i]-flxu_sw_clr_col[i-1])*tropo_i_frac
      flx_sw_dn_clr_trop[t,jlat,jlon] = flxd_sw_clr_col[i-1] + (flxd_sw_clr_col[i]-flxd_sw_clr_col[i-1])*tropo_i_frac
    end
  end

  # if :flx_lw_up_toa in keys(dset)

  # mo_srtm_config.f90:246

  # we need to add a correction, because ECHAM calculates its radiation assuming that faint sunlight is on in the nightside; a correction for diagnostic purposes shifts it over a little and turns off that light.
    # from radheat.f90:197
      # zflxs(jl,1)=pi0(jl)*ptrsol(jl,1)
      # pi0 is zi0
      # from physc.f90:625
        # zi0(jl)=flx_ratio_cur*solc*amu0_x(jl,krow)*rdayl_x(jl,krow)
        # flx_ratio_cur is 
        # from mo_radiation.f90:163
          # CASE (3) ->  mo_radiation_parameters.f90:73 INTEGER :: isolrad     =  3      !< mode of solar constant calculation
          #   solc = SUM(ssi_amip)
        # amu0_x is 
        # rdayl_x is
      # ptrsol is trsol
      # from radiation.f90:588
        # trsol(1:kproma,1:klevp1) = flx_dnsw(1:kproma,1:klevp1) / SPREAD(y1(1:kproma),2,klevp1)
        # from radiation.f90:586
          # y1(1:kproma) = (psctm*cos_mu0m(1:kproma))
          # from radiation.f90:243
            # psctm = flx_ratio_rad*solcm
              # flx_ratio_rad
              # from mo_radiation.f90:163
                # CASE (3) ->  mo_radiation_parameters.f90:73 INTEGER :: isolrad     =  3      !< mode of solar constant calculation
                #   solc = SUM(ssi_amip)
                #   #   REAL(wp), PARAMETER :: ssi_amip(14) =  (/ & !< solar flux (W/m2) in 14 SW bands for
                #                        ! AMIP-type CMIP5 simulation (average from 1979-1988)
                # & 11.95053_wp, 20.14766_wp, 23.40394_wp, 22.09458_wp, 55.41401_wp,  &
                # & 102.5134_wp, 24.69814_wp, 347.5362_wp, 217.2925_wp, 343.4221_wp,  &
                # & 129.403_wp, 47.14264_wp, 3.172126_wp, 13.18075_wp /)
                # 
                #  ssi_amip = [11.95053, 20.14766, 23.40394, 22.09458, 55.41401, 102.5134, 24.69814, 347.5362, 217.2925, 343.4221, 129.403, 47.14264, 3.172126, 13.18075]
  # fact = SW_correction ? input[:cos_mu0] ./ input[:cos_mu0m] : ones(input[:cos_mu0])
  fact = reshape(SW_correction ? input[:zi0][:,:,:] ./ (input[:solar_constant].*input[:cos_mu0m][:,:,:]) : ones(input[:cos_mu0m]),(ntime,1,nlat,nlon))
  flx_sw_up = fact .* flx_sw_up
  flx_sw_dn = fact .* flx_sw_dn
  flx_sw_up_clr = fact .* flx_sw_up_clr
  flx_sw_dn_clr = fact .* flx_sw_dn_clr
  
  if output_type == :profile
    i = (length(dims) == 1 ? 1 : :)
    Dict(
      :LW_up => .- flx_lw_up_vr[i,:,i,i],
      :LW_dn => flx_lw_dn_vr[i,:,i,i],
      :SW_up => .- flx_sw_up[i,end:-1:1,i,i],
      :SW_dn => flx_sw_dn[i,end:-1:1,i,i],
      :LW_up_clr => .- flx_lw_up_clr_vr[i,:,i,i],
      :LW_dn_clr => flx_lw_dn_clr_vr[i,:,i,i],
      :SW_up_clr => .- flx_sw_up_clr[i,end:-1:1,i,i],
      :SW_dn_clr => flx_sw_dn_clr[i,end:-1:1,i,i],
      :tropopause => tropopause[i,i,i]
    )      
  elseif output_type == :flux        
    # # surf
    # flx_lw_up_surf = flx_lw_up_vr[:,1,:,:]
    # flx_lw_dn_surf = flx_lw_dn_vr[:,1,:,:]
    # flx_lw_up_clr_surf = flx_lw_up_clr_vr[:,1,:,:]
    # flx_lw_dn_clr_surf = flx_lw_dn_clr_vr[:,1,:,:]
    # flx_sw_up_surf = flx_sw_up[:,end,:,:]
    # flx_sw_dn_surf = flx_sw_dn[:,end,:,:]
    # flx_sw_up_clr_surf = flx_sw_up_clr[:,end,:,:]
    # flx_sw_dn_clr_surf = flx_sw_dn_clr[:,end,:,:]
    # flx_sw_up_surf = .- fact .* flx_sw_up_surf
    # flx_sw_dn_surf = fact .* flx_sw_dn_surf
    # flx_sw_up_clr_surf = .- fact .* flx_sw_up_clr_surf
    # flx_sw_dn_clr_surf = fact .* flx_sw_dn_clr_surf
    
    fact = squeeze(fact,2)
    flx_sw_up_trop = .- fact .* flx_sw_up_trop
    flx_sw_dn_trop = fact .* flx_sw_dn_trop
    flx_sw_up_clr_trop = .- fact .* flx_sw_up_clr_trop
    flx_sw_dn_clr_trop = fact .* flx_sw_dn_clr_trop
    
    Dict(
      :LW_up_toa => .- flx_lw_up_vr[:,end,:,:],
      :LW_up_clr_toa => .- flx_lw_up_clr_vr[:,end,:,:],
      :SW_up_toa => .- flx_sw_up[:,1,:,:],
      :SW_dn_toa => flx_sw_dn[:,1,:,:],
      :SW_up_clr_toa => .- flx_sw_up_clr[:,1,:,:],
      :LW_up_toa => .- flx_lw_up_vr[:,end,:,:],
      :LW_up_clr_toa => .- flx_lw_up_clr_vr[:,end,:,:],
      :SW_up_toa => .- flx_sw_up[:,1,:,:],
      :SW_dn_toa => flx_sw_dn[:,1,:,:],
      :SW_up_clr_toa => .- flx_sw_up_clr[:,1,:,:],
      :LW_up_trop => -flx_lw_up_trop,
      :LW_dn_trop => flx_lw_dn_trop,
      :LW_up_clr_trop => -flx_lw_up_clr_trop,
      :LW_dn_clr_trop => flx_lw_dn_clr_trop,
      :SW_up_trop => flx_sw_up_trop,
      :SW_dn_trop => flx_sw_dn_trop,
      :SW_up_clr_trop => flx_sw_up_clr_trop,
      :SW_dn_clr_trop => flx_sw_dn_clr_trop,
      :tropopause => tropopause[:,:,:]
    )
  end
end

function radiation_by_column(nlay,nlev,col_dry_vr,wkl_vr,wx_vr,cld_frc_vr,tk_sfc,tk_hl_vr,tk_fl_vr,pm_sfc,pm_fl_vr,zsemiss,solar_constant,zi0,cos_mu0m,alb,aer_tau_lw_vr,aer_tau_sw_vr,aer_piz_sw_vr,aer_cg_sw_vr,cld_tau_lw_vr,cld_tau_sw_vr,cld_piz_sw_vr,cld_cg_sw_vr)
  # println("gases: ")
  pwvcm,jp,jt,jt1,laytrop,colh2o,colco2,colo3,coln2o,colco,colch4,colo2,colmol,colbrd,co2mult,selffac,forfac,indfor,forfrac,indself,selffrac,scaleminor,scaleminorn2,indminor,minorfrac,fac10,fac00,fac11,fac01,rat_h2oco2,rat_h2oco2_1,rat_o3co2,rat_o3co2_1,rat_h2oo3,rat_h2oo3_1,rat_h2on2o,rat_h2on2o_1,rat_h2och4,rat_h2och4_1,rat_n2oco2,rat_n2oco2_1,planklay,planklev,plankbnd = gases(nlay,nlev,col_dry_vr,wkl_vr,tk_sfc,tk_hl_vr,tk_fl_vr,pm_sfc,pm_fl_vr,zsemiss,true,true)
  
  # println("LW: ")
  flx_uplw_vr,flx_dnlw_vr,flx_uplw_clr_vr,flx_dnlw_clr_vr = LW(nlay,nlev,col_dry_vr,wx_vr,cld_tau_lw_vr,cld_frc_vr,zsemiss,pm_fl_vr,aer_tau_lw_vr,pwvcm,jp,jt,jt1,laytrop,colh2o,colco2,colo3,coln2o,colco,colch4,colo2,colbrd,selffac,forfac,indfor,forfrac,indself,selffrac,scaleminor,scaleminorn2,indminor,minorfrac,fac10,fac00,fac11,fac01,rat_h2oco2,rat_h2oco2_1,rat_o3co2,rat_o3co2_1,rat_h2oo3,rat_h2oo3_1,rat_h2on2o,rat_h2on2o_1,rat_h2och4,rat_h2och4_1,rat_n2oco2,rat_n2oco2_1,planklay,planklev,plankbnd)

  # println("SW: ")
  if zi0 != 0.0
    flxd_sw,flxu_sw,flxd_sw_clr,flxu_sw_clr = SW(nlay,nlev,solar_constant,cos_mu0m,alb,cld_frc_vr,jp,jt,jt1,laytrop,colh2o,colco2,colo3,coln2o,colco,colch4,colo2,colmol,co2mult,selffac,forfac,indfor,forfrac,indself,selffrac,fac10,fac00,fac11,fac01,aer_tau_sw_vr,aer_cg_sw_vr,aer_piz_sw_vr,cld_tau_sw_vr,cld_cg_sw_vr,cld_piz_sw_vr)
  else
    flxd_sw,flxu_sw,flxd_sw_clr,flxu_sw_clr = zeros(nlev),zeros(nlev),zeros(nlev),zeros(nlev)
  end
  # println("radiation: ")
  flx_uplw_vr,flx_dnlw_vr,flx_uplw_clr_vr,flx_dnlw_clr_vr,flxd_sw,flxu_sw,flxd_sw_clr,flxu_sw_clr
  # fill(nb_sw,nlev),fill(nb_sw,nlev),fill(nb_sw,nlev),fill(nb_sw,nlev),fill(nb_sw,nlev),fill(nb_sw,nlev),fill(nb_sw,nlev),fill(nb_sw,nlev),fill(nb_sw,nlev),fill(nb_sw,nlev)
end

function aerosols(philat,hyai,hybi,pp_hl,pp_fl,tk_hl)
  ntime, nlev, nlat, nlon = size(tk_hl)
  nlay = nlev - 1
  
  # println("Calculating aerosols...")
  aer_tau_lw_vr = zeros(ntime,nlay,nlat,nlon,nb_lw)
  aer_tau_sw_vr = zeros(ntime,nlay,nlat,nlon,nb_sw)
  aer_cg_sw_vr = zeros(ntime,nlay,nlat,nlon,nb_sw)
  aer_piz_sw_vr  = zeros(ntime,nlay,nlat,nlon,nb_sw)

  map_band = [1,5,2,2,2,3,4,3,1,1,1,1,1,1,1,1]  #< map between lw bands and model
  sinlon = zeros(nlon)     # sin[longitude].
  coslon = zeros(nlon)     # cos[longitude].
  gl_twomu = 2*sin.(philat*api/180.)
  
  for jlon = 1:nlon # double size for rotated domains
    zl = 2*api*(jlon-1.0)/nlon    # on decomposed grid
    sinlon[jlon] = sin(zl)
    coslon[jlon] = cos(zl)
  end
  
  aerosol_params = load("$(@__DIR__)/../jlds/aerosol_parameters.jld")
  caesc = aerosol_params["caesc"]
  caess = aerosol_params["caess"]
  caelc = aerosol_params["caelc"]
  caels = aerosol_params["caels"]
  caeuc = aerosol_params["caeuc"]
  caeus = aerosol_params["caeus"]
  caedc = aerosol_params["caedc"]
  caeds = aerosol_params["caeds"]
  caer = aerosol_params["caer"]

  naer=4

  taua = [ 0.730719, 0.912819, 0.725059, 0.682188 ] #< optical depth factors for four aerosol types
  cga  = [ 0.647596, 0.739002, 0.580845, 0.624246 ] #< asymmetry factor for four aerosol types
  piza = [ 0.872212, 0.982545, 0.623143, 0.997975 ] #< sngl sctr albedo for four aerosol typez

  # --- Pressure normalized optical depths, here the surface pressure is set to
  #     101325 hPa and the tropopause is fixed at 19330 hPa
  ctrbga = 0.03/(101325.0-19330.0)  #< troposphere background
  cstbga = 0.045/19330.0                 #< stratosphere background
  apzero = 101325.0
  zetam = hyai[1]/apzero + hybi[1]
  petah = zeros(nlay+1)
  petah[1] = zetam

  for jlev = 1:nlay
    zetap = hyai[jlev+1]/apzero + hybi[1+jlev]
    petah[jlev+1] = zetap
    zetam = zetap
  end
  
  zfaes = zeros(21)
  zfael = zeros(21)
  zfaeu = zeros(21)
  zfaed = zeros(21)

  zaes = zeros(nlon, nlat)
  zael = zeros(nlon, nlat)
  zaeu = zeros(nlon, nlat)
  zaed = zeros(nlon, nlat)

  zx1 = 8434.0/1000.0
  zx2 = 8434.0/3000.0

  cvdaes = petah.^max(1.,8434./1000.)
  cvdael = petah.^max(1.,8434./1000.)
  cvdaeu = petah.^max(1.,8434./1000.)
  cvdaed = petah.^max(1.,8434./3000.)

  if (petah[1] == 0.0)
    cvdaes[1] = 0.0
    cvdael[1] = 0.0
    cvdaeu[1] = 0.0
    cvdaed[1] = 0.0
  end

  caeops = 0.05
  caeopl = 0.2
  caeopu = 0.1
  caeopd = 1.9
  ctrpt  = 30.
  caeadm = 2.6E-10
  caeadk = [0.3876E-03,0.6693E-02,0.8563E-03]

  ppd_hl = pp_hl[:,2:(nlay+1),:,:]-pp_hl[:,1:nlay,:,:]

  for jlat = 1:nlat
    zsin  = 0.5*gl_twomu[jlat]
    zalp = legtri(zsin,11)
    zfaes[:] = 0.0
    zfael[:] = 0.0
    zfaeu[:] = 0.0
    zfaed[:] = 0.0
    imm  = 0
    imnc = 0
    imns = 0
    for jmm = 1:11
      imm = imm + 1
      for jnn = jmm:11
        imnc = imnc + 1
        zfaes[imm] += zalp[imnc]*caesc[imnc]
        zfael[imm] += zalp[imnc]*caelc[imnc]
        zfaeu[imm] += zalp[imnc]*caeuc[imnc]
        zfaed[imm] += zalp[imnc]*caedc[imnc]
      end
      if (jmm!=1)
        imm = imm + 1
        for jnn = jmm:11
          imns = imns + 1
          zfaes[imm] += zalp[imns+11]*caess[imns]
          zfael[imm] += zalp[imns+11]*caels[imns]
          zfaeu[imm] += zalp[imns+11]*caeus[imns]
          zfaed[imm] += zalp[imns+11]*caeds[imns]
        end
      end
    end
    #
    # --- Fourier transform
    #
    for jl = 1:nlon
      zcos1 = coslon[jl]
      zsin1 = sinlon[jl]
      zcos2 = zcos1*zcos1 - zsin1*zsin1
      zsin2 = zsin1*zcos1 + zcos1*zsin1
      zcos3 = zcos2*zcos1 - zsin2*zsin1
      zsin3 = zsin2*zcos1 + zcos2*zsin1
      zcos4 = zcos3*zcos1 - zsin3*zsin1
      zsin4 = zsin3*zcos1 + zcos3*zsin1
      zcos5 = zcos4*zcos1 - zsin4*zsin1
      zsin5 = zsin4*zcos1 + zcos4*zsin1
      zcos6 = zcos5*zcos1 - zsin5*zsin1
      zsin6 = zsin5*zcos1 + zcos5*zsin1
      zcos7 = zcos6*zcos1 - zsin6*zsin1
      zsin7 = zsin6*zcos1 + zcos6*zsin1
      zcos8 = zcos7*zcos1 - zsin7*zsin1
      zsin8 = zsin7*zcos1 + zcos7*zsin1
      zcos9 = zcos8*zcos1 - zsin8*zsin1
      zsin9 = zsin8*zcos1 + zcos8*zsin1
      zcos10= zcos9*zcos1 - zsin9*zsin1
      zsin10= zsin9*zcos1 + zcos9*zsin1

      zaes[jl,jlat] = zfaes[1] +
      2.0*(zfaes[2]*zcos1+zfaes[3]*zsin1+zfaes[4]*zcos2+
      zfaes[5]*zsin2+zfaes[6]*zcos3+zfaes[7]*zsin3+zfaes[8]*zcos4+
      zfaes[9]*zsin4+zfaes[10]*zcos5+zfaes[11]*zsin5+zfaes[12]*zcos6+
      zfaes[13]*zsin6+zfaes[14]*zcos7+zfaes[15]*zsin7+zfaes[16]*zcos8+
      zfaes[17]*zsin8+zfaes[18]*zcos9+zfaes[19]*zsin9+zfaes[20]*zcos10+
      zfaes[21]*zsin10)
      zael[jl,jlat] = zfael[1] +
      2.0*(zfael[2]*zcos1+zfael[3]*zsin1+zfael[4]*zcos2+
      zfael[5]*zsin2+zfael[6]*zcos3+zfael[7]*zsin3+zfael[8]*zcos4+
      zfael[9]*zsin4+zfael[10]*zcos5+zfael[11]*zsin5+zfael[12]*zcos6+
      zfael[13]*zsin6+zfael[14]*zcos7+zfael[15]*zsin7+zfael[16]*zcos8+
      zfael[17]*zsin8+zfael[18]*zcos9+zfael[19]*zsin9+zfael[20]*zcos10+
      zfael[21]*zsin10)
      zaeu[jl,jlat] = zfaeu[1] +
      2.0*(zfaeu[2]*zcos1+zfaeu[3]*zsin1+zfaeu[4]*zcos2+
      zfaeu[5]*zsin2+zfaeu[6]*zcos3+zfaeu[7]*zsin3+zfaeu[8]*zcos4+
      zfaeu[9]*zsin4+zfaeu[10]*zcos5+zfaeu[11]*zsin5+zfaeu[12]*zcos6+
      zfaeu[13]*zsin6+zfaeu[14]*zcos7+zfaeu[15]*zsin7+zfaeu[16]*zcos8+
      zfaeu[17]*zsin8+zfaeu[18]*zcos9+zfaeu[19]*zsin9+zfaeu[20]*zcos10+
      zfaeu[21]*zsin10)
      zaed[jl,jlat] = zfaed[1] +
      2.0*(zfaed[2]*zcos1+zfaed[3]*zsin1+zfaed[4]*zcos2+
      zfaed[5]*zsin2+zfaed[6]*zcos3+zfaed[7]*zsin3+zfaed[8]*zcos4+
      zfaed[9]*zsin4+zfaed[10]*zcos5+zfaed[11]*zsin5+zfaed[12]*zcos6+
      zfaed[13]*zsin6+zfaed[14]*zcos7+zfaed[15]*zsin7+zfaed[16]*zcos8+
      zfaed[17]*zsin8+zfaed[18]*zcos9+zfaed[19]*zsin9+zfaed[20]*zcos10+
      zfaed[21]*zsin10)
    end
  end

  zaetrn = zeros(nlat,nlon)
  zaetro = ones(nlat,nlon)
  zaero = zeros(ntime,nlay,nlat,nlon,naer)

  for jk = 1:nlay
    for jl = 1:nlon
      for krow = 1:nlat
        zaeqso = caeops*zaes[jl,krow]*cvdaes[jk]
        zaeqsn = caeops*zaes[jl,krow]*cvdaes[jk+1]
        zaeqlo = caeopl*zael[jl,krow]*cvdael[jk]
        zaeqln = caeopl*zael[jl,krow]*cvdael[jk+1]
        zaequo = caeopu*zaeu[jl,krow]*cvdaeu[jk]
        zaequn = caeopu*zaeu[jl,krow]*cvdaeu[jk+1]
        zaeqdo = caeopd*zaed[jl,krow]*cvdaed[jk]
        zaeqdn = caeopd*zaed[jl,krow]*cvdaed[jk+1]

        for t = 1:ntime
          if (pp_fl[t,jk,krow,jl] < 999.)
            zaetr= 1. # above 10 hPa
          else
            zaetrn[krow,jl] = zaetro[krow,jl]*(min(1.0,tk_hl[t,jk,krow,jl]/tk_hl[t,jk+1,krow,jl]))^ctrpt
            zaetr      = sqrt(zaetrn[krow,jl]*zaetro[krow,jl])
            zaetro[krow,jl] = zaetrn[krow,jl]
          end

          zaero[t,jk,krow,jl,1] = (1. - zaetr)*(ctrbga*ppd_hl[t,jk,krow,jl] + zaeqln - zaeqlo + zaeqdn - zaeqdo)
          zaero[t,jk,krow,jl,2] = (1. - zaetr)*(zaeqsn-zaeqso)
          zaero[t,jk,krow,jl,3] = (1. - zaetr)*(zaequn-zaequo)
          zaero[t,jk,krow,jl,4] = zaetr*cstbga*ppd_hl[t,jk,krow,jl]
        end
      end
    end
  end
  
  aer = max.(zaero,eps()) # ---optical thickness is not negative

  for ja = 1:naer, jk = 1:nlay
    zaer = aer[:,nlay+1-jk,:,:,ja]
    for jb = 1:nb_lw
      aer_tau_lw_vr[:,jk,:,:,jb] += zaer*caer[map_band[jb],ja]
    end
    aer_tau_sw_vr[:,jk,:,:,:] .+= zaer*taua[ja]
    aer_piz_sw_vr[:,jk,:,:,:] .+= zaer*taua[ja]*piza[ja]
    aer_cg_sw_vr[:,jk,:,:,:]  .+= zaer*taua[ja]*piza[ja]*cga[ja]
  end
  for t = 1:ntime, jl = 1:nlon, krow = 1:nlat, jk = 1:nlay, jb = 1:nb_sw
    if (aer_piz_sw_vr[t,jk,krow,jl,jb] > eps(Float64))
      aer_cg_sw_vr[t,jk,krow,jl,jb]  /= aer_piz_sw_vr[t,jk,krow,jl,jb]
      aer_piz_sw_vr[t,jk,krow,jl,jb] /= aer_tau_sw_vr[t,jk,krow,jl,jb]
    end
  end

  # println("Aerosols calculated.")
  aer_tau_lw_vr, aer_tau_sw_vr, aer_piz_sw_vr, aer_cg_sw_vr
end

function clouds(ntime,nlay,nlev,nlat,nlon,laland,laglac,ktype,zlwp_vr,ziwp_vr,zlwc_vr,ziwc_vr,cdnc_vr,icldlyr)
  # println("Calculating clouds...")

  # re_crystals2d = zeros(ntime,nlay,nlat,nlon)
  # re_droplets2d = zeros(ntime,nlay,nlat,nlon)
  cld_tau_lw_vr = zeros(ntime,nlay,nlat,nlon,nb_lw)
  cld_tau_sw_vr = zeros(ntime,nlay,nlat,nlon,nb_sw)
  cld_piz_sw_vr = zeros(ntime,nlay,nlat,nlon,nb_sw)
  cld_cg_sw_vr = zeros(ntime,nlay,nlat,nlon,nb_sw)

  n_mdl_bnds = 30
  n_sizes    = 61
  ccwmin = 1.e-7
  zkap_cont = 1.143
  zkap_mrtm = 1.077

  rebcug = [0.718, 0.726, 1.136, 1.320, 1.505, 1.290, 0.911, 0.949, 1.021, 1.193, 1.279, 0.626, 0.647, 0.668, 0.690, 0.690]
  rebcuh = [0.0069, 0.0060, 0.0024, 0.0004, -0.0016, 0.0003, 0.0043, 0.0038, 0.0030, 0.0013, 0.0005, 0.0054, 0.0052, 0.0050, 0.0048, 0.0048]
  l_variable_inhoml = true

  # Variable liquid cloud inhomogeneity is not used:
  l_variable_inhoml = false
  zinhoml1      = 0.77
  zinhoml2      = 0.77
  zinhomi       = 0.80
  zinpar  = 0.10

  cld_opt_props_dset = xr.open_dataset("$(@__DIR__)/../netcdfs/ECHAM6_CldOptProps.nc",decode_times=false)
  # if (p_parallel_io)
  wavenumber = cld_opt_props_dset["wavenumber"][:values]
  wavelength = cld_opt_props_dset["wavelength"][:values]
  re_droplet = cld_opt_props_dset["re_droplet"][:values]
  re_crystal = cld_opt_props_dset["re_crystal"][:values]
  z_ext_l = cld_opt_props_dset["extinction_per_mass_droplet"][:values]'
  z_coa_l = cld_opt_props_dset["co_albedo_droplet"][:values]'
  z_asy_l = cld_opt_props_dset["asymmetry_factor_droplet"][:values]'
  z_ext_i = cld_opt_props_dset["extinction_per_mass_crystal"][:values]'
  z_coa_i = cld_opt_props_dset["co_albedo_crystal"][:values]'
  z_asy_i = cld_opt_props_dset["asymmetry_factor_crystal"][:values]'

  reimin = minimum(re_crystal)
  reimax = maximum(re_crystal)
  del_rei= (re_crystal[2] - re_crystal[1])
  relmin = minimum(re_droplet)
  relmax = maximum(re_droplet)
  del_rel= (re_droplet[2] - re_droplet[1])

  if ((relmin < 1.5) || (relmin > 2.5))
    throw("Apparently unsuccessful loading of optical tables")
  end
  #
  # 1.0 Basic cloud properties
  # --------------------------------
  # if (l_variable_inhoml)
  #   zlwpt[1:nlon] = 0.0
  #   for jk = 1:nlay
  #     for jl = 1:nlon
  #       zlwpt[jl] = zlwpt[jl]+zlwp[jl,jk]
  #     end
  #   end
  #   WHERE (zlwpt[1:nlon] > 1.0)
  #   zinhoml[1:nlon] = zlwpt[1:nlon]**(-zinpar)
  #   elseWHERE
  #   zinhoml[1:nlon] = 1.0
  #   END WHERE
  # else
  zinhoml = zeros(ntime,nlat,nlon)
  zinhoml[ktype .== 0.0] = zinhoml1
  zinhoml[ktype .!= 0.0] = zinhoml2
  # end


  zkap = fill(zkap_mrtm,nlat,nlon)

  # possible source of error?
  zkap[(laland-laglac).==1] = zkap_cont
  rhoh2o = 1000.0
  #
  # 2.0 Cloud Optical Properties by interpolating tables in effective radius
  # --------------------------------
  zfact = 1.0e6*(3.0e-9/(4.0*api*rhoh2o))^(1.0/3.0)
  for jk=1:nlay
    for jl=1:nlon, t=1:ntime, krow=1:nlat
      ztau = zeros(n_mdl_bnds)
      zomg = zeros(n_mdl_bnds)
      zasy = zeros(n_mdl_bnds)

      if (icldlyr[t,jk,krow,jl]==1 && (zlwp_vr[t,jk,krow,jl]+ziwp_vr[t,jk,krow,jl])>ccwmin)

        re_crystals = max(reimin,min(reimax,83.8*ziwc_vr[t,jk,krow,jl]^0.216))
        re_droplets = max(relmin,min(relmax,zfact*zkap[krow,jl]*(zlwc_vr[t,jk,krow,jl]/cdnc_vr[t,jk,krow,jl])^(1.0/3.0)))

        # re_crystals2d[t,jk,krow,jl] = re_crystals
        # re_droplets2d[t,jk,krow,jl] = re_droplets

        ml1 = Int(max(1,min(n_sizes-1,floor(1.0+(re_droplets-relmin)/del_rel))))
        ml2 = ml1 + 1
        wl1 = 1.0 - (re_droplets - (relmin + del_rel* (ml1-1)) )/del_rel
        wl2 = 1.0 - wl1

        mi1 = Int(max(1,min(n_sizes-1,floor(1.0+(re_crystals-reimin)/del_rei))))
        mi2 = mi1 + 1
        wi1 = 1.0 - (re_crystals - (reimin + del_rei * (mi1-1)) )/del_rei
        wi2 = 1.0 - wi1

        for iband = 1:n_mdl_bnds
          ztol = zlwp_vr[t,jk,krow,jl]*(wl1*z_ext_l[ml1,iband] + wl2*z_ext_l[ml2,iband])
          ztoi = ziwp_vr[t,jk,krow,jl]*(wi1*z_ext_i[mi1,iband] + wi2*z_ext_i[mi2,iband])
          zol  = 1.0 - (wl1*z_coa_l[ml1,iband] + wl2*z_coa_l[ml2,iband])
          zoi  = 1.0 - (wi1*z_coa_i[mi1,iband] + wi2*z_coa_i[mi2,iband])
          zgl  = wl1*z_asy_l[ml1,iband] + wl2*z_asy_l[ml2,iband]
          zgi  = wi1*z_asy_i[mi1,iband] + wi2*z_asy_i[mi2,iband]

          zscratch = (ztol*zol+ztoi*zoi)
          ztau[iband] = ztol*zinhoml[t,krow,jl] + ztoi*zinhomi
          zomg[iband] = zscratch/(ztol+ztoi)
          zasy[iband] = (ztol*zol*zgl+ztoi*zoi*zgi)/zscratch
        end
        #
        # overwrite Kinne Optics with old Cloud Optics for LW Only
        #
        for iband = 1:16
          zmsald=0.025520637+0.2854650784*exp(-0.088968393014*re_droplets)
          zmsaid=(rebcuh[iband]+rebcug[iband]/re_crystals)
          ztau[iband]  = zmsald*zlwp_vr[t,jk,krow,jl]*zinhoml[t,krow,jl]+zmsaid*ziwp_vr[t,jk,krow,jl]*zinhomi
          zomg[iband] = 0.0
        end
      else
        ztau[:]  = 0.0
        zomg[:]  = 1.0
        zasy[:]  = 0.0
        # re_crystals2d[t,jk,krow,jl] = 0.0
        # re_droplets2d[t,jk,krow,jl] = 0.0
      end
      cld_tau_lw_vr[t,jk,krow,jl,:] = ztau[1:16] .* (1.0 - zomg[1:16])
      cld_tau_sw_vr[t,jk,krow,jl,:] = ztau[16:29]
      cld_piz_sw_vr[t,jk,krow,jl,:] = zomg[16:29]
      cld_cg_sw_vr[t,jk,krow,jl,:] = zasy[16:29]
    end
  end
    
  cld_tau_lw_vr, cld_tau_sw_vr, cld_piz_sw_vr, cld_cg_sw_vr # , re_drop, re_cryst
end

function gases(nlay,nlev,col_dry_vr,wkl_vr,tk_sfc,tk_hl_vr,tk_fl_vr,pm_sfc,pm_fl_vr,zsemiss,do_lw=true,do_sw=true)
  # println("Calculating gases...")
  # These are the temperatures associated with the respective pressures for the mls standard atmosphere.
  preflog = [6.9600e+00, 6.7600e+00, 6.5600e+00, 6.3600e+00, 6.1600e+00,5.9600e+00, 5.7600e+00, 5.5600e+00, 5.3600e+00, 5.1600e+00,4.9600e+00, 4.7600e+00, 4.5600e+00, 4.3600e+00, 4.1600e+00,3.9600e+00, 3.7600e+00, 3.5600e+00, 3.3600e+00, 3.1600e+00,2.9600e+00, 2.7600e+00, 2.5600e+00, 2.3600e+00, 2.1600e+00,1.9600e+00, 1.7600e+00, 1.5600e+00, 1.3600e+00, 1.1600e+00,9.6000e-01, 7.6000e-01, 5.6000e-01, 3.6000e-01, 1.6000e-01,-4.0000e-02,-2.4000e-01,-4.4000e-01,-6.4000e-01,-8.4000e-01,-1.0400e+00,-1.2400e+00,-1.4400e+00,-1.6400e+00,-1.8400e+00,-2.0400e+00,-2.2400e+00,-2.4400e+00,-2.6400e+00,-2.8400e+00,-3.0400e+00,-3.2400e+00,-3.4400e+00,-3.6400e+00,-3.8400e+00,-4.0400e+00,-4.2400e+00,-4.4400e+00,-4.6400e+00]
  tref = [2.9420e+02, 2.8799e+02, 2.7894e+02, 2.6925e+02, 2.5983e+02,2.5017e+02, 2.4077e+02, 2.3179e+02, 2.2306e+02, 2.1578e+02,2.1570e+02, 2.1570e+02, 2.1570e+02, 2.1706e+02, 2.1858e+02,2.2018e+02, 2.2174e+02, 2.2328e+02, 2.2479e+02, 2.2655e+02,2.2834e+02, 2.3113e+02, 2.3401e+02, 2.3703e+02, 2.4022e+02,2.4371e+02, 2.4726e+02, 2.5085e+02, 2.5457e+02, 2.5832e+02,2.6216e+02, 2.6606e+02, 2.6999e+02, 2.7340e+02, 2.7536e+02,2.7568e+02, 2.7372e+02, 2.7163e+02, 2.6955e+02, 2.6593e+02,2.6211e+02, 2.5828e+02, 2.5360e+02, 2.4854e+02, 2.4348e+02,2.3809e+02, 2.3206e+02, 2.2603e+02, 2.2000e+02, 2.1435e+02,2.0887e+02, 2.0340e+02, 1.9792e+02, 1.9290e+02, 1.8809e+02,1.8329e+02, 1.7849e+02, 1.7394e+02, 1.7212e+02]

  totplnk = LW_parameters["totplnk"]
  chi_mls = LW_parameters["chi_mls"]
  stpfac = 296./1013.
  indself = zeros(Int,nlay);
  indfor = zeros(Int,nlay);
  selffrac = zeros(nlay);
  forfac = zeros(nlay);
  forfrac = zeros(nlay);

  # just LW
  planklev = zeros(nlev,nb_lw);
  rat_h2oco2 =  zeros(nlay);
  rat_h2oco2_1 = zeros(nlay);
  rat_h2oo3 =  zeros(nlay);
  rat_h2oo3_1 = zeros(nlay);
  rat_h2on2o =  zeros(nlay);
  rat_h2on2o_1 = zeros(nlay);
  rat_h2och4 =  zeros(nlay);
  rat_h2och4_1 = zeros(nlay);
  rat_n2oco2 =  zeros(nlay);
  rat_n2oco2_1 = zeros(nlay);
  rat_o3co2 =  zeros(nlay);
  rat_o3co2_1 = zeros(nlay);

  wbrodl = col_dry_vr - squeeze(sum(wkl_vr[2:7,:],1),1)
  wvttl = sum(wkl_vr[1,:])
  amttl = sum(col_dry_vr) + wvttl

  wvsh = (amw * wvttl) / (amd * amttl)
  pwvcm = wvsh * (1.e3 * pm_sfc) / (1.e2 * g)

  function bound(input,high)
    min.(max.(fortran_int(input),1),high)
  end

  function bound(input,high,extra)
    x = bound(input,high)
    x, (input+extra) - float(x+extra)
  end

  indbounda,tbndfraca = bound(tk_sfc - 159.,180,0)
  indlev0a,t0fraca = bound(tk_hl_vr[1] - 159.,180,0)
  indlay,tlayfrac = bound(tk_fl_vr - 159.,180,0)
  indlev,tlevfrac = bound(tk_hl_vr[2:nlev] - 159.,180,0)

  dbdtlev = totplnk[indbounda+1,:] - totplnk[indbounda,:]
  plankbnd = zsemiss .* (totplnk[indbounda,:] + tbndfraca .* dbdtlev)
  dbdtlev = totplnk[indlev0a+1,:]-totplnk[indlev0a,:]
  planklev[1,:] .= totplnk[indlev0a,:] + t0fraca .* dbdtlev

  dbdtlev = totplnk[indlev+1,:] - totplnk[indlev,:]
  dbdtlay = totplnk[indlay+1,:] - totplnk[indlay,:]
  planklay = totplnk[indlay,:] + tlayfrac .* dbdtlay
  planklev[2:nlev,:] .= totplnk[indlev,:] + tlevfrac .* dbdtlev

  plog = log.(pm_fl_vr)
  jp = bound(36. - 5 *(plog + 0.04),58)
  jp1 = jp + 1

  fp = 5. * (preflog[jp] - plog)

  jt, ft   = bound(3. + (tk_fl_vr - tref[jp])./15.,4,-3)
  jt1, ft1 = bound(3. + (tk_fl_vr - tref[jp1])./15.,4,-3)

  water = @. wkl_vr[1,:]/col_dry_vr
  scalefac = @. pm_fl_vr * stpfac / tk_fl_vr

  scaleminor = @. pm_fl_vr / tk_fl_vr
  scaleminorn2 = scaleminor .* (wbrodl./(col_dry_vr+wkl_vr[1,:]))
  indminor,minorfrac = bound((tk_fl_vr-180.8)/7.2,18,0)

  # just LW
  get_chi_mls(input,lay,jp,chi1,chi2) = (input[lay]=chi_mls[chi1,jp]/chi_mls[chi2,jp])
  laytrop = 0
  function gas_level_function_1(lay,laytrop)
    p = pm_fl_vr[lay]
    tk = tk_fl_vr[lay]
    jpl = jp[lay]
    jp1l = jp1[lay]

    forfac[lay] = scalefac[lay] / (1.+water[lay])

    #  If the pressure is less than ~100mb, perform a different
    #  set of species interpolations.
    get_chi_mls(rat_h2oco2,lay,jpl,1,2)
    get_chi_mls(rat_h2oco2_1,lay,jp1l,1,2)

    if (plog[lay] <= 4.56)
      laytrop = laytrop
      #  Above laytrop.

      indfor[lay] = 3
      forfrac[lay] = (tk-188.0)/36.0 - 1.0

      # just LW
      get_chi_mls(rat_o3co2,lay,jpl,3,2)
      get_chi_mls(rat_o3co2_1,lay,jp1l,3,2)
    else
      laytrop = lay

      indfor[lay],forfrac[lay] = bound((332.0-tk)/36.0,2,0)
      indself[lay],selffrac[lay] = bound((tk-188.0)/7.2-7,9,-7)

      # just LW
      get_chi_mls(rat_h2oo3,lay,jpl,1,3)
      get_chi_mls(rat_h2oo3_1,lay,jp1l,1,3)
      get_chi_mls(rat_h2on2o,lay,jpl,1,4)
      get_chi_mls(rat_h2on2o_1,lay,jp1l,1,4)
      get_chi_mls(rat_h2och4,lay,jpl,1,6)
      get_chi_mls(rat_h2och4_1,lay,jp1l,1,6)
      get_chi_mls(rat_n2oco2,lay,jpl,4,2)
      get_chi_mls(rat_n2oco2_1,lay,jp1l,4,2)
    end

    laytrop
  end

  for lay=1:nlay
    laytrop = gas_level_function_1(lay,laytrop)
  end

  colco2 = 1.e-20 * wkl_vr[2,:]
  colo3  = 1.e-20 * wkl_vr[3,:]
  coln2o = 1.e-20 * wkl_vr[4,:]
  colco  = 1.e-20 * wkl_vr[5,:]
  colch4 = 1.e-20 * wkl_vr[6,:]
  colh2o = 1.e-20 * wkl_vr[1,:]
  colo2  = 1.e-20 * wkl_vr[7,:]
  colbrd = 1.e-20 * wbrodl
  colmol = 1.e-20 * col_dry_vr + colh2o
  @. colco2[colco2 .== 0.] = 1.e-32 * col_dry_vr[colco2 .== 0.]
  @. colo3[colo3 .== 0.] = 1.e-32 * col_dry_vr[colo3 .== 0.]
  @. coln2o[coln2o .== 0.] = 1.e-32 * col_dry_vr[coln2o .== 0.]
  @. colco[colco .== 0.] = 1.e-32 * col_dry_vr[colco .== 0.]
  @. colch4[colch4 .== 0.] = 1.e-32 * col_dry_vr[colch4 .== 0.]

  # just SW
  co2mult = (colco2 - 3.55e-24 * col_dry_vr) * 272.63 .* exp.(-1919.4 ./ tk_fl_vr) ./ (8.7604e-4 * tk_fl_vr)

  compfp = 1. - fp
  fac10 = @. compfp * ft
  fac00 = compfp .* (1. - ft)
  fac11 = @. fp * ft1
  fac01 = fp .* (1. - ft1)

  selffac = @. water * forfac

  # println("Gases calculated.")
  pwvcm,jp,jt,jt1,laytrop,colh2o,colco2,colo3,coln2o,colco,colch4,colo2,colmol,colbrd,co2mult,selffac,forfac,indfor,forfrac,indself,selffrac,scaleminor,scaleminorn2,indminor,minorfrac,fac10,fac00,fac11,fac01,rat_h2oco2,rat_h2oco2_1,rat_o3co2,rat_o3co2_1,rat_h2oo3,rat_h2oo3_1,rat_h2on2o,rat_h2on2o_1,rat_h2och4,rat_h2och4_1,rat_n2oco2,rat_n2oco2_1,planklay,planklev,plankbnd
  # 10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10
end

function LW(nlay,nlev,col_dry_vr,wx_vr,cld_tau_lw_vr,cld_frc_vr,zsemiss,pm_fl_vr,aer_tau_lw_vr,pwvcm,jp,jt,jt1,laytrop,colh2o,colco2,colo3,coln2o,colco,colch4,colo2,colbrd,selffac,forfac,indfor,forfrac,indself,selffrac,scaleminor,scaleminorn2,indminor,minorfrac,fac10,fac00,fac11,fac01,rat_h2oco2,rat_h2oco2_1,rat_o3co2,rat_o3co2_1,rat_h2oo3,rat_h2oo3_1,rat_h2on2o,rat_h2on2o_1,rat_h2och4,rat_h2och4_1,rat_n2oco2,rat_n2oco2_1,planklay,planklev,plankbnd)
  # println("Calculating LW radiation...")

  cldmin = 1.e-20 # minimum val for clouds
  ncbands = 1
  tauctot = sum(cld_tau_lw_vr,2)[:,1]
  taucloud = zeros(nlay,nb_lw) # [:,:,:] = 0.0
  forfac_LW = @. forfac * colh2o
  selffac_LW = @. selffac * colh2o

  for jk=1:nlay # loop over columns
    if (cld_frc_vr[jk] >= cldmin && tauctot[jk] >= cldmin)
      ncbands = 16
      for ib = 1:nb_lw
        taucloud[jk,:] = cld_tau_lw_vr[jk,:]
      end
    end
  end

  # println("rat_o3co2_1: ", rat_o3co2_1[1,2,1,1])

  fracs, taug, taut, secdiff = LW_optical_depths(nlay,aer_tau_lw_vr,pwvcm,laytrop,jp,jt,jt1,indself,indfor,pm_fl_vr,colbrd,scaleminor,scaleminorn2,indminor,selffac_LW,selffrac,forfac_LW,forfrac,minorfrac,colh2o,colco2,coln2o,colo3,colco,colch4,colo2,col_dry_vr,wx_vr,fac00,fac10,fac01,fac11,rat_h2oco2,rat_h2oco2_1,rat_o3co2,rat_o3co2_1,rat_h2oo3,rat_h2oo3_1,rat_h2on2o,rat_h2on2o_1,rat_h2och4,rat_h2och4_1,rat_n2oco2,rat_n2oco2_1,:calculate)


  odcld = zeros(nlay,nb_lw);
  cloudy = falses(nlay)
  for lay=1:nlay # loop over columns
    cloudy[lay] = (cld_frc_vr[lay] >= 1.e-6)
    if cloudy[lay]
      odcld[lay,1:ncbands] = secdiff[1:ncbands] .* taucloud[lay,1:ncbands]
    end
    # if lay == 1
    #   println("secdiff: ", secdiff[1:ncbands])
    #   println("taucloud: ", taucloud[lay,1:ncbands])
    # end
  end

  flx_uplw_vr = zeros(nlev);
  flx_dnlw_vr = zeros(nlev);
  flx_uplw_clr_vr = zeros(nlev);
  flx_dnlw_clr_vr = zeros(nlev);
  faccld1 = zeros(nlev);
  faccld2 = zeros(nlev);
  facclr1 = zeros(nlev);
  facclr2 = zeros(nlev);
  faccmb1 = zeros(nlev);
  faccmb2 = zeros(nlev);
  faccld1d = zeros(nlev);
  faccld2d = zeros(nlev);
  facclr1d = zeros(nlev);
  facclr2d = zeros(nlev);
  faccmb1d = zeros(nlev);
  faccmb2d = zeros(nlev);

  # Maximum/Random cloud overlap parameter
  istcld = trues(nlev);
  istcldd = trues(nlev);

  rat1 = 0.
  rat2 = 0.

  function LW_level_function_1(lev)
    # Maximum/random cloud overlap
    istcld[lev+1] = false
    if lev == nlay
    elseif (cld_frc_vr[lev+1] >= cld_frc_vr[lev])
      if istcld[lev]
        if (cld_frc_vr[lev] < 1.)
          facclr2[lev+1] = (cld_frc_vr[lev+1]-cld_frc_vr[lev])/(1. - cld_frc_vr[lev])
        end
      else
        fmax = max(cld_frc_vr[lev],cld_frc_vr[lev-1])
        if (cld_frc_vr[lev+1] > fmax)
          facclr1[lev+1] = rat2
          facclr2[lev+1] = (cld_frc_vr[lev+1]-fmax)/(1. - fmax)
        elseif (cld_frc_vr[lev+1] < fmax)
          facclr1[lev+1] = (cld_frc_vr[lev+1]-cld_frc_vr[lev])/ (cld_frc_vr[lev-1]-cld_frc_vr[lev])
        else
          facclr1[lev+1] = rat2
        end
      end
      if (facclr1[lev+1]>0. || facclr2[lev+1]>0.)
        rat1 = 1.
        rat2 = 0.
      else
        rat1 = 0.
        rat2 = 0.
      end
    else
      if istcld[lev]
        faccld2[lev+1] = (cld_frc_vr[lev]-cld_frc_vr[lev+1])/cld_frc_vr[lev]
      else
        fmin = min(cld_frc_vr[lev],cld_frc_vr[lev-1])
        if (cld_frc_vr[lev+1] <= fmin)
          faccld1[lev+1] = rat1
          faccld2[lev+1] = (fmin-cld_frc_vr[lev+1])/fmin
        else
          faccld1[lev+1] = (cld_frc_vr[lev]-cld_frc_vr[lev+1])/(cld_frc_vr[lev]-fmin)
        end
      end
      if (faccld1[lev+1]>0. || faccld2[lev+1]>0.)
        rat1 = 0.
        rat2 = 1.
      else
        rat1 = 0.
        rat2 = 0.
      end
    end
    if (lev == 1)
      faccmb2[lev+1] = faccld1[lev+1] * facclr2[lev]
    else
      faccmb1[lev+1] = facclr1[lev+1] * faccld2[lev] * cld_frc_vr[lev-1]
      faccmb2[lev+1] = faccld1[lev+1] * facclr2[lev] * (1. - cld_frc_vr[lev-1])
    end
  end

  for lev=(1:nlay)[cloudy]
    LW_level_function_1(lev)
  end

  function LW_level_function_2(lev)
    istcldd[lev] = false
    if lev == 1
    elseif (cld_frc_vr[lev-1] >= cld_frc_vr[lev])
      if istcldd[lev+1]
        if (cld_frc_vr[lev] < 1.)
          facclr2d[lev] = (cld_frc_vr[lev-1]-cld_frc_vr[lev])/(1. - cld_frc_vr[lev])
        end
      else
        fmax = max(cld_frc_vr[lev],cld_frc_vr[lev+1])
        if (cld_frc_vr[lev-1] > fmax)
          facclr1d[lev] = rat2
          facclr2d[lev] = (cld_frc_vr[lev-1]-fmax)/(1. - fmax)
        elseif (cld_frc_vr[lev-1] < fmax)
          facclr1d[lev] = (cld_frc_vr[lev-1]-cld_frc_vr[lev])/ (cld_frc_vr[lev+1]-cld_frc_vr[lev])
        else
          facclr1d[lev] = rat2
        end
      end
      if (facclr1d[lev]>0. || facclr2d[lev]>0.)
        rat1 = 1.
        rat2 = 0.
      else
        rat1 = 0.
        rat2 = 0.
      end
    else
      if istcldd[lev+1]
        faccld2d[lev] = (cld_frc_vr[lev]-cld_frc_vr[lev-1])/cld_frc_vr[lev]
      else
        fmin = min(cld_frc_vr[lev],cld_frc_vr[lev+1])
        if (cld_frc_vr[lev-1] <= fmin)
          faccld1d[lev] = rat1
          faccld2d[lev] = (fmin-cld_frc_vr[lev-1])/fmin
        else
          faccld1d[lev] = (cld_frc_vr[lev]-cld_frc_vr[lev-1])/(cld_frc_vr[lev]-fmin)
        end
      end
      if (faccld1d[lev]>0. || faccld2d[lev]>0.)
        rat1 = 0.
        rat2 = 1.
      else
        rat1 = 0.
        rat2 = 0.
      end
    end
    if (lev == nlay)
      faccmb2d[lev] = faccld1d[lev] * facclr2d[lev+1]
    else
      faccmb1d[lev] = facclr1d[lev] * faccld2d[lev+1] * cld_frc_vr[lev+1]
      faccmb2d[lev] = faccld1d[lev] * facclr2d[lev+1] * (1. - cld_frc_vr[lev+1])
    end
  end

  for lev=reverse((1:nlay)[cloudy])
    LW_level_function_2(lev)
  end
  
  # println(cloudy)

  # Loop over frequency bands.
  for iband = 1:nb_lw
    flx_uplw_vr,flx_dnlw_vr,flx_uplw_clr_vr,flx_dnlw_clr_vr = LW_radiative_transfer_per_band(nlay,nlev,iband,ncbands,planklay,planklev,secdiff,taut,fracs,odcld,cloudy,istcld,istcldd,cld_frc_vr,facclr1,facclr2,faccld1,faccld2,faccmb1,faccmb2,facclr1d,facclr2d,faccld1d,faccld2d,faccmb1d,faccmb2d,plankbnd,zsemiss,flx_uplw_vr,flx_dnlw_vr,flx_uplw_clr_vr,flx_dnlw_clr_vr)
  end
  
  api = 3.14159265358979323846
  fluxfac = 2.0e+04 * api

  # Calculate fluxes at surface
  @. flx_uplw_vr *= fluxfac
  @. flx_dnlw_vr *= fluxfac
  @. flx_uplw_clr_vr *= fluxfac
  @. flx_dnlw_clr_vr *= fluxfac
  
  # Calculate fluxes at model levels
  # #
  # # 5.0 Post Processing
  # # --------------------------------
  # flx_lw_net     = flx_dnlw_vr[end:-1:1]-flx_uplw_vr[end:-1:1]
  # flx_lw_net_clr = flx_dnlw_clr_vr[end:-1:1]-flx_uplw_clr_vr[end:-1:1]

  # println("LW radiation calculated.")
  flx_uplw_vr,flx_dnlw_vr,flx_uplw_clr_vr,flx_dnlw_clr_vr
end

function LW_optical_depths(nlay,aer_tau_lw_vr,pwvcm,laytrop,jp,jt,jt1,indself,indfor,pm_fl_vr,colbrd,scaleminor,scaleminorn2,indminor,selffac_LW,selffrac,forfac_LW,forfrac,minorfrac,colh2o,colco2,coln2o,colo3,colco,colch4,colo2,col_dry_vr,wx_vr,fac00,fac10,fac01,fac11,rat_h2oco2,rat_h2oco2_1,rat_o3co2,rat_o3co2_1,rat_h2oo3,rat_h2oo3_1,rat_h2on2o,rat_h2on2o_1,rat_h2och4,rat_h2och4_1,rat_n2oco2,rat_n2oco2_1,method)
  ngptlw = 140
  ngb = [1,1,1,1,1,1,1,1,1,1,           # band 1
       2,2,2,2,2,2,2,2,2,2,2,2,         # band 2
       3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3, # band 3
       4,4,4,4,4,4,4,4,4,4,4,4,4,4,     # band 4
       5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5, # band 5
       6,6,6,6,6,6,6,6,                 # band 6
       7,7,7,7,7,7,7,7,7,7,7,7,         # band 7
       8,8,8,8,8,8,8,8,                 # band 8
       9,9,9,9,9,9,9,9,9,9,9,9,         # band 9
       10,10,10,10,10,10,               # band 10
       11,11,11,11,11,11,11,11,         # band 11
       12,12,12,12,12,12,12,12,         # band 12
       13,13,13,13,                     # band 13
       14,14,                           # band 14
       15,15,                           # band 15
       16,16]                           # band 16
  fracs = zeros(nlay,ngptlw)
  taug = zeros(nlay,ngptlw)
  taut = zeros(nlay,ngptlw)
  secdiff = fill(1.66,nb_lw)

  # println("Calculating LW optical depths...")

  zlaytrop = fill(-1.0,nlay)
  zlaytrop[1:laytrop] = 1.0


  for iband in 1:nb_lw # 1:nb_lw
    LW_optical_depth_per_band(nlay,iband,laytrop,zlaytrop,jp,jt,jt1,indself,indfor,pm_fl_vr,colbrd,scaleminor,scaleminorn2,indminor,selffac_LW,selffrac,forfac_LW,forfrac,minorfrac,colh2o,colco2,coln2o,colo3,colco,colch4,colo2,col_dry_vr,wx_vr,fac00,fac10,fac01,fac11,rat_h2oco2,rat_h2oco2_1,rat_o3co2,rat_o3co2_1,rat_h2oo3,rat_h2oo3_1,rat_h2on2o,rat_h2on2o_1,rat_h2och4,rat_h2och4_1,rat_n2oco2,rat_n2oco2_1,fracs,taug)
  end

  # save("big_guys.jld","fracs",fracs,"taug",taug,"taut",taut)
  # taut = cached_dict["taut"]
  #
  # 4.0 Call the radiative transfer routine. (random-maximum overlap)
  # --------------------------------
  #
  a0 = [1.66,  1.55,  1.58,  1.66, 1.54, 1.454,  1.89,  1.33, 1.668,  1.66,  1.66,  1.66, 1.66,  1.66,  1.66,  1.66]
  a1 = [0.00,  0.25,  0.22,  0.00, 0.13, 0.446, -0.10,  0.40, -0.006,  0.00,  0.00,  0.00, 0.00,  0.00,  0.00,  0.00]
  a2 = [0.00, -12.0, -11.7,  0.00, -0.72,-0.243,  0.19,-0.062, 0.414,  0.00,  0.00,  0.00, 0.00,  0.00,  0.00,  0.00]
  ibnds = [2,3,5,6,7,8,9]
  # pwvcm = reshape(pwvcm,(size(pwvcm)...,1))
  @. secdiff[ibnds] = max(min(a0[ibnds] + a1[ibnds]*exp(a2[ibnds]*pwvcm),1.80),1.50)

  per_g_point(ig) = (@. taut[:,ig] = taug[:,ig] + aer_tau_lw_vr[:,ngb[ig]])
  for ig = 1:ngptlw
    per_g_point(ig)
  end

  # println("LW optical depths calculated.")
  fracs, taug, taut, secdiff
end

function LW_optical_depth_per_band(nlay,iband,laytrop,zlaytrop,jp,jt,jt1,indself,indfor,pm_fl_vr,colbrd,scaleminor,scaleminorn2,indminor,selffac_LW,selffrac,forfac_LW,forfrac,minorfrac,colh2o,colco2,coln2o,colo3,colco,colch4,colo2,col_dry_vr,wx_vr,fac00,fac10,fac01,fac11,rat_h2oco2,rat_h2oco2_1,rat_o3co2,rat_o3co2_1,rat_h2oo3,rat_h2oo3_1,rat_h2on2o,rat_h2on2o_1,rat_h2och4,rat_h2och4_1,rat_n2oco2,rat_n2oco2_1,fracs,taug)
  # println("Calculating optical depth for LW band ", iband, "...")

  ngc = [10,12,16,14,16,8,12,8,12,6,8,8,4,2,2,2]
  nspa = [1,1,9,9,9,1,9,1,9,1,1,9,9,1,9,9]
  nspb = [1,1,5,5,5,0,1,1,1,1,1,0,0,1,0,0]
  chi_mls = LW_parameters["chi_mls"]
  lw_params = LW_parameters["lw_params"]
  function spec_items(colh, r, col)
    specc = colh + r*col
    specp = colh/specc
    if (specp >= oneminus)
      specp = oneminus
    end
    specm = 8. * (specp)
    j = 1 + fortran_int(specm)
    f = specm % 1.0

    specc, specp, specm, j, f
  end

  refrat_planck_a_dict = Dict(
    3 => chi_mls[1,9]/chi_mls[2,9],
    4 => chi_mls[1,11]/chi_mls[2,11],
    5 => chi_mls[1,5]/chi_mls[2,5],
    7 => chi_mls[1,3]/chi_mls[3,3],
    9 => chi_mls[1,9]/chi_mls[6,9],
    12 => chi_mls[1,10]/chi_mls[2,10],
    13 => chi_mls[1,5]/chi_mls[4,5],
    15 => chi_mls[4,1]/chi_mls[2,1],
    16 => chi_mls[1,6]/chi_mls[6,6]
  )

  refrat_planck_b_dict = Dict(
    3 => chi_mls[1,13]/chi_mls[2,13],
    4 => chi_mls[3,13]/chi_mls[2,13],
    5 => chi_mls[3,43]/chi_mls[2,43]
  )

  refrat_m_a_dict = Dict(
    3 => chi_mls[1,3]/chi_mls[2,3],
    5 => chi_mls[1,7]/chi_mls[2,7],
    7 => chi_mls[1,3]/chi_mls[3,3],
    9 => chi_mls[1,3]/chi_mls[6,3],
    13 => chi_mls[1,1]/chi_mls[4,1],
    15 => chi_mls[4,1]/chi_mls[2,1]
  )

  refrat_m_b_dict = Dict(
    3 => chi_mls[1,13]/chi_mls[2,13]
  )

  refrat_m_a3_dict = Dict(
    13 => chi_mls[1,3]/chi_mls[4,3]
  )

  fracrefa = lw_params[iband]["fracrefa"]
  absa = lw_params[iband]["absa"]
  selfref = lw_params[iband]["selfref"]
  forref = lw_params[iband]["forref"]
  if "fracrefb" in keys(lw_params[iband])
    fracrefb = lw_params[iband]["fracrefb"]
  end
  if "absb" in keys(lw_params[iband])
    absb = lw_params[iband]["absb"]
  end

  if iband in keys(refrat_planck_a_dict)
    refrat_planck_a = refrat_planck_a_dict[iband]
  end

  if iband in keys(refrat_planck_b_dict)
    refrat_planck_b = refrat_planck_b_dict[iband]
  end

  if iband in keys(refrat_m_a_dict)
    refrat_m_a = refrat_m_a_dict[iband]
  end

  if iband in keys(refrat_m_b_dict)
    refrat_m_b = refrat_m_b_dict[iband]
  end

  if iband in keys(refrat_m_a3_dict)
    refrat_m_a3 = refrat_m_a3_dict[iband]
  end

  fks(p) = (p^4, 1.0 - p - 2.0*p^4, p + p^4)

  g_range = sum(ngc[1:(iband-1)])+(1:ngc[iband])

    # Lower atmosphere loop
  for lay = 1:laytrop
    if iband == 3
      # computing the interpolation lookup indices and fractions
      # is (almost) independent of lower/higher atmosphere flag


      # If the value of X is greater than or equal to zero, then the value of Y is returned. If the value of X is smaller than zero or is a NaN, then the value of Z is returned.

      # mpuetz: select correct factors depending on lower/higher atmosphere flag
      # zlaytrop is of type REAL, so fast pipelined fsel instructions can be used
      # had to use FSEL intrinsic here to get XLF not to do gather/scatter

      refrat_m      = lay <= laytrop ? refrat_m_a : refrat_m_b
      refrat_planck = lay <= laytrop ? refrat_planck_a : refrat_planck_b
      ztabsize      = lay <= laytrop ? 8. : 4.

      speccomb    = colh2o[lay] + rat_h2oco2[lay]  *colco2[lay]
      speccomb1   = colh2o[lay] + rat_h2oco2_1[lay]*colco2[lay]
      speccomb_mn2o   = colh2o[lay] + refrat_m            *colco2[lay]
      speccomb_planck = colh2o[lay] + refrat_planck       *colco2[lay]

      specparm    = min(colh2o[lay]/speccomb,oneminus)
      specparm1   = min(colh2o[lay]/speccomb1,oneminus)
      specparm_mn2o   = min(colh2o[lay]/speccomb_mn2o,oneminus)
      specparm_planck = min(colh2o[lay]/speccomb_planck,oneminus)

      specmult        = ztabsize*specparm
      specmult1       = ztabsize*specparm1
      specmult_mn2o   = ztabsize*specparm_mn2o
      specmult_planck = ztabsize*specparm_planck

      fs    = specmult        - Int(floor(specmult))
      fs1   = specmult1       - Int(floor(specmult1))
      fmn2o = specmult_mn2o   - Int(floor(specmult_mn2o))
      fpl   = specmult_planck - Int(floor(specmult_planck))

      # mpuetz: this is never used ?
      # fmn2omf = minorfrac[lay]*fmn2o
      js    = Int(floor(specmult + 1.0))
      js1   = Int(floor(specmult1 + 1.0))
      jmn2o = Int(floor(specmult_mn2o + 1.0))
      jpl   = Int(floor(specmult_planck + 1.0))

      #  In atmospheres where the amount of N2O is too great to be considered
      #  a minor species, adjust the column amount of N2O by an empirical factor
      #  to obtain the proper contribution.
      chi_n2o = coln2o[lay]/col_dry_vr[lay]
      ratn2o = 1.e20*chi_n2o/chi_mls[4,jp[lay]+1] - 0.5
      adjcoln2o = coln2o[lay]
      if (ratn2o > 1.0)
        adjfac = 0.5 + ratn2o^0.65
        adjcoln2o = adjfac*chi_mls[4,jp[lay]+1]*col_dry_vr[lay]*1.e-20
      end
    end

    if iband in [4,5,12]
      speccomb, specparm, specmult, js, fs = spec_items(colh2o[lay],rat_h2oco2[lay],colco2[lay])
      speccomb1, specparm1, specmult1, js1, fs1 = spec_items(colh2o[lay],rat_h2oco2_1[lay],colco2[lay])
      speccomb_planck, specparm_planck, specmult_planck, jpl, fpl = spec_items(colh2o[lay],refrat_planck_a,colco2[lay])
      if iband == 5
        speccomb_mo3, specparm_mo3, specmult_mo3, jmo3, fmo3 = spec_items(colh2o[lay],refrat_m_a,colco2[lay])
      end
    end

    if iband == 7
      speccomb, specparm, specmult, js, fs = spec_items(colh2o[lay],rat_h2oo3[lay],colo3[lay])
      speccomb1, specparm1, specmult1, js1, fs1 = spec_items(colh2o[lay],rat_h2oo3_1[lay],colo3[lay])
      speccomb_mco2, specparm_mco2, specmult_mco2, jmco2, fmco2 = spec_items(colh2o[lay],refrat_m_a,colo3[lay])
      speccomb_planck, specparm_planck, specmult_planck, jpl, fpl = spec_items(colh2o[lay],refrat_planck_a,colo3[lay])
    end

    if iband in [9,16]
      speccomb, specparm, specmult, js, fs = spec_items(colh2o[lay],rat_h2och4[lay],colch4[lay])
      speccomb1, specparm1, specmult1, js1, fs1 = spec_items(colh2o[lay],rat_h2och4_1[lay],colch4[lay])
      speccomb_planck, specparm_planck, specmult_planck, jpl, fpl = spec_items(colh2o[lay],refrat_planck_a,colch4[lay])
      if iband == 9
        speccomb_mn2o, specparm_mn2o, specmult_mn2o, jmn2o, fmn2o = spec_items(colh2o[lay],refrat_m_a,colch4[lay])
      end
    end

    if iband == 13
      speccomb, specparm, specmult, js, fs = spec_items(colh2o[lay],rat_h2on2o[lay],coln2o[lay])
      speccomb1, specparm1, specmult1, js1, fs1 = spec_items(colh2o[lay],rat_h2on2o_1[lay],coln2o[lay])
      speccomb_mco2, specparm_mco2, specmult_mco2, jmco2, fmco2 = spec_items(colh2o[lay],refrat_m_a,coln2o[lay])
      speccomb_mco, specparm_mco, specmult_mco, jmco, fmco = spec_items(colh2o[lay],refrat_m_a3,coln2o[lay])
      speccomb_planck, specparm_planck, specmult_planck, jpl, fpl = spec_items(colh2o[lay],refrat_planck_a,coln2o[lay])
    end

    if iband == 15
      speccomb, specparm, specmult, js, fs = spec_items(coln2o[lay],rat_n2oco2[lay],colco2[lay])
      speccomb1, specparm1, specmult1, js1, fs1 = spec_items(coln2o[lay],rat_n2oco2_1[lay],colco2[lay])
      speccomb_mn2, specparm_mn2, specmult_mn2, jmn2, fmn2 = spec_items(coln2o[lay],refrat_m_a,colco2[lay])
      speccomb_planck, specparm_planck, specmult_planck, jpl, fpl = spec_items(coln2o[lay],refrat_planck_a,colco2[lay])
    end

    if iband in [3,4,5,7,9,12,13,15,16]
      specparm_solo = specparm
      specparm1_solo = specparm1
      fs_solo = fs
      fs1_solo = fs1
      if (0.125 < specparm_solo < 0.875)
        fac000 = (1. - fs_solo)*fac00[lay]
        fac010 = (1. - fs_solo)*fac10[lay]
        fac100 = fs_solo*fac00[lay]
        fac110 = fs_solo*fac10[lay]
      else
        fk0, fk1, fk2 = fks(specparm_solo < 0.125 ? fs_solo - 1.0 : -fs_solo)
        fac000 = fk0*fac00[lay]
        fac100 = fk1*fac00[lay]
        fac200 = fk2*fac00[lay]
        fac010 = fk0*fac10[lay]
        fac110 = fk1*fac10[lay]
        fac210 = fk2*fac10[lay]
      end

      if (0.125 < specparm1_solo < 0.875)
        fac001 = (1. - fs1_solo)*fac01[lay]
        fac011 = (1. - fs1_solo)*fac11[lay]
        fac101 = fs1_solo*fac01[lay]
        fac111 = fs1_solo*fac11[lay]
      else
        fk0, fk1, fk2 = fks(specparm1_solo < 0.125 ? fs1_solo - 1.0 : -fs1_solo)
        fac001 = fk0*fac01[lay]
        fac101 = fk1*fac01[lay]
        fac201 = fk2*fac01[lay]
        fac011 = fk0*fac11[lay]
        fac111 = fk1*fac11[lay]
        fac211 = fk2*fac11[lay]
      end
    end

    if !(iband in keys(refrat_planck_a_dict))
      ind0 = ((jp[lay]-1)*5+(jt[lay]-1))*nspa[iband] + 1
      ind1 = (jp[lay]*5+(jt1[lay]-1))*nspa[iband] + 1
    else
      ind0 = ((jp[lay]-1)*5+(jt[lay]-1))*nspa[iband] + js
      ind1 = (jp[lay]*5+(jt1[lay]-1))*nspa[iband] + js1
    end

    inds = indself[lay]
    indf = indfor[lay]
    if iband in [1, 2]
      pp = pm_fl_vr[lay]
      if iband == 1
        corradj =  1.0
        if (pp < 250.)
          corradj = 1. - 0.15 * (250. - pp) / 154.4
        end
      else
        corradj = 1. - .05 * (pp - 100.) / 900.
      end
    else
      corradj = 1.0
    end

    if iband == 3
      ng3 = ngc[iband]
      ind0 = Int(ind0)
      ind1 = Int(ind1)
      tau_major = speccomb * if (specparm < 0.125)
        (fac000 * absa[ind0,1:ng3] + fac100 * absa[ind0+1,1:ng3] + fac200 * absa[ind0+2,1:ng3] + fac010 * absa[ind0+9,1:ng3] + fac110 * absa[ind0+10,1:ng3] + fac210 * absa[ind0+11,1:ng3])
      elseif (specparm > 0.875)
        (fac200 * absa[ind0-1,1:ng3] + fac100 * absa[ind0,1:ng3] + fac000 * absa[ind0+1,1:ng3] + fac210 * absa[ind0+8,1:ng3] + fac110 * absa[ind0+9,1:ng3] + fac010 * absa[ind0+10,1:ng3])
      else
        (fac000 * absa[ind0,1:ng3] + fac100 * absa[ind0+1,1:ng3] + fac010 * absa[ind0+9,1:ng3] + fac110 * absa[ind0+10,1:ng3])
      end

      tau_major1 = speccomb1 * if (specparm1 < 0.125)
        (fac001 * absa[ind1,1:ng3] + fac101 * absa[ind1+1,1:ng3] + fac201 * absa[ind1+2,1:ng3] + fac011 * absa[ind1+9,1:ng3] + fac111 * absa[ind1+10,1:ng3] + fac211 * absa[ind1+11,1:ng3])
      elseif (specparm1 > 0.875)
        (fac201 * absa[ind1-1,1:ng3] + fac101 * absa[ind1,1:ng3] + fac001 * absa[ind1+1,1:ng3] + fac211 * absa[ind1+8,1:ng3] + fac111 * absa[ind1+9,1:ng3] + fac011 * absa[ind1+10,1:ng3])
      else
        (fac001 * absa[ind1,1:ng3] + fac101 * absa[ind1+1,1:ng3] + fac011 * absa[ind1+9,1:ng3] + fac111 * absa[ind1+10,1:ng3])
      end
    end

    if iband in [6,7,8,13]
      chi_co2 = colco2[lay]/(col_dry_vr[lay])
      ratco2 = 1.e20*chi_co2/chi_mls[2,jp[lay]+1]
      if (ratco2 > 3.0)
        if iband == 6
          adjfac = 2.0+(ratco2-2.0)^0.77
        elseif iband == 7
          adjfac = 3.0+(ratco2-3.0)^0.79
        elseif iband == 8
          adjfac = 2.0+(ratco2-2.0)^0.65
        elseif iband == 13
          adjfac = 2.0+(ratco2-2.0)^0.68
        end
        adjcolco2 = adjfac*chi_mls[2,jp[lay]+1]*col_dry_vr[lay]*1.e-20
      else
        adjcolco2 = colco2[lay]
      end
    end

    if iband == 9
      chi_n2o = coln2o[lay]/(col_dry_vr[lay])
      ratn2o = 1.e20*chi_n2o/chi_mls[4,jp[lay]+1]
      if (ratn2o > 1.5)
        adjfac = 0.5+(ratn2o-0.5)^0.65
        adjcoln2o = adjfac*chi_mls[4,jp[lay]+1]*col_dry_vr[lay]*1.e-20
      else
        adjcoln2o = coln2o[lay]
      end
    end

    iband in [1] && (scalen2 = colbrd[lay]*scaleminorn2[lay])
    iband in [15] && (scalen2 = colbrd[lay]*scaleminor[lay])
    iband in [11] && (scaleo2 = colo2[lay]*scaleminor[lay])
    iband in [1, 3, 5, 6, 7, 8, 9, 11, 13, 15] && (indm = indminor[lay])

    tauself = selffac_LW[lay] * (selfref[inds,:] + selffrac[lay] * (selfref[inds+1,:] - selfref[inds,:]))
    taufor = forfac_LW[lay] * (forref[indf,:] + forfrac[lay] * (forref[indf+1,:] -  forref[indf,:]))

    # if (iband == 15) && (lay == 8)
    #   println("tauself_15: ", tauself)
    #   println("taufor_15: ", taufor)
    # end

    if iband in [4,5,7,9,12,13,15,16]
      tau_major = speccomb * if (specparm < 0.125)
        (fac000 * absa[ind0,:] + fac100 * absa[ind0+1,:] + fac200 * absa[ind0+2,:] + fac010 * absa[ind0+9,:] + fac110 * absa[ind0+10,:] + fac210 * absa[ind0+11,:])
      elseif (specparm > 0.875)
        (fac200 * absa[ind0-1,:] + fac100 * absa[ind0,:] + fac000 * absa[ind0+1,:] + fac210 * absa[ind0+8,:] + fac110 * absa[ind0+9,:] + fac010 * absa[ind0+10,:])
      else
        (fac000 * absa[ind0,:] + fac100 * absa[ind0+1,:] + fac010 * absa[ind0+9,:] + fac110 * absa[ind0+10,:])
      end
      tau_major1 = speccomb1 * if (specparm1 < 0.125)
        (fac001 * absa[ind1,:] + fac101 * absa[ind1+1,:] + fac201 * absa[ind1+2,:] + fac011 * absa[ind1+9,:] + fac111 * absa[ind1+10,:] + fac211 * absa[ind1+11,:])
      elseif (specparm1 > 0.875)
        (fac201 * absa[ind1-1,:] + fac101 * absa[ind1,:] + fac001 * absa[ind1+1,:] + fac211 * absa[ind1+8,:] + fac111 * absa[ind1+9,:] + fac011 * absa[ind1+10,:])
      else
        (fac001 * absa[ind1,:] + fac101 * absa[ind1+1,:] + fac011 * absa[ind1+9,:] + fac111 * absa[ind1+10,:])
      end
      
      # if (iband == 15) && (lay == 8)
      #   println("tau_major_15: ", tau_major)
      #   println("tau_major1_15: ", tau_major1)
      # end
    end


    taug[lay,g_range] .= tauself + taufor
    if iband in [3]
      n2om1 = lw_params[iband]["ka_mn2o"][Int(jmn2o),indm,:] + fmn2o * (lw_params[iband]["ka_mn2o"][Int(jmn2o)+1,indm,:] - lw_params[iband]["ka_mn2o"][Int(jmn2o),indm,:])
      n2om2 = lw_params[iband]["ka_mn2o"][Int(jmn2o),indm+1,:] + fmn2o * (lw_params[iband]["ka_mn2o"][Int(jmn2o)+1,indm+1,:] - lw_params[iband]["ka_mn2o"][Int(jmn2o),indm+1,:])
      taug[lay,g_range] += adjcoln2o*(n2om1 + minorfrac[lay] * (n2om2 - n2om1))
    end

    if iband in [5]
      o3m1 = lw_params[iband]["ka_mo3"][jmo3,indm,:] + fmo3 * (lw_params[iband]["ka_mo3"][jmo3+1,indm,:]-lw_params[iband]["ka_mo3"][jmo3,indm,:])
      o3m2 = lw_params[iband]["ka_mo3"][jmo3,indm+1,:] + fmo3 * (lw_params[iband]["ka_mo3"][jmo3+1,indm+1,:]-lw_params[iband]["ka_mo3"][jmo3,indm+1,:])
      taug[lay,g_range] += (o3m1 + minorfrac[lay]*(o3m2-o3m1))*colo3[lay] + wx_vr[1,lay] * lw_params[iband]["ccl4"]
    end

    if iband in [6]
      taug[lay,g_range] += wx_vr[2,lay] * lw_params[iband]["cfc11adj"] + wx_vr[3,lay] * lw_params[iband]["cfc12"]
    end

    if iband in [6,8]
      absco2 = (lw_params[iband]["ka_mco2"][indm,:] + minorfrac[lay] * (lw_params[iband]["ka_mco2"][indm+1,:] - lw_params[iband]["ka_mco2"][indm,:]))
    end

    if iband in [7,13]
      co2m1 = lw_params[iband]["ka_mco2"][jmco2,indm,:] + fmco2 * (lw_params[iband]["ka_mco2"][jmco2+1,indm,:] - lw_params[iband]["ka_mco2"][jmco2,indm,:])
      co2m2 = lw_params[iband]["ka_mco2"][jmco2,indm+1,:] + fmco2 * (lw_params[iband]["ka_mco2"][jmco2+1,indm+1,:] - lw_params[iband]["ka_mco2"][jmco2,indm+1,:])
      absco2 = co2m1 + minorfrac[lay] * (co2m2 - co2m1)
    end

    if iband in [8]
      abso3 = (lw_params[iband]["ka_mo3"][indm,:] + minorfrac[lay] * (lw_params[iband]["ka_mo3"][indm+1,:] - lw_params[iband]["ka_mo3"][indm,:]))
      absn2o = (lw_params[iband]["ka_mn2o"][indm,:] + minorfrac[lay] * (lw_params[iband]["ka_mn2o"][indm+1,:] - lw_params[iband]["ka_mn2o"][indm,:]))
      taug[lay,g_range] += colo3[lay] * abso3 + coln2o[lay] * absn2o + wx_vr[3,lay] * lw_params[iband]["cfc12"] + wx_vr[4,lay] * lw_params[iband]["cfc22adj"]
    end

    if iband in [9]
      n2om1 = lw_params[iband]["ka_mn2o"][jmn2o,indm,:] + fmn2o * (lw_params[iband]["ka_mn2o"][jmn2o+1,indm,:] - lw_params[iband]["ka_mn2o"][jmn2o,indm,:])
      n2om2 = lw_params[iband]["ka_mn2o"][jmn2o,indm+1,:] + fmn2o * (lw_params[iband]["ka_mn2o"][jmn2o+1,indm+1,:] - lw_params[iband]["ka_mn2o"][jmn2o,indm+1,:])
      taug[lay,g_range] += adjcoln2o*(n2om1 + minorfrac[lay] * (n2om2 - n2om1))
    end

    if iband in [11]
      taug[lay,g_range] += scaleo2 * (lw_params[iband]["ka_mo2"][indm,:] + minorfrac[lay] * (lw_params[iband]["ka_mo2"][indm+1,:] - lw_params[iband]["ka_mo2"][indm,:]))
    end

    if iband in [13]
      com1 = lw_params[iband]["ka_mco"][jmco,indm,:] + fmco * (lw_params[iband]["ka_mco"][jmco+1,indm,:] - lw_params[iband]["ka_mco"][jmco,indm,:])
      com2 = lw_params[iband]["ka_mco"][jmco,indm+1,:] + fmco * (lw_params[iband]["ka_mco"][jmco+1,indm+1,:] - lw_params[iband]["ka_mco"][jmco,indm+1,:])
      taug[lay,g_range] += colco[lay]*(com1 + minorfrac[lay] * (com2 - com1))
    end

    if iband in [1,15]
      taun2 = scalen2 * if iband == 1
        (lw_params[iband]["ka_mn2"][indm,:] + minorfrac[lay] * (lw_params[iband]["ka_mn2"][indm+1,:] - lw_params[iband]["ka_mn2"][indm,:]))
      elseif iband == 15
        n2m1 = lw_params[iband]["ka_mn2"][jmn2,indm,:] + fmn2 * (lw_params[iband]["ka_mn2"][jmn2+1,indm,:] - lw_params[iband]["ka_mn2"][jmn2,indm,:])
        n2m2 = lw_params[iband]["ka_mn2"][jmn2,indm+1,:] + fmn2 * (lw_params[iband]["ka_mn2"][jmn2+1,indm+1,:] - lw_params[iband]["ka_mn2"][jmn2,indm+1,:])
        (n2m1 + minorfrac[lay] * (n2m2 - n2m1))
      end
      # if (iband == 15) && (lay == 8)
      #   println("taun2_15: ", taun2)
      # end
      taug[lay,g_range] += taun2
    end

    iband in [6,7,8,13] && (taug[lay,g_range] += adjcolco2*absco2)

    if iband in [1,2,6,8,10,11,14]
      taug[lay,g_range] += (iband == 14 ? colco2[lay] : colh2o[lay]) * (fac00[lay] * absa[ind0,:] + fac10[lay] * absa[ind0+1,:] + fac01[lay] * absa[ind1,:] + fac11[lay] * absa[ind1+1,:])
      fracs[lay,g_range] = fracrefa
    elseif iband == 3
      taug[lay,g_range] += tau_major + tau_major1
      fracs[lay,g_range] = fracrefa[Int(jpl),:] + fpl * (fracrefa[Int(jpl)+1,:]-fracrefa[Int(jpl),:])
    else
      taug[lay,g_range] += tau_major + tau_major1
      fracs[lay,g_range] = fracrefa[jpl,:] + fpl * (fracrefa[jpl+1,:]-fracrefa[jpl,:])
    end

    taug[lay,g_range] *= corradj
  end

    # Upper atmosphere loop (1,2,3,4,5,NOT 6,7,8,9,10,11,NOT 12,13,14,NOT 15,16)
  if !(iband in [6, 12, 15])
    for lay in (laytrop+1):nlay
      if iband == 3
        # computing the interpolation lookup indices and fractions
        # is (almost) independent of lower/higher atmosphere flag


        # If the value of X is greater than or equal to zero, then the value of Y is returned. If the value of X is smaller than zero or is a NaN, then the value of Z is returned.

        # mpuetz: select correct factors depending on lower/higher atmosphere flag
        # zlaytrop is of type REAL, so fast pipelined fsel instructions can be used
        # had to use FSEL intrinsic here to get XLF not to do gather/scatter

        refrat_m      = lay <= laytrop ? refrat_m_a : refrat_m_b
        refrat_planck = lay <= laytrop ? refrat_planck_a : refrat_planck_b
        ztabsize      = lay <= laytrop ? 8. : 4.

        speccomb    = colh2o[lay] + rat_h2oco2[lay]  *colco2[lay]
        speccomb1   = colh2o[lay] + rat_h2oco2_1[lay]*colco2[lay]
        speccomb_mn2o   = colh2o[lay] + refrat_m            *colco2[lay]
        speccomb_planck = colh2o[lay] + refrat_planck       *colco2[lay]

        specparm    = min(colh2o[lay]/speccomb,oneminus)
        specparm1   = min(colh2o[lay]/speccomb1,oneminus)
        specparm_mn2o   = min(colh2o[lay]/speccomb_mn2o,oneminus)
        specparm_planck = min(colh2o[lay]/speccomb_planck,oneminus)

        specmult        = ztabsize*specparm
        specmult1       = ztabsize*specparm1
        specmult_mn2o   = ztabsize*specparm_mn2o
        specmult_planck = ztabsize*specparm_planck

        fs    = specmult        - Int(floor(specmult))
        fs1   = specmult1       - Int(floor(specmult1))
        fmn2o = specmult_mn2o   - Int(floor(specmult_mn2o))
        fpl   = specmult_planck - Int(floor(specmult_planck))

        # mpuetz: this is never used ?
        # fmn2omf = minorfrac[lay]*fmn2o
        js    = Int(floor(specmult + 1.0))
        js1   = Int(floor(specmult1 + 1.0))
        jmn2o = Int(floor(specmult_mn2o + 1.0))
        jpl   = Int(floor(specmult_planck + 1.0))

        #  In atmospheres where the amount of N2O is too great to be considered
        #  a minor species, adjust the column amount of N2O by an empirical factor
        #  to obtain the proper contribution.
        chi_n2o = coln2o[lay]/col_dry_vr[lay]
        ratn2o = 1.e20*chi_n2o/chi_mls[4,jp[lay]+1] - 0.5
        adjcoln2o = coln2o[lay]
        if (ratn2o > 1.0)
          adjfac = 0.5 + ratn2o^0.65
          adjcoln2o = adjfac*chi_mls[4,jp[lay]+1]*col_dry_vr[lay]*1.e-20
        end
      end


      iband in [1,2,3,10,11] && (indf = indfor[lay])
      iband in [1,3,7,8,9,11,13] && (indm = indminor[lay])

      if iband == 1
        pp = pm_fl_vr[lay]
        corradj =  1. - 0.15 * (pp / 95.6)
        scalen2 = colbrd[lay] * scaleminorn2[lay]
      else
        corradj = 1.
      end

      if iband in [4,5]
        speccomb = colo3[lay] + rat_o3co2[lay]*colco2[lay]
        specparm = colo3[lay]/speccomb
        if (specparm >= oneminus)
          specparm = oneminus
        end
        specmult = 4.*(specparm)
        js = 1 + fortran_int(specmult)
        fs = specmult % 1.0

        speccomb1 = colo3[lay] + rat_o3co2_1[lay]*colco2[lay]
        specparm1 = colo3[lay]/speccomb1
        if (specparm1 >= oneminus)
          specparm1 = oneminus
        end
        specmult1 = 4.*(specparm1)
        js1 = 1 + fortran_int(specmult1)
        fs1 = specmult1 % 1.0

        speccomb_planck = colo3[lay]+refrat_planck_b*colco2[lay]
        specparm_planck = colo3[lay]/speccomb_planck
        if (specparm_planck >= oneminus)
          specparm_planck=oneminus
        end
        specmult_planck = 4.*specparm_planck
        jpl= 1 + fortran_int(specmult_planck)
        fpl = specmult_planck % 1.0
      end

      if iband in [1,2,7,8,9,10,11,14,16]
        ind0 = ((jp[lay]-13)*5+(jt[lay]-1))*nspb[iband] + 1
        ind1 = ((jp[lay]-12)*5+(jt1[lay]-1))*nspb[iband] + 1
      end

      if iband in [4,5]
        ind0 = ((jp[lay]-13)*5+(jt[lay]-1))*nspb[iband] + js
        ind1 = ((jp[lay]-12)*5+(jt1[lay]-1))*nspb[iband] + js1
      end

      if iband in [3,4,5]
        fs_solo = fs
        fs1_solo = fs1
        fac000 = (1. - fs_solo) * fac00[lay]
        fac010 = (1. - fs_solo) * fac10[lay]
        fac100 = fs_solo * fac00[lay]
        fac110 = fs_solo * fac10[lay]
        fac001 = (1. - fs1_solo) * fac01[lay]
        fac011 = (1. - fs1_solo) * fac11[lay]
        fac101 = fs1_solo * fac01[lay]
        fac111 = fs1_solo * fac11[lay]
      end

      if iband == 3
        ind0 = Int(((jp[lay]-13)*5+(jt[lay]-1))*nspb[iband] + js)
        ind1 = Int(((jp[lay]-12)*5+(jt1[lay]-1))*nspb[iband] + js1)
        ng3 = ngc[iband]
        tau_major = speccomb * (fac000 * absb[ind0,1:ng3] + fac100 * absb[ind0+1,1:ng3] + fac010 * absb[ind0+5,1:ng3] + fac110 * absb[ind0+6,1:ng3])
        tau_major1 = speccomb1 * (fac001 * absb[ind1,1:ng3] + fac101 * absb[ind1+1,1:ng3] + fac011 * absb[ind1+5,1:ng3] + fac111 * absb[ind1+6,1:ng3])
      end

      if iband in [7,8,9]
        #  In atmospheres where the amount of CO2 is too great to be considered
        #  a minor species, adjust the column amount of CO2 by an empirical factor
        #  to obtain the proper contribution.
        chi_co2 = colco2[lay]/(col_dry_vr[lay])
        ratco2 = 1.e20*chi_co2/chi_mls[2,jp[lay]+1]
        if (ratco2 > 3.0)
          if iband == 7
            adjfac = 2.0+(ratco2-2.0)^0.79
          elseif iband == 8
            adjfac = 2.0+(ratco2-2.0)^0.65
          elseif iband == 9
            adjfac = 0.5+(ratco2-0.5)^0.65
          end
          adjcolco2 = adjfac*chi_mls[2,jp[lay]+1]*col_dry_vr[lay]*1.e-20
        else
          adjcolco2 = colco2[lay]
        end
      end

      if iband == 9
        #  In atmospheres where the amount of N2O is too great to be considered
        #  a minor species, adjust the column amount of N2O by an empirical factor
        #  to obtain the proper contribution.
        chi_n2o = coln2o[lay]/(col_dry_vr[lay])
        ratn2o = 1.e20*chi_n2o/chi_mls[4,jp[lay]+1]
        if (ratn2o > 1.5)
          adjfac = 0.5+(ratn2o-0.5)^0.65
          adjcoln2o = adjfac*chi_mls[4,jp[lay]+1]*col_dry_vr[lay]*1.e-20
        else
          adjcoln2o = coln2o[lay]
        end
      end

      if iband == 11
        scaleo2 = colo2[lay]*scaleminor[lay]
      end

      if iband in [1,2,7,8,9,10,11,13,14,16]
        fracs[lay,g_range] = fracrefb
      elseif iband == 3
        fracs[lay,g_range] = fracrefb[Int(jpl),:] + fpl * (fracrefb[Int(jpl)+1,:]-fracrefb[Int(jpl),:])
      else
        fracs[lay,g_range] = fracrefb[jpl,:] + fpl * (fracrefb[jpl+1,:]-fracrefb[jpl,:])
      end

      if iband in [1]
        taun2 = scalen2*(lw_params[iband]["kb_mn2"][indm,:] + minorfrac[lay] * (lw_params[iband]["kb_mn2"][indm+1,:] - lw_params[iband]["kb_mn2"][indm,:]))
        taug[lay,g_range] += taun2
      end

      if iband in [1,2,3,10,11]
        taufor = forfac_LW[lay] * (forref[indf,:] + forfrac[lay] * (forref[indf+1,:] - forref[indf,:]))
        taug[lay,g_range] += taufor
      end

      if iband in [1,2,10,11]
        tauh2o = (colh2o[lay] * (fac00[lay] * absb[ind0,:] + fac10[lay] * absb[ind0+1,:] + fac01[lay] * absb[ind1,:] + fac11[lay] * absb[ind1+1,:]))
        taug[lay,g_range] += tauh2o
      end

      if iband == 3
        n2om1 = lw_params[iband]["kb_mn2o"][Int(jmn2o),indm,:] + fmn2o * (lw_params[iband]["kb_mn2o"][Int(jmn2o)+1,indm,:]-lw_params[iband]["kb_mn2o"][Int(jmn2o),indm,:])
        n2om2 = lw_params[iband]["kb_mn2o"][Int(jmn2o),indm+1,:] + fmn2o * (lw_params[iband]["kb_mn2o"][Int(jmn2o)+1,indm+1,:]-lw_params[iband]["kb_mn2o"][Int(jmn2o),indm+1,:])
        absn2o = n2om1 + minorfrac[lay] * (n2om2 - n2om1)
        taug[lay,g_range] += tau_major + tau_major1 + adjcoln2o*absn2o
      end

      if iband in [4,5]
        taug[lay,g_range] += speccomb * (fac000 * absb[ind0,:] + fac100 * absb[ind0+1,:] + fac010 * absb[ind0+5,:] + fac110 * absb[ind0+6,:]) + speccomb1 * (fac001 * absb[ind1,:] + fac101 * absb[ind1+1,:] + fac011 * absb[ind1+5,:] + fac111 * absb[ind1+6,:])
      end

      if iband == 5
        taug[lay,g_range] += wx_vr[1,lay] * lw_params[iband]["ccl4"]
      end

      if iband in [7,8]
        absco2 = lw_params[iband]["kb_mco2"][indm,:] + minorfrac[lay] * (lw_params[iband]["kb_mco2"][indm+1,:] - lw_params[iband]["kb_mco2"][indm,:])
        taug[lay,g_range] += adjcolco2 * absco2 + colo3[lay] * (fac00[lay] * absb[ind0,:] + fac10[lay] * absb[ind0+1,:] + fac01[lay] * absb[ind1,:] + fac11[lay] * absb[ind1+1,:])
      end

      if iband in [8,9]
        absn2o = lw_params[iband]["kb_mn2o"][indm,:] + minorfrac[lay] * (lw_params[iband]["kb_mn2o"][indm+1,:] - lw_params[iband]["kb_mn2o"][indm,:])
      end

      if iband == 8
        taug[lay,g_range] += coln2o[lay]*absn2o + wx_vr[3,lay] * lw_params[iband]["cfc12"] + wx_vr[4,lay] * lw_params[iband]["cfc22adj"]
      end

      if iband in [9, 16]
        taug[lay,g_range] += colch4[lay] * (fac00[lay] * absb[ind0,:] + fac10[lay] * absb[ind0+1,:] + fac01[lay] * absb[ind1,:] + fac11[lay] * absb[ind1+1,:])
      end

      if iband == 9
        taug[lay,g_range] += adjcoln2o*absn2o
      end

      if iband == 11
        tauo2 = scaleo2 * (lw_params[iband]["kb_mo2"][indm,:] + minorfrac[lay] * (lw_params[iband]["kb_mo2"][indm+1,:] - lw_params[iband]["kb_mo2"][indm,:]))
        taug[lay,g_range] += tauo2
      end

      if iband == 13
        abso3 = lw_params[iband]["kb_mo3"][indm,:] + minorfrac[lay] * (lw_params[iband]["kb_mo3"][indm+1,:] - lw_params[iband]["kb_mo3"][indm,:])
        taug[lay,g_range] += colo3[lay]*abso3
      end

      if iband == 14
        taug[lay,g_range] += colco2[lay] * (fac00[lay] * absb[ind0,:] + fac10[lay] * absb[ind0+1,:] + fac01[lay] * absb[ind1,:] + fac11[lay] * absb[ind1+1,:])
      end

      @. taug[lay,g_range] *= corradj
    end
  end

  # just in 4, 7
  if iband == 4
    for lay in (laytrop+1):nlay
      ngs3 = sum(ngc[1:(iband-1)])
      taug[lay,ngs3+8]=taug[lay,ngs3+8]*0.92
      taug[lay,ngs3+9]=taug[lay,ngs3+9]*0.88
      taug[lay,ngs3+10]=taug[lay,ngs3+10]*1.07
      taug[lay,ngs3+11]=taug[lay,ngs3+11]*1.1
      taug[lay,ngs3+12]=taug[lay,ngs3+12]*0.99
      taug[lay,ngs3+13]=taug[lay,ngs3+13]*0.88
      taug[lay,ngs3+14]=taug[lay,ngs3+14]*0.943
    end
  end

  if iband == 7
    for lay in (laytrop+1):nlay
      ngs6 = sum(ngc[1:(iband-1)])
      taug[lay,ngs6+6]=taug[lay,ngs6+6]*0.92
      taug[lay,ngs6+7]=taug[lay,ngs6+7]*0.88
      taug[lay,ngs6+8]=taug[lay,ngs6+8]*1.07
      taug[lay,ngs6+9]=taug[lay,ngs6+9]*1.1
      taug[lay,ngs6+10]=taug[lay,ngs6+10]*0.99
      taug[lay,ngs6+11]=taug[lay,ngs6+11]*0.855
    end
  end

  # just in 6, 12, 15
  if iband == 6
    for lay in (laytrop+1):nlay
      taug[lay,g_range] = wx_vr[2,lay] * lw_params[iband]["cfc11adj"] + wx_vr[3,lay] * lw_params[iband]["cfc12"]
      fracs[lay,g_range] = fracrefa
    end
  end
  
  # println(taug[8,:])
  # println("Optical depth for LW band ", iband, " calculated.")
end

function LW_radiative_transfer_per_band(nlay,nlev,iband,ncbands,planklay,planklev,secdiff,taut,fracs,odcld,cloudy,istcld,istcldd,cld_frc_vr,facclr1,facclr2,faccld1,faccld2,faccmb1,faccmb2,facclr1d,facclr2d,faccld1d,faccld2d,faccmb1d,faccmb2d,plankbnd,zsemiss,flx_uplw_vr,flx_dnlw_vr,flx_uplw_clr_vr,flx_dnlw_clr_vr)
  # println("Calculating LW radiative transfer for LW band ", iband , "...")
  pade   = 0.278
  bpade = 1.0 / pade
  tblint = 10000.0
  ipat = reshape([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,3,3,3,4,4,4,5,5,5,5,5,5,5,5,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],(16,3))
  rec_6 = 0.166667
  ngc = [10,12,16,14,16,8,12,8,12,6,8,8,4,2,2,2]

  tau_tbl = LW_parameters["tau_tbl"]
  exp_tbl = LW_parameters["exp_tbl"]
  tfn_tbl = LW_parameters["tfn_tbl"]

  drad = zeros(nlev);
  urad = zeros(nlev);
  clrdrad = zeros(nlev);
  clrurad = zeros(nlev);
  
  g_range = (sum(ngc[1:(iband-1)])+1):cumsum(ngc)[iband]

  cldradd = 0.0 # zeros(ntime);
  clrradd = 0.0 # zeros(ntime);
  cldradu = 0.0 # zeros(ntime);
  clrradu = 0.0 # zeros(ntime);

  for igc in g_range

    # confused about cloud variables? check it out:
    # icldidx 1D array, has cloudy indices
    # iclridx 1D array, has non-cloudy indices
    # iclddn 3D array, traces if you've passed through a cloud yet
    # istcld 4D (nlev), currently unkonwn, but perhaps a cloud overlap var
    # istcldd 4D (nlev), currently unknown, but perhaps a cloud overlap var
    radld = 0.0 # zeros(ntime);
    radclrd = 0.0 # zeros(ntime);
    rad = 0.0 # zeros(ntime);
    bbd = 0.0 # zeros(ntime);
    bbdtot = 0.0 # zeros(ntime);
    gassrc = 0.0 # zeros(ntime);
    zodcld = 0.0 # zeros(ntime);
    odtot = 0.0 # zeros(ntime);
    atrans = zeros(nlay);
    atot = zeros(nlay);
    bbugas = zeros(nlay);
    bbutot = zeros(nlay);
    iclddn = false # have you touched a cloud at some point in your trajectory gone down, then gone back up?

    function do_cloudy_step(lev,iband,igc,bbd_coeff,bbdtot_coeff,blay,dplankup,dplankdn,odepth)
      plfrac = fracs[lev,igc]
      blay = planklay[lev,iband]
      bbd   = plfrac * (blay + bbd_coeff * dplankdn)
      bbdtot = plfrac * (blay + bbdtot_coeff * dplankdn)
      gassrc = atrans[lev] * bbd
      bbugas[lev] = plfrac * (blay + bbd_coeff * dplankup)
      bbutot[lev] = plfrac * (blay + bbdtot_coeff * dplankup)
    end

    # Clear layer
    function do_clear_step(lev,iband,igc,bbd_coeff,blay,dplankup,dplankdn,odepth,taut)
      plfrac = fracs[lev,igc]
      blay  = planklay[lev,iband]
      bbd = plfrac * (blay + dplankdn * bbd_coeff)
      bbugas[lev] = plfrac * (blay + bbd_coeff * dplankup)
      radld = radld + (bbd-radld)*atrans[lev]
      # if lev == 2 && igc == 69
      #   println("bbd: ",bbd)
      #   println("atrans[lev]: ",atrans[lev])
      #   println("radld: ", radld)
      # end
      drad[lev] = drad[lev] + radld
    end

    # First, go downwards, and as you do, calculate downwards radiative flux
    for lev = nlay:-1:1
      blay = planklay[lev,iband]
      dplankup = planklev[lev+1,iband] - blay
      dplankdn = planklev[lev,iband] - blay
      odepth = @. max(0.0,secdiff[iband] * taut[lev,igc])
      zodcld = ncbands == 16 ? odcld[lev,iband] : odcld[lev,1]
      
      # Cloudy layer
      if cloudy[lev]
        # println("here!")
        iclddn = true
        odtot = odepth + zodcld
        bbd_coeff,bbdtot_coeff = if (odtot < 0.06)
          atrans[lev] = odepth - 0.5 * odepth * odepth
          atot[lev] = odtot - 0.5 * odtot * odtot
          rec_6 * odepth, rec_6 * odtot
        elseif (odepth <= 0.06)
          odtot = odepth + zodcld
          ittot = fortran_int(tblint * odtot/(bpade+odtot) + 0.5)
          atrans[lev] = odepth - 0.5 * odepth*odepth
          atot[lev] = 1. - exp_tbl[ittot+1]

          rec_6 * odepth, tfn_tbl[ittot+1]
        else
          itgas = fortran_int(tblint*odepth/(bpade+odepth) + 0.5)
          odepth = tau_tbl[itgas+1]
          odtot = odepth + zodcld
          ittot = fortran_int(tblint*odtot/(bpade+odtot) + 0.5)
          atrans[lev] = 1. - exp_tbl[itgas+1]
          atot[lev] = 1. - exp_tbl[ittot+1]

          tfn_tbl[itgas+1],tfn_tbl[ittot+1]
        end

        do_cloudy_step(lev,iband,igc,bbd_coeff,bbdtot_coeff,blay,dplankup,dplankdn,odepth)

        if istcldd[lev+1]
          cldradd = cld_frc_vr[lev] * radld
          clrradd = radld - cldradd
          # if iband == 1 && igc == 1
          #   println("lev: ", lev)
          #   println("at istcldd thing")
          # end
          rad = 0.
        end

        ttot = 1. - atot[lev]
        cldsrc = bbdtot * atot[lev]
        cldradd = cldradd * ttot + cld_frc_vr[lev] * cldsrc
        clrradd = clrradd * (1. - atrans[lev]) + (1. - cld_frc_vr[lev]) * gassrc
        radld = cldradd + clrradd
        drad[lev] = drad[lev] + radld

        radmod = rad * (facclr1d[lev] * (1. - atrans[lev]) + faccld1d[lev] * ttot) - faccmb1d[lev] * gassrc + faccmb2d[lev] * cldsrc
        oldcld = cldradd - radmod
        oldclr = clrradd + radmod
        rad = -radmod + facclr2d[lev]*oldclr - faccld2d[lev]*oldcld
        cldradd = cldradd + rad
        clrradd = clrradd - rad
      else
        coeff = if (odepth <= 0.06)
          atrans[lev] = odepth - 0.5 * odepth * odepth
          rec_6*odepth
        else
          itr = fortran_int(tblint*odepth/(bpade+odepth)+0.5)
          atrans[lev] = 1. - exp_tbl[itr+1]
          tfn_tbl[itr+1]
        end

        do_clear_step(lev,iband,igc,coeff,blay,dplankup,dplankdn,odepth,taut)
      end


      #  Set clear sky stream to total sky stream as long as layers
      #  remain clear.  Streams diverge when a cloud is reached (iclddn=1),
      #  and clear sky stream must be computed separately from that point.
      if iclddn
        radclrd = radclrd + (bbd-radclrd) * atrans[lev]
        clrdrad[lev] = clrdrad[lev] + radclrd
      else
        radclrd = radld
        clrdrad[lev] = drad[lev]
      end
    end

    # Next, do the surface...

    # Spectral emissivity  reflectance
    #  Include the contribution of spectrally varying longwave emissivity
    #  and reflection from the surface to the upward radiative transfer.
    #  Note: Spectral and Lambertian reflection are identical for the
    #  diffusivity angle flux integration used here.
    #  Note: The emissivity is applied to plankbnd and dplankbnd_dt when
    #  they are defined in subroutine setcoef.
    rad0 = fracs[1,igc] * plankbnd[iband]
    #  Add in reflection of surface downward radiance.
    reflect = 1. - zsemiss
    radlu = rad0 + reflect * radld
    radclru = rad0 + reflect * radclrd
    urad[1] += radlu
    clrurad[1] += radclru

    # And then, head back up!
    for lev = 1:nlay
      if cloudy[lev]
        gassrc = bbugas[lev] * atrans[lev]
        if istcld[lev]
          cldradu = cld_frc_vr[lev] * radlu
          clrradu = radlu - cldradu
          rad = 0.
        end
        ttot = 1. - atot[lev]
        cldsrc = bbutot[lev] * atot[lev]
        cldradu = cldradu * ttot + cld_frc_vr[lev] * cldsrc
        clrradu = clrradu * (1.0 - atrans[lev]) + (1. - cld_frc_vr[lev]) * gassrc
        # if (lev == 8) && (iband == 1)
        #   println("atot: ", atot[lev])
        #   println("bbutot: ", bbutot[lev])
        #   println("atrans: ", atrans[lev])
        #   println("gassrc: ", gassrc)
        #   println("cldradu: ", cldradu)
        #   println("clrradu: ", clrradu)
        # end
        # Total sky radiance
        radlu = cldradu + clrradu
        urad[lev+1] = urad[lev+1] + radlu
        radmod = rad * (facclr1[lev+1]*(1.0 - atrans[lev]) + faccld1[lev+1] * ttot) - faccmb1[lev+1] * gassrc + faccmb2[lev+1] * cldsrc
        oldcld = cldradu - radmod
        oldclr = clrradu + radmod
        rad = -radmod + facclr2[lev+1]*oldclr - faccld2[lev+1]*oldcld
        cldradu = cldradu + rad
        clrradu = clrradu - rad
      else # Clear layer
        radlu = radlu + (bbugas[lev]-radlu)*atrans[lev]
        urad[lev+1] = urad[lev+1] + radlu
      end

      #  Set clear sky stream to total sky stream as long as all layers
      #  are clear (iclddn=0).  Streams must be calculated separately at
      #  all layers when a cloud is present (iclddn=1), because surface
      #  reflectance is different for each stream.
      if iclddn
        radclru += (bbugas[lev]-radclru)*atrans[lev]
        # if (lev == 8)
        #   println("bbugas[lev] ", bbugas[lev])
        #   println("atrans[lev] ", atrans[lev])
        #   println("radclru ", radclru)
        # end
        clrurad[lev+1] += radclru
      else
        radclru = radlu
        clrurad[lev+1] = urad[lev+1]
      end
    end
    # throw("oops")
  end

  # Process longwave output from band.
  # Calculate upward, downward, and net flux.

  wtdiff = 0.5


  # println("drad[1]: ", drad[1])

  flx_uplw_vr     += urad * wtdiff * delwave[iband]
  flx_dnlw_vr     += drad * wtdiff * delwave[iband]
  flx_uplw_clr_vr += clrurad * wtdiff * delwave[iband]
  flx_dnlw_clr_vr += clrdrad * wtdiff * delwave[iband]

  # println("Radiative transfer for LW band ", iband , " calculated.")
  # End spectral band loop
  flx_uplw_vr,flx_dnlw_vr,flx_uplw_clr_vr,flx_dnlw_clr_vr
end

function SW(nlay,nlev,solar_constant,cos_mu0m,alb,cld_frc_vr,jp,jt,jt1,laytrop,colh2o,colco2,colo3,coln2o,colco,colch4,colo2,colmol,co2mult,selffac,forfac,indfor,forfrac,indself,selffrac,fac10,fac00,fac11,fac01,aer_tau_sw_vr,aer_cg_sw_vr,aer_piz_sw_vr,cld_tau_sw_vr,cld_cg_sw_vr,cld_piz_sw_vr)
  # println("Calculating SW radiation...")
  nir_vis_boundary = 14500.
  par_lower_boundary = 14285.7143 # equivalent to
  par_upper_boundary = 25000.     # equivalent to
  i_overlap = 1
  ndaylen = 86400
  rcday      = ndaylen*g/3.5/rd
  zeps       = 1.e-06
  frc_vis = max.(0.0, min.(1.0, (wavenum2 - nir_vis_boundary) ./ delwave))
  frc_nir = 1.0 - frc_vis


  ssi_preind = [11.95005, 20.14612, 23.40302, 22.09443, 55.41679, 102.512, 24.69536, 347.4719, 217.2217, 343.2816, 129.3001, 47.07624, 3.130199, 13.17521]
  ssi_default = [12.1095682699999987, 20.3650825429849398, 23.7297328613475429, 22.4276934179066849, 55.6266126199999960, 1.0293153082385436E+02, 24.2936128100154392, 3.4574251380000004E+02, 2.1818712729860400E+02, 3.4719231470000005E+02, 1.2949501812000000E+02, 50.1522503011060650, 3.0799387010047838, 12.8893773299999985 ]
  solcm = 1360.874719
  ssi_preind /= solcm
  #   --- weight radiation within a band for the solar cycle ---
  #   psctm contains TSI (the "solar constant") scaled with the
  #   Sun-Earth distance. ssi_factor contains the relative contribution
  #   of each band to TSI. ssi_default is the (originally only
  #   implicitly defined) solar flux in the 14 bands.
  # bnd_wght(:) = psctm * ssi_factor(:) / ssi_default(:)
  bnd_wght = @. solar_constant * ssi_preind / ssi_default
  #--hs

  pfuvf = 0.
  zclear = 1.
  zcloud = 0.
  ztotcc = 0.

  for jk = 1:nlay
    zclear *= (1.0-max(cld_frc_vr[jk],zcloud))/(1.0-min(zcloud,1.0-zeps))
    zcloud = cld_frc_vr[jk]
    ztotcc = 1.0-zclear
  end

  # cloud overlap, assuming max-ran (i_overlap above)
  zcloud = 1 - zclear

  jpb1 = 16
  jpb2 = jpb1 + nb_sw - 1
  zbbcu = zeros(nlev,nb_sw)
  zbbcd = zeros(nlev,nb_sw)
  zbbfu = zeros(nlev,nb_sw)
  zbbfd = zeros(nlev,nb_sw)

  zincflux = 0.0
  zinctot = 0.0
  zrmu0i = 1.0 ./ cos_mu0m

  for iband = jpb1:jpb2
   ztaur,ztaug,zsflxzen = SW_optical_depth_per_band(iband,nlay,nlev,fac00,fac01,fac10,fac11,jp,jt,jt1,colh2o,colco2,colch4,colo3,colo2,colmol,laytrop,selffac,selffrac,indself,forfac,forfrac,indfor)
   SW_radiative_transfer_per_band(iband,nlay,nlev,ztaur,ztaug,zsflxzen,alb,zbbfd,zbbfu, zbbcd, zbbcu, cos_mu0m, zrmu0i, zincflux, zinctot, aer_tau_sw_vr,aer_cg_sw_vr,aer_piz_sw_vr,cld_tau_sw_vr,cld_cg_sw_vr,cld_piz_sw_vr,cld_frc_vr)
  end

  band_weighted_sum(fluxes) = squeeze(sum(reshape(bnd_wght,(1,nb_sw)) .* fluxes,2),2)
  flxu_sw = band_weighted_sum(zcloud .* zbbfu + zclear .* zbbcu)
  flxd_sw = band_weighted_sum(zcloud .* zbbfd + zclear .* zbbcd)
  flxu_sw_clr = band_weighted_sum(zbbcu)
  flxd_sw_clr = band_weighted_sum(zbbcd)

  # println("SW radiation calculated.")
  flxd_sw,flxu_sw,flxd_sw_clr,flxu_sw_clr
end

function SW_optical_depth_per_band(iband,nlay,nlev,fac00,fac01,fac10,fac11,jp,jt,jt1,colh2o,colco2,colch4,colo3,colo2,colmol,laytrop,selffac,selffrac,indself,forfac,forfrac,indfor)
  # println("Calculating optical depth for SW band ", iband, "...")
  
  ngc = [6, 12, 8, 8, 10, 10, 2, 10, 8, 6, 6, 8, 6, 12]
  nspa = [9, 9, 9, 9, 1, 9, 9, 1, 9, 1, 0, 1, 9, 1]
  nspb = [1, 5, 1, 1, 1, 5, 1, 0, 1, 0, 0, 1, 5, 1]
  repclc = 1.e-12
  ibm = iband-15
  ng = ngc[ibm]

  ztaur = zeros(nlay,ng) # output from taumol
  ztaug = zeros(nlay,ng) # output from taumol
  zsflxzen = zeros(ng) # output from taumol

  sw_dict = SW_parameters["sw_params"][iband-15]
  #-- for each band:computes the gaseous and Rayleigh optical thickness
  #  for all g-points within the band

  # here it is
  # 20 trop, p_taur[iplon,i_lay,ig] = z_tauray???

  laysolfr = iband in [16,17,27,28,29] ? nlay : laytrop

  z_o2adj = 1.6 # The following factor is the ratio of total O2 band intensity (lines and Mate continuum) to O2 band intensity (line only).  It is needed to adjust the optical depths since the k's include only lines.

  function make_zs(p1,p2,firstfac,fact,lay)
    z_speccomb = p1[lay] + fact*p2[lay]
    z_specmult = firstfac*min(oneminus,p1[lay] / z_speccomb)
    z_speccomb,z_specmult,1+fortran_int(z_specmult),z_specmult%1
  end

  if iband != 26
    absa = sw_dict["absa"]
    if !(iband in [23,25])
      absb = sw_dict["absb"]
    end
  end
  
  # println("laytrop: ", laytrop)

  for lay=1:laytrop # trop
    # moving "z_speccomb,z_specmult,js,z_fs =" out breaks it
    if iband in [16,18]
      z_speccomb,z_specmult,js,z_fs = make_zs(colh2o,colch4,8.,sw_dict[iband == 16 ? "strrat1" : "strrat"],lay)
    elseif iband in [17,19,21]
      z_speccomb,z_specmult,js,z_fs = make_zs(colh2o,colco2,8.,sw_dict["strrat"],lay)
    elseif iband in [22,24]
      z_speccomb,z_specmult,js,z_fs = make_zs(colh2o,colo2,8.,(iband == 22 ? z_o2adj : 1.0)*sw_dict["strrat"],lay)
    elseif iband in [28]
      z_speccomb,z_specmult,js,z_fs = make_zs(colo3,colo2,8.,sw_dict["strrat"],lay)
    else
      z_speccomb = 0.0
    end

    rayl_term = (iband == 24) ? (sw_dict["raylac"][js,:] + z_fs * (sw_dict["raylac"][js+1,:] - sw_dict["raylac"][js,:])) : (iband in [23,25,26,27] ? sw_dict["raylc"] : sw_dict["rayl"])
    @. ztaur[lay,:] = colmol[lay] * rayl_term

    if (z_speccomb != 0.0) || (iband in [20,23,25,27,29])
      ab, nsp = (absa, nspa)
      ind0 = ((jp[lay]-1) * 5 + (jt[lay]-1)) * nsp[iband-15] + ((z_speccomb != 0.0) ? js : 1)
      ind1 = ((jp[lay]) * 5 + (jt1[lay]-1)) * nsp[iband-15] + ((z_speccomb != 0.0) ? js : 1)

      if (z_speccomb != 0.0)
        ztaug[lay,:] += z_speccomb * (
          (1. - z_fs) * (ab[ind0,:] * fac00[lay] + ab[ind0+9,:] * fac10[lay] + ab[ind1,:] * fac01[lay] + ab[ind1+9,:] * fac11[lay]) +
          z_fs * (ab[ind0+1,:] * fac00[lay] + ab[ind0+10,:] * fac10[lay] + ab[ind1+1,:] * fac01[lay] + ab[ind1+10,:] * fac11[lay]))
      else
        if (iband in [16,18])
          p1 = colch4[lay]
        elseif iband in [19]
          p1 = colco2[lay]
        elseif (iband in [20,23,25,29])
          p1 = colh2o[lay] * (iband == 23 ? sw_dict["givfac"] : 1.0)
        elseif (iband in [22,24])
          p1 = colo2[lay] * (iband == 22 ? z_o2adj : 1.0)
        elseif (iband in [27])
          p1 = colo3[lay]
        end

        ztaug[lay,:] += p1 * (fac00[lay] * ab[ind0,:] + fac10[lay] * ab[ind0+1,:] + fac01[lay] * ab[ind1,:] + fac11[lay] * ab[ind1+1,:])
      end
    end

    if iband in [16,17,18,19,20,21,22,23,24,29]
      indf = indfor[lay]
      forrefc = sw_dict["forrefc"]
      ztaug[lay,:] += colh2o[lay] * forfac[lay] * (forrefc[indf,:] + forfrac[lay] * (forrefc[indf+1,:] - forrefc[indf,:]))

      inds = indself[lay]
      selfrefc = sw_dict["selfrefc"]
      ztaug[lay,:] += colh2o[lay] * selffac[lay] * (selfrefc[inds,:] + selffrac[lay] * (selfrefc[inds+1,:] - selfrefc[inds,:]))      
    end

    if iband in [20]
      ztaug[lay,:] += colch4[lay] * sw_dict["absch4c"]
    end

    if iband in [22]
      ztaug[lay,:] += 4.35e-4*colo2[lay]/(350.0*2.0)
    end

    if iband in [24,25]
      ztaug[lay,:] += colo3[lay] * sw_dict["abso3ac"]
    end

    if (iband in [29])
      ztaug[lay,:] += colco2[lay] * sw_dict["absco2c"]
    end

    if !(iband in [16,17,27,28,29])
      if !(iband in [26])
        layreffr = sw_dict["layreffr"]
        if (jp[lay] < layreffr && jp[lay+1] >= layreffr)
          laysolfr = min(lay+1,laytrop)
        end
      end
      if lay == laysolfr
        sfluxrefc = sw_dict["sfluxrefc"]
        if iband in [17,18,19,21,22,24,28]
          zsflxzen[:] = sfluxrefc[js,:] + z_fs * (sfluxrefc[js+1,:] - sfluxrefc[js,:])
        else
          zsflxzen[:] .= sfluxrefc * (iband in [27] ? sw_dict["scalekur"] : 1.0)
        end
      end
    end
  end

  for lay=(laytrop+1):nlay
    # moving "z_speccomb,z_specmult,js,z_fs =" out breaks it
    if iband in [17,21]
      z_speccomb,z_specmult,js,z_fs = make_zs(colh2o,colco2,4.,sw_dict["strrat"],lay)
    elseif iband in [28]
      z_speccomb,z_specmult,js,z_fs = make_zs(colo3,colo2,4.,sw_dict["strrat"],lay)
    else
      z_speccomb = 0.0
    end

    rayl_term = (iband == 24) ? sw_dict["raylbc"] : (iband in [23,25,26,27] ? sw_dict["raylc"] : sw_dict["rayl"])
    @. ztaur[lay,:] = colmol[lay] * rayl_term

    if (z_speccomb != 0.0) || (iband in [16,18,19,20,22,24,27,29])
      ab, nsp = (absb, nspb)
      ind0 = ((jp[lay]-13) * 5 + ( jt[lay]-1)) * nsp[iband-15] + ((z_speccomb != 0.0) ? js : 1)
      ind1 = ((jp[lay]-12) * 5 + (jt1[lay]-1)) * nsp[iband-15] + ((z_speccomb != 0.0) ? js : 1)

      if (z_speccomb != 0.0)
        ztaug[lay,:] += z_speccomb * (
          (1. - z_fs) * (ab[ind0,:] * fac00[lay] + ab[ind0+5,:] * fac10[lay] + ab[ind1,:] * fac01[lay] + ab[ind1+5,:] * fac11[lay]) +
          z_fs * (ab[ind0+1,:] * fac00[lay] + ab[ind0+6,:] * fac10[lay] + ab[ind1+1,:] * fac01[lay] + ab[ind1+6,:] * fac11[lay]))
      else
        if (iband in [16,18])
          p1 = colch4[lay]
        elseif iband in [19,29]
          p1 = colco2[lay]
        elseif (iband in [20,23,25])
          p1 = colh2o[lay] * (iband == 23 ? sw_dict["givfac"] : 1.0)
        elseif (iband in [22,24])
          p1 = colo2[lay] * (iband == 22 ? z_o2adj : 1.0)
        elseif (iband in [27])
          p1 = colo3[lay]
        end

        ztaug[lay,:] += p1 * (fac00[lay] * ab[ind0,:] + fac10[lay] * ab[ind0+1,:] + fac01[lay] * ab[ind1,:] + fac11[lay] * ab[ind1+1,:])
      end
    end

    if iband in [17,20,21]
      indf = indfor[lay]
      forrefc = sw_dict["forrefc"]
      ztaug[lay,:] += colh2o[lay] * forfac[lay] * (forrefc[indf,:] + forfrac[lay] * (forrefc[indf+1,:] - forrefc[indf,:]))
    end

    if iband in [20]
      ztaug[lay,:] += colch4[lay] * sw_dict["absch4c"]
    end

    if iband in [22]
      ztaug[lay,:] += 4.35e-4*colo2[lay]/(350.0*2.0)
    end

    if iband in [24,25]
      ztaug[lay,:] += colo3[lay] * sw_dict["abso3bc"]
    end

    if iband in [29]
      ztaug[lay,:] += colh2o[lay] * sw_dict["absh2oc"]
    end

    if iband in [16,17,27,28,29]
      layreffr = sw_dict["layreffr"]
      if (jp[lay-1] < layreffr && jp[lay] >= layreffr)
        laysolfr = lay
      end
      if lay == laysolfr
        sfluxrefc = sw_dict["sfluxrefc"]

        if iband in [17,18,19,21,22,24,28]
          zsflxzen = sfluxrefc[js,:] + z_fs * (sfluxrefc[js+1,:] - sfluxrefc[js,:])
        else
          zsflxzen = sfluxrefc * (iband in [27] ? sw_dict["scalekur"] : 1.0)
        end
      end
    end
  end

  # println("Optical depth for SW band ", iband, " calculated.")
  ztaur, ztaug, zsflxzen
end

function SW_radiative_transfer_per_band(iband,nlay,nlev,ztaur,ztaug,zsflxzen,alb,zbbfd,zbbfu, zbbcd, zbbcu, cos_mu0m, zrmu0i, zincflux, zinctot, aer_tau_sw_vr,aer_cg_sw_vr,aer_piz_sw_vr,cld_tau_sw_vr,cld_cg_sw_vr,cld_piz_sw_vr,cld_frc_vr)
  # println("Calculating SW radiative transfer for SW band ", iband , "...")
  ngc = [6, 12, 8, 8, 10, 10, 2, 10, 8, 6, 6, 8, 6, 12]
  ibm = iband-15

  zref  = zeros(nlev) # direct reflectance
  zrefc = zeros(nlev) # direct albedo for clear
  zrefo = zeros(nlev) # direct albedo for cloud

  zrefd  = zeros(nlev) # diffuse reflectance
  zrefdc = zeros(nlev) # diffuse albedo for clear
  zrefdo = zeros(nlev) # diffuse albedo for cloud

  ztra  = zeros(nlay) # direct transmittence
  ztrac = zeros(nlay) # direct transmittance for clear
  ztrao = zeros(nlay) # direct transmittance for cloud

  ztrad  = zeros(nlay) # diffuse transmittence
  ztradc = zeros(nlay) # diffuse transmittance for clear
  ztrado = zeros(nlay) # diffuse transmittance for cloud

  ztdbt  = zeros(nlev) # total direct beam transmittance at levels
  ztdbtc = zeros(nlev)

  zrup  = zeros(nlev)
  zrupd  = zeros(nlev)

  zrupc = zeros(nlev)
  zrupdc = zeros(nlev)

  repclc = 1.e-12
  
  # println("iband: ", iband)

  for jg=1:ngc[iband-15]
    zincflx = zsflxzen[jg] * cos_mu0m
    zincflux += zsflxzen[jg] * cos_mu0m
    zinctot += zsflxzen[jg]

    #-- clear-sky
    # TOA
    ztdbtc[1] = 1.
    # surface values
    zrefc[nlev]  = alb # zalbp[ibm]
    zrefdc[nlev] = alb # zalbd[ibm]
    zrupc[nlev]  = alb # zalbp[ibm]
    zrupdc[nlev] = alb # zalbd[ibm]

    #-- total-sky
    ztdbt[1] = 1.
    # surface values
    zref[nlev]  = alb # zalbp[ibm]
    zrefd[nlev] = alb # zalbd[ibm]
    zrup[nlev]  = alb # zalbp[ibm]
    zrupd[nlev] = alb # zalbd[ibm]
    
    # println("jg: ", jg)
    # println("first")
    # println("zrupc: ", zrupc[1])
    # println("zrupdc: ", zrupdc[1])
    # println("zrup: ", zrup[1])
    # println("zrupd: ", zrupd[1])
    

    up_range = nlev-(1:nlay)

    # ztaur[up_range,jg] = 1e-6 # take this out when you're ready for more fun
    ztauc = ztaur[up_range,jg] + ztaug[up_range,jg] + aer_tau_sw_vr[up_range,ibm]
    ztauo = ztauc + cld_tau_sw_vr[up_range,ibm]
    
    zomcc = ztaur[up_range,jg] + aer_tau_sw_vr[up_range,ibm].*aer_piz_sw_vr[up_range,ibm]
    zomco = zomcc + cld_tau_sw_vr[up_range,ibm].*cld_piz_sw_vr[up_range,ibm]

    zgcc = aer_cg_sw_vr[up_range,ibm].*aer_piz_sw_vr[up_range,ibm].*aer_tau_sw_vr[up_range,ibm]
    zgco = zgcc + cld_tau_sw_vr[up_range,ibm].*cld_piz_sw_vr[up_range,ibm].*cld_cg_sw_vr[up_range,ibm]

    zgcc  ./= zomcc
    zgco  ./= zomco
    zomcc ./= ztauc
    zomco ./= ztauo

    #-- Delta scaling for clear-sky / aerosol optical quantities
    zf = zgcc .* zgcc
    zwf = zomcc .* zf
    ztauc = (1. - zwf) .* ztauc
    zomcc = (zomcc-zwf)./(1. - zwf)
    zgcc = (zgcc-zf)./(1. - zf)

    #-- Delta scaling for cloudy quantities
    zf = zgco .* zgco
    zwf = zomco .* zf
    ztauo = (1. - zwf).*ztauo
    zomco = (zomco-zwf)./(1. - zwf)
    zgco = (zgco-zf)./(1. - zf)

    # if any(ztauc .< 0) || any(ztauo .< 0)
    #   println("jg: ", jg)
    #   # println("ztauc: ", ztauc)
    #   # println("ztauo: ", ztauo)
    # end


    # mpuetz: llrtchk = TRUE always in this to reftra()()
    zrefc,zrefdc,ztrac,ztradc = srtm_reftra(nlay,zgcc,cos_mu0m,zrmu0i,ztauc,zomcc,zrefc,zrefdc,ztrac,ztradc)

    llrtchk = @. cld_frc_vr[end:-1:1] > repclc

    zrefo,zrefdo,ztrao,ztrado = srtm_reftra(nlay,zgco,cos_mu0m,zrmu0i,ztauo,zomco,zrefo,zrefdo,ztrao,ztrado,llrtchk)

    zclear = 1.0 - cld_frc_vr[up_range]
    zcloud = cld_frc_vr[up_range]
    zref[1:nlay] .= zclear.*zrefc[1:nlay] + zcloud.*zrefo[1:nlay]
    zrefd[1:nlay] .= zclear.*zrefdc[1:nlay] + zcloud.*zrefdo[1:nlay]
    ztra = zclear.*ztrac + zcloud.*ztrao
    ztrad= zclear.*ztradc+ zcloud.*ztrado
    zdbtmc = @. exp(-ztauc*zrmu0i)
    zdbtmo = @. exp(-ztauo*zrmu0i)
    zdbt = zclear.*zdbtmc+zcloud.*zdbtmo
    zdbtc =zdbtmc

    for jk=1:nlay
      ztdbt[jk+1] = zdbt[jk]*ztdbt[jk]
      ztdbtc[jk+1] = zdbtc[jk]*ztdbtc[jk]
    end

    #-- vertical quadrature producing clear-sky fluxes
    # zrdndc,zrupc,zrupdc,zcd,zcu = srtm_vrtqdr(ntime,nlay,nlev,nlat,nlon,zrefc,zrefdc,ztrac,ztradc,zdbtc,zrdndc,zrupc,zrupdc,ztdbtc,zcd,zcu)
    zcd,zcu = srtm_vrtqdr(nlay,nlev,zrefc,zrefdc,ztrac,ztradc,zdbtc,zrupc,zrupdc,ztdbtc)

    #-- vertical quadrature producing cloudy fluxes
    zfd,zfu = srtm_vrtqdr(nlay,nlev,zref ,zrefd ,ztra ,ztrad ,zdbt ,zrup ,zrupd ,ztdbt)
    
    # println("second")
    # println("zrupc: ", zrupc[1])
    # println("zrupdc: ", zrupdc[1])
    # println("zrup: ", zrup[1])
    # println("zrupd: ", zrupd[1])
    
    #-- up and down-welling fluxes at levels
    zbbfu[:,iband-15] += zincflx .* zfu
    zbbfd[:,iband-15] += zincflx .* zfd
    zbbcu[:,iband-15] += zincflx .* zcu
    zbbcd[:,iband-15] += zincflx .* zcd

    # # -- direct flux
    # zsudu[:,:,:,:,iband-15] += zincflx .* ztdbt[:,nlev,:,:]
    # zsuduc[:,:,:,:,iband-15] += zincflx .* ztdbtc[:,nlev,:,:]
  end
  #-- end loop on JG
  # println("Radiative transfer for SW band ", iband , " calculated.")
end

function srtm_reftra(nlay,zgc,cos_mu0m,zrmu0i,ztau,zomc,zref,zrefd,ztra,ztrad,llrtchk = false)
  zsr3 = sqrt(3.)
  zwcrit = 0.9995
  kmodts = 2
  replog = 1e-12

  zg3     = 3. * zgc
  zgamma1 = (8. - zomc .* (5. + zg3)) * 0.25
  zgamma2 = 3. * (zomc .* (1. - zgc )) * 0.25
  zgamma3 = (2. - zg3 .* cos_mu0m) * 0.25
  zzz     = (1. - zgc).^2
  zcrit   = zomc.*zzz - zwcrit.*(zzz - (1. - zomc).*(zgc.^2))

  zcrit_trues = zcrit .>= 0.
  if any(zcrit_trues) ##-- conservative scattering,
    zcrit_trues_tall = falses(nlay+1)
    zcrit_trues_tall[1:nlay] = zcrit_trues

    zem2 = exp.(-min.(ztau[zcrit_trues] * zrmu0i,500.))
    za  = zgamma1[zcrit_trues] * cos_mu0m
    za1 = za - zgamma3[zcrit_trues]
    zgt = zgamma1[zcrit_trues] .* ztau[zcrit_trues]

    # collimated beam
    ztemp=1.0./(1. + zgt)
    zref[zcrit_trues_tall] .= (zgt - za1 .* (1. - zem2)) .* ztemp
    ztra[zcrit_trues] .= 1. - zref[zcrit_trues_tall]

    # isotropic incidence
    zrefd[zcrit_trues_tall] .= zgt .* ztemp
    ztrad[zcrit_trues] .= 1. - zrefd[zcrit_trues_tall]
  end

  zcrit_falses = .!zcrit_trues
  if any(zcrit_falses) #-- non-conservative scattering
    zcrit_falses_tall = falses(nlay+1)
    zcrit_falses_tall[1:nlay] = zcrit_falses

    # homogeneous reflectance and transmittance
    zrk = sqrt.(max.(zgamma1[zcrit_falses].^2 - zgamma2[zcrit_falses].^2,replog))

    zep1 = exp.(min.(zrk .* ztau[zcrit_falses], 500.))
    zep2 = exp.(min.(ztau[zcrit_falses] * zrmu0i,500.))

    zem1 = 1.0./zep1
    zem2 = 1.0./zep2
    zomc = zomc[zcrit_falses]

    zgamma4 = 1. - zgamma3[zcrit_falses]

    za1 = zgamma1[zcrit_falses] .* zgamma4 + zgamma2[zcrit_falses] .* zgamma3[zcrit_falses]
    za2 = zgamma1[zcrit_falses] .* zgamma3[zcrit_falses] + zgamma2[zcrit_falses] .* zgamma4

    zrp = zrk * cos_mu0m
    zrp1 = 1. + zrp
    zrm1 = 1. - zrp
    zrk2 = 2. * zrk
    zrpp = 1. - zrp.*zrp
    zrkg = zrk + zgamma1[zcrit_falses]
    zr1  = zrm1 .* (za2 + zrk .* zgamma3[zcrit_falses])
    zr2  = zrp1 .* (za2 - zrk .* zgamma3[zcrit_falses])
    zr3  = zrk2 .* (zgamma3[zcrit_falses] - za2 * cos_mu0m )
    zr4  = zrpp .* zrkg
    zr5  = zrpp .* (zrk - zgamma1[zcrit_falses])
    zt1  = zrp1 .* (za1 + zrk .* zgamma4)
    zt2  = zrm1 .* (za1 - zrk .* zgamma4)
    zt3  = zrk2 .* (zgamma4 + za1 * cos_mu0m )
    zt4  = zr4
    zt5  = zr5
    zbeta = .-zr5 ./ zr4

    # collimated beam
    zdenr = zr4.*zep1 + zr5.*zem1
    zref[zcrit_falses_tall] = zomc .* (zr1 .* zep1 - zr2 .* zem1 - zr3 .* zem2) ./ zdenr

    zdent = zt4 .* zep1 + zt5 .* zem1
    ztra[zcrit_falses] = zem2 .* (1. - zomc .* (zt1 .* zep1 - zt2 .* zem1 - zt3 .* zep2) ./ zdent)

    # diffuse beam
    zemm = zem1 .* zem1
    zdend = 1. ./ ( (1. - zbeta .* zemm ) .* zrkg)
    zrefd[zcrit_falses_tall] =  zgamma2[zcrit_falses] .* (1. - zemm) .* zdend
    ztrad[zcrit_falses] =  zrk2 .* zem1 .* zdend
  end

  # if
  if llrtchk != false
    zref[1:nlay][.!llrtchk] = 0.0
    ztra[1:nlay][.!llrtchk] = 1.0
    zrefd[1:nlay][.!llrtchk] = 0.0
    ztrad[1:nlay][.!llrtchk] = 1.0
  end
  
  zref,zrefd,ztra,ztrad
end

function srtm_vrtqdr(nlay,nlev,pref ,prefd, ptra, ptrad, pdbt,prup,prupd,ptdbt)
  ztdn = zeros(nlev)
  prdnd = zeros(nlev)
  pfd = zeros(nlev)
  pfu = zeros(nlev)

  zreflect = @. 1.0 / (1.0 -prefd[nlev]*prefd[nlay])
  prup[nlay] = pref[nlay] + (ptrad[nlay] .* ((ptra[nlay]-pdbt[nlay]).*prefd[nlev]+pdbt[nlay].*pref[nlev])).*zreflect
  prupd[nlay] = prefd[nlay] + ptrad[nlay] .* ptrad[nlay] .* prefd[nlev] .* zreflect

  for jk=1:(nlay-1)
    ikp=nlev-jk
    ikx=ikp-1

    zreflect = 1.0 ./ (1.0 -prupd[ikp].*prefd[ikx])
    prup[ikx]=pref[ikx]+(ptrad[ikx].*((ptra[ikx]-pdbt[ikx]).*prupd[ikp]+pdbt[ikx].*prup[ikp])).*zreflect
    prupd[ikx]=prefd[ikx]+ptrad[ikx].*ptrad[ikx].*prupd[ikp].*zreflect
  end
  
  # println("pdbt: ",pdbt[1])

  #-- upper boundary conditions

  ztdn[1]=1.0
  prdnd[1]=0.0
  ztdn[2]=ptra[1]
  prdnd[2]=prefd[1]

  #-- pass from top to bottom

  for jk=2:nlay
    ikp=jk+1

    zreflect=1.0 ./ (1.0 -prefd[jk].*prdnd[jk])
    ztdn[ikp]=ptdbt[jk].*ptra[jk]+(ptrad[jk].*((ztdn[jk]-ptdbt[jk])+ ptdbt[jk].*pref[jk].*prdnd[jk])) .* zreflect
    prdnd[ikp]=prefd[jk]+ptrad[jk].*ptrad[jk].*prdnd[jk].*zreflect
  end

  #-- up and down-welling fluxes at levels

  for jk=1:nlev
    zreflect=1.0 ./ (1.0 - prdnd[jk].*prupd[jk])
    pfu[jk]=(ptdbt[jk].*prup[jk] + (ztdn[jk]-ptdbt[jk]).*prupd[jk]).*zreflect
    pfd[jk]=ptdbt[jk] + (ztdn[jk]-ptdbt[jk]+ptdbt[jk].*prup[jk].*prdnd[jk]).*zreflect
  end
  #     ------------------------------------------------------------------
  # prdnd,prup,prupd,pfd,pfu
  pfd,pfu
end

# function setup_RRTM()
#   setup_aerosol_parameters()
#   setup_LW_parameters()
#   setup_SW_parameters()
# end

# println("started radiaton")
# println(ARGS)
# println("after")
# if length(ARGS) > 2
#   @time radiation(ARGS[1],Int(float(ARGS[2])),Int(float(ARGS[3])))
# else
#   @time radiation(ARGS[1],Int(float(ARGS[2])))
# end
# # if length(ARGS) > 1
# #   @time radiation(ARGS[1],Int(float(ARGS[2])))
# # else
# #   @time radiation(ARGS[1])
# # end
# println("finished radiation")

end # module
