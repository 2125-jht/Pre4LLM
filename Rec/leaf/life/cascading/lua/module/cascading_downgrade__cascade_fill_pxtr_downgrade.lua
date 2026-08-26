function cascade_fill_pxtr_downgrade()
  local pctr = empirical_ctr or 0.0
  local pltr = empirical_ltr or 0.0
  local pwtr = empirical_wtr or 0.0
  local pftr = empirical_ftr or 0.0
  local plvtr = empirical_lvtr or 0.0
  local psvtr = empirical_svtr or 0.0
  local ptr = empirical_ptr or 0.0
  local phtr = empirical_htr or 0.0
  local pcmtr = empirical_cmtr or 0.0
  return pctr, pltr, pwtr, pftr, plvtr, psvtr, ptr, phtr, pcmtr
end