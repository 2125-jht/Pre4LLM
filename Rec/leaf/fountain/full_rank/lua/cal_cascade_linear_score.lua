function cal_cascade_linear_score()
  local pctr = cascade_pctr or 0.0
  local pltr = cascade_pltr or 0.0
  local pwtr = cascade_pwtr or 0.0
  local pftr = cascade_pftr or 0.0
  local pcmtr = cascade_pcmtr or 0.0
  local pcestr = cascade_pcestr or 0.0
  local ptr = cascade_ptr or 0.0
  local pepstr = cascade_pepstr or 0.0
  local psvtr = cascade_psvtr or 0.0
  local plvtr = cascade_plvtr or 0.0
  local pwtd = cascade_pwtd or 0.0
  local pcltr = cascade_pcltr or 0.0
  local phtr = cascade_phtr or 0.0
  local pwatch_time = cascade_pwatch_time or 0.0

  local pctr_weight = fountain_fr_cascade_linear_score_ctr_weight or 0.0
  local pltr_weight = fountain_fr_cascade_linear_score_ltr_weight or 0.0
  local pwtr_weight = fountain_fr_cascade_linear_score_ctr_weight or 0.0
  local pftr_weight = fountain_fr_cascade_linear_score_ltr_weight or 0.0
  local pcmtr_weight = fountain_fr_cascade_linear_score_cmtr_weight or 0.0
  local pcestr_weight = fountain_fr_cascade_linear_score_cestr_weight or 0.0
  local ptr_weight = fountain_fr_cascade_linear_score_ptr_weight or 0.0
  local pepstr_weight = fountain_fr_cascade_linear_score_epstr_weight or 0.0
  local psvtr_weight = fountain_fr_cascade_linear_score_svtr_weight or 0.0
  local plvtr_weight = fountain_fr_cascade_linear_score_lvtr_weight or 0.0
  local pwtd_weight = fountain_fr_cascade_linear_score_wtd_weight or 0.0
  local pcltr_weight = fountain_fr_cascade_linear_score_cltr_weight or 0.0
  local phtr_weight = fountain_fr_cascade_linear_score_htr_weight or 0.0
  local pwatch_time_weight = fountain_fr_cascade_linear_score_watchtime_weight or 0.0

  local linear_score = 0.0
  linear_score = pctr_weight * pctr + pltr_weight * pltr + pwtr_weight * pwtr + pftr_weight * pftr + pcmtr_weight * pcmtr + pcestr_weight * pcestr + ptr_weight * ptr + pepstr_weight * pepstr + psvtr_weight * psvtr + plvtr_weight * plvtr + pwtd_weight * pwtd + pcltr_weight * pcltr + phtr_weight * phtr + pwatch_time_weight * pwatch_time
  return linear_score
end
