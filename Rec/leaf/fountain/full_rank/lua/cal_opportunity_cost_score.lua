function cal_opportunity_cost_score()
  -- interact
  local pctr = fullrank_sim_pevtr and fullrank_sim_pevtr * 1.0 or 0.0
  local pltr = fullrank_sim_pltr and fullrank_sim_pltr * 1.0 or 0.0
  local pwtr = fullrank_sim_pwtr and fullrank_sim_pwtr * 1.0 or 0.0
  local pftr = fullrank_sim_pftr and fullrank_sim_pftr * 1.0 or 0.0
  local pcmtr = fullrank_sim_pcmtr and fullrank_sim_pcmtr * 1.0 or 0.0
  local pcmef = fullrank_sim_pcmef and fullrank_sim_pcmef * 1.0 or 0.0
  local pptr = fullrank_sim_pptr and fullrank_sim_pptr * 1.0 or 0.0
  local pepstr = fullrank_sim_pepstr and fullrank_sim_pepstr * 1.0 or 0.0
  local lstr = fullrank_sim_lstr and fullrank_sim_lstr * 1.0 or 0.0
  local evtr_v2 = fullrank_detail_new_pevtr_v2 and fullrank_detail_new_pevtr_v2 * 1.0 or 0.0
  local trans_pvtr = fullrank_trans_pvtr_score and fullrank_trans_pvtr_score * 1.0 or 0.0
  local pcltr = fullrank_sim_pcltr and fullrank_sim_pcltr * 1.0 or 0.0
  local plvtr = fullrank_sim_plvtr and fullrank_sim_plvtr * 1.0 or 0.0
  local pfintr = fullrank_sim_pfintr and fullrank_sim_pfintr * 1.0 or 0.0
  local psvtr = fullrank_sim_psvr and fullrank_sim_psvr * 1.0 or 0.0
  local phtr = fullrank_sim_phtr and fullrank_sim_phtr * 1.0 or 0.0
  -- cal action_once_score
  local pctr_score = math.min(pctr * fullrank_action_once_score_pctr_weight, 1.0)
  local pltr_score = math.min(pltr * fullrank_action_once_score_pltr_weight, 1.0)
  local pwtr_score = math.min(pwtr * fullrank_action_once_score_pwtr_weight, 1.0)
  local pftr_score = math.min(pftr * fullrank_action_once_score_pftr_weight, 1.0)
  local pcmtr_score = math.min(pcmtr * fullrank_action_once_score_pcmtr_weight, 1.0)
  local pcmef_score = math.min(pcmef * fullrank_action_once_score_pcmef_weight, 1.0)
  local pptr_score = math.min(pptr * fullrank_action_once_score_pptr_weight, 1.0)
  local pepstr_score = math.min(pepstr * fullrank_action_once_score_pepstr_weight, 1.0)
  local lstr_score = math.min(lstr * fullrank_action_once_score_lstr_weight, 1.0)
  local evtr_v2_score = math.min(evtr_v2 * fullrank_action_once_score_evtr_v2_weight, 1.0)
  local pcltr_score = math.min(pcltr * fullrank_action_once_score_pcltr_weight, 1.0)
  local plvtr_score = math.min(plvtr * fullrank_action_once_score_plvtr_weight, 1.0)
  local pfintr_score = math.min(pfintr * fullrank_action_once_score_pfintr_weight, 1.0)
  local phtr_score = math.min(phtr * fullrank_action_once_score_phtr_weight, 1.0)

  local action_once_score = (1.0 - phtr_score) * (1.0 - (1.0 - pctr_score) * (1.0 - pltr_score) * (1.0 - pwtr_score) * (1.0 - pftr_score) * (1.0 - pcmtr_score) * (1.0 - pcmef_score) * (1.0 - pptr_score) * (1.0 - pepstr_score) * (1.0 - lstr_score) * (1.0 - evtr_v2_score) * (1.0 - pcltr_score) * (1.0 - plvtr_score) * (1.0 - pfintr_score))
  -- cal interact score
  local interact_score = fullrank_opportunity_score_ctr_weight * pctr + fullrank_opportunity_score_ltr_weight * pltr + fullrank_opportunity_score_wtr_weight * pwtr + fullrank_opportunity_score_ftr_weight * pftr + fullrank_opportunity_score_cmtr_weight * pcmtr + fullrank_opportunity_score_cmef_weight * pcmef + fullrank_opportunity_score_ptr_weight * pptr + fullrank_opportunity_score_epstr_weight * pepstr + fullrank_opportunity_score_lstr_weight * lstr + fullrank_opportunity_score_evtrv2_weight * evtr_v2 + pcltr * fullrank_opportunity_score_pcltr_weight + plvtr * fullrank_opportunity_score_plvtr_weight
  interact_score = interact_score + fullrank_opportunity_score_pfintr_weight * pfintr + fullrank_opportunity_score_psvtr_weight * psvtr + fullrank_opportunity_score_phtr_weight * phtr
  -- watchtime
  local pvtr = fullrank_sim_pvtr and fullrank_sim_pvtr * 1.0 or 0.0
  local pvtr_score = 0.0
  if fountain_enable_opportunity_score_use_trans_pvtr_score > 0.0 then
    pvtr_score = math.min(fountain_fullrank_trans_pvtr_score_max, trans_pvtr) * 1.0 / fountain_fullrank_trans_pvtr_score_max
    pvtr_score = pvtr_score ^ fullrank_opportunity_score_vtr_power_weight
  else
    pvtr_score = pvtr ^ fullrank_opportunity_score_vtr_power_weight
  end
  -- opportunity_cost_score
  local opportunity_cost_score = 0.0
  -- replace linear_score use action_once_score
  if fountain_enable_opportunity_score_use_action_once_score > 0.0 then
    interact_score = action_once_score
  end
  if fountain_enable_opportunity_score_use_linear_score > 0.0 then
    opportunity_cost_score = fullrank_opportunity_score_alpha * pvtr_score + fullrank_opportunity_score_beta * interact_score
  else
    opportunity_cost_score = fullrank_opportunity_score_alpha * pvtr_score + fullrank_opportunity_score_beta * interact_score / (pvtr_score + 0.001 + fullrank_opportunity_score_smooth)
  end
  return opportunity_cost_score, action_once_score
end

function cal_satisfy_score()
  local pfintr = fullrank_sim_pfintr and fullrank_sim_pfintr * 1.0 or 0.0
  local duration_max = fullrank_satisfy_score_duration_max or 180.0
  local duration = (duration_ms and duration_ms * 1.0 or 7000.0) / 1000.0
  duration = math.min(duration, duration_max)
  local sat_score = pfintr ^ fullrank_satisfy_score_pfintr_weight / (duration + 0.1)
  return sat_score
end

function cal_fit_ptime_score()
  local duration = (duration_ms and duration_ms * 1.0 or 0.0) / 1000.0
  local pevtr = fullrank_sim_pevtr and fullrank_sim_pevtr * 1.0 or 0.0
  local plvtr = fullrank_sim_plvtr and fullrank_sim_plvtr * 1.0 or 0.0
  local fit_ptime = 0.0
  local duration_thre_lvtr = math.floor((duration * 28.0 + 180.0) / 33.0)
  local duration_thre_evtr = duration
  if duration < 3.0 then
    fit_ptime = 7.0 * pevtr + (8.0 - 7.0) * plvtr
  elseif duration < 7.0 then
    fit_ptime = 7.0 * pevtr + (duration_thre_lvtr - 7.0) * plvtr
  elseif duration < 18.0 then
    fit_ptime = duration_thre_evtr * pevtr + (duration_thre_lvtr - duration_thre_evtr) * plvtr
  elseif duration < 36.0 then
    fit_ptime = 18.0 * pevtr + (duration_thre_lvtr - 18.0) * plvtr
  else
    fit_ptime = 18.0 * pevtr + 18.0 * plvtr
  end
  return fit_ptime
end

function cal_value_multiply_score()
  -- interact
  local pctr = fullrank_sim_pevtr and fullrank_sim_pevtr * 1.0 or 0.0
  local pltr = fullrank_sim_pltr and fullrank_sim_pltr * 1.0 or 0.0
  local pwtr = fullrank_sim_pwtr and fullrank_sim_pwtr * 1.0 or 0.0
  local pftr = fullrank_sim_pftr and fullrank_sim_pftr * 1.0 or 0.0
  local pcmtr = fullrank_sim_pcmtr and fullrank_sim_pcmtr * 1.0 or 0.0
  local pcmef = fullrank_sim_pcmef and fullrank_sim_pcmef * 1.0 or 0.0
  local pptr = fullrank_sim_pptr and fullrank_sim_pptr * 1.0 or 0.0
  local pepstr = fullrank_sim_pepstr and fullrank_sim_pepstr * 1.0 or 0.0
  local lstr = fullrank_sim_lstr and fullrank_sim_lstr * 1.0 or 0.0
  local evtr_v2 = fullrank_detail_new_pevtr_v2 and fullrank_detail_new_pevtr_v2 * 1.0 or 0.0
  local pcltr = fullrank_sim_pcltr and fullrank_sim_pcltr * 1.0 or 0.0
  local plvtr = fullrank_sim_plvtr and fullrank_sim_plvtr * 1.0 or 0.0
  local pfintr = fullrank_sim_pfintr and fullrank_sim_pfintr * 1.0 or 0.0
  local trans_pvtr = fullrank_trans_pvtr_score and fullrank_trans_pvtr_score * 1.0 or 0.0
  local pvtr = fullrank_sim_pvtr and fullrank_sim_pvtr * 1.0 or 0.0
  local pctr_score = 1.0 + (fullrank_multiply_score_pctr_alpha * pctr) ^ fullrank_multiply_score_pctr_beta
  local pltr_score = 1.0 + (fullrank_multiply_score_pltr_alpha * pltr) ^ fullrank_multiply_score_pltr_beta
  local pwtr_score = 1.0 + (fullrank_multiply_score_pwtr_alpha * pwtr) ^ fullrank_multiply_score_pwtr_beta
  local pftr_score = 1.0 + (fullrank_multiply_score_pftr_alpha * pftr) ^ fullrank_multiply_score_pftr_beta
  local pcmtr_score = 1.0 + (fullrank_multiply_score_pcmtr_alpha * pcmtr) ^ fullrank_multiply_score_pcmtr_beta
  local pcmef_score = 1.0 + (fullrank_multiply_score_pcmef_alpha * pcmef) ^ fullrank_multiply_score_pcmef_beta
  local pptr_score = 1.0 + (fullrank_multiply_score_pptr_alpha * pptr) ^ fullrank_multiply_score_pptr_beta
  local pepstr_score = 1.0 + (fullrank_multiply_score_pepstr_alpha * pepstr) ^ fullrank_multiply_score_pepstr_beta
  local lstr_score = 1.0 + (fullrank_multiply_score_lstr_alpha * lstr) ^ fullrank_multiply_score_lstr_beta
  local evtr_v2_score = 1.0 + (fullrank_multiply_score_evtr_v2_alpha * evtr_v2) ^ fullrank_multiply_score_evtr_v2_beta
  local pcltr_score = 1.0 + (fullrank_multiply_score_pcltr_alpha * pcltr) ^ fullrank_multiply_score_pcltr_beta
  local plvtr_score = 1.0 + (fullrank_multiply_score_plvtr_alpha * plvtr) ^ fullrank_multiply_score_plvtr_beta
  local pfintr_score = 1.0 + (fullrank_multiply_score_pfintr_alpha * pfintr) ^ fullrank_multiply_score_pfintr_beta
  local trans_pvtr_score = 1.0 + (fullrank_multiply_score_trans_pvtr_alpha * trans_pvtr) ^ fullrank_multiply_score_trans_pvtr_beta
  local pvtr_score = 1.0 + (fullrank_multiply_score_pvtr_alpha * pvtr) ^ fullrank_multiply_score_pvtr_beta
  local value_multiply_score = pctr_score * pltr_score * pwtr_score * pftr_score * pcmtr_score * pcmef_score * pptr_score * pepstr_score * lstr_score * evtr_v2_score * pcltr_score * plvtr_score * pfintr_score * trans_pvtr_score * pvtr_score
  return value_multiply_score
end

function cal_action_once_score()
  -- interact
  local pctr = fullrank_sim_pevtr and fullrank_sim_pevtr * 1.0 or 0.0
  local pltr = fullrank_sim_pltr and fullrank_sim_pltr * 1.0 or 0.0
  local pwtr = fullrank_sim_pwtr and fullrank_sim_pwtr * 1.0 or 0.0
  local pftr = fullrank_sim_pftr and fullrank_sim_pftr * 1.0 or 0.0
  local pcmtr = fullrank_sim_pcmtr and fullrank_sim_pcmtr * 1.0 or 0.0
  local pcmef = fullrank_sim_pcmef and fullrank_sim_pcmef * 1.0 or 0.0
  local pptr = fullrank_sim_pptr and fullrank_sim_pptr * 1.0 or 0.0
  local pepstr = fullrank_sim_pepstr and fullrank_sim_pepstr * 1.0 or 0.0
  local lstr = fullrank_sim_lstr and fullrank_sim_lstr * 1.0 or 0.0
  local evtr_v2 = fullrank_detail_new_pevtr_v2 and fullrank_detail_new_pevtr_v2 * 1.0 or 0.0
  local pcltr = fullrank_sim_pcltr and fullrank_sim_pcltr * 1.0 or 0.0
  local plvtr = fullrank_sim_plvtr and fullrank_sim_plvtr * 1.0 or 0.0
  local pfintr = fullrank_sim_pfintr and fullrank_sim_pfintr * 1.0 or 0.0
  local phtr = fullrank_sim_phtr and fullrank_sim_phtr * 1.0 or 0.0
  local trans_pvtr = fullrank_trans_pvtr_score and fullrank_trans_pvtr_score * 1.0 or 0.0
  local pvtr = fullrank_sim_pvtr and fullrank_sim_pvtr * 1.0 or 0.0
  -- cal action_once_watchtime_score
  local pctr_score = math.min(pctr * fullrank_action_once_watchtime_score_pctr_weight, 1.0)
  local evtr_v2_score = math.min(evtr_v2 * fullrank_action_once_watchtime_score_evtr_v2_weight, 1.0)
  local pvtr_score = math.min(pvtr * fullrank_action_once_watchtime_score_pvtr_weight, 1.0)
  local trans_pvtr_score = math.min(phtr * fullrank_action_once_watchtime_score_trans_pvtr_weight, 1.0)
  local plvtr_score = math.min(plvtr * fullrank_action_once_watchtime_score_plvtr_weight, 1.0)
  local pfintr_score = math.min(pfintr * fullrank_action_once_watchtime_score_pfintr_weight, 1.0)
  -- cal action_once_interact_score
  local pltr_score = math.min(pltr * fullrank_action_once_interact_score_pltr_weight, 1.0)
  local pwtr_score = math.min(pwtr * fullrank_action_once_interact_score_pwtr_weight, 1.0)
  local pftr_score = math.min(pftr * fullrank_action_once_interact_score_pftr_weight, 1.0)
  local pcmtr_score = math.min(pcmtr * fullrank_action_once_interact_score_pcmtr_weight, 1.0)
  local pcmef_score = math.min(pcmef * fullrank_action_once_interact_score_pcmef_weight, 1.0)
  local pptr_score = math.min(pptr * fullrank_action_once_interact_score_pptr_weight, 1.0)
  local pepstr_score = math.min(pepstr * fullrank_action_once_interact_score_pepstr_weight, 1.0)
  local lstr_score = math.min(lstr * fullrank_action_once_interact_score_lstr_weight, 1.0)
  local pcltr_score = math.min(pcltr * fullrank_action_once_interact_score_pcltr_weight, 1.0)
  local phtr_score = math.min(phtr * fullrank_action_once_interact_score_phtr_weight, 1.0)

  local action_once_interact_score = (1.0 - phtr_score) * (1.0 - (1.0 - pltr_score) * (1.0 - pwtr_score) * (1.0 - pftr_score) * (1.0 - pcmtr_score) * (1.0 - pcmef_score) * (1.0 - pptr_score) * (1.0 - pepstr_score) * (1.0 - lstr_score) * (1.0 - pcltr_score))
  local action_once_watchtime_score = (1.0 - phtr_score) * (1.0 - (1.0 - pctr_score) * (1.0 - evtr_v2_score) * (1.0 - plvtr_score) * (1.0 - pfintr_score) * (1.0 - pvtr_score) * (1.0 - trans_pvtr_score))
  return action_once_interact_score, action_once_watchtime_score
end

function cal_opportunity_cost_score_v2()
  local pctr = fullrank_sim_pevtr and fullrank_sim_pevtr * 1.0 or 0.0
  local pltr = fullrank_sim_pltr and fullrank_sim_pltr * 1.0 or 0.0
  local pwtr = fullrank_sim_pwtr and fullrank_sim_pwtr * 1.0 or 0.0
  local pftr = fullrank_sim_pftr and fullrank_sim_pftr * 1.0 or 0.0
  local pcmtr = fullrank_sim_pcmtr and fullrank_sim_pcmtr * 1.0 or 0.0
  local pcmef = fullrank_sim_pcmef and fullrank_sim_pcmef * 1.0 or 0.0
  local pcltr = fullrank_sim_pcltr and fullrank_sim_pcltr * 1.0 or 0.0
  local pptr = fullrank_sim_pptr and fullrank_sim_pptr * 1.0 or 0.0
  local pepstr = fullrank_sim_pepstr and fullrank_sim_pepstr * 1.0 or 0.0
  local pvtr = fullrank_sim_pvtr and fullrank_sim_pvtr * 1.0 or 0.0
  local plvtr = fullrank_sim_plvtr and fullrank_sim_plvtr * 1.0 or 0.0
  local reverse_score = fountain_ada_xtr_use_reverse_score * 1.0 or 1.0
  local queues = {
    pctr,
    pltr,
    pwtr,
    pftr,
    pptr,
    pepstr,
    pcltr,
    pcmtr,
    pcmef,
    plvtr,
    pvtr
  }
  local ada_weight = user_ada_weight_tensor or {}
  local opportunity_cost_score = 0.0
  local linear_score = 0.0
  local prod_score = 1.0
  local size = 11
  if fountain_ada_xtr_use_linear_score > 0.0 then
    if #queues > 0 and #ada_weight > 0 and #queues == #ada_weight then
      for i = 1, #queues do
        opportunity_cost_score = opportunity_cost_score + queues[i] * ada_weight[i]
      end
    end
  else
    if #queues > 0 and #ada_weight > 0 and #ada_weight == 33 then
      for i = 1, #queues do
        linear_score = linear_score + queues[i] * ada_weight[i]
        prod_score = prod_score * (ada_weight[size+i] * queues[i] + ada_weight[2*size+i])
      end
    end
    opportunity_cost_score = linear_score + prod_score
  end
  opportunity_cost_score = opportunity_cost_score * reverse_score
  return opportunity_cost_score
end
function get_ada_weight()
  local ada_weight = user_ada_weight_tensor or {}
  local que_dim = 11
  if #ada_weight == 0
  then
    for i=1, que_dim do
      ada_weight[i] = 0.0
    end
  end
  return ada_weight[1], ada_weight[2], ada_weight[3], ada_weight[4], ada_weight[5], ada_weight[6], ada_weight[7], ada_weight[8], ada_weight[9], ada_weight[10], ada_weight[11]
end

function cal_rl_xtr_score()
  local pctr = fullrank_sim_pevtr and fullrank_sim_pevtr * 1.0 or 0.0
  local plvtr = fullrank_sim_plvtr and fullrank_sim_plvtr * 1.0 or 0.0
  local phtr = fullrank_sim_phtr and fullrank_sim_phtr * 1.0 or 0.0
  local pltr = fullrank_sim_pltr and fullrank_sim_pltr * 1.0 or 0.0
  local pwtr = fullrank_sim_pwtr and fullrank_sim_pwtr * 1.0 or 0.0
  local pftr = fullrank_sim_pftr and fullrank_sim_pftr * 1.0 or 0.0
  local pcmtr = fullrank_sim_pcmtr and fullrank_sim_pcmtr * 1.0 or 0.0
  local pcmef = fullrank_sim_pcmef and fullrank_sim_pcmef * 1.0 or 0.0
  local pptr = fullrank_sim_pptr and fullrank_sim_pptr * 1.0 or 0.0
  local pepstr = fullrank_sim_pepstr and fullrank_sim_pepstr * 1.0 or 0.0
  local pvtr = fullrank_sim_pvtr and fullrank_sim_pvtr * 1.0 or 0.0
  local pfintr = fullrank_sim_pfintr and fullrank_sim_pfintr * 1.0 or 0.0

  local ctr_weight = fullrank_rl_ctr_weight and fullrank_rl_ctr_weight or 0.0
  local lvtr_weight = fullrank_rl_lvtr_weight and fullrank_rl_lvtr_weight or 0.0
  local htr_weight = fullrank_rl_htr_weight and fullrank_rl_htr_weight or 0.0
  local ltr_weight = fullrank_rl_ltr_weight and fullrank_rl_ltr_weight or 0.0
  local wtr_weight = fullrank_rl_wtr_weight and fullrank_rl_wtr_weight or 0.0
  local ftr_weight = fullrank_rl_ftr_weight and fullrank_rl_ftr_weight or 0.0
  local cmtr_weight = fullrank_rl_cmtr_weight and fullrank_rl_cmtr_weight or 0.0
  local cmef_weight = fullrank_rl_cmef_weight and fullrank_rl_cmef_weight or 0.0
  local ptr_weight = fullrank_rl_ptr_weight and fullrank_rl_ptr_weight or 0.0
  local epstr_weight = fullrank_rl_epstr_weight and fullrank_rl_epstr_weight or 0.0
  local vtr_weight = fullrank_rl_vtr_weight and fullrank_rl_vtr_weight or 0.0
  local fintr_weight = fullrank_rl_fintr_weight and fullrank_rl_fintr_weight or 0.0
  local bias = fullrank_rl_bias and fullrank_rl_bias or 0.0

  local score = pctr * ctr_weight + plvtr * lvtr_weight + phtr * htr_weight + pltr * ltr_weight + pwtr * wtr_weight + pftr * ftr_weight + pcmtr * cmtr_weight + pcmef * cmef_weight + pptr * ptr_weight + pepstr * epstr_weight + pvtr * vtr_weight + pfintr * fintr_weight + bias 
  return score
end

