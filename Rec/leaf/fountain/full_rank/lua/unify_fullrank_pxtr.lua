-- 用于精排所有打分的统一
function unify_fullrank_pxtr()
  local pfr_v4 = fullrank_ltr_v4_fountain_finish_rate and fullrank_ltr_v4_fountain_finish_rate or 1.0
  local pfr_v4_final = pfr_v4
  local dura_cdf_pfr = fullrank_dura_cdf_pfr or 0.0
  local duration = duration_ms and duration_ms or 2.0
  local svr = fullrank_sim_psvr and fullrank_sim_psvr or 0.0

  if skip_fountain_finish_rate_adjust == 0 then
    duration = math.min(duration, fountain_fullrank_finish_duration_factor_max_value)
    pfr_v4_final = ((duration / 1000.0 + fountain_fullrank_fr_duration_factor_offset)^fountain_fullrank_finish_duration_factor_pow_weight) * pfr_v4
  end

  if skip_fountain_finish_rate_adjust_v3 == 0 then
    pfr_v4_final = pfr_v4 * ((1.0 - svr)^fountain_fullrank_not_svr_pow_weight_for_pfr)
  end
  local local_lt_interest_ee_score = 1.0
  if (page ~= nil and page >1) then
    local_lt_interest_ee_score = long_term_interest_ee_score
  end
  -- 开启只预估非首屏，则填充首屏分数统一默认值
  if (fountain_skip_fr_pred_only_fast_v1 == 0 and page ~= nil and page <=1) then
    pfr_v4_final = 1.0
  end 
  if fountain_fullrank_enable_cdf_fr_smooth == 0 then
    dura_cdf_pfr = fountain_fullrank_cdf_fr_smooth_alpha / (dura_cdf_pfr + 1e-6) + fountain_fullrank_cdf_fr_smooth_beta
  end
  return pfr_v4_final, local_lt_interest_ee_score, dura_cdf_pfr
end

function unify_fullrank_common_attr()
  local fast_v1_next = fountain_ensemble_power_weight_fullrank_ltr_v4_next
  if (fountain_ensemble_power_weight_fullrank_ltr_v4_next > 0) then
    if (page ~= nil and page > 1) then
      fast_v1_next = (page^fountain_fullrank_next_score_debias_pow_weight) * fountain_ensemble_power_weight_fullrank_ltr_v4_next
    elseif skip_fountain_fullrank_ltr_v4_next_splash == 0 then
      fast_v1_next = 0.0
    end
  end
  return fast_v1_next
end