function cascade_pxtr()
    local duration_sec = duration_ms/1000 or 0
    if (duration_sec < 0) then
        duration_sec = 0
    elseif (duration_sec > fountain_cascade_duration_sec_bound) then
        duration_sec = fountain_cascade_duration_sec_bound
    end
    local cascade_pwatch_time = cascade_pwatch_time or 0
    local cascade_pfinish_rate = cascade_pwatch_time * 64  / (duration_sec + 0.1)
    local pctr = cascade_pctr or 0.0
    local pltr = cascade_pltr or 0.0
    local pwtr = cascade_pwtr or 0.0
    local pftr = cascade_pftr or 0.0
    local ptr = cascade_ptr or 0.0
    local plvtr = cascade_plvtr or 0.0
    local psvtr = cascade_psvtr or 0.0
    local pwatch_time = cascade_pwatch_time or 0.0
    local hot_pctr = hot_cascade_pctr or 0.0
    local is_splash_request = (common_request_type == "fountain_splash") and 1 or 0
    local ctr_param = (is_splash_request > 0) and fountain_cascade_weighted_score_pctr_param or fountain_cascade_weighted_score_pctr_param_splash
    local ltr_param = (is_splash_request > 0) and fountain_cascade_weighted_score_pltr_param or fountain_cascade_weighted_score_pltr_param_splash
    local wtr_param = (is_splash_request > 0) and fountain_cascade_weighted_score_pwtr_param or fountain_cascade_weighted_score_pwtr_param_splash
    local ftr_param = (is_splash_request > 0) and fountain_cascade_weighted_score_pftr_param or fountain_cascade_weighted_score_pftr_param_splash
    local ptr_param = (is_splash_request > 0) and fountain_cascade_weighted_score_ptr_param or fountain_cascade_weighted_score_ptr_param_splash
    local lvtr_param = (is_splash_request > 0) and fountain_cascade_weighted_score_plvtr_param or fountain_cascade_weighted_score_plvtr_param_splash
    local svtr_param = (is_splash_request > 0) and fountain_cascade_weighted_score_psvtr_param or fountain_cascade_weighted_score_psvtr_param_splash
    local svtr_power_param = (is_splash_request > 0) and fountain_cascade_weighted_score_psvtr_power_param or fountain_cascade_weighted_score_psvtr_power_param_splash
    local watch_time_param = (is_splash_request > 0) and fountain_cascade_weighted_score_pwatch_time_param or fountain_cascade_weighted_score_pwatch_time_param_splash
    local hot_pctr_param = (is_splash_request > 0) and fountain_cascade_weighted_score_hot_pctr_power_param or fountain_cascade_weighted_score_hot_pctr_power_param_splash
    local enalbe_plvtr_modulate = fountain_cascade_weighted_score_enable_plvtr_modulate or 0
    local lvtr_max_value = fountain_cascade_lvtr_max_value or 2.0
    local lvtr_sigmoid_decay_rate = fountain_cascade_lvtr_sigmoid_decay_rate or 0.5
    local lvtr_sigmoid_bias = fountain_cascade_lvtr_sigmoid_bias or 8.5
    -- 短播参数调整
    local svtr_factor = 1.0 - (psvtr ^ svtr_power_param) * svtr_param
    -- 长播纠偏
    local no_bias_plvtr = plvtr
    local view_length_sum = _G['explore_stat__view_length_sum'] or 0.0
    local click_count = _G['explore_stat__click_count'] or 0
    local confident_click_count = 100
    local duration_ms = duration_ms or 0
    local smooth_watch_time_ms = math.min(duration_ms, lvtr_sigmoid_bias * 1000.0)
    local avg_watch_time = 1.0 * (view_length_sum +
      math.max(0, (confident_click_count - click_count)) * smooth_watch_time_ms)
        / math.max(click_count, confident_click_count) / 1000
    if (enalbe_plvtr_modulate > 0) then
        no_bias_plvtr = no_bias_plvtr * lvtr_max_value / (1.0 + math.exp(-lvtr_sigmoid_decay_rate * (avg_watch_time - lvtr_sigmoid_bias)))
    end
    -- 加权分计算
    local score = (hot_pctr ^ hot_pctr_param) * (pctr * ctr_param + pltr * ltr_param + pwtr * wtr_param + no_bias_plvtr * lvtr_param + ptr * ptr_param + pwatch_time * watch_time_param) * svtr_factor
    return cascade_pfinish_rate, score
end

-- ftr 和 duration 结合纠偏
function cascade_ftr_duration()
    local duration_sec = duration_ms and duration_ms or 0
    duration_sec = duration_sec / 1000
    local duration_max = cascade_ftr_kai_duration_max and cascade_ftr_kai_duration_max or 300
    local duration_min = cascade_ftr_kai_duration_min and cascade_ftr_kai_duration_min or 3
    local cascade_ftr_kai = cascade_ftr_kai or 0.0
    local cascade_ftr_kai_enable_transfer_1 = cascade_ftr_kai_enable_transfer_1 or 0
    local cascade_ftr_kai_enable_transfer_2 = cascade_ftr_kai_enable_transfer_2 or 0
    if cascade_ftr_kai_enable_transfer_1 > 0 then
      cascade_ftr_kai = math.max(cascade_ftr_kai, 0.0)
    end
    if cascade_ftr_kai_enable_transfer_2 > 0 then
      cascade_ftr_kai = cascade_transfer_ftr_kai_2(cascade_ftr_kai)
    end
    if (duration_sec < 0) then
        duration_sec = 0
    elseif (duration_sec > duration_max) then
        duration_sec = duration_max
    end
    local duration_sec_kelly = duration_sec
    if (duration_sec_kelly < duration_min) then
      duration_sec_kelly = duration_min
    end
    -- 完播*时长
    local cascade_ftr_kai_duration_weight = cascade_ftr_kai_duration_weight or 0.0
    local ftr_power = cascade_ftr_kai_duration_ftr_power or 1.0
    local ftr_offset = cascade_ftr_kai_duration_ftr_offset or 0.0
    local cascade_ftr_kai_duration = (ftr_offset + cascade_ftr_kai ^ ftr_power) * duration_sec ^ cascade_ftr_kai_duration_weight
    -- 凯利公式
    local avg_duration_kelly = cascade_ftr_kelly_avg_duration or 30.0
    local odd = duration_sec_kelly / avg_duration_kelly + 1
    local cascade_ftr_kai_kelly = cascade_ftr_kai - (1 - cascade_ftr_kai) / odd + 1.0
    return cascade_ftr_kai_duration, cascade_ftr_kai_kelly
end

-- IPW
function cascade_ftr_ipw_debias()
  local cascade_ftr_kai = cascade_ftr_kai or 0.0
  local cascade_ipw_opt_ftr = cascade_ftr_kai
  local cascade_ftr_ipw_debias_v1 = cascade_ftr_ipw_debias_v1 or 0
  local cascade_ftr_ipw_debias_v2 = cascade_ftr_ipw_debias_v2 or 0
  local cascade_ftr_ipw_debias_v3 = cascade_ftr_ipw_debias_v3 or 0
  local cascade_ftr_kai_ipw_value = cascade_ftr_kai_ipw_value or ""
  local cascade_ftr_kai_ipw_value_double = cascade_ftr_kai_ipw_value_default or 0.0
  if cascade_ftr_kai_ipw_value ~= "" then
    cascade_ftr_kai_ipw_value_double = tonumber(cascade_ftr_kai_ipw_value)
  end
  if cascade_ftr_ipw_debias_v1 > 0 then
    cascade_ipw_opt_ftr = cascade_ftr_kai/(1-cascade_ftr_kai_ipw_value_double)
  end
  if cascade_ftr_ipw_debias_v2 > 0 then
    cascade_ipw_opt_ftr = cascade_ftr_kai_ipw_value_double
  end
  if cascade_ftr_ipw_debias_v3 > 0 then
    local ftr_alpha = cascade_ftr_ipw_debias_ftr_alpha or 0.0
    local ftr_factor = cascade_ftr_ipw_debias_ftr_factor or 0.0
    local ftr_beta = cascade_ftr_ipw_debias_ftr_beta or 0.0
    local pct_beta = cascade_ftr_ipw_debias_pct_beta or 0.0
    cascade_ipw_opt_ftr = cascade_ftr_kai_ipw_value_double ^ pct_beta * (ftr_alpha + ftr_factor * cascade_ftr_kai) ^ ftr_beta
  end
  return cascade_ipw_opt_ftr
end

function cascade_ftr_redis_key()
    local enable_opt_cascade_ftr_ipw_bucket = enable_opt_cascade_ftr_ipw_bucket or 0
    local ftr_redis_key_opt_prefix = ftr_redis_key_opt_prefix or "ft1_"
    local  cascade_ftr_kai = cascade_ftr_kai or 0.0
    local  duration_ms = duration_ms or 0
    if (enable_opt_cascade_ftr_ipw_bucket == 0) then
      local duration_sec = math.min(600, math.floor(duration_ms / 1000))
      local cascade_ftr_kai = math.min(500, math.floor(cascade_ftr_kai*100))
      local cascade_ftr_kai_redis_key = "fountain_finish_".. duration_sec .. "_" .. cascade_ftr_kai
      return cascade_ftr_kai_redis_key
    else
      -- 不处理图片, 设置一个不存在的key, value为0
      if (duration_ms == 0) then
        return ftr_redis_key_opt_prefix .."pic"
      else
        -- 5s一分桶, 最多120个桶, 时长最大600S
        local duration_bucket = math.min(120, math.floor(duration_ms / 5000))
        -- 0.001一分桶, 最多5000分桶, 完播率最大5.0
        local cascade_ftr_kai_bucket = math.min(5000, math.floor(cascade_ftr_kai*1000))
        local cascade_ftr_kai_redis_key = ftr_redis_key_opt_prefix .. duration_bucket .. "_" .. cascade_ftr_kai_bucket
        return cascade_ftr_kai_redis_key
      end
    end
end

function cascade_longterm_parse_weight()
    local w = {}
    for weight in string.gmatch(fountain_longterm_value_score_weight, "[0-9.]+") do
        table.insert(w, tonumber(weight))
    end
    if (#w == 6) then
        return w
    end
    return {0, 0, 0, 0, 0, 0}
end

function cascade_lvtr_sigmoid_bias_fix_common()
    local fountain_cascade_lvtr_sigmoid_bias_double = fountain_cascade_lvtr_sigmoid_bias_double or 1.92417e+161
    return fountain_cascade_lvtr_sigmoid_bias_double
end

function cascade_longterm_score()
    local w = longterm_weights or {0, 0, 0, 0, 0, 0}
    local longterm_pctr = longterm_pctr or 0
    local longterm_pltr = longterm_pltr or 0
    local longterm_pwtr = longterm_pwtr or 0
    local longterm_pftr = longterm_pftr or 0
    local longterm_plvtr = longterm_plvtr or 0
    local longterm_psvtr = longterm_psvtr or 0
    return w[1] * longterm_pctr + w[2] * longterm_pltr + w[3] * longterm_pwtr + w[4] * longterm_pftr + w[5] * longterm_plvtr + w[6] * longterm_psvtr
end

function parse_prerank_weights()
  local weights = {}
  for w in string.gmatch(fountain_prerank_weights, "[0-9.]+") do
    table.insert(weights, tonumber(w))
  end
  if (#weights == 19) then
    return weights
  end
  return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
end

function cal_prerank_score()
  local w = fountain_prerank_pxtr_weights or {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
  local prerank_pctr = cascade_pctr or 0
  local prerank_pltr = cascade_pltr or 0
  local prerank_pwtr = cascade_pwtr or 0
  local prerank_pftr = cascade_pftr or 0
  local prerank_plvtr = cascade_plvtr or 0
  local prerank_psvtr = cascade_psvtr or 0
  local prerank_ptr = cascade_ptr or 0
  local prerank_pwatch_time = cascade_pwatch_time or 0
  local prerank_pepstr = cascade_pepstr or 0
  local prerank_pcestr = cascade_pcestr or 0
  local prerank_pcmtr = cascade_pcmtr or 0
  local prerank_pwtd = cascade_pwtd or 0
  local prerank_pcltr = cascade_pcltr or 0
  local prerank_phtr = cascade_phtr or 0
  local prerank_pcotr = cascade_pcotr or 0
  local duration = duration_ms / 1000.0 or 0
  local score =  w[1] * prerank_pctr + w[2] * prerank_pltr + w[3] * prerank_pwtr + w[4] * prerank_pftr + w[5] * prerank_plvtr
  local score2 = w[6] * prerank_psvtr + w[7] * prerank_ptr + w[8] * prerank_pwatch_time + w[9] * prerank_pepstr + w[10] * prerank_pcestr
  local score3 = w[11] * prerank_pcmtr + w[12] * prerank_pwtd + w[13] * prerank_pcltr + w[14] * prerank_phtr + w[15] * prerank_pcotr
  local score4 = w[16] * prerank_pcotr * duration + w[17] * prerank_pcotr * prerank_plvtr + w[18] * prerank_pcotr * prerank_pwatch_time + w[19] * prerank_pcotr * prerank_pctr
  return score + score2 + score3 + score4
end

function calc_pc_combine_score()
  local pctr = cascade_pctr or 0
  local pc = cascade_pcotr or 0
  local plvtr = cascade_plvtr or 0
  local pvtr = cascade_pwatch_time or 0
  local duration = duration_ms / 1000.0 or 0
  local pc_evtr = pctr * pc
  local pc_duration = duration * pc
  local pc_lvtr = plvtr * pc
  local pc_vtr = pvtr * pc
  return pc_duration, pc_evtr, pc_lvtr, pc_vtr
end


function calc_cascade_ftr_diff_score()
  local view_length_sum = _G['explore_stat__view_length_sum'] and _G["explore_stat__view_length_sum"] or 0
  local click_count = _G['explore_stat__click_count'] and _G["explore_stat__click_count"] or 0
  local cascade_ftr_kai = cascade_ftr_kai or 0.0
  local duration_ms = duration_ms or 0
  local is_picture = is_picture or 0
  local duration_cluster_default_ftr = duration_cluster_default_ftr or 0.0
  local avg_ftr_min_click_count = fountain_avg_ftr_min_click_count or 100
  local ftr_kai_max_value = fountain_cascade_ftr_kai_max_value or 2.0
  local ftr_diff_score_bias = fountain_cascade_ftr_diff_score_bias or 10.0
  local photo_emp_ftr = 0.0
  if is_picture == 1 or duration_ms == 0 then
    return 0.0, 0.0
  end
  if view_length_sum == 0 or click_count < avg_ftr_min_click_count then
    photo_emp_ftr = duration_cluster_default_ftr
  else
    -- 跟预估的label对齐: 经验值不加平滑, 限制范围
    photo_emp_ftr = math.max(math.min(1.0 * view_length_sum / (click_count * duration_ms), ftr_kai_max_value), 0.0)
  end
  return photo_emp_ftr, cascade_ftr_kai - photo_emp_ftr + ftr_diff_score_bias
end

--questionnaire_score
function calc_questionnaire_score()
  local fountain_questionnaire_filter_min_total_count = fountain_questionnaire_filter_min_total_count or 100
  local pos_threshold = fountain_questionnaire_score_pos_threshold or 0.5
  local neg_threshold = fountain_questionnaire_score_neg_threshold or 0.25
  local unsure_threshold = fountain_questionnaire_score_unsure_threshold or 0.5
  local use_global = fountain_questionnaire_score_use_global or 0
  local positive_count = _G['explore_questionnaire_info__positive_count'] or 0
  local negative_count = _G['explore_questionnaire_info__negative_count'] or 0
  local unsure_count = _G['explore_questionnaire_info__unsure_count'] or 0
  if use_global > 0 then
    positive_count = positive_count + (_G['questionnaire_info__positive_count'] or 0)
    negative_count = negative_count + (_G['questionnaire_info__negative_count'] or 0)
    unsure_count = unsure_count + (_G['questionnaire_info__unsure_count'] or 0)
  end
  local total_count = positive_count + negative_count + unsure_count
  local questionnaire_score = 0.0
  if (total_count > 0 and total_count > fountain_questionnaire_filter_min_total_count) then
    pos_rate = positive_count / total_count
    neg_rate = negative_count / total_count
    unsure_rate = unsure_count / total_count
    if (pos_rate > pos_threshold and neg_rate < neg_threshold and unsure_rate < unsure_threshold) then
      questionnaire_score = 1.0
    end
  end
  return questionnaire_score
end

--questionnaire_score boost
function calc_questionnaire_boost()
  local questionnaire_boost_ratio = cascade_questionnaire_boost_ratio or 1.0
  local questionnaire_boost_threshold = cascade_questionnaire_boost_threshold or 1.0
  local questionnaire_score = questionnaire_score or 0.0
  local cascade_ensemble_score = cascade_ensemble_score
  local cascade_ensemble_score_adjust = cascade_ensemble_score or 1.0
  if (questionnaire_score > questionnaire_boost_threshold) then
    cascade_ensemble_score_adjust = cascade_ensemble_score_adjust * questionnaire_boost_ratio
  end
  return cascade_ensemble_score_adjust
end

function cal_action_once_score()
  local pctr = cascade_pctr and cascade_pctr * 1.0 or 0.0
  local plvtr = cascade_plvtr and cascade_plvtr * 1.0 or 0.0
  local pfintr_quantile = cascade_ipw_opt_ftr and cascade_ipw_opt_ftr * 1.0 or 0.0 -- 完播率分位数
  local pslide = cascade_slide_kai and cascade_slide_kai * 1.0 or 0.0
  local pwatch_time = cascade_pwatch_time and cascade_pwatch_time * 1.0 or 0.0
  local pwtd = cascade_pwtd and cascade_pwtd * 1.0 or 0.0
  local pwtd_kai = cascade_wtd_kai and cascade_wtd_kai * 1.0 or 0.0
  local pftr_duration = cascade_ftr_kai_duration and cascade_ftr_kai_duration * 1.0 or 0.0 -- fintr * duration
  local pltr = cascade_pltr and cascade_pltr * 1.0 or 0.0
  local pwtr = cascade_pwtr and cascade_pwtr * 1.0 or 0.0
  local pftr = cascade_pftr and cascade_pftr * 1.0 or 0.0
  local pcmtr = cascade_pcmtr and cascade_pcmtr * 1.0 or 0.0
  local pcmef = cascade_pcestr and cascade_pcestr * 1.0 or 0.0
  local pptr = cascade_ptr and cascade_ptr * 1.0 or 0.0
  local pepstr = cascade_pepstr and cascade_pepstr * 1.0 or 0.0
  local pcltr = cascade_pcltr and cascade_pcltr * 1.0 or 0.0
  local phtr = cascade_phtr and cascade_phtr * 1.0 or 0.0
  -- cal action_once_watchtime_score
  local pctr_score = math.min(pctr * cascade_action_once_watchtime_score_pctr_weight, 1.0)
  local plvtr_score = math.min(plvtr * cascade_action_once_watchtime_score_plvtr_weight, 1.0)
  local pfintr_quantile_score = math.min(pfintr_quantile * cascade_action_once_watchtime_score_pfintr_quantile_weight, 1.0)
  local pslide_score = math.min(pslide * cascade_action_once_watchtime_score_pslide_weight, 1.0)
  local pwatch_time_score = math.min(pwatch_time * cascade_action_once_watchtime_score_pwatch_time_weight, 1.0)
  local pwtd_score = math.min(pwtd * cascade_action_once_watchtime_score_pwtd_weight, 1.0)
  local pwtd_kai_score = math.min(pwtd_kai * cascade_action_once_watchtime_score_pwtd_kai_weight, 1.0)
  local pftr_duration_score = math.min(pftr_duration * cascade_action_once_watchtime_score_pftr_duration_weight, 1.0)
  local phtr_score = math.min(phtr * cascade_action_once_interact_score_phtr_weight, 1.0)
  -- cal action_once_interact_score
  local pltr_score = math.min(pltr, 1.0)
  local pwtr_score = math.min(pwtr, 1.0)
  local pftr_score = math.min(pftr, 1.0)
  local pcmtr_score = math.min(pcmtr, 1.0)
  local pcmef_score = math.min(pcmef, 1.0)
  local pptr_score = math.min(pptr, 1.0)
  local pepstr_score = math.min(pepstr, 1.0)
  local pcltr_score = math.min(pcltr, 1.0)

  local action_once_interact_score = (1.0 - phtr_score) * (1.0 - (1.0 - pltr_score) * (1.0 - pwtr_score) * (1.0 - pftr_score) * (1.0 - pcmtr_score) * (1.0 - pcmef_score) * (1.0 - pptr_score) * (1.0 - pepstr_score) * (1.0 - pcltr_score))
  local action_once_watchtime_score = (1.0 - phtr_score) * (1.0 - (1.0 - pctr_score) * (1.0 - plvtr_score) * (1.0 - pslide_score) * (1.0 - pwatch_time_score) * (1.0 - pwtd_score) * (1.0 - pfintr_quantile_score) * (1.0 - pftr_duration_score) * (1.0 - pwtd_kai_score))
  return action_once_interact_score, action_once_watchtime_score
end

function cascade_filter_score_calc()
  local pltr_threshold = fountain_cascade_filter_pltr_threshold or 0.0
  local pwtr_threshold = fountain_cascade_filter_pwtr_threshold or 0.0
  local pcmtr_threshold = fountain_cascade_filter_pcmtr_threshold or 0.0
  local pcestr_threshold = fountain_cascade_filter_pcestr_threshold or 0.0
  local psvtr_threshold = fountain_cascade_filter_psvtr_threshold or 1.0
  local pwatch_time_threshold = fountain_cascade_filter_pwatch_time_threshold or 0.0
  local pwtd_threshold = fountain_cascade_filter_pwtd_threshold or 0.0
  local ftr_kai_threshold = fountain_cascade_filter_ftr_kai_threshold or 0.0
  local pctr_threshold = fountain_cascade_filter_pctr_threshold or 0.0
  local slide_kai_threshold = fountain_cascade_filter_slide_kai_threshold or 0.0
  local pepstr_threshold = fountain_cascade_filter_pepstr_threshold or 0.0

  local pltr = cascade_pltr or 1.0
  local pwtr = cascade_pwtr or 1.0
  local pcmtr = cascade_pcmtr or 1.0
  local pcestr = cascade_pcestr or 1.0
  local psvtr = cascade_psvtr or 0.0
  local pwatch_time = cascade_pwatch_time or 1.0
  local pwtd = cascade_pwtd or 1.0
  local ftr_kai = cascade_ftr_kai or 1.0
  local pctr = cascade_pctr or 1.0
  local slide_kai = cascade_slide_kai or 1.0
  local pepstr = cascade_pepstr or 1.0


  local cascade_filter_score  = 0
  if pltr < pltr_threshold and pwtr < pwtr_threshold and pcmtr < pcmtr_threshold and pcestr < pcestr_threshold and psvtr > psvtr_threshold and pwatch_time < pwatch_time_threshold and pwtd < pwtd_threshold and ftr_kai < ftr_kai_threshold and pctr < pctr_threshold and slide_kai < slide_kai_threshold and pepstr < pepstr_threshold then
    cascade_filter_score = 1
  end
  return cascade_filter_score
end

-- 将 [-2, 1e-4] 范围值映射到 [0, 1e-4]
function cascade_transfer_ftr_kai_2(ftr_kai)
  if ftr_kai <= -2.0 then
    return 0.0
  elseif(ftr_kai >= 1e-4) then
    return ftr_kai
  else
    return 1e-4 * (ftr_kai + 2.0) / (2.0 + 1e-4)
  end
end

function calc_cascade_phtr_discount_score()
  local pow_weight = fountain_cascade_phtr_discount_score_pow_weight or 1.0
  local weight = fountain_cascade_phtr_discount_score_weight or 0.0
  local cascade_score =  cascade_score or 0.0
  local cascade_phtr = cascade_phtr  or 0.0

  local discount_htr = 1.0 - (cascade_phtr ^ pow_weight) * weight
  local discount_score = cascade_score * discount_htr

  return discount_score

end