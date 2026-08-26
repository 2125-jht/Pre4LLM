-- 精排打分变换公式
function calc_fullrank_score()
  local fullrank_sim_pvtr = fullrank_sim_pvtr or 0
  local fullrank_sim_click_score = fullrank_sim_click_score or 0
  local fullrank_sim_plvtr = fullrank_sim_plvtr or 0
  local fullrank_sim_pwtr = fullrank_sim_pwtr or 0
  local fullrank_sim_pptr = fullrank_sim_pptr or 0
  local duration_ms = duration_ms or 0
  local picture_variant_attr = picture_variant_attr or 0
  local pwatchtime_ms = math.exp(math.min(6, fullrank_sim_pvtr)) * 1000
  local duration_discount = math.log(duration_ms / 1000 + 1) + fountain_fullrank_duration_discount_weight
  local pwatchtime_no_bias = pwatchtime_ms / duration_discount
  local vtr_max_value = fountain_vtr_max_value and fountain_vtr_max_value or 0.08
  local vtr_sigmoid_decay_rate = fountain_vtr_sigmoid_decay_rate and fountain_vtr_sigmoid_decay_rate or 0.15
  local vtr_smooth_rate = fountain_vtr_smooth_rate and fountain_vtr_smooth_rate or 1500
  local vtr_sigmoid_bias = fountain_vtr_sigmoid_bias and fountain_vtr_sigmoid_bias or 9
  local smooth_watch_time_ms = picture_variant_attr < 1 and math.min(duration_ms, 100000) or 100000
  local click_count = _G['explore_stat__click_count']
  local view_length_sum = _G['explore_stat__view_length_sum']
  local explore_view_length_sum = view_length_sum or 0
  local explore_click_count = click_count or 0
  local lvtr_sigmoid_bias = fountain_fullrank_lvtr_sigmoid_bias or 0
  local avg_watch_time_ms = 1.0 * (explore_view_length_sum + math.max(0, 100 - explore_click_count) * smooth_watch_time_ms) / math.max(100, explore_click_count)
  local no_bias_plvtr = fullrank_sim_plvtr * 2.0 / (1.0 + math.exp(-0.5 * (avg_watch_time_ms / 1000.0 - lvtr_sigmoid_bias)))
  local longview_score = fullrank_sim_click_score * no_bias_plvtr
  if (enable_fountain_longview_score_remove_click_coef > 0) then
     longview_score = no_bias_plvtr
  end

  local vtr_coefficient = 1
  local big_duration_discount = 1.0
  local big_duration_discount_bias = fountian_vtr_big_duration_discount_bias and fountian_vtr_big_duration_discount_bias or 0.005
  local vtr_slope = fountian_vtr_big_duration_discount_slope and fountian_vtr_big_duration_discount_slope or 0.001
  if (enable_fountain_pwatch_time_sigmoid_bias_new > 0) then
    vtr_coefficient = vtr_max_value / (1.0 + math.exp(-1 * vtr_sigmoid_decay_rate * (-1 * duration_ms / vtr_smooth_rate  + vtr_sigmoid_bias))) + 1
    if(duration_ms > 100000) then
      big_duration_discount = 1.0 - math.max(big_duration_discount_bias, 0.001*(duration_ms - 100000)/10000.0)
      if (fountain_vtr_score_discount_fix > 0) then
        big_duration_discount = 1.0 - math.min(big_duration_discount_bias, vtr_slope * (duration_ms - 100000)/10000.0)
      end
    end
    pwatchtime_no_bias = fullrank_sim_pvtr * vtr_coefficient * big_duration_discount
  end

  if (enable_fr_origin_pvtr > 0) then
    pwatchtime_no_bias = fullrank_sim_pvtr
  end

  -- 高阶队列
  local time_weight = fountain_fullrank_sim_pevtr_coef_weight * fullrank_sim_click_score + fountain_fullrank_sim_pvtr_coef_weight * fullrank_sim_pvtr
  local evtr_v2_power_weight = fountain_fullrank_distill_score_evtr_v2_weight and fountain_fullrank_distill_score_evtr_v2_weight or 1.0
  local pvtr_multi_pwtr = time_weight * fullrank_sim_pwtr
  local pvtr_multi_pptr = time_weight * fullrank_sim_pptr
  local evtr_v2 = fullrank_detail_new_pevtr_v2 and fullrank_detail_new_pevtr_v2 or 0.0
  local pwtd_v2 = fullrank_sim_pwtd_v2_playtime and fullrank_sim_pwtd_v2_playtime or 0.0
  local evtr_v2_multi_pwtd_v2 = evtr_v2 * pwtd_v2
  local distill_score = fullrank_distill_rerank_score and fullrank_distill_rerank_score or 0.0
  local evtr_distill_score = (evtr_v2 ^ evtr_v2_power_weight) * distill_score

  -- pvtr
  local pvtr = fullrank_sim_pvtr or 0.0
  if pvtr < 0.0 or pvtr > 1.0 then
    pvtr = 0.0
  elseif pvtr == 1.0 then
    pvtr = fullrank_pvtr_trans_score_threshold
  else
    pvtr = math.min(pvtr / (1.0 - pvtr), fullrank_pvtr_trans_score_threshold)
  end

  -- act vtr
  local act_wtd = fullrank_act_wtd and fullrank_act_wtd or 0.0
  local act_vtr_max = fountain_act_vtr_max and fountain_act_vtr_max or 1.0
  if skip_fountain_act_vtr_norm < 1 then
    act_wtd = act_wtd * act_vtr_max / 200.0
  end
  
  l2r_score = fullrank_ltr_score and fullrank_ltr_score or 0.0
  if skip_fountain_act_vtr_merge < 1 then
    l2r_max = fountain_fullrank_act_l2r_max and fountain_fullrank_act_l2r_max or 0.0
    l2r_weight = fountain_fullrank_act_l2r_merge_weight and fountain_fullrank_act_l2r_merge_weight or 0.0
    vtr_weight = fountain_fullrank_act_vtr_merge_weight and fountain_fullrank_act_vtr_merge_weight or 0.0
    if l2r_score < l2r_max then
      l2r_score = 0.0
    end
    l2r_score = l2r_score + 1.0
    act_wtd = (l2r_score ^ l2r_weight) * (act_wtd ^ vtr_weight)
    if skip_fountain_act_l2r_replace < 1 then
      l2r_score = act_wtd
    end
  end

  return pwatchtime_no_bias, longview_score, pvtr_multi_pwtr, pvtr_multi_pptr, evtr_v2_multi_pwtd_v2, pvtr, act_wtd, l2r_score, evtr_distill_score
end

function wtd_transfer()
  -- wtd score
  local ctr_weight = fountain_ltr_score_ctr_weight and fountain_ltr_score_ctr_weight or 1
  local ltr_ctr = fullrank_ltr_ctr and fullrank_ltr_ctr or 0
  local ltr_score = ctr_weight * ltr_ctr
  return ltr_score
end

function transfer_pvtr()
  local pvtr = fullrank_sim_pvtr or 0.0
  if pvtr < 0.0 or pvtr > 1.0 then
    pvtr = 0.0
  elseif pvtr == 1.0 then
    pvtr = fullrank_pvtr_trans_score_threshold
  else
    pvtr = math.min(pvtr / (1.0 - pvtr), fullrank_pvtr_trans_score_threshold)
  end
  return pvtr
end

function pxtr_sample_debais()
  follow_sample_weight = fountain_fullrank_follow_upsample_weight and fountain_fullrank_follow_upsample_weight or 1.0
  commont_sample_weight = fountain_fullrank_commont_upsample_weight and fountain_fullrank_commont_upsample_weight or 1.0
  pwtr = fullrank_sim_follow_score and fullrank_sim_follow_score or 0.0
  pcmtr = fullrank_sim_pcmtr and fullrank_sim_pcmtr or 0.0
  pwtr = pwtr / (pwtr + follow_sample_weight * (1 - pwtr))
  pcmtr = pcmtr / (pcmtr + follow_sample_weight * (1 - pcmtr))
  return pwtr, pcmtr
end


function calc_diff_score()
  local pctr = fullrank_sim_click_score or 0
  local pctr2 = fullrank_act_ctr or 0
  local plvtr = fullrank_sim_plvtr or 0
  local pwtr = fullrank_sim_pwtr or 0
  local pptr = fullrank_sim_pptr or 0
  local pltr = fullrank_sim_like_score or 0
  local pcmtr = fullrank_sim_pcmtr or 0
  local fintr = fullrank_sim_pfintr or 0
  local pevtr = fullrank_detail_new_pevtr_v2 or 0
  local ctr_weight = fountain_fullrank_diff_ctr_weight or 0
  local ctr_weight2 = fountain_fullrank_diff_ctr_weight_2 or 0
  local lvtr_weight = fountain_fullrank_diff_lvtr_weight or 0
  local wtr_weight = fountain_fullrank_diff_wtr_weight or 0
  local ptr_weight = fountain_fullrank_diff_ptr_weight or 0
  local ltr_weight = fountain_fullrank_diff_ltr_weight or 0
  local cmtr_weight = fountain_fullrank_diff_cmtr_weight or 0
  local fintr_weight = fountain_fullrank_diff_fintr_weight or 0
  local evtr_weight = fountain_fullrank_diff_evtr_weight or 0
  local act_weight = fountain_fullrank_diff_act_weight or 0
  local act_score = wtr_weight * pwtr + ptr_weight * pptr + ltr_weight * pltr + cmtr_weight * pcmtr
  act_score = act_score ^ act_weight
  local vtr_score = (fintr * fintr_weight + plvtr * lvtr_weight) * (pctr * ctr_weight + pctr2 * ctr_weight2 + pevtr * evtr_weight)
  vtr_score = math.exp(vtr_score)
  return act_score * vtr_score
end