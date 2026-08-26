-- 用于rerank
function handle_common_attr()
    if (enable_fountain_rerank_v4_dpp_add_new_server == 1) then
        return fountain_rerank_limit_size_produce_v4, fountain_rerank_hetu_level_one_winsize_produce_v4, fountain_rerank_kess_service_v2_produce_v4, fountain_rerank_hetu_level_one_39_type_times_produce_v4
    else
        return fountain_rerank_limit_size, fountain_rerank_hetu_level_one_winsize, fountain_rerank_kess_service_v2, fountain_rerank_hetu_level_one_39_type_times
    end
end

function handle_common_attr_new_ab()
    if (fountain_rerank_enable_dpp == 1 and enable_fountain_rerank_v4_dpp_add_new_server == 1) then
        fountain_rerank_enable_dpp = 1
    else
        fountain_rerank_enable_dpp = 0
    end

    if (fountain_rerank_the_diversity_bug == 0 and enable_fountain_rerank_v4_dpp_add_new_server == 1) then
        fountain_rerank_the_diversity_bug = 0
    else
        fountain_rerank_the_diversity_bug = 1
    end
    return fountain_rerank_enable_dpp, fountain_rerank_the_diversity_bug
end

function handle_common_attr_new_ab_splash()
    if (enable_use_fountain_splash_rerank == 1 and enable_fountain_rerank_v4_dpp_add_new_server == 1) then
        enable_use_fountain_splash_rerank = 1
    else
        enable_use_fountain_splash_rerank = 0
    end
    return enable_use_fountain_splash_rerank
end

function convert_photo_info_attr()
    local hetu_level_one_array = _G['hetu_tag_level_info__hetu_level_one'] or {}
    local hetu_level_two_array = _G['hetu_tag_level_info__hetu_level_two'] or {}
    local hetu_level_five_array = _G['hetu_tag_level_info__hetu_level_five'] or {}
    local explore_view_length_sum = _G['explore_stat__view_length_sum'] or 0
    local explore_click_count = _G['explore_stat__click_count'] or 0

    local hetu_level_one_attr = hetu_level_one_array[1] or nil
    local hetu_level_two_attr = hetu_level_two_array[1] or nil
    local hetu_level_two_attr2 = hetu_level_two_array[2] or nil
    local hetu_level_two_attr3 = hetu_level_two_array[3] or nil
    local hetu_level_five_attr = hetu_level_five_array[1] or nil
    local hetu_level_five_attr2 = hetu_level_five_array[2] or nil
    local hetu_level_five_attr3 = hetu_level_five_array[3] or nil
    local empirical_watchtime = explore_view_length_sum / (explore_click_count + 100.0)

    local duration_ms = _G['duration_ms'] or 0
    local duration_0_7s = nil
    local duration_7_9s = nil
    local duration_9_12s = nil
    local duration_12_17s = nil
    local duration_17_20s = nil
    local duration_20_58s = nil
    local duration_gt_58s = nil
    local duration_gt_120s = nil
    local duration_40_60s = nil
    local hetu_level_one_6_type = 0
    local hetu_level_one_9_type = 0
    local hetu_level_one_28_type = 0
    local hetu_level_one_39_type = 0
    if (fountain_rerank_the_diversity_bug > 0) then
        duration_0_7s = 0
        duration_7_9s = 0
        duration_9_12s = 0
        duration_12_17s = 0
        duration_17_20s = 0
        duration_20_58s = 0
        duration_gt_58s = 0
        duration_gt_120s = 0
    end
    if (duration_ms > 120000) then
        duration_gt_120s = 1
    elseif (duration_ms > 58000) then
        duration_gt_58s = 1
    elseif (duration_ms > 20000) then
        duration_20_58s = 1
    elseif (duration_ms > 17000) then
        duration_17_20s = 1
    elseif (duration_ms > 12000) then
        duration_12_17s = 1
    elseif (duration_ms > 9000) then
        duration_9_12s = 1
    elseif (duration_ms > 7000) then
        duration_7_9s = 1
    else
        duration_0_7s = 1
    end

    if (duration_ms > 40000 and duration_ms < 60000) then
        duration_40_60s = 1
    end
    if (#hetu_level_one_array > 0) then
        for k,v in ipairs(hetu_level_one_array) do
            if (v ~= nil and v == 6) then
                hetu_level_one_6_type = 1
            end
            if (v ~= nil and v == 9) then
                hetu_level_one_9_type = 1
            end
            if (v ~= nil and v == 28) then
                hetu_level_one_28_type = 1
            end
            if (v ~= nil and v == 39) then
                hetu_level_one_39_type = 1
            end
        end
    end
    return hetu_level_one_attr, hetu_level_two_attr, hetu_level_two_attr2, hetu_level_two_attr3, hetu_level_five_attr, hetu_level_five_attr2, hetu_level_five_attr3, empirical_watchtime, duration_0_7s, duration_7_9s, duration_9_12s, duration_12_17s, duration_17_20s, duration_20_58s, duration_gt_58s, duration_gt_120s, duration_40_60s, hetu_level_one_6_type, hetu_level_one_9_type, hetu_level_one_28_type, hetu_level_one_39_type
end

function full_rank_score_cal_splash()
    local score_factor = duration_ms > (fountain_rerank_duration_adjust_level * 1000) and fountain_rerank_duration_adjust_param or 1.0
    local pctr = (fullrank_sim_pevtr or 0.0) * score_factor * (1.0 + (pctr_duration_debias_coffe or 0.0))
    local pltr = (fullrank_sim_pltr or 0.0) * score_factor * (1.0 + (pltr_duration_debias_coffe or 0.0))
    local pwtr = (fullrank_sim_pwtr or 0.0) * score_factor * (1.0 + (pwtr_duration_debias_coffe or 0.0))
    local pftr = (fullrank_sim_pftr or 0.0) * score_factor * (1.0 + (pftr_duration_debias_coffe or 0.0))
    local plvtr = (fullrank_sim_longview_score_no_bias or 0.0) * score_factor * (1.0 + (plvtr_duration_debias_coffe or 0.0))
    local psvr = (fullrank_sim_psvr or 0.0) * score_factor
    local pptr = (fullrank_sim_pptr or 0.0) * score_factor
    local pcmtr = (fullrank_sim_pcmtr or 0.0) * score_factor * (1.0 + (pcmtr_duration_debias_coffe or 0.0))
    local pcmef = (fullrank_sim_pcmef or 0.0) * score_factor
    local pepstr = (fullrank_sim_pepstr or 0.0) * score_factor
    local pwatchtime = (fullrank_sim_pwatchtime_no_bias or 0.0) * score_factor
    local pwatchtimeori= (fullrank_sim_pvtr or 0.0) * score_factor
    local l2r_score = (fullrank_ltr_score or 0.0) * score_factor
    local out_ctr = (fullrank_sim_out_pctr or 0.0) * score_factor
    local evtr_v2 = (fullrank_detail_new_pevtr_v2 or 0.0) * score_factor
    local plstr = (fullrank_sim_lstr or 0.0) * score_factor
    local neg_feedback_discount = (fullrank_ensemble_score or 0.0) * score_factor
    local cltr = (fullrank_sim_pcltr or 0.0) * score_factor * (1.0 + (pcltr_duration_debias_coffe or 0.0))
    local lvtr_ori = (fullrank_sim_plvtr or 0.0) * score_factor
    local pfintr = (fullrank_sim_pfintr or 0.0) * score_factor
    local finish_score = (fullrank_ltr_v4_fountain_finish_rate or 0.0) * score_factor
    local next_score = (fullrank_ltr_v4_fountain_next or 0.0) * score_factor
    local slide_score = (fountain_splash_slide or 0.0) * score_factor
    local opportunity_cost_score = (fullrank_opportunity_cost_score or 0.0) * score_factor
    local ada_score = (fullrank_ada_xtr_score or 0.0) * score_factor
    local trans_pvtr = (fullrank_trans_pvtr_score or 0.0) * score_factor
    local act_ctr = (fullrank_act_ctr or 0.0) * score_factor
    local topk_mgs_expected_score = (topk_mgs_expected_score or 0.0) * score_factor
    local pcpr = (fullrank_sim_pcpr or 0.0) * score_factor
    local act_wtd = (fullrank_act_wtd or 0.0) * score_factor
    local pure_val = (fullrank_pure_value_score or 0.0) * score_factor
    local fusion_pctr = pctr * score_factor^score_factor_coffe
    local fusion_pcltr = cltr * score_factor^score_factor_coffe
    local cl_play_time = (fullrank_cl_play_time or 0.0) * score_factor
    local fullrank_min_act_rank_reci = (fullrank_min_act_rank_reci or 0.0) * score_factor
    local ori_pswptr = (fullrank_ori_pswptr or 0.0) * score_factor
    return pctr, pltr, pwtr, pftr, pptr, pcmtr, plvtr, psvr, psvr, pwatchtime, pwatchtimeori, l2r_score, pepstr, out_ctr, neg_feedback_discount, evtr_v2, plstr, cltr, pcmef, lvtr_ori, pfintr, finish_score, next_score, slide_score, opportunity_cost_score, ada_score, trans_pvtr, act_ctr, topk_mgs_expected_score, pcpr, act_wtd, pure_val, fusion_pctr, fusion_pcltr, cl_play_time, fullrank_min_act_rank_reci, ori_pswptr
end

function full_rank_score_cal_new()
    local score_factor = duration_ms > (fountain_rerank_duration_adjust_level * 1000) and fountain_rerank_duration_adjust_param or 1.0
    local pctr = (fullrank_sim_pevtr or 0.0) * score_factor * (1.0 + (pctr_duration_debias_coffe or 0.0))
    local pltr = (fullrank_sim_pltr or 0.0) * score_factor * (1.0 + (pltr_duration_debias_coffe or 0.0))
    local pwtr = (fullrank_sim_pwtr or 0.0) * score_factor * (1.0 + (pwtr_duration_debias_coffe or 0.0))
    local pftr = (fullrank_sim_pftr or 0.0) * score_factor * (1.0 + (pftr_duration_debias_coffe or 0.0))
    local plvtr = (fullrank_sim_longview_score_no_bias or 0.0) * score_factor * (1.0 + (plvtr_duration_debias_coffe or 0.0))
    local psvr = (fullrank_sim_psvr or 0.0) * score_factor
    local pptr = (fullrank_sim_pptr or 0.0) * score_factor
    local pcmtr = (fullrank_sim_pcmtr or 0.0) * score_factor * (1.0 + (pcmtr_duration_debias_coffe or 0.0))
    local pcmef = (fullrank_sim_pcmef or 0.0) * score_factor
    local pepstr = (fullrank_sim_pepstr or 0.0) * score_factor
    local pwatchtime = (fullrank_sim_pwatchtime_no_bias or 0.0) * score_factor
    local pwatchtimeori= (fullrank_sim_pvtr or 0.0) * score_factor
    local l2r_score = (fullrank_ltr_score or 0.0) * score_factor
    local out_ctr = (fullrank_sim_out_pctr or 0.0) * score_factor
    local evtr_v2 = (fullrank_detail_new_pevtr_v2 or 0.0) * score_factor
    local plstr = (fullrank_sim_lstr or 0.0) * score_factor
    local neg_feedback_discount = (fullrank_ensemble_score or 0.0) * score_factor
    local cltr = (fullrank_sim_pcltr or 0.0) * score_factor * (1.0 + (pcltr_duration_debias_coffe or 0.0))
    local lvtr_ori = (fullrank_sim_plvtr or 0.0) * score_factor
    local pfintr = (fullrank_sim_pfintr or 0.0) * score_factor
    local finish_score = (fullrank_ltr_v4_fountain_finish_rate or 0.0) * score_factor
    local next_score = (fullrank_ltr_v4_fountain_next or 0.0) * score_factor
    local slide_score = (fountain_splash_slide or 0.0) * score_factor
    local opportunity_cost_score = (fullrank_opportunity_cost_score or 0.0) * score_factor
    local ada_score = (fullrank_ada_xtr_score or 0.0) * score_factor
    local trans_pvtr = (fullrank_trans_pvtr_score or 0.0) * score_factor
    local act_ctr = (fullrank_act_ctr or 0.0) * score_factor
    local topk_mgs_expected_score = (topk_mgs_expected_score or 0.0) * score_factor
    local pcpr = (fullrank_sim_pcpr or 0.0) * score_factor
    local act_wtd = (fullrank_act_wtd or 0.0) * score_factor
    local pure_val = (fullrank_pure_value_score or 0.0) * score_factor
    local fusion_pctr = pctr * score_factor^score_factor_coffe
    local fusion_pcltr = cltr * score_factor^score_factor_coffe
    local cl_play_time = (fullrank_cl_play_time or 0.0) * score_factor
    local fullrank_min_act_rank_reci = (fullrank_min_act_rank_reci or 0.0) * score_factor
    local ori_pswptr = (fullrank_ori_pswptr or 0.0) * score_factor
    local lt_interest = (long_term_interest_ee_score or 0.0) * score_factor
    local comment_ltr = (comment_ltr or 0.0) * score_factor
    local xgb_ltr = (xgb_ltr or 0.0) * score_factor
    local pos0 = (rerank_generate_pos0 or 0.0) * score_factor
    local next0 = (rerank_generate_next0 or 0.0) * score_factor
    local pctr_hetu = (fullrank_sim_click_score_debias_hetu or 0.0) * score_factor
    local pltr_hetu = (fullrank_sim_pltr_debias_hetu or 0.0) * score_factor
    local pwtr_hetu = (fullrank_sim_pwtr_debias_hetu or 0.0) * score_factor
    local pftr_hetu = (fullrank_sim_pftr_debias_hetu or 0.0) * score_factor
    local pcmtr_hetu = (fullrank_sim_pcmtr_debias_hetu or 0.0) * score_factor
    local pptr_hetu = (fullrank_sim_pptr_debias_hetu or 0.0) * score_factor
    return pctr, pltr, pwtr, pftr, pptr, pcmtr, plvtr, psvr, psvr, pwatchtime, pwatchtimeori, l2r_score, pepstr, out_ctr, neg_feedback_discount, evtr_v2, plstr, cltr, pcmef, lvtr_ori, pfintr, finish_score, next_score, slide_score, opportunity_cost_score, ada_score, trans_pvtr, act_ctr, topk_mgs_expected_score, pcpr, act_wtd, pure_val, fusion_pctr, fusion_pcltr, cl_play_time, fullrank_min_act_rank_reci, ori_pswptr, lt_interest, comment_ltr, xgb_ltr, pos0, next0, pctr_hetu, pltr_hetu, pwtr_hetu, pftr_hetu, pcmtr_hetu, pptr_hetu
end

function convert_duration_attr()
    local duration_0_7s_new = duration_0_7s and duration_0_7s * 1.0 or 0.0
    local duration_7_9s_new = duration_7_9s and duration_7_9s * 1.0 or 0.0
    local duration_9_12s_new = duration_9_12s and duration_9_12s * 1.0 or 0.0
    local duration_12_17s_new = duration_12_17s and duration_12_17s * 1.0 or 0.0
    local duration_17_20s_new = duration_17_20s and duration_17_20s * 1.0 or 0.0
    local duration_20_58s_new = duration_20_58s and duration_20_58s * 1.0 or 0.0
    local duration_gt_58s_new = duration_gt_58s and duration_gt_58s * 1.0 or 0.0
    local duration_gt_120s_new = duration_gt_120s and duration_gt_120s * 1.0 or 0.0
    local duration_40_60s_new = duration_40_60s and duration_40_60s * 1.0 or 0.0

    return duration_0_7s_new, duration_7_9s_new, duration_9_12s_new, duration_12_17s_new, duration_17_20s_new, duration_20_58s_new, duration_gt_58s_new, duration_gt_120s_new, duration_40_60s_new
end

function splash_calculate()
    local real_show_count = _G['explore_stat__real_show_count'] or 10
    real_show_count = real_show_count > 0 and real_show_count or 10
    local click_count = _G['explore_stat__click_count'] or 1
    click_count = click_count > 0 and click_count or 1
    local like_count = _G['explore_stat__like_count'] or 0
    like_count = like_count > 0 and like_count or 0
    local follow_count = _G['explore_stat__follow_count'] or 0
    follow_count = follow_count > 0 and follow_count or 0
    local forward_count = _G['explore_stat__forward_count'] or 0
    forward_count = forward_count > 0 and forward_count or 0
    local profile_enter_count = _G['explore_stat__profile_enter_count'] or 0
    profile_enter_count = profile_enter_count > 0 and profile_enter_count or 0
    local comment_count = _G['explore_stat__comment_count'] or 0
    comment_count = comment_count > 0 and comment_count or 0
    local negative_count = _G['explore_stat__negative_count'] or 0
    negative_count = negative_count > 0 and negative_count or 0
    local empirical_ctr = click_count * 1.0 / real_show_count
    local empirical_ltr = like_count * 1.0 / click_count
    local empirical_wtr = follow_count * 1.0 / click_count
    local empirical_ftr = forward_count * 1.0 / click_count
    local empirical_ptr = profile_enter_count * 1.0 / click_count
    local empirical_cmtr = comment_count * 1.0 / click_count
    local empirical_htr = negative_count * 1.0 / click_count
    return empirical_ctr, empirical_ltr, empirical_wtr, empirical_ftr, empirical_ptr, empirical_cmtr, empirical_htr
end

function cal_user_weight_factor(request_xtr, request_ratio, user_xtr, emp_xtr, min_ratio, max_ratio)
  local weight_ori = 1.0
  if emp_xtr <= 0 then
    return weight_ori
  end
  local user_xtr = user_xtr or 0.0
  local request_xtr = request_xtr or 0.0
  local adjust_user_xtr = request_ratio * request_xtr + (1 - request_ratio) * user_xtr
  if adjust_user_xtr <= 0 then
    return 1.0
  end
  local weight_factor = math.min(math.max(adjust_user_xtr / emp_xtr, min_ratio), max_ratio)
  return weight_factor
end

function cal_user_emp_ada_weight_factor()
  local ratio_min = fountain_rerank_ensemble_power_weight_adjust_ratio_min or 1.0
  local ratio_max = fountain_rerank_ensemble_power_weight_adjust_ratio_max or 1.0
  local request_ratio = fountain_rerank_ensemble_power_weight_adjust_request_ratio or 0.0
  local like_weight = cal_user_weight_factor(pltr_avg, request_ratio, userExpLtr, fountain_rerank_ensemble_power_weight_fullrank_ltr_emp, ratio_min, ratio_max)
  local follow_weight = cal_user_weight_factor(pwtr_avg, request_ratio, userExpWtr, fountain_rerank_ensemble_power_weight_fullrank_wtr_emp, ratio_min, ratio_max)
  local comment_weight = cal_user_weight_factor(pcmtr_avg, request_ratio, userExpCmtr, fountain_rerank_ensemble_power_weight_fullrank_cmtr_emp, ratio_min, ratio_max)
  local profile_weight = cal_user_weight_factor(pptr_avg, request_ratio, userExpPtr, fountain_rerank_ensemble_power_weight_fullrank_ptr_emp, ratio_min, ratio_max)
  local forward_weight = cal_user_weight_factor(pftr_avg, request_ratio, userExpFtr, fountain_rerank_ensemble_power_weight_fullrank_ftr_emp, ratio_min, ratio_max)
  local epstr_weight = cal_user_weight_factor(pepstr_avg, request_ratio, userExpEptr, fountain_rerank_ensemble_power_weight_fullrank_epstr_emp, ratio_min, ratio_max)
  local evtr_weight = cal_user_weight_factor(pevtr_avg, request_ratio, user_emp_evtr, fountain_rerank_ensemble_power_weight_fullrank_evtr_emp, ratio_min, ratio_max)
  local lvtr_weight = cal_user_weight_factor(plvtr_avg, request_ratio, user_emp_lvtr, fountain_rerank_ensemble_power_weight_fullrank_lvtr_emp, ratio_min, ratio_max)
  local fintr_weight = cal_user_weight_factor(pfintr_avg, request_ratio, user_emp_watch_time, fountain_rerank_ensemble_power_weight_fullrank_fintr_emp, ratio_min, ratio_max)
  return like_weight, follow_weight, comment_weight, profile_weight, forward_weight, epstr_weight, evtr_weight, lvtr_weight, fintr_weight
end

function calculate()
    local tag = tag or 0
    tag = tag > 0 and tag or nil
    local fourth_level_id = _G['author__category_detail__fourth_level_id'] or 0 
    fourth_level_id = fourth_level_id > 0 and fourth_level_id or nil
    local hetu_cluster = _G['hetu_tag_level_info_v2__hetu_cluster_id'] or 0 
    hetu_cluster = hetu_cluster > 0 and hetu_cluster or nil
    local duration_ms = duration_ms or 0
    local real_show_count = _G['explore_stat__real_show_count'] or 10
    real_show_count = real_show_count > 0 and real_show_count or 10
    local click_count = _G['explore_stat__click_count'] or 1
    click_count = click_count > 0 and click_count or 1
    local like_count = _G['explore_stat__like_count'] or 0
    like_count = like_count > 0 and like_count or 0
    local follow_count = _G['explore_stat__follow_count'] or 0
    follow_count = follow_count > 0 and follow_count or 0
    local forward_count = _G['explore_stat__forward_count'] or 0
    forward_count = forward_count > 0 and forward_count or 0
    local profile_enter_count = _G['explore_stat__profile_enter_count'] or 0
    profile_enter_count = profile_enter_count > 0 and profile_enter_count or 0
    local comment_count = _G['explore_stat__comment_count'] or 0
    comment_count = comment_count > 0 and comment_count or 0
    local negative_count = _G['explore_stat__negative_count'] or 0
    negative_count = negative_count > 0 and negative_count or 0
    local short_duration_threshold = 7000
    local short_duration_variant_attr = duration_ms < short_duration_threshold and 1 or nil
    local long_duration_threshold = 32000
    local long_duration_variant_attr = duration_ms > long_duration_threshold and 1 or nil
    local lt20s_duration_variant_attr = duration_ms <= long_duration_threshold and 1 or nil
    local empirical_ctr = click_count * 1.0 / real_show_count
    local empirical_ltr = like_count * 1.0 / click_count
    local empirical_wtr = follow_count * 1.0 / click_count
    local empirical_ftr = forward_count * 1.0 / click_count
    local empirical_ptr = profile_enter_count * 1.0 / click_count
    local empirical_cmtr = comment_count * 1.0 / click_count
    local empirical_htr = negative_count * 1.0 / click_count
    return tag, fourth_level_id, hetu_cluster, short_duration_variant_attr, long_duration_variant_attr, lt20s_duration_variant_attr, empirical_ctr, empirical_ltr, empirical_wtr, empirical_ftr, empirical_ptr, empirical_cmtr, empirical_htr
end

function only_trick()
    return 0
end

function splash_convert_photo_info_attr()

    local duration_ms = _G['duration_ms'] or 0
    local duration_0_7s = nil
    local duration_7_9s = nil
    local duration_9_12s = nil
    local duration_12_17s = nil
    local duration_17_20s = nil
    local duration_20_58s = nil
    local duration_gt_58s = nil
    local duration_gt_120s = nil
    if (duration_ms > 120000) then
        duration_gt_120s = 1
    elseif (duration_ms > 58000) then
        duration_gt_58s = 1
    elseif (duration_ms > 20000) then
        duration_20_58s = 1
    elseif (duration_ms > 17000) then
        duration_17_20s = 1
    elseif (duration_ms > 12000) then
        duration_12_17s = 1
    elseif (duration_ms > 9000) then
        duration_9_12s = 1
    elseif (duration_ms > 7000) then
        duration_7_9s = 1
    else
        duration_0_7s = 1
    end
    return duration_0_7s, duration_7_9s, duration_9_12s, duration_12_17s, duration_17_20s, duration_20_58s, duration_gt_58s, duration_gt_120s
end


function cal_rerank_list_score()
    local rerank_context_new = rerank_context_new or {}
    local rerank_list_next = rerank_list_next or {}
    local next_score_weight = fountain_rerank_list_next_weight or 0.0
    local use_odd = fountain_rerank_predict_use_odd_score or 0
    local score = 0
    if #rerank_context_new == 6 and #rerank_list_next == 1 then
        for k,v in ipairs(rerank_context_new) do
            if use_odd > 0 then
                score = score +  v / math.max(1.0 - v, 1e-4)
            else
                score = score + v
            end
        end
        score = score + rerank_list_next[1] ^ next_score_weight
    end
    return score
end

function cal_rerank_list_score_splash()
    local rerank_context_new = rerank_context_new or {}
    local use_odd = fountain_rerank_predict_use_odd_score or 0
    local score = 0
    if #rerank_context_new == 2 then
        for k,v in ipairs(rerank_context_new) do
            if use_odd > 0 then
                score = score +  v / math.max(1.0 - v, 1e-4)
            else
                score = score + v
            end
        end
    end
    return score
end


function splash_seq_max_size() 
    local request_num = request_num or 10
    if (request_num == 10) then
        return fountain_splash_rerank_gen_final_seq_max_size
    end
    return fountain_splash_rerank_gen_seed_ensemble_seq_max_size_4
end

function adjust_quota_size(size, factor, increase_quota_status)
    local factor = factor and factor or 1.0
    local increase_quota_status = increase_quota_status and increase_quota_status or 0
    local size = size
    if (increase_quota_status > 0) then
        size = math.floor(size * factor)
    end
    return size
end

function adjust_rerank_limit_size()
    return adjust_quota_size(fountain_rerank_limit_size,fountain_rerank_limit_size_increase_quota_factor, increase_quota_status)
end

function adjust_splash_rerank_limit_size()
    return adjust_quota_size(fountain_splash_rerank_limit_size, fountain_splash_rerank_limit_size_increase_quota_factor, increase_quota_status)
end

function duration_change(seq)
    local duration_ms = duration_ms or 0
    local duration_0_7s = nil
    local duration_7_9s = nil
    local duration_9_12s = nil
    local duration_12_17s = nil
    local duration_17_20s = nil
    local durtaion_20_58s = nil
    local durtaion_58_120s = nil
    local duration_gt_120s = nil
    if (duration_ms < 7000) then
        duration_0_7s = 1
    elseif (duration_ms < 9000) then
        duration_7_9s = 1
    elseif (duration_ms < 12000) then
        duration_9_12s = 1
    elseif (duration_ms < 17000) then
        duration_12_17s = 1
    elseif (duration_ms < 20000) then
        duration_17_20s = 1
    elseif (duration_ms < 58000) then
        durtaion_20_58s = 1
    elseif (duration_ms < 120000) then
        durtaion_58_120s = 1
    else
        duration_gt_120s = 1
    end

    local score = 0.98 ^ seq

    return duration_0_7s, duration_7_9s, duration_9_12s, duration_12_17s, duration_17_20s, durtaion_20_58s, durtaion_58_120s, duration_gt_120s, score
end

function score_change()
    local score = score_before_rerank or 0.0
    local duration_s = duration_s or 0
    local coeff_list = fountain_rerank_duration_coeff_double_list or {}
    local score_coeff = 1.0

    if (duration_s > 0 and #coeff_list == 8) then
        score_coeff = coeff_list[duration_s]
    end

    return score * score_coeff
end
    