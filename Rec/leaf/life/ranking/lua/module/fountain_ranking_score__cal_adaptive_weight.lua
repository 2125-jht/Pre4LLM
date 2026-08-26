function cal_adaptive_factor(_avg, _dev, _norm)
    if _avg == nil or _dev == nil then
        return 1.0
    end
    local factor = _norm * _avg / ((1 - _dev ^ 0.5) ^ 2 + 0.0001)
    if factor > 2.0 then
        factor = 2.0
    end
    if factor < 0.1 then
        factor = 0.1
    end
    return factor
end

function cal_fullrank_adaptive_weights_v2()
    local ratio_min = fountain_ensemble_power_weight_adjust_ratio_min or 1.0
    local ratio_max = fountain_ensemble_power_weight_adjust_ratio_max or 1.0
    local pxtr_avg_weight = fountain_fullrank_user_ada_pxtr_avg_weight or 1.0
    local pxtr_cnt = fullrank_splash_pre_filter_keep_photo_size or 500
    local emp_click_cnt = user_colossus_click_count_ada or 500
    local pxtr_w = pxtr_avg_weight * pxtr_cnt / (pxtr_cnt + emp_click_cnt)
    local pltr_avg = pltr_avg and pltr_avg or 0.0
    local pwtr_avg = pwtr_avg and pwtr_avg or 0.0
    local pcmtr_avg = pcmtr_avg and pcmtr_avg or 0.0
    local pptr_avg = pptr_avg and pptr_avg or 0.0
    local psvr_avg = psvr_avg and psvr_avg or 0.0
    local plvtr_avg = plvtr_avg and plvtr_avg or 0.0
    local pftr_avg = pftr_avg and pftr_avg or 0.0
    local enable_filter_weight_lower_one = enable_filter_fountain_ada_weight_lower_one or 0
    local enable_filter_weight_over_one = enable_filter_fountain_ada_weight_over_one or 0
    if fountain_fullrank_skip_ada_user_emp_xtr == 0 then
        if fountain_fullrank_skip_ada_wt_user_emp_xtr == 0 then
            userExpLvtr = pxtr_w * plvtr_avg + (1 - pxtr_w) * user_emp_lvtr_ada
        end
        if fountain_fullrank_skip_ada_act_user_emp_xtr == 0 then
            userExpLtr = pxtr_w * pltr_avg + (1 - pxtr_w) * user_emp_ltr_ada
            userExpWtr = pxtr_w * pwtr_avg + (1- pxtr_w) * user_emp_wtr_ada
            userExpCmtr = pxtr_w * pcmtr_avg + (1 - pxtr_w) * user_emp_cmtr_ada
            userExpPtr = pxtr_w * pptr_avg + (1 - pxtr_w) * user_emp_eptr_ada
            userExpFtr = pxtr_w * pftr_avg + (1 - pxtr_w) * user_emp_ftr_ada
        end
        if fountain_fullrank_skip_ada_vv_user_emp_xtr == 0 then
            userExpSvtr = pxtr_w * psvr_avg + (1 - pxtr_w) * user_emp_svtr_ada
        end
    end
    local like_weight = calc_xtr_weight_adaptive_ratio(enable_filter_weight_lower_one, enable_filter_weight_over_one, 
        xlife_fountain_ensemble_power_weight_fullrank_like_score,userExpLtr, fountain_ensemble_power_weight_fullrank_like_emp, ratio_min, ratio_max)
    local follow_weight = calc_xtr_weight_adaptive_ratio(enable_filter_weight_lower_one, enable_filter_weight_over_one, 
        xlife_fountain_ensemble_power_weight_fullrank_follow_score,userExpWtr, fountain_ensemble_power_weight_fullrank_follow_emp, ratio_min, ratio_max)
    local comment_weight = calc_xtr_weight_adaptive_ratio(enable_filter_weight_lower_one, enable_filter_weight_over_one, 
        fountain_ensemble_power_weight_fullrank_pcmtr_score,userExpCmtr, fountain_ensemble_power_weight_fullrank_pcmtr_emp, ratio_min, ratio_max)
    local profile_weight = calc_xtr_weight_adaptive_ratio(enable_filter_weight_lower_one, enable_filter_weight_over_one, 
        fountain_ensemble_power_weight_fullrank_pptr_emp,userExpPtr, fountain_ensemble_power_weight_fullrank_pptr_emp, ratio_min, ratio_max)
    local epstr_weight = calc_xtr_weight_adaptive_ratio(enable_filter_weight_lower_one, enable_filter_weight_over_one, 
        fountain_ensemble_power_weight_fullrank_pepstr_score, userExpPtr, fountain_ensemble_power_weight_fullrank_pptr_emp, ratio_min, ratio_max)
    local vtr_multi_wtr = calc_xtr_weight_adaptive_ratio(enable_filter_weight_lower_one, enable_filter_weight_over_one, 
        fountain_ensemble_power_weight_fullrank_pvtr_multi_pwtr, userExpWtr, fountain_ensemble_power_weight_fullrank_follow_emp, ratio_min, ratio_max)
    local vtr_multi_pptr = calc_xtr_weight_adaptive_ratio(enable_filter_weight_lower_one, enable_filter_weight_over_one, 
        fountain_ensemble_power_weight_fullrank_pvtr_multi_pptr, userExpPtr, fountain_ensemble_power_weight_fullrank_pptr_emp, ratio_min, ratio_max)
    local forward_weight = calc_xtr_weight_adaptive_ratio(enable_filter_weight_lower_one, enable_filter_weight_over_one, 
        fountain_ensemble_weight_forward_score, userExpFtr, fountain_ensemble_power_weight_fullrank_forward_emp, ratio_min, ratio_max)
    local like_raw_power_weight = xlife_fountain_fullrank_ensemble_like_raw_pow_weight_attr
    local follow_raw_power_weight = xlife_fountain_fullrank_ensemble_follow_raw_pow_weight_attr
    local comment_raw_power_weight = fountain_fullrank_ensemble_comment_raw_pow_weight_attr
    local epstr_raw_power_weight = fountain_fullrank_ensemble_pepstr_raw_pow_weight_attr
    local forward_raw_power_weight = fountain_fullrank_ensemble_pftr_raw_pow_weight_attr
    
    if fountain_fullrank_ensemble_use_absolute_score_queue_power_weight == 0 then
        like_raw_power_weight = calc_xtr_weight_adaptive_ratio(enable_filter_weight_lower_one, enable_filter_weight_over_one, 
            xlife_fountain_fullrank_ensemble_like_raw_pow_weight_attr, userExpLtr, fountain_ensemble_power_weight_fullrank_like_emp, ratio_min, ratio_max)
        follow_raw_power_weight = calc_xtr_weight_adaptive_ratio(enable_filter_weight_lower_one, enable_filter_weight_over_one, 
            xlife_fountain_fullrank_ensemble_follow_raw_pow_weight_attr, userExpWtr, fountain_ensemble_power_weight_fullrank_follow_emp, ratio_min, ratio_max)
        comment_raw_power_weight = calc_xtr_weight_adaptive_ratio(enable_filter_weight_lower_one, enable_filter_weight_over_one, 
            fountain_fullrank_ensemble_comment_raw_pow_weight_attr, userExpCmtr, fountain_ensemble_power_weight_fullrank_pcmtr_emp, ratio_min, ratio_max)
        epstr_raw_power_weight = calc_xtr_weight_adaptive_ratio(enable_filter_weight_lower_one, enable_filter_weight_over_one, 
            fountain_fullrank_ensemble_pepstr_raw_pow_weight_attr, userExpPtr, fountain_ensemble_power_weight_fullrank_pptr_emp, ratio_min, ratio_max)
        forward_raw_power_weight = calc_xtr_weight_adaptive_ratio(enable_filter_weight_lower_one, enable_filter_weight_over_one, 
            fountain_fullrank_ensemble_pftr_raw_pow_weight_attr, userExpFtr, fountain_ensemble_power_weight_fullrank_forward_emp, ratio_min, ratio_max)
    end
        
    -- svtr
    local svtr_weight = fountain_ensemble_weight_fullrank_pthanos_svr
    local svtr_inorder_weight = fountain_ensemble_power_weight_fullrank_svr_in_order_score
    if (fountain_ensemble_power_weight_fullrank_psvtr_emp > 0.0 and userExpSvtr > 0.0) then
        svtr_weight = calc_xtr_weight_adaptive_ratio(enable_filter_weight_lower_one, enable_filter_weight_over_one, 
            fountain_ensemble_weight_fullrank_pthanos_svr,userExpSvtr, fountain_ensemble_power_weight_fullrank_psvtr_emp, ratio_min, ratio_max)
        svtr_inorder_weight = calc_xtr_weight_adaptive_ratio(enable_filter_weight_lower_one, enable_filter_weight_over_one, 
            fountain_ensemble_power_weight_fullrank_svr_in_order_score,userExpSvtr, fountain_ensemble_power_weight_fullrank_psvtr_emp, ratio_min, ratio_max)
    end
    -- lvtr
    local lvtr_weight = fountain_ensemble_weight_fullrank_sim_longview_score_no_bias_debias
    if (fountain_ensemble_power_weight_fullrank_plvtr_emp > 0.0 and userExpLvtr > 0.0) then
        lvtr_weight = calc_xtr_weight_adaptive_ratio(enable_filter_weight_lower_one, enable_filter_weight_over_one, 
            fountain_ensemble_weight_fullrank_sim_longview_score_no_bias_debias,userExpLvtr, fountain_ensemble_power_weight_fullrank_plvtr_emp, ratio_min, ratio_max)
    end
    return like_weight, follow_weight, comment_weight, profile_weight, epstr_weight, vtr_multi_wtr, vtr_multi_pptr,
      svtr_weight, svtr_inorder_weight, lvtr_weight, forward_weight, like_raw_power_weight, follow_raw_power_weight, comment_raw_power_weight, epstr_raw_power_weight,forward_raw_power_weight
end

function calc_xtr_weight_adaptive_ratio(enable_filter_weight_lower_one, enable_filter_weight_over_one, ori_weight, user_xtr, emp_xtr, min_ratio, max_ratio)
    if emp_xtr <= 0 then
        return ori_weight
    end
    local user_emp_xtr = user_xtr or emp_xtr
    local ada_weight =  math.min(math.max(user_emp_xtr / emp_xtr, min_ratio), max_ratio)
    if ((enable_filter_weight_lower_one > 0  and ada_weight > 1.0) or (enable_filter_weight_over_one > 0  and ada_weight < 1.0)) then 
        weight = ori_weight
    else
        weight = ori_weight * ada_weight
    end 

    return weight
end

function boost_teenager_comment_weights()
    local cmtr_weight = fountain_ensemble_power_weight_fullrank_pcmtr_score
    local cmef_weight = fountain_ensemble_power_weight_fullrank_cmef_score
    local teenager_age_segment_upper_bound = fountain_comment_targeted_teenager_age_segment_upper_bound or 2
    local age_segment = age_segment or teenager_age_segment_upper_bound + 1
    if (age_segment > 0 and age_segment <= teenager_age_segment_upper_bound) then
        local cmtr_score_teenage_coeff = fountain_ensemble_power_weight_fullrank_pcmtr_score_teenager_coeff or 1.0
        local cmef_score_teenage_coeff = fountain_ensemble_power_weight_fullrank_cmef_score_teenager_coeff or 1.0
        cmtr_weight = cmtr_weight * cmtr_score_teenage_coeff
        cmef_weight = cmef_weight * cmef_score_teenage_coeff
    end
    return cmtr_weight, cmef_weight
end

function cal_request_pxtr_weight_factor(request_pxtr, emp_request_pxtr, min_ratio, max_ratio, pow_w, bias)
  local weight_factor = 1.0
  if emp_request_pxtr ~= nil and emp_request_pxtr > 0 then
  local request_or_emp_request_pxtr = request_pxtr or emp_request_pxtr
  weight_factor = math.min(math.max(request_or_emp_request_pxtr / emp_request_pxtr, min_ratio), max_ratio)
  weight_factor = weight_factor^pow_w + bias
  end
  return weight_factor
end

function cal_request_pxtr_ada_weight()
    local pevtr_w = fountain_ensemble_power_weight_fullrank_click_score
    local pevtr_v2_w = fountain_ensemble_weight_fullrank_detail_new_pevtr_v2
    local plvtr_w = fountain_ensemble_power_weight_fullrank_longview_score
    local pfintr_w = fountain_ensemble_power_weight_fullrank_pfintr_score
    local watchtime_w = fountain_ensemble_power_weight_fullrank_pvtr_score
    local ratio_min = fountain_fullrank_ensemble_req_adjust_ratio_min_list or {}
    local ratio_max = fountain_fullrank_ensemble_req_adjust_ratio_max_list or {}
    local pow_w = fountain_fullrank_ensemble_req_adjust_ratio_pow_w or 1.0
    local bias = fountain_fullrank_ensemble_req_adjust_ratio_bias or 0.0
    if #ratio_min > 0 and #ratio_min == #ratio_max then
        pevtr_w = pevtr_w * cal_request_pxtr_weight_factor(pevtr_avg, user_emp_evtr, ratio_min[1], ratio_max[1], pow_w, bias)
        pevtr_v2_w = pevtr_v2_w * cal_request_pxtr_weight_factor(pevtr_v2_avg, user_emp_evtr, ratio_min[2], ratio_max[2], pow_w, bias)
        plvtr_w = plvtr_w * cal_request_pxtr_weight_factor(plvtr_avg, user_emp_lvtr, ratio_min[3], ratio_max[3], pow_w, bias)
        pfintr_w = pfintr_w * cal_request_pxtr_weight_factor(pfintr_avg, user_emp_watch_time, ratio_min[4], ratio_max[4], pow_w, bias)
        watchtime_w = watchtime_w * cal_request_pxtr_weight_factor(pwatchtime_avg, user_emp_evtr, ratio_min[5], ratio_max[5], pow_w, bias)
    end
    return pevtr_w, pevtr_v2_w, plvtr_w, pfintr_w, watchtime_w
end

function cal_multi_stage_size_limit()
    local fast_watchtime_limit = fountain_fullrank_fast_watchtime_limit_size
    local splash_watchtime_limit = fountain_fullrank_splash_watchtime_limit_size
    local fast_interact_limit = fountain_fullrank_fast_interact_limit_size
    local splash_interact_limit = fountain_fullrank_splash_interact_limit_size
    local fast_vv_limit = fountain_fullrank_fast_vv_limit_size
    local splash_vv_limit = fountain_fullrank_splash_vv_limit_size
    local page = page or 1
    local watchtime_limit = page > 1 and fast_watchtime_limit or splash_watchtime_limit
    local interact_limit = page > 1 and fast_interact_limit or splash_interact_limit
    local vv_limit = page > 1 and fast_vv_limit or splash_vv_limit
    return watchtime_limit, interact_limit, vv_limit
end