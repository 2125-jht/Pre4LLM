-- 用于rerank
function convert_photo_info_attr(seq)
    local hetu_level_one_array = hetu_tag_level_info__hetu_level_one or {}
    local hetu_level_two_array = hetu_tag_level_info__hetu_level_two or {}
    local explore_view_length_sum = explore_stat__view_length_sum or 0
    local explore_click_count = explore_stat__click_count or 0
    local ensemble_score = explore_fr_ensemble_score or 0
    ensemble_score = 1 / (seq + 10.0) * 0.0

    local hetu_level_one_attr = hetu_level_one_array[1] or nil
    local hetu_level_two_attr = hetu_level_two_array[1] or nil
    local hetu_level_two_attr2 = hetu_level_two_array[2] or nil
    local hetu_level_two_attr3 = hetu_level_two_array[3] or nil
    local empirical_watchtime = explore_view_length_sum / (explore_click_count + 100.0)

    local duration_ms = duration_ms or 0
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
    return hetu_level_one_attr, hetu_level_two_attr, hetu_level_two_attr2, hetu_level_two_attr3, empirical_watchtime, duration_0_7s, duration_7_9s, duration_9_12s, duration_12_17s, duration_17_20s, duration_20_58s, duration_gt_58s, duration_gt_120s, ensemble_score
end

function lvtr_coeff(duration_ms, is_picture, avg_watch_time_ms)
    local duration_ms = duration_ms or 0.0
    local is_picture = is_picture or 0.0
    local avg_watch_time_ms = avg_watch_time_ms or 0.0
    local duration = duration_ms / 1000.0
    if (is_picture > 0) then
        duration = 18.0
    end
    local avg_watch_time = avg_watch_time_ms / 1000.0
    local coeff = 1.0
    if (avg_watch_time <= 3 or duration <= 7) then
        coeff = 0.4 * math.max(0.0, math.min(18.0, duration) - 3) / 15.0 + 0.1;
    else 
        local ratio = avg_watch_time / duration * 100.0;
        if (duration < 18.0) then
            ratio = ratio * 0.7
        end
        coeff = 2.2 / (1.0 + math.exp(-0.05 * (ratio - 30.0)));
    end
    return coeff
end

function full_rank_score_cal()
    local picture_discount_param = dpp_rerank_picture_discount_param_new or 1.0
    local photo_level_discount = fr_rerank_photo_level_discount_param or 1.0
    local duration_discount = fr_rerank_duration_lt_58_discount_param or 1.0
    local photo_level = content_safety_level_with_namespace__level_hot_online or 2
    local duration_ms = duration_ms or 0
    local picture_variant_attr = is_picture or 0
    local no_pctr_multiply = explore_rerank_enable_no_pctr_multiply or 0
    local rerank_pic_coff_attr_transfer = rerank_pic_coff_attr_transfer or 1.0 
    local diversity_pfntr_multiply = explore_rerank_enable_diversity_pfntr_multiply or 0
    local diversity_pfntr_multiply_coff = explore_rerank_enable_diversity_pfntr_multiply_coff or 0.0
    local is_explore_photo = is_explore_photo or 0
    local interest_explore_boost = fr_rerank_interest_explore_boost_param or 1.0
    local photo_age_hour = photo_age_hour or 145
    local age_bucket = math.floor(photo_age_hour / 24) + 1
    local age_weight_number = age_weight_number or {}
    local pic_age_weight_number = pic_age_weight_number or {}
    local photo_age_boost_fans_threshold = explore_photo_age_boost_fans_threshold or 0
    local author_fans = author__fans_count or 0
    local score_factor = picture_variant_attr < 1 and 1.0 or (picture_discount_param * rerank_pic_coff_attr_transfer)
    score_factor = photo_level < 2 and score_factor * photo_level_discount or score_factor
    score_factor = duration_ms > 58000 and score_factor * duration_discount or score_factor
    score_factor = is_explore_photo == 1 and score_factor * interest_explore_boost or score_factor
    if #age_weight_number == 7 and #pic_age_weight_number == 7 and author_fans >= photo_age_boost_fans_threshold then
        if picture_variant_attr < 1 then
            score_factor = age_bucket <= 7 and score_factor * age_weight_number[age_bucket] or score_factor
        else
            score_factor = age_bucket <= 7 and score_factor * pic_age_weight_number[age_bucket] or score_factor
        end
    end
    local pctr = corr_pctr or 0.0
    local pwtr = corr_pwtr or 0.0
    local pltr = pltr or 0.0
    local fr_score1 = fr_score1 or 0.0
    local fr_score2 = fr_score2 or 0.0
    local l2r_score = consume_time_ltr or 0.0
    local pftr = pftr or 0.0
    local duration_gt_58s = duration_ms > 58000 and 1.0 or 0.0
    local pptr = pptr or 0.0
    local plvtr = plvtr or 0.0
    local pepstr = pepstr or 0.0
    local explore_fr_ensemble_score = explore_fr_ensemble_score or 0.0
    --预留score 实验看是否可以去掉
    local pcltr = pcltr or 0.0
    local fetr = fetr or 0.0
    local fountain_eff = fountain_eff or 0.0
    --预留diversity 要修改c++代码后续加
    local pcmtr = pcmtr or 0.0
    local pcmef = pcmef or 0.0
    local pctr_pfr2r = pctr_pfr2r or 0.0
    local pcltr_pfr2r = pcltr_pfr2r or 0.0
    local ada_xtr_score = ada_xtr_score or 0.0
    local watchtime_interact_score = watchtime_interact_score or 0.0
    local awesome_wtd = awesome_wtd or 0.0
    local pdtr = pdtr or 0.0
    local interact_fusion_score = interact_fusion_score or 0.0
    local watch_time_fusion_score = watch_time_fusion_score or 0.0
    local pure_value = explore_fullrank_pure_value_score or 0.0
    local pcpr = corr_cpr or 0.0
    local pevtr = pevtr or 0.0
    local coeff = lvtr_coeff(duration_ms, picture_variant_attr, avg_watch_time)
    --local coeff = 1.0
    local pctr_score = pctr * score_factor * (1 + (pctr_duration_debias_coffe or 0.0))
    local pwtr_score = pwtr * score_factor * (1 + (pwtr_duration_debias_coffe or 0.0))
    local pltr_score = (no_pctr_multiply > 0 and pltr * score_factor or pltr * score_factor * pctr) * (1.0 + (pltr_duration_debias_coffe or 0.0))
    local fr_score1_score = no_pctr_multiply > 0 and fr_score1 * score_factor or fr_score1 * score_factor * pctr
    local fr_score2_score = no_pctr_multiply > 0 and fr_score2 * score_factor or fr_score2 * score_factor * pctr
    local l2r_score_score = l2r_score * score_factor
    local pftr_score = (no_pctr_multiply > 0 and pftr * score_factor or pftr * score_factor * pctr) * (1.0 + (pftr_duration_debias_coffe or 0.0))
    local duration_gt_58s_score = duration_gt_58s * score_factor
    local pptr_score = pptr * score_factor
    local plvtr_score = (no_pctr_multiply > 0 and plvtr * score_factor * coeff or plvtr * score_factor * pctr * coeff) * (1.0 + (plvtr_duration_debias_coffe or 0.0))
    local pepstr_score = no_pctr_multiply > 0 and pepstr * score_factor or pepstr * score_factor * pctr
    local explore_fr_ensemble_score_score = explore_fr_ensemble_score
    --预留score 实验看是否可以去掉
    local pcltr_score = (no_pctr_multiply > 0 and pcltr * score_factor or pcltr * score_factor * pctr) * (1.0 + (pcltr_duration_debias_coffe or 0.0))
    local fetr_score = no_pctr_multiply > 0 and fetr * score_factor or fetr * score_factor * pctr
    local fountain_eff_score = no_pctr_multiply > 0 and fountain_eff * score_factor or fountain_eff * score_factor * pctr
    --预留diversity 要修改c++代码后续加
    local pcmtr_score = (no_pctr_multiply > 0 and pcmtr * score_factor or pcmtr * score_factor * pctr) * (1.0 + (pcmtr_duration_debias_coffe or 0.0))
    local pcmef_score = no_pctr_multiply > 0 and pcmef * score_factor or pcmef * score_factor * pctr
    local ada_xtr_score_score = ada_xtr_score * score_factor
    local interact_cost_score = watchtime_interact_score * score_factor
    local awesome_wtd_score = awesome_wtd * score_factor
    local pdtr_score = pdtr * score_factor
    local pdbfrtr = consume_time_pf2r_score or 0.0
    local interact_fusion = interact_fusion_score * score_factor
    local watch_time_fusion = watch_time_fusion_score * score_factor
    local diversity = diversity_pfntr_multiply > 0 and score_factor * pdbfrtr^diversity_pfntr_multiply_coff or score_factor
    local pctr_pfr2r_score = pctr_pfr2r * score_factor
    local pcltr_pfr2r_score = pcltr_pfr2r * score_factor
    local pure_value_score = pure_value * score_factor
    local pcpr_score = pcpr * score_factor
    local pevtr_score = pevtr * score_factor
    local min_act_rank_score = (min_act_rank_score or 0.0) * score_factor
    local gen_l2r_score = (gen_l2r_score or 0.0) * score_factor
    return pctr_score, pwtr_score, pltr_score, fr_score1_score, fr_score2_score, l2r_score_score, pftr_score, duration_gt_58s_score, pptr_score, plvtr_score, pepstr_score, explore_fr_ensemble_score_score, pcltr_score, fetr_score, fountain_eff_score, pcmtr_score, pcmef_score, diversity, ada_xtr_score_score, interact_cost_score, awesome_wtd_score, pdtr_score, pdbfrtr,interact_fusion,watch_time_fusion, pctr_pfr2r_score, pcltr_pfr2r_score, pure_value_score, pcpr_score, pevtr_score, min_act_rank_score, gen_l2r_score
end

function calculate()
    local dnn_cluster = photo_dnn_cluster_id or 0
    dnn_cluster = dnn_cluster > 0 and dnn_cluster or nil
    local hetu_cluster = _G['hetu_tag_level_info_v2.hetu_cluster_id'] or 0 
    hetu_cluster = hetu_cluster > 0 and hetu_cluster or nil
    local duration_ms = duration_ms or 0
    local show_count = explore_stat__show_count or 10
    show_count = show_count > 0 and show_count or 10
    local real_show_count = explore_stat__real_show_count or 10
    real_show_count = real_show_count > 0 and real_show_count or 10
    local click_count = explore_stat__click_count or 1
    click_count = click_count > 0 and click_count or 1
    local like_count = explore_stat__like_count or 0
    like_count = like_count >= 0 and like_count or 0
    local follow_count = explore_stat__follow_count or 0
    follow_count = follow_count >= 0 and follow_count or 0
    local forward_count = explore_stat__forward_count or 0
    forward_count = forward_count >= 0 and forward_count or 0
    local profile_enter_count = explore_stat__profile_enter_count or 0
    profile_enter_count = profile_enter_count >= 0 and profile_enter_count or 0
    local comment_count = explore_stat__comment_count or 0
    comment_count = comment_count >= 0 and comment_count or 0
    local negative_count = explore_stat__negative_count or 0
    negative_count = negative_count >= 0 and negative_count or 0
    local view_length_sum = view_length_sum or 0
    view_length_sum = view_length_sum >= 0 and view_length_sum or 0
    local total_report_count = explore_stat__report_detail__total_report_count or 0
    total_report_count = total_report_count >= 0 and total_report_count or 0
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
    local empirical_watchtime = view_length_sum * 1.0 / click_count
    local empirical_rrr = total_report_count / show_count

    local confident_click_count = 100
    local lvtr_sigmoid_bias_sec = 10.0
    local picture_variant_attr = is_picture or 0
    if (picture_variant_attr > 0) then
        duration_ms = lvtr_sigmoid_bias_sec * 1000.0;
    end
    local smooth_watch_time_ms = math.min(duration_ms, lvtr_sigmoid_bias_sec * 1000.0)
    local avg_watch_time_ms = 1.0 * (view_length_sum +
            math.max(0, confident_click_count - click_count) * smooth_watch_time_ms)
              / math.max(click_count, confident_click_count);


    local time = os.time() * 1000
    local upload_time = upload_time or 0
    local photo_age_hour = (time - upload_time) / (1000 * 60 *60)
    local photo_age_hour_int = math.floor(photo_age_hour)
    return dnn_cluster, hetu_cluster, short_duration_variant_attr, long_duration_variant_attr, lt20s_duration_variant_attr, empirical_ctr, empirical_ltr, empirical_wtr, empirical_ftr, empirical_ptr, empirical_cmtr, empirical_htr, empirical_watchtime, empirical_rrr, photo_age_hour_int, avg_watch_time_ms
end

function other_name()
    local is_pic = is_picture or 0.0
    local explore_fr_ensemble_score_new = explore_fr_ensemble_score or 0.0
    local consume_time_ltr_new = consume_time_ltr or 0.0
    local picture_discount = rerank_picture_discount_param or 1.0
    if (is_pic > 0) then
        explore_fr_ensemble_score_new = explore_fr_ensemble_score_new * picture_discount
        consume_time_ltr_new = consume_time_ltr_new * picture_discount
    end

    return explore_fr_ensemble_score_new, consume_time_ltr_new
end
