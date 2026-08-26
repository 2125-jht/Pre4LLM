function report_discount_calculate()
    local photo_rstr_norm_show = rerank_photo_rstr_norm_show or 1000000.0
    local photo_rstr_norm_report = rerank_photo_rstr_norm_report or 2.0
    local report_count = explore_stat__report_detail__total_report_count or 0.0
    local show_count = explore_stat__show_count or 0.0
    local rstr = (report_count + photo_rstr_norm_report)
      / (show_count + photo_rstr_norm_show)
    local photo_level = audit_hot_high_tag_level or 1
    
    local param_n = rerank_rstr_discount_param_n or 0.000004
    local param_o = rerank_rstr_discount_param_o or 0.000002
    if (photo_level >= 4) then
        param_n = rerank_rstr_discount_param_n_high_good or 0.00001
    end
    if (param_o < 1e-10 or param_n < 1e-10 or rstr < 0) then
        return 1.0
    end
    local discount = 1 - 1 / (1 + math.exp((param_n - rstr) / param_o));
    local calibration = 1 - 1 / (1 + math.exp(param_n / param_o));
    return discount / calibration;
end

function discount_function(x1, y1, x2, y2, x)
    local y = 0
    if (x <= x1) then
      y = y1
    elseif (x >= x2) then
      y = y2
    else
      y = ((y2 - y1) * x + (x2 * y1 - x1 * y2)) / (x2 - x1)
    end
    return y
end

function jury_grading_calculate()
    local jury_discount = 1.0

    local cover_mu = jury_grading_cover_title_mu or 2.0
    local cover_slope = jury_grading_cover_title_slope or -5.5
    local cover_title_score = mmu_photo_score__ds_cover_title_score or 0.0
    if (cover_title_score > 1e-6) then
        jury_discount = jury_discount / (1 + math.exp(cover_slope * (cover_title_score - cover_mu)))
    end

    local disgust_score_max = jury_grading_ds_disgust_score_max or 1.0
    local disgust_score_min = jury_grading_ds_disgust_score_min or 0.6
    local mid_disgust_discount_max = jury_grading_mid_disgust_discount_max or 1.0
    local mid_disgust_discount_min = jury_grading_mid_disgust_discount_min or 0.99
    local high_disgust_discount_max = jury_grading_high_disgust_discount_max or 1.0
    local high_disgust_discount_min = jury_grading_high_disgust_discount_min or 0.99
    local user_aesthetic_score_new = user_aesthetic_score 
    local low_to_mid_threshold = jury_grading_mmu_aesthetic_low_to_mid_threshold or 2.2
    local mid_to_high_threshold = jury_grading_mmu_aesthetic_mid_to_high_threshold or 3.1
    local disgust_score = mmu_photo_score__ds_disgust_score or 0.0
    if disgust_score > 1e-6 then
        if (disgust_score_max - disgust_score_min > 1e-6) then
            if (user_aesthetic_score_new > low_to_mid_threshold and user_aesthetic_score_new < mid_to_high_threshold) then
                jury_discount = jury_discount * discount_function(
                disgust_score_min,
                mid_disgust_discount_max,
                disgust_score_max,
                mid_disgust_discount_min,
                disgust_score);
            elseif (user_aesthetic_score_new > mid_to_high_threshold) then
                jury_discount = jury_discount * discount_function(
                disgust_score_min,
                high_disgust_discount_max,
                disgust_score_max,
                high_disgust_discount_min,
                disgust_score);
            end
        end
    end

    local like_bait_score_max = jury_grading_ds_like_bait_score_max_new or 1.0
    local like_bait_score_min = jury_grading_ds_like_bait_score_min_new or 0.0
    local like_bait_discount_max = jury_grading_like_bait_discount_max_new or 1.0
    local like_bait_discount_min = jury_grading_like_bait_discount_min_new or 0.99
    local like_bait_score = mmu_photo_score__ds_like_bait_score or 0.0
    if (like_bait_score > 1e-6) then
        if (like_bait_score_max - like_bait_score_min > 1e-6) then
            jury_discount = jury_discount * discount_function(
                like_bait_score_min,
                like_bait_discount_max,
                like_bait_score_max,
                like_bait_discount_min,
                like_bait_score);
        end
    end
    return jury_discount
end

function hate_discount()
    local phtr_new = phtr or 0.0
    local htr_discount_param_n = 0.0001
    local htr_discount_param_o = 0.03
    local discount = 1.0;
    local upload_decay_rate = 0.95
    local recent_upload_bitmap = recentUploadBitMap or 0
    local hate_time_ms = hateTimeMs or nil
    if htr_discount_param_n > 1e-6 and phtr_new > 1e-6 then
        discount = 1  - 1 / (1 + math.exp((htr_discount_param_n - phtr_new)  / htr_discount_param_o));
    end

    -- 是否是活跃作者
    local weight = 0.0;
    for i = 1, 30 do
        local weight_tmp = 0.0
        if ((2 ^ (30 - i) & recent_upload_bitmap) > 0) then
            weight_tmp = 1.0
        end
        weight = weight * upload_decay_rate + weight_tmp
    end
    if (weight > 1.5) then
        return discount
    end
    
    -- 最近是否有hate
    local has_recent_hate = 0
    local currentTime = os.time() * 1000
    local active_hate_in_minutes =  4320
    if hate_time_ms ~= nil then
        for i, v in ipairs(hate_time_ms) do
            if (currentTime - v < active_hate_in_minutes * 60 * 1000) then
                has_recent_hate = 1
                break
            end
        end
    end
    if (has_recent_hate < 1) then
        return discount
    end

    -- 正常情况
    htr_discount_param_n = 0.00005
    htr_discount_param_o = 0.005
    local htr_discount_param_threshold = 0.00005
    if (phtr_new < htr_discount_param_threshold) then
      local tmp = math.exp((htr_discount_param_threshold - htr_discount_param_n)  / htr_discount_param_o);
      local k = (-1) * tmp / ((1 + tmp) * htr_discount_param_threshold);
      discount = k * phtr_new + 1;
    else
      discount = 1 / (1 + math.exp((phtr_new - htr_discount_param_n)  / htr_discount_param_o));
    end

    return discount
end

function score_coeff_calculate_stage2()
    local pctr_power_beta_time = explore_fr_es_pctr_x_pxtr_power_beta_time or 1.0
    local pctr_power_beta_action = explore_fr_es_pctr_x_pxtr_power_beta_action or 1.0
    local pctr_power_beta_wtd = awesome_wtd_pctr_weight or 0.0
    local hate_discount_new = hate_discount_new or 1.0
    local consume_time_ltr_new = consume_time_ltr or 0.0
    local pctr_new = corr_pctr or 0.0
    local pctr_new_power_time = pctr_new ^ pctr_power_beta_time
    local pctr_new_power_action = pctr_new ^ pctr_power_beta_action
    local pftr_new = pftr or 0.0
    local pptr_new = pptr or 0.0
    local pdtr_new = pdtr or 0.0
    local pepstr_new = pepstr or 0.0
    local pcltr_new = pcltr or 0.0
    local pevtr_new = pevtr or 0.0
    local fr_score2_new = fr_score2 or 0.0
    local awesome_wtd_new = awesome_wtd or 0.0
    local awesome_wtd_score = pctr_new ^ pctr_power_beta_wtd * awesome_wtd_new
    local score_consume_time_ltr = hate_discount_new * consume_time_ltr_new
    local score_pctr_x_pcltr = pctr_new_power_action * pcltr_new
    local score_pctr_x_pepstr = pctr_new_power_action * pepstr_new
    local score_pctr_x_pptr = pctr_new_power_action * pptr_new
    local score_pctr_x_pdtr = pctr_new_power_action * pdtr_new
    local score_pctr_x_pftr = pctr_new_power_action * pftr_new
    local score_pctr_x_pevtr = pctr_new_power_time * pevtr_new
    local score_pctr_x_fr_score2 = pctr_new_power_time * fr_score2_new
    local score_pctr_x_awesome_wtd = pctr_new_power_time * awesome_wtd_new
    return awesome_wtd_score, score_consume_time_ltr, score_pctr_x_pcltr, score_pctr_x_pepstr, score_pctr_x_pptr, score_pctr_x_pdtr, score_pctr_x_pftr, score_pctr_x_pevtr, score_pctr_x_fr_score2, score_pctr_x_awesome_wtd
end

function collect_garbage()
    collectgarbage()
end
