function split(input, delimiter)
    local arr = {}
    local ids = {}
    string.gsub(input, '[^' .. delimiter ..']+', function(w) table.insert(arr, w) end)
    for _, str in ipairs(arr) do
        table.insert(ids, tonumber(str))
    end
    return ids
end

function pxtr_change(seq)
    local pctr = pctr or 0.0
    local fetr = fetr or 0.0
    local fountain_eff = fountain_eff or 0.0
    local pwtr = pwtr or 0.0

    if (pctr < 0.0) then
        pctr = 0.0
    end

    local alpha = explore_fullrank_calibration_ctr_param or 0.1614
    pctr = alpha * pctr / (1 - (1 - alpha) * pctr)

    local pcmtr = pcmtr or 0.0
    local pctr_x_pcmtr = pctr * pcmtr

    fetr = pctr^0.5 * fetr;
    fountain_eff = pctr^0.5 * fountain_eff;

    local enable_follow_author_pwtr_corr = enable_follow_author_pwtr_corr or 0
    local is_follow_author = is_follow_author or 0
    if enable_follow_author_pwtr_corr > 0 then
        if is_follow_author > 0 then
            local pwtr_corr_coef = ranking_follow_author_pwtr_corr_coef or 1.0
            pwtr = pwtr * pwtr_corr_coef
        end
    end

    local remitted_exptags_str = exploreRank_cls_pctr_filter_remitted_exptags or ""
    local remitted_exptags = split(remitted_exptags_str, ":")
    local reason = reason or -1

    local is_satisfy_ctr_filter = 0
    local is_save = 0
    -- 命中了豁免的reason
    for _, tag in ipairs(remitted_exptags) do
        if (reason == tag) then
            return pctr, fetr, fountain_eff, is_satisfy_ctr_filter, pctr_x_pcmtr, is_save, pwtr
        end
    end

    -- 优质视频豁免
    local enable_audit_hot_skip_filter = enable_audit_hot_skip_rank_pctr_filter or 0.0
    if enable_audit_hot_skip_filter > 0 then
        local audit = audit_hot_high_tag_level or 0
        if audit == 4 then
            return pctr, fetr, fountain_eff, is_satisfy_ctr_filter, pctr_x_pcmtr, is_save, pwtr
        end
    end

    -- 其他正常filter逻辑
    local enable_v4_fr_refactor = enable_produce_v4_fr_refactor or 0.0
    local ctr_threshold = 0.0
    if enable_v4_fr_refactor > 0 then
        ctr_threshold = exploreRank_cls_pctr_filter_threshold or 0.0
    else
        ctr_threshold = exploreRank_cls_pctr_filter_threshold_old or 0.0
    end
    local filter_flag = exploreRank_cls_pctr_filter_flag or 0.0
    if filter_flag > 0 then
        if pctr < ctr_threshold then
            is_satisfy_ctr_filter = 1
        end
    end

    local user_risk_level = user_risk_level or 2
    local user_risk_min = user_risk_min or 3
    local enable_user_high_level_skip_pctr_filter = enable_user_high_level_skip_pctr_filter or 0

    if (enable_user_high_level_skip_pctr_filter > 0 and user_risk_level >= user_risk_min) then
        is_satisfy_ctr_filter = 0
    end

    local enable_picture_skip_pctr_filter = enable_picture_skip_pctr_filter or 0
    local is_picture = is_picture or 0
    if enable_picture_skip_pctr_filter > 0 and is_picture > 0 then
        is_satisfy_ctr_filter = 0
    end

    local fr_pctr_filter_top_pcltr_save_num = fr_pctr_filter_top_pcltr_save_num or 0

    if (seq < fr_pctr_filter_top_pcltr_save_num and is_satisfy_ctr_filter == 1) then
        is_satisfy_ctr_filter = 0
        is_save = 1
    end

    return pctr, fetr, fountain_eff, is_satisfy_ctr_filter, pctr_x_pcmtr, is_save, pwtr

end

function svr_act_queue()
    local pctr = corr_pctr or 0.0
    pctr = pctr >= 0.0 and pctr or 0.0
    local pwtr = pwtr or 0.0
    pwtr = pwtr >= 0.0 and pwtr or 0.0
    local fr_score2 = fr_score2 or 0.0
    fr_score2 = fr_score2 >= 0.0 and fr_score2 or 0.0
    local pltr = pltr or 0.0
    pltr = pltr >= 0.0 and pltr or 0.0
    local pcmtr = pcmtr or 0.0
    pcmtr = pcmtr >= 0.0 and pcmtr or 0.0
    local pepstr = pepstr or 1.0
    pepstr = pepstr >= 0.0 and pepstr or 1.0
    local psvr = psvr or 0.0
    psvr = psvr >= 0.0 and psvr or 0.0
    local plvtr = plvtr or 0.0
    plvtr = plvtr >= 0.0 and plvtr or 0.0
    local score = math.exp(100.0 * (2.0 - psvr))
                        + math.exp(1.0 * plvtr)
                        + math.exp(1.0 * fr_score2 * pctr)
    local act_score = pwtr^4.0
                        * pltr
                        * pcmtr
                        * pepstr * 10^8
                        + 30
    local svr_act_score = score * act_score * pctr^7.0
    return svr_act_score
end
