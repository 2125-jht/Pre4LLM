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
    local pwtr = pwtr or 0.0
    local pctr_upper_bound = pctr_upper_bound or 1.0

    if (pctr < 0.0) then
        pctr = 0.0
    end

    -- pctr校准
    local alpha = explore_fullrank_calibration_ctr_param or 0.1614
    pctr = alpha * pctr / (1 - (1 - alpha) * pctr)
    local enable_explore_pctr_upper_bound_limit = enable_explore_pctr_upper_bound_limit or 0
    if enable_explore_pctr_upper_bound_limit > 0 then
        pctr = math.min(pctr_upper_bound, pctr)
    end 
    
    -- pctr融合
    local enable_ctr_socre_ensemble = enable_ctr_socre_ensemble or 0
    local ensemble_coef = ctr_socre_ensemble_coef or 0.0
    local diversity_ctr_score = explore_diversity_ctr_score or 0.0
    if enable_ctr_socre_ensemble > 0 then
        pctr = ensemble_coef * diversity_ctr_score + (1 - ensemble_coef) * pctr
    end

    -- 对已关注作者的pwtr进行矫正
    local enable_follow_author_pwtr_corr = enable_follow_author_pwtr_corr or 0
    local is_follow_author = is_follow_author or 0
    if enable_follow_author_pwtr_corr > 0 then
        if is_follow_author > 0 then
            local pwtr_corr_coef = ranking_follow_author_pwtr_corr_coef or 1.0
            pwtr = pwtr * pwtr_corr_coef
        end
    end

    return pctr, pwtr

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
