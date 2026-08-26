function print1()
    local explore_fr_ensemble_score = explore_fr_ensemble_score or 0.0
    local pctr = pctr or 0.0

    return explore_fr_ensemble_score
end

function cal_act_vtr_combo()
    local wtr = pwtr or 0.0
    local ltr = pltr or 0.0
    local cmtr = pcmtr or 0.0
    local epstr = pepstr or 0.0
    if (epstr < 0.0) then
        epstr = 0.0
    end
    local fr_score2 = fr_score2 or 0.0
    if (fr_score2 < 0.0) then
        fr_score2 = 0.0
    end
    local ctr = corr_pctr or 0.0

    local params = xtr_norm_str_rank_number
    local combo_score = 0.0
    if #params == 4 and enable_act_combo_vtr_rank > 0 then
        combo_score = math.max(wtr * params[1], ltr * params[2], cmtr * params[3], epstr * params[4]) * (fr_score2 ^ vtr_pow_weight_rank) * (ctr ^ 7.0)
    end
    return combo_score
end

function split(input, delimiter)
    local t={}
    for str in string.gmatch(input, "([^"..delimiter.."]+)") do
        table.insert(t, str)
    end
    return t
end

function cal_normal_value(input_value, xtr_avg, xtr_dev, bias, max_value, norm_factor)
    local xtr_avg = xtr_avg or 0.0
    local xtr_dev = xtr_dev or 0.0
    local input_value = input_value or 0.0
    local weight_factor = math.abs(xtr_avg) / ((1 - xtr_dev) ^ 2 + 1e-8) * norm_factor
    weight_factor = weight_factor + bias
    weight_factor = math.min(max_value, weight_factor)

    return weight_factor * input_value
end

function cal_xtr_adaptive_weight()
    if enable_ranking_statistics_adaptive <= 0 then
        return
        explore_ensemble_power_weight_fullrank_pctr_score,
        explore_ensemble_power_weight_fullrank_pltr_score,
        explore_ensemble_power_weight_fullrank_pwtr_score,
        explore_ensemble_power_weight_fullrank_pepstr_score,
        fr_pmctr_rank_weight
    end
    local statistics_adaptive_xtr_norm_str = statistics_adaptive_xtr_norm_str or ""
    statistics_adaptive_xtr_norm = split(statistics_adaptive_xtr_norm_str, ";")
    local norm_factor_map = {}
    for _, tag in ipairs(statistics_adaptive_xtr_norm) do
        for k, v in string.gmatch(tag, "(%w+):(%w+)") do
            norm_factor_map[k] = tonumber(v)
        end
    end

    local wtr_weight = explore_ensemble_power_weight_fullrank_pwtr_score
    if norm_factor_map["wtr"] ~= nil then
        wtr_weight = cal_normal_value(explore_ensemble_power_weight_fullrank_pwtr_score,
                pwtr_avg, pwtr_dev, statistics_adaptive_factor_bias, max_tatistics_adaptive_factor, norm_factor_map['wtr'])
    end

    local ltr_weight = explore_ensemble_power_weight_fullrank_pltr_score
    if norm_factor_map["ltr"] ~= nil then
        ltr_weight = cal_normal_value(explore_ensemble_power_weight_fullrank_pltr_score,
                pltr_avg, pltr_dev, statistics_adaptive_factor_bias, max_tatistics_adaptive_factor, norm_factor_map['ltr'])
    end

    local ctr_weight = explore_ensemble_power_weight_fullrank_pctr_score
    if norm_factor_map["ctr"] ~= nil then
        ctr_weight = cal_normal_value(explore_ensemble_power_weight_fullrank_pctr_score,
                pctr_avg, pctr_dev, statistics_adaptive_factor_bias, max_tatistics_adaptive_factor, norm_factor_map['ctr'])
    end

    local cmtr_weight = fr_pmctr_rank_weight
    if norm_factor_map["cmtr"] ~= nil then
        cmtr_weight = cal_normal_value(fr_pmctr_rank_weight,
                pcmtr_avg, pcmtr_dev, statistics_adaptive_factor_bias, max_tatistics_adaptive_factor, norm_factor_map['cmtr'])
    end

    local epstr_weight = explore_ensemble_power_weight_fullrank_pepstr_score
    if norm_factor_map["epstr"] ~= nil then
        epstr_weight = cal_normal_value(explore_ensemble_power_weight_fullrank_pepstr_score,
                pepstr_avg, pepstr_dev, statistics_adaptive_factor_bias, max_tatistics_adaptive_factor, norm_factor_map['epstr'])
    end
    return ctr_weight, ltr_weight, wtr_weight, epstr_weight, cmtr_weight
end

function watchtime_interact_score_calc()
    local pctr_weight = user_interact_watchtime_cost_score_pctr_weight or 0.0
    local inter_cost_weight = user_interact_watchtime_cost_score_cost_weight or 0.0
    local alpha = user_interact_watchtime_cost_score_alpha or 0.0
    local beta = user_interact_watchtime_cost_score_beta or 0.0
    local weight_str = user_interact_watchtime_cost_score_weight_str or ""
    local watchtime_weight = user_interact_watchtime_cost_score_watchtime_weight or 0.0
    local enable_cltr = enable_user_interact_watchtime_cost_score_cltr or false
    local cltr_weight = user_interact_watchtime_cost_score_cltr_weight or 0.0

    local pctr = corr_pctr or 0.0
    local pltr = pltr or 0.0
    local pwtr = pwtr or 0.0
    local pftr = pftr or 0.0
    local pcmtr = pcmtr or 0.0
    local pcmef = pcmef or 0.0
    local pptr = pptr or 0.0
    local pepstr = pepstr or 0.0
    local fetr = fetr or 0.0
    local fountain_eff = fountain_eff or 0.0
    local fr_score2 = fr_score2 or 0.0
    local pcltr = pcltr or 0.0
    
    local weight_str_split = split(weight_str, ":")
    local weight_number = {}
    for _, weight in ipairs(weight_str_split) do
        table.insert(weight_number, tonumber(weight))
    end

    local interact_score = 0.0
    if (#weight_number == 10) then
        interact_score = pctr * weight_number[1] + pltr * weight_number[2] + pwtr * weight_number[3]
            + pftr * weight_number[4] + pcmtr * weight_number[5] + pcmef * weight_number[6]
            + pptr * weight_number[7] + pepstr * weight_number[8] + fetr * weight_number[9]
            + fountain_eff * weight_number[10]
        if (enable_cltr and cltr_weight > 0) then
            interact_score = interact_score + pcltr * cltr_weight
        end
    end

    local watchtime_score = fr_score2 ^ watchtime_weight
    local inter_cost_score = alpha * watchtime_score + beta * interact_score / (watchtime_score + 1e-3)
    local score = (pctr ^ pctr_weight) * (inter_cost_score ^ inter_cost_weight)

    return score
end

function ewatch_score_change()
    local duration_ms_new = duration_ms or 0.0
    local duration = duration_ms_new / 1000.0
    local fr_score1_new = fr_score1 or 0.0

    local alpha = 18.0
    if (duration <= 3.0) then
        alpha = 18.0
    elseif (duration <= 36.0) then
        alpha = (duration * 28.0 + 180.0) / 33.0
    else
        alpha = 36.0
    end

    local score = alpha * fr_score1_new
    return score
end

function get_cmef_debias_bucket_name()
    local hetu_level_one_array = hetu_tag_level_info__hetu_level_one or {}
    local hetu_level_one_attr = hetu_level_one_array[1] or 0

    local gender_attr = "U";
    if (gender == 0) then
        gender_attr = "M"
    elseif (gender == 1) then
        gender_attr = "F"
    else
        gender_attr = "U"
    end

    local pcmef_debias_bucket_name = table.concat({"cmef-", hetu_level_one_attr, "T-", gender_attr})
    return pcmef_debias_bucket_name
end

function get_cmef_debias_score()
    local pctr = corr_pctr or 0.0
    local pcmef = pcmef or 0.0
    local pcmef_debias_bucket_score = tonumber(pcmef_debias_bucket_score) or 1.0;
    local pcmef_debias_score = pctr * pcmef / pcmef_debias_bucket_score;
    return pcmef_debias_score;
end