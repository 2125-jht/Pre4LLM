function calc_mc_max_hetu_one_rate()
    local local_hetu_one_cascade =  hetu_level_one_v2_index_cascade_list_no_dedup and hetu_level_one_v2_index_cascade_list_no_dedup or {}
    local lt_ee_weight = fountain_fast_ensemble_weight_cascade_long_term_interest_ee_score
    local max_hetu_rate = 0.0
    local lt_weight_adjust_threshold = fountain_mc_lt_weight_adjust_threshold and fountain_mc_lt_weight_adjust_threshold or 0.4
    local lt_weight_adjust_coef = fountain_mc_lt_weight_adjust_coef and fountain_mc_lt_weight_adjust_coef or 1.5
    if (fountain_mc_enable_lt_weight_adjust and lt_ee_weight > 0 and #local_hetu_one_cascade > 0) then
        local hetu_mc_rate_tb = {}
        for i = 1, #local_hetu_one_cascade do
            if local_hetu_one_cascade[i] ~= nil then
                if hetu_mc_rate_tb[local_hetu_one_cascade[i]] ~= nil then
                    hetu_mc_rate_tb[local_hetu_one_cascade[i]] = hetu_mc_rate_tb[local_hetu_one_cascade[i]] + 1.0
                else 
                    hetu_mc_rate_tb[local_hetu_one_cascade[i]] = 1.0
                end
                max_hetu_rate = math.max(max_hetu_rate, hetu_mc_rate_tb[local_hetu_one_cascade[i]] / (#local_hetu_one_cascade * 1.0))
            end
        end
        if (max_hetu_rate > lt_weight_adjust_threshold) then
            lt_ee_weight = lt_ee_weight * lt_weight_adjust_coef
        end
    end
    return lt_ee_weight
end

function calc_mc_high_value_hetu_debias()
    -- get hetu id
    local hetu_level_one_v2_original = _G['hetu_tag_level_info_v2__hetu_level_one'] and _G['hetu_tag_level_info_v2__hetu_level_one'] or {}
    local hetu_level_one_v2 = {}
    local index = 1
    for i = 1, #hetu_level_one_v2_original do
        local tagId = getTagIdFromHetuTag(hetu_level_one_v2_original[i])
        table.insert(hetu_level_one_v2, index, tagId)
        index = index + 1
    end
    
    local longterm_score = cascade_long_term_interest_ee_score and cascade_long_term_interest_ee_score or 1.0
    if (#hetu_level_one_v2 > 0 and #high_value_hetu_list > 0 and isInTable(hetu_level_one_v2[1], high_value_hetu_list)) then
        if (fountain_mc_enable_only_longterm_debias) then
            if (longterm_score > 1.0) then
                longterm_score = longterm_score * fountain_mc_high_value_hetu_debias_coef
            end
        else 
            longterm_score = longterm_score * fountain_mc_high_value_hetu_debias_coef
        end
    end
    return longterm_score
end

function calc_fullrank_high_value_hetu_debias()
    -- get hetu id
    local hetu_level_one_v2_original = _G['hetu_tag_level_info_v2__hetu_level_one'] and _G['hetu_tag_level_info_v2__hetu_level_one'] or {}
    local hetu_level_one_v2 = {}
    local index = 1
    for i = 1, #hetu_level_one_v2_original do
        local tagId = getTagIdFromHetuTag(hetu_level_one_v2_original[i])
        table.insert(hetu_level_one_v2, index, tagId)
        index = index + 1
    end
    
    local longterm_score = long_term_interest_ee_score and long_term_interest_ee_score or 1.0
    if (#hetu_level_one_v2 > 0 and #high_value_hetu_list > 0 and isInTable(hetu_level_one_v2[1], high_value_hetu_list)) then
        if (fountain_fullrank_enable_only_longterm_debias) then
            if (longterm_score > 1.0) then
                longterm_score = longterm_score * fountain_fullrank_high_value_hetu_debias_coef
            end
        else 
            longterm_score = longterm_score * fountain_fullrank_high_value_hetu_debias_coef
        end
    end
    return longterm_score
end

function getTagIdFromHetuTag(hetuTag)
    local tagId = ((hetuTag >> 8) & 0xffffff)
    return tagId
end

function isInTable(tagId, tb)
    if (#tb > 0) then
        for i = 1, #tb do
            if (tb[i] ~= nil and tagId == tb[i]) then
                return true
            end
        end
    end
    return false
end

