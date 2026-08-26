function cal_long_term_interest_score()
    local hetu_level_one_list = _G["hetu_tag_level_info_v2__hetu_level_one"] and _G["hetu_tag_level_info_v2__hetu_level_one"] or {}
    local hetu_level_one = (#hetu_level_one_list > 0) and hetu_level_one_list[1] or nil
    local hetu_level_two_list = _G["hetu_tag_level_info_v2__hetu_level_two"] and _G["hetu_tag_level_info_v2__hetu_level_two"] or {}
    local hetu_level_two = (#hetu_level_two_list > 0) and hetu_level_two_list[1] or nil
    local hetu_level_three_list = _G["hetu_tag_level_info_v2__hetu_level_three"] and _G["hetu_tag_level_info_v2__hetu_level_three"] or {}
    local hetu_level_three = (#hetu_level_three_list > 0) and hetu_level_three_list[1] or nil
    -- 长期兴趣一级标签提权
    local hetu_level_one = hetu_level_one and hetu_level_one or -1 
    local fullrank_longterm_interest_score1 = 0.0
    if (hetuLevelOneLongTermId ~= nil and common_request_type ~= fountain_splash_request_type) then
        for i = 1, #hetuLevelOneLongTermId do
            if hetu_level_one == hetuLevelOneLongTermId[i] then
                fullrank_longterm_interest_score1 = 0.5
            end
        end
    end
    if (userBrowseSetHetuLevel1 ~= nil and common_request_type ~= fountain_splash_request_type) then
        for i = 1, #userBrowseSetHetuLevel1 do
            if hetu_level_one == userBrowseSetHetuLevel1[i] then
                fullrank_longterm_interest_score1 = 0.0
            end
        end
    end
    -- 长期兴趣二级标签提权
    local hetu_level_two = hetu_level_two and hetu_level_two or -1 
    local fullrank_longterm_interest_score2 = 0.0
    if hetuLevelTwoLongTermId ~= nil then
        for i = 1, #hetuLevelTwoLongTermId do
            if hetu_level_two == hetuLevelTwoLongTermId[i] then
                fullrank_longterm_interest_score2 = 0.3
            end
        end
    end
    if userBrowseSetHetuLevel2 ~= nil then
        for i = 1, #userBrowseSetHetuLevel2 do
            if hetu_level_two == userBrowseSetHetuLevel2[i] then
                fullrank_longterm_interest_score2 = 0.0
            end
        end
    end
    -- 长期兴趣三级标签提权
    local hetu_level_three = hetu_level_three and hetu_level_three or -1 
    local fullrank_longterm_interest_score3 = 0.0
    if hetuLevelThreeLongTermId ~= nil then
        for i = 1, #hetuLevelThreeLongTermId do
            if hetu_level_three == hetuLevelThreeLongTermId[i] then
                fullrank_longterm_interest_score3 = 0.2
            end
        end
    end
    if userBrowseSetHetuLevel3 ~= nil then
        for i = 1, #userBrowseSetHetuLevel3 do
            if hetu_level_three == userBrowseSetHetuLevel3[i] then
                fullrank_longterm_interest_score3 = 0.0
            end
        end
    end  
    return fullrank_longterm_interest_score1,fullrank_longterm_interest_score2,fullrank_longterm_interest_score3
end