
function split(input, delimiter)
    if delimiter == nil then
        delimiter = "%s"  -- 默认按空格分割
    end
    
    local result = {}
    for match in (input..delimiter):gmatch("(.-)"..delimiter) do
        table.insert(result, match)
    end
    
    return result
end

function check_common()
    local follow_aids = follow_aids or {}
    local recent_7d_aids = recent_7d_aids or {}

    local follow_aids_valid = {}
    local recent_7d_aids_valid = {}

    for i, aid in pairs(follow_aids) do
        if aid > 0 then
            table.insert(follow_aids_valid, aid)
        end
    end
    for i, aid in pairs(recent_7d_aids) do
        if aid > 0 then
            table.insert(recent_7d_aids_valid, aid)
        end
    end

    return follow_aids_valid, recent_7d_aids_valid
end

function process_feature(author_fans_count_list, comment_stay_time_list, play_time_ms_list)
    local author_fans_count_list = author_fans_count_list or {0,0,0,0,0,0,0,0,0,0}
    local play_time_ms_list = play_time_ms_list or {0,0,0,0,0,0,0,0,0,0}
    local comment_stay_time_list = comment_stay_time_list  or {0,0,0,0,0,0,0,0,0,0}

    local author_fans_count_rel_list = {}
    local comment_stay_time_rel_list = {}
    local play_time_ms_rel_list = {}
    
    if #author_fans_count_list == 10 then
        for _, value in ipairs(author_fans_count_list) do
            local val = math.min(10000000, (value or 0))
            table.insert(author_fans_count_rel_list, math.floor(5 * math.log(val + 1)))
        end
    end

    if #comment_stay_time_list == 10 then
        for _, value in ipairs(comment_stay_time_list) do
            table.insert(comment_stay_time_rel_list, (value or 0) / 1000)
        end
    end

    if #play_time_ms_list == 10 then
        for _, value in ipairs(play_time_ms_list) do
            table.insert(play_time_ms_rel_list, (value or 0) / 1000)
        end
    end

    return author_fans_count_rel_list, play_time_ms_rel_list, comment_stay_time_rel_list
end

function transfer_author_age()
    local author_age_segment_str_list = author_age_segment_str or {}
    local author_age_segment_list = {}    
    if #author_age_segment_str_list == 10 then
        for index, value in ipairs(author_age_segment_str_list) do
            local index = 0
            if author_age_segment_str == "0-12" then
                index = 1
            elseif author_age_segment_str == "12-17" then
                index = 2
            elseif author_age_segment_str == "18-23" then
                index = 3
            elseif author_age_segment_str == "24-30" then
                index = 4
            elseif author_age_segment_str == "31-40" then
                index = 5
            elseif author_age_segment_str == "41-49" then
                index = 6
            elseif author_age_segment_str == "50+" then
                index = 7
            end
            table.insert(author_age_segment_list, index)
        end
    end
    return author_age_segment_list
end

function gen_label()
    local is_click_list = is_click_list or {0,0,0,0,0,0,0,0,0,0}
    local is_like_list = is_like_list or {0,0,0,0,0,0,0,0,0,0}
    local is_follow_list = is_follow_list or {0,0,0,0,0,0,0,0,0,0}
    local is_forward_list = is_forward_list or {0,0,0,0,0,0,0,0,0,0}
    local is_comment_list = is_comment_list or {0,0,0,0,0,0,0,0,0,0}
    local is_collect_list = is_collect_list or {0,0,0,0,0,0,0,0,0,0}
    local is_hate_list = is_hate_list or {0,0,0,0,0,0,0,0,0,0}
    local real_show_index_list = real_show_index_list or {0,0,0,0,0,0,0,0,0,0}
 
    local play_time_ms_list = play_time_ms_list or {0,0,0,0,0,0,0,0,0,0}
    local duration_ms_list = duration_ms_list or {0,0,0,0,0,0,0,0,0,0}
    local max_time = max_time or 0
    local min_time = min_time or 0
    local timelist = timelist or ""
    local d_session_label = d_session_label or 0

    local point_ltr_label = {0,0,0,0,0,0,0,0,0,0}
    local point_ltr_wt = {1,1,1,1,1,1,1,1,1,1}
    local is_next_label = {0,0,0,0,0,0,0,0,0,0}
    local is_real_show_list = {0,0,0,0,0,0,0,0,0,0}
    local session_inner_time = 0
    -- local session_out_time = 0
    local session_vv = 0
    local is_revisit = 0
    
    if #is_click_list == 10 and #is_like_list == 10 and #is_follow_list == 10 and #is_forward_list == 10 and #is_comment_list == 10 and 
        #is_collect_list == 10 and #is_hate_list == 10 and #play_time_ms_list == 10 and #duration_ms_list == 10 then
        for index, value in ipairs(is_click_list) do
            local wt = 10 * is_like_list[index] + 10 * is_follow_list[index] + 10 * is_forward_list[index]
            + 10 * is_comment_list[index] + 10 * is_collect_list[index]
            if play_time_ms_list[index] > 7000 and  play_time_ms_list[index]/duration_ms_list[index] > 0.8 then
                wt = wt + 10
            end
            if is_click_list[index] == 1 or wt > 10 then
                point_ltr_label[index] = 1
            end
            if wt > 1 then
                point_ltr_wt[index] = wt
            end
        end
    end
    local difftime  = (max_time - min_time)/1000
    session_inner_time = difftime
    local time_list = split(timelist, ",")
    session_vv = #time_list - 1
    -- local tmp_time = 24*60*60
    -- if #time_list > 0 then
    --     for index, value in ipairs(time_list) do
    --         max_min_table = split(value, ",")
    --         min_time_before = tonumber(max_min_table[1]) or 24*60*60
    --         local diff = (min_time_before - max_time)/1000
    --         if diff > 0  and diff < tmp_time then
    --             tmp_time = diff
    --         end
    --     end
    -- end
    -- session_out_time = 24*60*60 - tmp_time

    if #real_show_index_list == 10  then
        for index, value in ipairs(real_show_index_list) do
            if index + 1 <= 10 and real_show_index_list[index + 1] > 0 then
                is_next_label[index] = 1
            end
            if real_show_index_list[10] > 0 then
                is_next_label[10] = 1
            end
            if real_show_index_list[index] > 0 then
                is_real_show_list[index] = 1
            end
        end
    end

    if d_session_label/1000 < 24*60*60 then
        is_revisit = 1
    end

    return point_ltr_label, point_ltr_wt, session_inner_time, session_vv, is_revisit, is_next_label ,is_real_show_list, list_size
end

function transform_string_number(input, output)
    local output = {0,0,0,0,0,0,0,0,0,0}
    if #input == 10 then
        for index, value in ipairs(input) do
            local vals = split(value, "_")
            if #vals > 0 and vals[1] ~= "" and vals[1] ~= "-1" then
                output[index] = tonumber(vals[1])
            end
        end
    end
    return output
end

function process_hetu_feature()
    local hetu_level_one_tag_list = hetu_level_one_tag_list or {"0","0","0","0","0","0","0","0","0","0"}
    local hetu_level_two_tag_list = hetu_level_two_tag_list or {"0","0","0","0","0","0","0","0","0","0"}
    local hetu_level_three_tag_list = hetu_level_three_tag_list or {"0","0","0","0","0","0","0","0","0","0"}
    local emp_ctr_list = emp_ctr_list or {"0","0","0","0","0","0","0","0","0","0"}
    local emp_ltr_list = emp_ltr_list or {"0","0","0","0","0","0","0","0","0","0"}
    local emp_wtr_list = emp_wtr_list or {"0","0","0","0","0","0","0","0","0","0"}
    local emp_ftr_list = emp_ftr_list or {"0","0","0","0","0","0","0","0","0","0"}
    local emp_lvtr_list = emp_lvtr_list or {"0","0","0","0","0","0","0","0","0","0"}
    local pctr_list = pctr_list or {"0","0","0","0","0","0","0","0","0","0"}
    local pltr_list = pltr_list or {"0","0","0","0","0","0","0","0","0","0"}
    local pwtr_list = pwtr_list or {"0","0","0","0","0","0","0","0","0","0"}
    local pftr_list = pftr_list or {"0","0","0","0","0","0","0","0","0","0"}
    local plvtr_list = plvtr_list or {"0","0","0","0","0","0","0","0","0","0"}
    local pptr_list = pptr_list or {"0","0","0","0","0","0","0","0","0","0"}
    local psvtr_list = psvtr_list or {"0","0","0","0","0","0","0","0","0","0"}
    local pwtd_score_list = pwtd_score_list or {"0","0","0","0","0","0","0","0","0","0"}
    local pcltr_list = pcltr_list or {"0","0","0","0","0","0","0","0","0","0"}
    local pcmef_list = pcmef_list or {"0","0","0","0","0","0","0","0","0","0"}
    local pcmtr_list = pcmtr_list or {"0","0","0","0","0","0","0","0","0","0"}
    local mc_htr_list = mc_htr_list or {"0","0","0","0","0","0","0","0","0","0"}
    local mc_pcestr_list = mc_pcestr_list or {"0","0","0","0","0","0","0","0","0","0"}
    local mc_pcltr_list = mc_pcltr_list or {"0","0","0","0","0","0","0","0","0","0"}
    local mc_pcmtr_list = mc_pcmtr_list or {"0","0","0","0","0","0","0","0","0","0"}
    local mc_pctr_list = mc_pctr_list or {"0","0","0","0","0","0","0","0","0","0"}
    local mc_pepstr_list = mc_pepstr_list or {"0","0","0","0","0","0","0","0","0","0"}
    local mc_pltr_list = mc_pltr_list or {"0","0","0","0","0","0","0","0","0","0"}
    local mc_plvtr_list = mc_plvtr_list or {"0","0","0","0","0","0","0","0","0","0"}
    local mc_psvtr_list = mc_psvtr_list or {"0","0","0","0","0","0","0","0","0","0"}
    local mc_pwtd_list = mc_pwtd_list or {"0","0","0","0","0","0","0","0","0","0"}
    local mc_pwtr_list = mc_pwtr_list or {"0","0","0","0","0","0","0","0","0","0"}

    local output = {0,0,0,0,0,0,0,0,0,0}

    return  transform_string_number(hetu_level_one_tag_list, output),
            transform_string_number(hetu_level_two_tag_list, output),
            transform_string_number(hetu_level_three_tag_list, output),
            transform_string_number(emp_ctr_list, output),
            transform_string_number(emp_ltr_list, output),
            transform_string_number(emp_wtr_list, output),
            transform_string_number(emp_ftr_list, output),
            transform_string_number(emp_lvtr_list, output),
            transform_string_number(pctr_list, output),
            transform_string_number(pltr_list, output),
            transform_string_number(pwtr_list, output),
            transform_string_number(pftr_list, output),
            transform_string_number(plvtr_list, output),
            transform_string_number(pptr_list, output),
            transform_string_number(psvtr_list, output),
            transform_string_number(pwtd_score_list, output),
            transform_string_number(pcltr_list, output),
            transform_string_number(pcmef_list, output),
            transform_string_number(pcmtr_list, output),
            transform_string_number(mc_htr_list, output),
            transform_string_number(mc_pcestr_list, output),
            transform_string_number(mc_pcltr_list, output),
            transform_string_number(mc_pcmtr_list, output),
            transform_string_number(mc_pctr_list, output),
            transform_string_number(mc_pepstr_list, output),
            transform_string_number(mc_pltr_list, output),
            transform_string_number(mc_plvtr_list, output),
            transform_string_number(mc_psvtr_list, output),
            transform_string_number(mc_pwtd_list, output),
            transform_string_number(mc_pwtr_list, output)
end