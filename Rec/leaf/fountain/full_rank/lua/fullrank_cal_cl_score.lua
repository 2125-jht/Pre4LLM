--fullrank cl score cal with duration
function fullrank_cal_cl_score()
    local cl_time_weight = fountain_fullrank_cl_time_score_weight and fountain_fullrank_cl_time_score_weight or 1.0
    local cl_duration_weight = fountain_fullrank_cl_duration_weight and fountain_fullrank_cl_duration_weight or 0.0
    local cl_click_weight = fountain_fullrank_cl_click_weight and fountain_fullrank_cl_click_weight or 0.0
    local cl_threshold_weight = fountain_fullrank_cl_threshold_weight and fountain_fullrank_cl_threshold_weight or 0.0
    local cl_duration_seg =   fountain_fullrank_cl_duration_seg and fountain_fullrank_cl_duration_seg or 1000
    local cl_duration_max =   fountain_fullrank_cl_duration_max and fountain_fullrank_cl_duration_max or 120
    local cl_score = fullrank_cl_score and fullrank_cl_score or 0.0
    local fullrank_sim_pevtr = fullrank_sim_pevtr or 0
    local enable_bias = fountain_fullrank_cl_enable_threshold_bias or 0.0
    local enable_bias_v2 = fountain_fullrank_cl_enable_threshold_bias_v2 or 0.0
    local enable_duration = fountain_fullrank_cl_enable_duration or 1.0

    if enable_duration > 0 then
        cl_score = cl_score ^ cl_time_weight * math.min(duration_ms / cl_duration_seg, cl_duration_max) ^ cl_duration_weight * fullrank_sim_pevtr ^ cl_click_weight
    end
    if enable_bias > 0 then
        local index = math.min(200, math.max(4, math.floor(duration_ms / 1000))) - 3
        local duration_finish_threshold = duration_finish_threshold or {}
        if #duration_finish_threshold == 197 then
            cl_score = (cl_score * duration_finish_threshold[index]) ^ cl_time_weight * math.min(duration_ms / cl_duration_seg, cl_duration_max) ^ cl_duration_weight * fullrank_sim_pevtr ^ cl_click_weight
        end
    end
    if enable_bias_v2 > 0 then
        local tmp_score = fullrank_cl_score and fullrank_cl_score or 0.0
        local index = math.min(200, math.max(4, math.floor(duration_ms / 1000))) - 3
        local duration_finish_threshold = duration_finish_threshold or {}
        if #duration_finish_threshold == 197 then
            cl_score = tmp_score ^ cl_time_weight
                    * (math.min(duration_ms / cl_duration_seg, cl_duration_max)) ^ cl_duration_weight
                    * fullrank_sim_pevtr ^ cl_click_weight
                    * duration_finish_threshold[index] ^ cl_threshold_weight
        end
    end
    return cl_score
end