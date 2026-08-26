function htr_filter_threshold()
    local cascade_candidates_count = photo_count_cascading_begin and photo_count_cascading_begin or 3000
    local htr_filter_rate_threshold = explore_mc_phtr_max_filter_rate and explore_mc_phtr_max_filter_rate or 0.5
    local htr_filter_reserved_num = cascade_candidates_count - cascade_candidates_count * htr_filter_rate_threshold
    return math.floor(htr_filter_reserved_num)
end

function get_intere_lists()
    local intere_str_list = intere_str_list or {}
    local intere_score_str = ""
    local cluster_id_str = ""
    local cluster_vv_str =""
    if #intere_str_list < 2 then 
        return cluster_id_str, intere_score_str, cluster_vv_str
    end
    intere_score_str = intere_str_list[1]
    cluster_id_str = intere_str_list[2]
    if  #intere_str_list >=3 then
        cluster_vv_str = intere_str_list[3]
    end
    
    return intere_score_str, cluster_id_str, cluster_vv_str
end

function get_intere_scores()
    local intere_score_list = intere_score_list or {}
    local cluster_id_list = cluster_id_list or {}
    local cluster_intere_score_map = {}
    local cluster_num = 633
    for i = 1, cluster_num do
        cluster_intere_score_map[i] = 0.0
    end
    if #intere_score_list == 0 or #cluster_id_list == 0 
        or #intere_score_list ~= #cluster_id_list then
            return cluster_intere_score_map
    end
    for i, cid in ipairs(cluster_id_list) do 
        cluster_intere_score_map[cid] = intere_score_list[i]
    end
    return cluster_intere_score_map
end

function get_item_intere_score()
    local cluster_id_1k = hetu_tag_level_info__hetu_cluster_id or -1
    local remap_cluster_id_632_list = remap_cluster_id_632_list or {}
    local cluster_id_632 = -1
    if #remap_cluster_id_632_list > 0 and  cluster_id_1k > 0 then
        cluster_id_632 = remap_cluster_id_632_list[cluster_id_1k]
    end
    local cluster_intere_score_map = cluster_intere_score_map or {}
    local intere_score = 1.0
    intere_score = cluster_intere_score_map[cluster_id_632] or 0.0
    
    local is_eff_intere = 0
    local is_not_intere = 0
    if intere_score >= 3 then
        is_eff_intere = 1
    end 
    if intere_score < -3.6 then
        is_not_intere = 1
    end
    intere_score = math.min(intere_score, 10.0)
    intere_score = math.max(intere_score, -10.0)
    nonneg_intere_score = math.max(intere_score+1.8, 0.0)
    return intere_score, is_eff_intere, is_not_intere, nonneg_intere_score
end