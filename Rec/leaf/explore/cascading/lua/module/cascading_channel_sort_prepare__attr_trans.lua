function htr_filter_threshold()
    local cascade_candidates_count = photo_count_cascading_begin and photo_count_cascading_begin or 3000
    local htr_filter_rate_threshold = explore_mc_phtr_max_filter_rate and explore_mc_phtr_max_filter_rate or 0.5
    local htr_filter_reserved_num = cascade_candidates_count - cascade_candidates_count * htr_filter_rate_threshold
    return math.floor(htr_filter_reserved_num)
end
