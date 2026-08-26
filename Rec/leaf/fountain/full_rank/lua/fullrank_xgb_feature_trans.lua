-- 精排特征转换
function fullrank_xgb_feature_trans()
    local explore_real_show_count = _G["explore_stat__real_show_count"] and _G["explore_stat__real_show_count"] or 0
    local explore_profile_enter_count = _G["explore_stat__profile_enter_count"] and _G["explore_stat__profile_enter_count"] or 0
    local explore_like_count = _G["explore_stat__like_count"] and _G["explore_stat__like_count"] or 0
    local explore_forward_count = _G["explore_stat__forward_count"] and _G["explore_stat__forward_count"] or 0
    local explore_long_play_count = _G["explore_stat__long_play_count"] and _G["explore_stat__long_play_count"] or 0
    local explore_short_play_count = _G["explore_stat__short_play_count"] and _G["explore_stat__short_play_count"] or 0
    local explore_click_count = _G["explore_stat__click_count"] and _G["explore_stat__click_count"] or 0
    local explore_comment_count = _G["explore_stat__comment_count"] and _G["explore_stat__comment_count"] or 0
    local explore_follow_count = _G["explore_stat__follow_count"] and _G["explore_stat__follow_count"] or 0
    return featurePDurationMs,explore_real_show_count,explore_profile_enter_count,explore_forward_count,explore_long_play_count,explore_like_count,explore_click_count,explore_follow_count,explore_comment_count,explore_short_play_count
  end