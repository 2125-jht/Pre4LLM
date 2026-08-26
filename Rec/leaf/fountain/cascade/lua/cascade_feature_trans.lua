-- 精排特征转换
function cascade_feature_trans()
  local current_time_ms = currentTimeMs and currentTimeMs or 0
  local featurePUploadTimeDiff = (currentTimeMs - upload_time) // (1000 * 3600)
  local hot_click_count = _G["explore_stat__click_count"] and _G["explore_stat__click_count"] or 0
  local hot_realshow_count = _G["explore_stat__real_show_count"] and _G["explore_stat__real_show_count"] or 0
  local hot_like_count = _G["explore_stat__like_count"] and _G["explore_stat__like_count"] or 0
  local hot_follow_count = _G["explore_stat__follow_count"] and _G["explore_stat__follow_count"] or 0
  local hot_forward_count = _G["explore_stat__forward_count"] and _G["explore_stat__forward_count"] or 0
  local hot_longview_count = _G["explore_stat__long_play_count"] and _G["explore_stat__long_play_count"] or 0
  local hot_shortview_count = _G["explore_stat__short_play_count"] and _G["explore_stat__short_play_count"] or 0
  local hot_view_length_sum = _G["explore_stat__view_length_sum"] and _G["explore_stat__view_length_sum"] or 0
  local featurePHotClickCount = (hot_click_count > 100) and hot_click_count or 0
  local featurePHotLikeCount = (hot_like_count > 100) and hot_like_count or 0
  local featurePHotFollowCount = (hot_follow_count > 100) and hot_follow_count or 0
  local featurePHotLongViewCount = (hot_longview_count > 100) and hot_longview_count or 0
  local featurePHotCtr = (hot_click_count + 1.0) / (hot_realshow_count + 400.0)
  local featurePHotLtr = hot_like_count / (hot_click_count + 100.0)
  local featurePHotWtr = hot_follow_count / (hot_click_count + 1000.0)
  local featurePHotFtr = hot_forward_count / (hot_click_count + 1000.0)
  local featurePHotLvtr = hot_longview_count / (hot_click_count + 100.0)
  local featurePHotSvtr = hot_shortview_count / (hot_click_count + 100.0)
  local featurePHotAvgWatchTime = (hot_click_count > 100) and ((hot_view_length_sum + 0.0) / hot_click_count) or 0.0

  local author_click_count = _G["author__exp_stat__exp_click"] and _G["author__exp_stat__exp_click"] or 0
  local author_realshow_count = _G["author__exp_stat__exp_realshow"] and _G["author__exp_stat__exp_realshow"] or 0
  local author_like_count = _G["author__exp_stat__exp_like"] and _G["author__exp_stat__exp_like"] or 0
  local author_follow_count = _G["author__exp_stat__exp_follow"] and _G["author__exp_stat__exp_follow"] or 0
  local author_forward_count = _G["author__exp_stat__exp_forward"] and _G["author__exp_stat__exp_forward"] or 0
  local author_longview_count = _G["author__exp_stat__exp_long_view"] and _G["author__exp_stat__exp_long_view"] or 0
  local author_shortview_count = _G["author__exp_stat__exp_short_view"] and _G["author__exp_stat__exp_short_view"] or 0
  local author_watch_time = _G["author__exp_stat__exp_watch_time"] and _G["author__exp_stat__exp_watch_time"] or 0
  local featurePAClickCount = author_click_count and author_click_count or nil
  local featurePAClickCount = (author_click_count > 2000) and author_click_count or (math.min((author_click_count // 500) * 500, 100000000))
  local featurePALikeCount = (author_like_count > 2000) and author_like_count or (math.min((author_like_count // 500) * 500, 100000000))
  local featurePAFollowCount = (author_follow_count > 2000) and author_follow_count or (math.min((author_follow_count // 500) * 500, 100000000))
  local featurePALongViewCount = (author_longview_count > 2000) and author_longview_count or (math.min((author_longview_count // 500) * 500, 100000000))
  local featurePACtr = (author_click_count + 1.0) / (author_realshow_count + 400.0)
  local featurePALtr = author_like_count / (author_click_count + 100.0)
  local featurePAWtr = author_follow_count / (author_click_count + 1000.0)
  local featurePAFtr = author_forward_count / (author_click_count + 1000.0)
  local featurePALvtr = author_longview_count / (author_click_count + 100.0)
  local featurePASvtr = author_shortview_count / (author_click_count + 100.0)
  local featurePAAvgWatchTime = (author_click_count > 100) and (author_watch_time / author_click_count) or 0.0
  local duration = duration_ms or 0
  local duration_seg = cascade_wtd_table_seg and cascade_wtd_table_seg or {}
  local fountainDurationPercent = get_duration_percent(duration, duration_seg)
  return featurePUploadTimeDiff, featurePHotClickCount, featurePHotLikeCount, featurePHotFollowCount, featurePHotLongViewCount, featurePHotCtr, featurePHotLtr, featurePHotWtr, featurePHotFtr, featurePHotLvtr, featurePHotSvtr, featurePHotAvgWatchTime, featurePAClickCount, featurePALikeCount, featurePAFollowCount, featurePALongViewCount, featurePACtr, featurePALtr, featurePAWtr, featurePAFtr, featurePALvtr, featurePASvtr, featurePAAvgWatchTime, fountainDurationPercent
end

function get_duration_percent(photo_duration, duration_seg)
  local left = 1
  local right = #duration_seg
  while left <= right do
    local mid = math.floor((left + right) / 2)
    if photo_duration < duration_seg[mid] then
      right = mid - 1
    else
      left = mid + 1
    end
  end
  return right
end
