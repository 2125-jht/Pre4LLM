-- 精排特征转换
function fullrank_feature_trans()
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
  local profile_enter_count = _G['explore_stat__profile_enter_count'] or 0
  local comment_count = _G['explore_stat__comment_count'] or 0
  local negative_count = _G['explore_stat__negative_count'] or 0
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

  local real_show_count = hot_realshow_count > 0 and hot_realshow_count or 10
  local click_count = hot_click_count > 0 and hot_click_count or 1
  local profile_enter_count = profile_enter_count > 0 and profile_enter_count or 0
  local comment_count = comment_count > 0 and comment_count or 0
  local negative_count = negative_count > 0 and negative_count or 0
  local empirical_ctr = click_count * 1.0 / real_show_count
  local empirical_ltr = hot_like_count * 1.0 / click_count
  local empirical_wtr = hot_follow_count * 1.0 / click_count
  local empirical_ftr = hot_forward_count * 1.0 / click_count
  local empirical_ptr = profile_enter_count * 1.0 / click_count
  local empirical_cmtr = comment_count * 1.0 / click_count
  local empirical_htr = negative_count * 1.0 / click_count
  local empirical_watchtime = hot_view_length_sum / (hot_click_count + 100.0)
   
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
  local hetu_level_one_list = _G["hetu_tag_level_info__hetu_level_one"] and _G["hetu_tag_level_info__hetu_level_one"] or {}
  local featurePHetu0 = (#hetu_level_one_list > 0) and hetu_level_one_list[1] or -1
  local hetu_level_one_list = _G["hetu_tag_level_info_v2__hetu_level_one"] and _G["hetu_tag_level_info_v2__hetu_level_one"] or {}
  local hetu_level_one = (#hetu_level_one_list > 0) and hetu_level_one_list[1] or nil
  local hetu_level_two_list = _G["hetu_tag_level_info_v2__hetu_level_two"] and _G["hetu_tag_level_info_v2__hetu_level_two"] or {}
  local hetu_level_two = (#hetu_level_two_list > 0) and hetu_level_two_list[1] or nil
  local hetu_level_three_list = _G["hetu_tag_level_info_v2__hetu_level_three"] and _G["hetu_tag_level_info_v2__hetu_level_three"] or {}
  local hetu_level_three = (#hetu_level_three_list > 0) and hetu_level_three_list[1] or nil
  local hetu_level_one_v2_list = _G["hetu_level_one_v2"] and _G["hetu_level_one_v2"] or {}
  local hetu_level_one_v2_index = (#hetu_level_one_v2_list > 0) and hetu_level_one_v2_list[1] or nil
  local duration = duration_ms or 0
  local fountainDurationPercent = get_duration_percent(duration)
  return featurePUploadTimeDiff, featurePHotClickCount, featurePHotLikeCount, featurePHotFollowCount, featurePHotLongViewCount, featurePHotCtr, featurePHotLtr, featurePHotWtr, featurePHotFtr, featurePHotLvtr, featurePHotSvtr, featurePHotAvgWatchTime, featurePAClickCount, featurePALikeCount, featurePAFollowCount, featurePALongViewCount, featurePACtr, featurePALtr, featurePAWtr, featurePAFtr, featurePALvtr, featurePASvtr, featurePAAvgWatchTime, featurePHetu0, hetu_level_one,hetu_level_two,hetu_level_three,hetu_level_one_v2_index, fountainDurationPercent,empirical_ctr, empirical_ltr, empirical_wtr, empirical_ftr, empirical_ptr, empirical_cmtr, empirical_htr,empirical_watchtime
end

function get_simple_ltr_feature()
  local user_emp_ltr = user_emp_ltr or 0.0
  local user_emp_wtr = user_emp_wtr or 0.0
  local user_emp_ftr = user_emp_ftr or 0.0
  local user_emp_cmtr = user_emp_cmtr or 0.0
  local user_emp_eptr = user_emp_eptr or 0.0
  local user_emp_htr = user_emp_htr or 0.0
  local all_emp_ltr = all_emp_ltr or 1.0
  if (all_emp_ltr <= 0.0) then
    all_emp_ltr = 1.0
  end
  local all_emp_wtr = all_emp_wtr or 1.0
  if (all_emp_wtr <= 0.0) then
    all_emp_wtr = 1.0
  end
  local all_emp_ftr = all_emp_ftr or 1.0
  if (all_emp_ftr <= 0.0) then
    all_emp_ftr = 1.0
  end
  local all_emp_cmtr = all_emp_cmtr or 1.0
  if (all_emp_cmtr <= 0.0) then
    all_emp_cmtr = 1.0
  end
  local all_emp_eps = all_emp_eps or 1.0
  if (all_emp_eps <= 0.0) then
    all_emp_eps = 1.0
  end
  local all_emp_htr = all_emp_htr or 1.0
  if (all_emp_htr <= 0.0) then
    all_emp_htr = 1.0
  end

  local feature_ltr = user_emp_ltr > 0.0 and user_emp_ltr / all_emp_ltr or 1.0
  local feature_wtr = user_emp_wtr > 0.0 and user_emp_wtr / all_emp_wtr or 1.0
  local feature_ftr = user_emp_ftr > 0.0 and user_emp_ftr / all_emp_ftr or 1.0
  local feature_cmtr = user_emp_cmtr > 0.0 and user_emp_cmtr / all_emp_cmtr or 1.0
  local feature_eptr = user_emp_eptr > 0.0 and user_emp_eptr / all_emp_eps or 1.0
  local feature_htr = user_emp_htr > 0.0 and user_emp_htr / all_emp_htr or 1.0

  return feature_ltr, feature_wtr, feature_ftr, feature_cmtr, feature_eptr, feature_htr
end

function get_duration_percent(photo_duration)
  local duration_seg = {7532, 8958, 10545, 11945, 14606, 
		16668, 20391, 24579, 30240, 35872, 44725, 57053, 
		70659, 86381, 104715, 127714, 156791, 198041, 274216, 100000000}
  local left = 1
  local right = #duration_seg
  local mid = math.ceil((left + right) / 2)
  while left ~= mid do
    if duration_seg[mid] == photo_duration then
	   break
    elseif duration_seg[mid] < photo_duration then
	   left = mid + 1
	else
      right = mid - 1
    end
	mid = math.ceil((left + right) / 2)
  end
  return left - 1
end