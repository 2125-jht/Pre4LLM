-- 非首屏精排整体排序后的调权
function  fullrank_score_adjust_fast()
  local fullrank_ensemble_score = fullrank_ensemble_score and fullrank_ensemble_score or 0.0
  local skip_neg_feedback_discount = skip_fullrank_negative_feedback_discount and skip_fullrank_negative_feedback_discount or 1
  local skip_neg_feedback_discount_v2 = skip_fullrank_negative_feedback_discount_v2 and skip_fullrank_negative_feedback_discount_v2 or 1
  local fullrank_discount_ratio = fullrank_discount_ratio and fullrank_discount_ratio or 0
  local fullrank_neg_feedback_discount_score = fullrank_ensemble_score
  if ((skip_neg_feedback_discount == 0 or skip_neg_feedback_discount_v2 == 0) and fullrank_discount_ratio > 0) then
    fullrank_neg_feedback_discount_score = fullrank_neg_feedback_discount_score * fullrank_discount_ratio
  end
  local duration_ms = duration_ms and duration_ms or 0.0
  local plvtr = fullrank_sim_plvtr and fullrank_sim_plvtr or 0.0
  local explore_operation_c_review_level = explore_operation_c_review_level and explore_operation_c_review_level or 0
  -- 打压社区内容
  if (enable_community_discount == 1 and community_discount_ratio > 0 and explore_operation_c_review_level == 2000001078) then
    fullrank_neg_feedback_discount_score = fullrank_neg_feedback_discount_score * community_discount_ratio
  end
  if (plvtr > long_duration_boost_min_plvtr and duration_ms > 58000.0 and long_duration_boost > 1.0) then
    fullrank_neg_feedback_discount_score = fullrank_neg_feedback_discount_score * long_duration_boost
  end
  -- questionnaire_score boost
  local enable_questionnaire_boost = fullrank_enable_questionnaire_boost or 0
  if (enable_questionnaire_boost > 0) then
    local questionnaire_boost_ratio = fullrank_questionnaire_boost_ratio or 1.0
    local questionnaire_boost_threshold = fullrank_questionnaire_boost_threshold or 1.0
    local questionnaire_score = questionnaire_score or 0.0
    if (questionnaire_score > questionnaire_boost_threshold) then
      fullrank_neg_feedback_discount_score = fullrank_neg_feedback_discount_score * questionnaire_boost_ratio
    end
  end
  return fullrank_neg_feedback_discount_score
end

-- 首屏精排整体排序后的调权
function  fullrank_score_adjust_splash()
  local fullrank_ensemble_score = fullrank_ensemble_score and fullrank_ensemble_score or 0.0
  local fullrank_ensemble_score_after_adjust = fullrank_ensemble_score
  local related_score = source_related_score and source_related_score or 0
  local enable_fountain_movie_ip_boost = enable_fountain_movie_ip_boost and enable_fountain_movie_ip_boost or 0
  local fountain_movie_ip_boost_ratio = fountain_movie_ip_boost_ratio and fountain_movie_ip_boost_ratio or 0.0
  local duration_ms = duration_ms and duration_ms or 0.0
  local plvtr = fullrank_sim_plvtr and fullrank_sim_plvtr or 0.0
  -- 相关性提权
  if (related_score > 0) then
    fullrank_ensemble_score_after_adjust = fullrank_ensemble_score_after_adjust * (1.0 + related_score)
  end
  -- 影视 ip boost
  if (enable_fountain_movie_ip_boost == 1 and reason == 10305 and fountain_movie_ip_boost_ratio > 0) then
    fullrank_ensemble_score_after_adjust = fullrank_ensemble_score_after_adjust * fountain_movie_ip_boost_ratio
  end
  if (plvtr > long_duration_boost_min_plvtr and duration_ms > 58000.0 and long_duration_boost > 1.0) then
    fullrank_ensemble_score_after_adjust = fullrank_ensemble_score_after_adjust * long_duration_boost
  end
  -- questionnaire_score boost
  local enable_questionnaire_boost = fullrank_enable_questionnaire_boost or 0
  if (enable_questionnaire_boost > 0) then
    local questionnaire_boost_ratio = fullrank_questionnaire_boost_ratio or 1.0
    local questionnaire_boost_threshold = fullrank_questionnaire_boost_threshold or 1.0
    local questionnaire_score = questionnaire_score or 0.0
    if (questionnaire_score > questionnaire_boost_threshold) then
      fullrank_ensemble_score_after_adjust = fullrank_ensemble_score_after_adjust * questionnaire_boost_ratio
    end
  end

  return fullrank_ensemble_score_after_adjust
end