function calculate()
  local enable_fountain_related_score_calc_v3 = enable_fountain_related_score_calc_v3
  local fountain_skip_filter_photo_by_not_related_reason_splash_v2 = fountain_skip_filter_photo_by_not_related_reason_splash_v2
  local fountain_skip_filter_photo_by_not_related_information_splash = fountain_skip_filter_photo_by_not_related_information_splash
  if ((source_hetu_level_one_v2 == nil or #source_hetu_level_one_v2 == 0)
    and (source_hetu_level_two_v2 == nil or #source_hetu_level_two_v2 == 0)
    and (source_hetu_level_three_v2 == nil or #source_hetu_level_three_v2 == 0)
    and (source_hetu_level_four_v2 == nil or #source_hetu_level_four_v2 == 0)
    and sourcePidFourthLevelCategory == nil
    and sourcePidThirdLevelCategory == nil
    and (source_hetu_face_id_v2 == nil or #source_hetu_face_id_v2 == 0)
    and (source_hetu_tag_v2 == nil or #source_hetu_tag_v2 == 0)
    and source_hetu_cluster_id_v2 == nil) then
      enable_fountain_related_score_calc_v3 = 0
      fountain_skip_filter_photo_by_not_related_reason_splash_v2 = 1
      fountain_skip_filter_photo_by_not_related_information_splash = 1
  end
  return enable_fountain_related_score_calc_v3, fountain_skip_filter_photo_by_not_related_reason_splash_v2, fountain_skip_filter_photo_by_not_related_information_splash
end
