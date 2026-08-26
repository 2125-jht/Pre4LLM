function calculate()
  -- 返回值
  if (sourcePidAuthorId == nil) then
      sourcePidAuthorId = -1
  end

  local enable_cal_information_score_splash = 0
  local is_information_category = 0
  local use_v3 = enable_explore_fountain_leaf_use_hetu_v3 or 0
  if (source_hetu_level_one_v2 ~= nil) then
    for i = 1, #source_hetu_level_one_v2 do
      if source_hetu_level_one_v2[i] == 28 or (use_v3 > 0 and source_hetu_level_one_v2[i] == 23) then
        is_information_category = 1
      end
    end
  end
  -- 仅对资讯垂类（hetu_level_one == 28）开启新版相关分计算 
  if (enable_cal_information_score_init == 1 and is_information_category == 1) then
    enable_cal_information_score_splash = 1
  end
  return sourcePidAuthorId, enable_cal_information_score_splash
end
