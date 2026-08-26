function calculate()
  local result = {}
  if videoPlayRawAids == nil or videoDurations == nil or videoPlayTime == nil or #videoPlayRawAids ~= #videoDurations or #videoDurations ~= #videoPlayTime then
    return result
  end
  local total_cnt = 0
  for i, v in ipairs(videoPlayRawAids) do
    if total_cnt >= 200 then
      break
    end
    if videoDurations[i] > duration_lower_limit and ((videoDurations[i] <= duration_upper_limit and videoPlayTime[i] >= videoDurations[i]) or (videoDurations[i] > duration_upper_limit and videoPlayTime[i] >= duration_upper_limit)) then
      table.insert(result, v)
      total_cnt = total_cnt + 1
    end
  end
  return result
end