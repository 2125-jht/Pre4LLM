function calculate()
  local hetu_level_one_list = _G["hetu_tag_level_info__hetu_level_one"] and _G["hetu_tag_level_info__hetu_level_one"] or {}
  local hetu_level_one = (#hetu_level_one_list > 0) and hetu_level_one_list[1] or nil
  local hetu_level_two_list = _G["hetu_tag_level_info__hetu_level_two"] and _G["hetu_tag_level_info__hetu_level_two"] or {}
  local hetu_level_two = (#hetu_level_two_list > 0) and hetu_level_two_list[1] or nil
  local hetu_level_five_list = _G["hetu_tag_level_info__hetu_level_five"] and _G["hetu_tag_level_info__hetu_level_five"] or {}
  local hetu_level_five = (#hetu_level_five_list > 0) and hetu_level_five_list[1] or nil
  return hetu_level_one, hetu_level_two, hetu_level_five
end

function fetch_duration_group_id()
  local duration_id = 0
  local duration_s = math.floor((duration_ms and duration_ms or 0) / 1000)
  local duration_s_max = fountain_duration_s_id_max and fountain_duration_s_id_max or 200
  local threshold_list = faActionL2rV4DurationId_threshold_list and faActionL2rV4DurationId_threshold_list or {}
  if #threshold_list > 0 then
    for i = 1, #threshold_list do
      if duration_s <= threshold_list[i] then
        duration_id = i - 1
        break
      end
    end
  end
  local duration_s_id = math.min(duration_s_max, duration_s)
  local vtr_max_list = fountain_fullrank_ltr_v4_vtr_max_list and fountain_fullrank_ltr_v4_vtr_max_list or {}
  local vtr_max = 1.0
  if #vtr_max_list > 0 then
    local idx = math.min(#vtr_max_list, duration_s + 1)
    vtr_max = vtr_max_list[idx]
  end
  return duration_id, duration_s_id, vtr_max
end
