function calculate()
  local hetu_level_one_list = _G["hetu_tag_level_info__hetu_level_one"] and _G["hetu_tag_level_info__hetu_level_one"] or {}
  local hetu_level_one = (#hetu_level_one_list > 0) and hetu_level_one_list[1] or -1
  local hetu_level_two_list = _G["hetu_tag_level_info__hetu_level_two"] and _G["hetu_tag_level_info__hetu_level_two"] or {}
  local hetu_level_two = (#hetu_level_two_list > 0) and hetu_level_two_list[1] or nil
  return hetu_level_one, hetu_level_two
end
