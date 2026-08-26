function getTagIdFromHetuTag(hetuTag)
  local tagId = ((hetuTag >> 8) & 0xffffff)
  return tagId
end

function calculate()
  local hetu_level_one_v2_original = _G['hetu_tag_level_info_v2__hetu_level_one'] and _G['hetu_tag_level_info_v2__hetu_level_one'] or {}
  local hetu_level_one_v2 = {}
  local index = 1
  for i = 1, #hetu_level_one_v2_original do
    local tagId = getTagIdFromHetuTag(hetu_level_one_v2_original[i])
    table.insert(hetu_level_one_v2, index, tagId)
    index = index + 1
  end
  local hetu_level_two_v2_original = _G['hetu_tag_level_info_v2__hetu_level_two'] and _G['hetu_tag_level_info_v2__hetu_level_two'] or {}
  local hetu_level_two_v2 = {}
  local index = 1
  for i = 1, #hetu_level_two_v2_original do
    local tagId = getTagIdFromHetuTag(hetu_level_two_v2_original[i])
    table.insert(hetu_level_two_v2, index, tagId)
    index = index + 1
  end
  local hetu_level_three_v2_original = _G['hetu_tag_level_info_v2__hetu_level_three'] and _G['hetu_tag_level_info_v2__hetu_level_three'] or {}
  local hetu_level_three_v2 = {}
  index = 1
  for i = 1, #hetu_level_three_v2_original do
    local tagId = getTagIdFromHetuTag(hetu_level_three_v2_original[i])
    table.insert(hetu_level_three_v2, index, tagId)
    index = index + 1
  end
  local hetu_level_five_v2_original = _G['hetu_tag_level_info_v2__hetu_level_five'] and _G['hetu_tag_level_info_v2__hetu_level_five'] or {}
  local hetu_level_five_v2 = {}
  index = 1
  for i = 1, #hetu_level_five_v2_original do
    local tagId = getTagIdFromHetuTag(hetu_level_five_v2_original[i])
    table.insert(hetu_level_five_v2, index, tagId)
    index = index + 1
  end
  local hetu_tag_v2_original = _G['hetu_tag_level_info_v2__hetu_tag'] and _G['hetu_tag_level_info_v2__hetu_tag'] or {}
  local hetu_tag_v2 = {}
  index = 1
  for i = 1, #hetu_tag_v2_original do
    local tagId = getTagIdFromHetuTag(hetu_tag_v2_original[i])
    table.insert(hetu_tag_v2, index, tagId)
    index = index + 1
  end
  local hetu_face_id_v2_original = _G['hetu_tag_level_info_v2__hetu_face_id'] and _G['hetu_tag_level_info_v2__hetu_face_id'] or {}
  local hetu_face_id_v2 = {}
  index = 1
  for i = 1, #hetu_face_id_v2_original do
    local tagId = getTagIdFromHetuTag(hetu_face_id_v2_original[i])
    table.insert(hetu_face_id_v2, index, tagId)
    index = index + 1
  end
  local limit_hetu_table = {}
  for i = 1, #hetu_level_one_v2 do
    local id = hetu_level_one_v2[i]
    if id == 6 or id == 9 or id ==39 then  
      table.insert(limit_hetu_table, i, id)
    else 
      table.insert(limit_hetu_table, i, 10002)
    end
  end
  return hetu_level_one_v2, hetu_level_two_v2, hetu_level_three_v2, hetu_level_five_v2, hetu_tag_v2, hetu_face_id_v2, limit_hetu_table
end