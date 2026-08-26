function calculate()
  -- 返回值
  if (sourcePidAuthorId == nil) then
      sourcePidAuthorId = -1
  end

  local enable_cal_information_score_splash = 0
  local is_information_category = 0
  if (source_hetu_level_one_v2 ~= nil) then
    for i = 1, #source_hetu_level_one_v2 do
      if source_hetu_level_one_v2[i] == 28 or source_hetu_level_one_v2[i] == 23 then
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

function getTagIdFromHetuTag(hetuTag)
  local tagId = ((hetuTag >> 8) & 0xffffff)
  return tagId
end

function retrieval_splash_control()
  local sourcePidHetuFaceId = sourcePidHetuFaceId and sourcePidHetuFaceId or nil
  local source_hetu_level_one_v2_original = source_hetu_level_one_v2_original and source_hetu_level_one_v2_original or {}
  local source_hetu_level_one_v2 = {}
  local index = 1
  for i = 1, #source_hetu_level_one_v2_original do
    local tagId = getTagIdFromHetuTag(source_hetu_level_one_v2_original[i])
    table.insert(source_hetu_level_one_v2, index, tagId)
    index = index + 1
  end
  local source_hetu_level_two_v2_original = source_hetu_level_two_v2_original and source_hetu_level_two_v2_original or {}
  local source_hetu_level_two_v2 = {}
  local index = 1
  for i = 1, #source_hetu_level_two_v2_original do
    local tagId = getTagIdFromHetuTag(source_hetu_level_two_v2_original[i])
    table.insert(source_hetu_level_two_v2, index, tagId)
    index = index + 1
  end
  local source_hetu_level_three_v2_original = source_hetu_level_three_v2_original and source_hetu_level_three_v2_original or {}
  local source_hetu_level_three_v2 = {}
  index = 1
  for i = 1, #source_hetu_level_three_v2_original do
    local tagId = getTagIdFromHetuTag(source_hetu_level_three_v2_original[i])
    table.insert(source_hetu_level_three_v2, index, tagId)
    index = index + 1
  end
  local source_hetu_level_four_v2_original = source_hetu_level_four_v2_original and source_hetu_level_four_v2_original or {}
  local source_hetu_level_four_v2 = {}
  index = 1
  for i = 1, #source_hetu_level_four_v2_original do
    local tagId = getTagIdFromHetuTag(source_hetu_level_four_v2_original[i])
    table.insert(source_hetu_level_four_v2, index, tagId)
    index = index + 1
  end
  local source_hetu_tag_v2_original = source_hetu_tag_v2_original and source_hetu_tag_v2_original or {}
  local source_hetu_tag_v2 = {}
  index = 1
  for i = 1, #source_hetu_tag_v2_original do
    local tagId = getTagIdFromHetuTag(source_hetu_tag_v2_original[i])
    table.insert(source_hetu_tag_v2, index, tagId)
    index = index + 1
  end
  local source_hetu_face_id_v2_original = source_hetu_face_id_v2_original and source_hetu_face_id_v2_original or {}
  local source_hetu_face_id_v2 = {}
  index = 1
  for i = 1, #source_hetu_face_id_v2_original do
    local tagId = getTagIdFromHetuTag(source_hetu_face_id_v2_original[i])
    table.insert(source_hetu_face_id_v2, index, tagId)
    index = index + 1
  end
  local source_hetu_cluster_id_v2 = source_hetu_cluster_id_v2_original and source_hetu_cluster_id_v2_original or nil
  return
      source_hetu_level_one_v2, source_hetu_level_two_v2, source_hetu_level_three_v2, source_hetu_level_four_v2,
      source_hetu_tag_v2, source_hetu_face_id_v2, source_hetu_cluster_id_v2
end
