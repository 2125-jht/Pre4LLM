function getTagIdFromHetuTag(hetuTag)
  local tagId = ((hetuTag >> 8) & 0xffffff)
  return tagId
end

function calculate()
  local page = page and page or -1
  -- 首屏nn召回
  if (topSubdivisionBucket == nil or topSubdivisionBucket == "" or string.len(topSubdivisionBucket) == 0) then
    skip_top_subdivision_nn_retrieval_v2 = 1
  end
  if (exploreSubdivisionBucket == nil or exploreSubdivisionBucket == "" or string.len(exploreSubdivisionBucket) == 0) then
    skip_explore_subdivision_nn_retrieval_v2 = 1
  end
  if (colossusRetrievalTrigger == nil or #colossusRetrievalTrigger <= 0) then
    skip_fountain_colossus_retr = 1
  end
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
  local userBrowseSetOriginalHetuLevel1 = userBrowseSetOriginalHetuLevel1 and userBrowseSetOriginalHetuLevel1 or {}
  local userBrowseSetHetuLevel1 = {}
  local index = 1
  for i = 1, #userBrowseSetOriginalHetuLevel1 do
    local tagId = getTagIdFromHetuTag(userBrowseSetOriginalHetuLevel1[i])
    table.insert(userBrowseSetHetuLevel1, index, tagId)
    index = index + 1
  end
  local userBrowseSetOriginalHetuLevel2 = userBrowseSetOriginalHetuLevel2 and userBrowseSetOriginalHetuLevel2 or {}
  local userBrowseSetHetuLevel2 = {}
  local index = 1
  for i = 1, #userBrowseSetOriginalHetuLevel2 do
    local tagId = getTagIdFromHetuTag(userBrowseSetOriginalHetuLevel2[i])
    table.insert(userBrowseSetHetuLevel2, index, tagId)
    index = index + 1
  end
  local userBrowseSetOriginalHetuLevel3 = userBrowseSetOriginalHetuLevel3 and userBrowseSetOriginalHetuLevel3 or {}
  local userBrowseSetHetuLevel3 = {}
  local index = 1
  for i = 1, #userBrowseSetOriginalHetuLevel3 do
    local tagId = getTagIdFromHetuTag(userBrowseSetOriginalHetuLevel3[i])
    table.insert(userBrowseSetHetuLevel3, index, tagId)
    index = index + 1
  end
  local skip_interact_author_retr = 1
  if find_v4_skip_fountain_interact_author_retr ~= nil and find_v4_skip_fountain_interact_author_retr == 0 and skip_fountain_mid_photo_gnn_i2i_retr ~= nil and skip_fountain_mid_photo_gnn_i2i_retr == 0 then
    skip_interact_author_retr = 0
  end
  local skip_fountain_colossus_retr_emb_fetch_old = skip_fountain_colossus_retr
  if skip_fountain_colossus_retr == nil or skip_fountain_colossus_retr == 1 then
    skip_fountain_colossus_retr_emb_fetch_new = 1
  else
    skip_fountain_colossus_retr_emb_fetch_old = 1 - skip_fountain_colossus_retr_emb_fetch_new
  end
  return skip_top_subdivision_nn_retrieval_v2, skip_explore_subdivision_nn_retrieval_v2, skip_fountain_colossus_retr, source_hetu_level_one_v2, source_hetu_level_two_v2, source_hetu_level_three_v2,
    source_hetu_level_four_v2, source_hetu_tag_v2, source_hetu_face_id_v2, userBrowseSetHetuLevel1, userBrowseSetHetuLevel2, userBrowseSetHetuLevel3, skip_interact_author_retr,
    skip_fountain_colossus_retr_emb_fetch_old, skip_fountain_colossus_retr_emb_fetch_new
end
