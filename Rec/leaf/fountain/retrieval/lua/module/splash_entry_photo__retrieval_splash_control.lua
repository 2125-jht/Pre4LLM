function getTagIdFromHetuTag(hetuTag)
  local tagId = ((hetuTag >> 8) & 0xffffff)
  return tagId
end

-- 生成 times 个 [1, len] 间不重复的随机数
function GetRandomNumList(len, times)
  local rsList = {}
  for i = 1, len do
      table.insert(rsList,i)
  end
  local num, tmp
  for i = 1, times do
      math.randomseed(currentTimeMs)
      num = math.random(1, len)
      tmp = rsList[i]
      rsList[i] = rsList[num]
      rsList[num] = tmp
  end
  return rsList
end

function retrieval_splash_control()
  -- 首屏nn召回
  local skip_fountain_top_subdivision_nn_retrieval_tag_splash = skip_fountain_top_subdivision_nn_retrieval_tag_splash or 1
  if (topSubdivisionHetuBucket == nil or topSubdivisionHetuBucket == "" or string.len(topSubdivisionHetuBucket) == 0) then
    skip_fountain_top_subdivision_nn_retrieval_tag_splash = 1
    skip_fountain_reco_emb_hetu_retrieval_splash = 1
  end
  local fountain_enable_first_page_skip_u2i_retrieval = fountain_enable_first_page_skip_u2i_retrieval and fountain_enable_first_page_skip_u2i_retrieval or 0
  local fountain_skip_reco_emb_u2i_retr_splash = fountain_skip_reco_emb_u2i_retr_splash and fountain_skip_reco_emb_u2i_retr_splash or 1
  local fountain_skip_gcse_u2i_retrieval_splash = fountain_skip_gcse_u2i_retrieval_splash and fountain_skip_gcse_u2i_retrieval_splash or 1
  if (fountainHetuTagBucket == nil or fountainHetuTagBucket == "" or string.len(fountainHetuTagBucket) == 0 or fountain_enable_first_page_skip_u2i_retrieval == 1) then
    fountain_skip_reco_emb_u2i_retr_splash = 1
    fountain_skip_gcse_u2i_retrieval_splash = 1
  end
  if (topSubdivisionBucket == nil or topSubdivisionBucket == "" or string.len(topSubdivisionBucket) == 0) then
    fountain_retrieval_skip_top_subdivision_nn_retrieval = 1
  end
  local sourcePidHetuFaceId = sourcePidHetuFaceId and sourcePidHetuFaceId or nil
  local fountain_swing_retr_redis_key = 'fountain_swing_pid_' .. tostring(featureSourcePId)
  local fountain_relation_interaction_retr_redis_key = 'cc_relation_feed_'.. tostring(featureUId)
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
  table_index = 1
  local source_related_tag_key_list_tmp = {}
  for i = 1, #source_hetu_tag_v2 do
    local tagId = source_hetu_tag_v2[i]
    if (tagId >= 500006 and tagId < 4000000) or (tagId >= 55000 and tagId < 60000) then 
      source_related_tag_key = 'hetutag:tag2tag:v2:ids:' .. tostring(tagId)
      table.insert(source_related_tag_key_list_tmp, table_index, source_related_tag_key)
      table_index = table_index + 1
    end
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
  local fountain_skip_ip2tag2ip_retr_splash = fountain_skip_ip2tag2ip_retr_splash and fountain_skip_ip2tag2ip_retr_splash or 1
  local fountain_enable_ip2tag2ip_retr_opt = fountain_enable_ip2tag2ip_retr_opt and fountain_enable_ip2tag2ip_retr_opt or 0
  local source_movie_related_ips_key = {}
  local source_movie_ip_extends_key = {}
  if (sourceMovieIp == nil or #sourceMovieIp <= 0) then
    fountain_skip_ip2tag2ip_retr_splash = 1
    fountain_enable_ip2tag2ip_retr_opt = 0
  else
    if (fountain_skip_ip2tag2ip_retr_splash == 0) then
      for _, ip in ipairs(sourceMovieIp) do
        table.insert(source_movie_related_ips_key, 'kgtag:ip2tag:' .. tostring(ip))
        if fountain_ip2tag2ip_retr_movie2movie_level >= 1 and fountain_enable_ip2tag2ip_retr_opt == 0 then
          table.insert(source_movie_ip_extends_key, 'kgtag:movie2movie:series:' .. tostring(ip))
        end
        if fountain_ip2tag2ip_retr_movie2movie_level >= 2 then
          table.insert(source_movie_ip_extends_key, 'kgtag:movie2movie:behaviors:' .. tostring(ip))
        end
        if fountain_ip2tag2ip_retr_movie2movie_level >= 3 then
          table.insert(source_movie_ip_extends_key, 'kgtag:movie2movie:subjecttag:' .. tostring(ip))
        end
        if fountain_ip2tag2ip_retr_movie2movie_level >= 4 then
          table.insert(source_movie_ip_extends_key, 'kgtag:movie2movie:commonactor:' .. tostring(ip))
        end
        table.insert(source_movie_ip_extends_key, 'kgtag:person2movie:coreactor:' .. tostring(ip))
        table.insert(source_movie_ip_extends_key, 'kgtag:person2person:merged:' .. tostring(ip))
      end
    end
  end
  local skip_fountain_icf_splash_retr_final = skip_fountain_icf_splash_retr or 1
  if (skip_fountain_icf_splash_retr_mobile ~= nil and  skip_fountain_icf_splash_retr_mobile == 0)  then
    skip_fountain_icf_splash_retr_final = 0
  end
  return skip_fountain_top_subdivision_nn_retrieval_tag_splash, fountain_retrieval_skip_top_subdivision_nn_retrieval,
      skip_fountain_reco_emb_hetu_retrieval_splash,
      fountain_swing_retr_redis_key,
      fountain_skip_reco_emb_u2i_retr_splash, fountain_skip_gcse_u2i_retrieval_splash, source_hetu_level_one_v2, source_hetu_level_two_v2, source_hetu_level_three_v2, source_hetu_level_four_v2,
      source_hetu_tag_v2, source_hetu_face_id_v2, source_hetu_cluster_id_v2,
      source_movie_related_ips_key, source_movie_ip_extends_key, fountain_skip_ip2tag2ip_retr_splash, fountain_enable_ip2tag2ip_retr_opt, fountain_relation_interaction_retr_redis_key,
      skip_fountain_icf_splash_retr_final
end