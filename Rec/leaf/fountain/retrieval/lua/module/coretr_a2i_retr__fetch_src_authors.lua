function add_trigger(result_list, result_set, hate_set, target_list, num_limit)
  if target_list == nil or #target_list == 0 then
    return
  end
  local cnt = 0
  for i, aid in ipairs(target_list) do
    if cnt >= num_limit then
      break
    end
    if not result_set[aid] and not hate_set[aid] then
      result_set[aid] = true
      table.insert(result_list, aid)
      cnt = cnt + 1
    end
  end
end

function fetch_src_authors()
  local result_list = {}
  local result_set = {}
  local hate_set = {}
  if hateAidList ~= nil and #hateAidList > 0 then
    for i, aid in ipairs(hateAidList) do
      hate_set[aid] = true
    end
  end
  if featureFountainProfileShortViewAidList ~= nil and #featureFountainProfileShortViewAidList then
    for i, aid in ipairs(featureFountainProfileShortViewAidList) do
      hate_set[aid] = true
    end
  end
  add_trigger(result_list, result_set, hate_set, featureFountainProfileFollowAidList, 30)
  add_trigger(result_list, result_set, hate_set, featureFountainProfileLikeAidList, 30)
  add_trigger(result_list, result_set, hate_set, featureFountainProfileLongViewAidList, 50)
  add_trigger(result_list, result_set, hate_set, featureUserProfileV1ProfileEnterAidList, 30)
  add_trigger(result_list, result_set, hate_set, featureUserProfileV1FollowAidList, 30)
  return result_list
end

function rm_interact_authors()
  if sim_author_retr_author_list == nil or #sim_author_retr_author_list == 0 then
    return sim_author_retr_author_list
  end
  local trigger_list = {}
  local interact_set = {}
  if featureFountainProfileLikeAidList ~= nil and #featureFountainProfileLikeAidList > 0 then
    for i, aid in ipairs(featureFountainProfileLikeAidList) do
      interact_set[aid] = true 
    end
  end
  if featureFountainProfileFollowAidList ~= nil and #featureFountainProfileFollowAidList > 0 then
    for i, aid in ipairs(featureFountainProfileFollowAidList) do
      interact_set[aid] = true 
    end
  end
  if featureFountainProfileCommentAidList ~= nil and #featureFountainProfileCommentAidList > 0 then
    for i, aid in ipairs(featureFountainProfileCommentAidList) do
      interact_set[aid] = true 
    end
  end 
  for i, aid in ipairs(sim_author_retr_author_list) do
    if not interact_set[aid] then
      table.insert(trigger_list, aid)
    end
  end
  return trigger_list
end