function calc_icf_enlarge_trigger()
    local result = commonRetrievalPhotos or {}
    if sourceSearchIcfResult == nil or fountain_icf_splash_enlarge_trigger_num <= 0 then
      return result
    end
    j = 1
    cnt = 0
    if_break = 0
    while true do
      i = string.find(sourceSearchIcfResult, ",", j)
      if i == nil then
        photo_attr = string.sub(sourceSearchIcfResult, j, -1)
        if_break = 1
      else
        photo_attr = string.sub(sourceSearchIcfResult, j, i-1)
        j = i + 1
      end
      id_pos = string.find(photo_attr, ":", 1)
      if id_pos ~= nil then
        table.insert(result, tonumber(string.sub(photo_attr, 1, id_pos-1)))
        cnt = cnt + 1
      end
      if cnt >= fountain_icf_splash_enlarge_trigger_num or if_break > 0 then
        break
      end
    end
    return result
  end

function shuffle(t, topk)
  if type(t)~="table" then
    return
  end
  local tab={}
  local index=1
  math.randomseed(os.time())
  while #t~=0 and #tab < topk do
    local n=math.random(1,#t)
    if t[n]~=nil then
      tab[index]=t[n]
      table.remove(t,n)
      index=index+1
    end
  end
  return tab
end

function shuffle_trigger_list()
  local result = featureFountainProfileEffViewPidList or {}
  if #result <= 0 then
    return result
  end
  result = shuffle(featureFountainProfileEffViewPidList, #featureFountainProfileEffViewPidList)
  return result
end

function filter(t, hate_list)
  if type(t)~="table" then
    return {}
  end
  if type(hate_list)~="table" then
    return t
  end
  if #t <= 0 or #hate_list <= 0 then
    return t
  end
  local res={}
  local set = {}
  for k,v in ipairs(hate_list) do
	  set[v] = true
  end
  local index = 1
  for k,v in ipairs(t) do  
	  if set[v] == nil then
	    res[index] = v
	    index = index + 1
	  end
  end
  return res
end

function filter_trigger_list()
  local result1 = featureFountainProfileEffViewPidList or {}
  local result2 = colossusRetrievalTrigger or {}
  local result3 = featureFountainProfileLongViewPidList or {}
  local hate_list = featureUserHateList or {}
  if skip_fountain_eff_view_filter == 0 then
    result1 = filter(result1, hate_list)
  end
  if skip_fountain_colossus_filter == 0 then
    result2 = filter(result2, hate_list)
  end
  if skip_fountain_longview_filter == 0 then
    result3 = filter(result3, hate_list)
  end
  return result1, result2, result3
end

function filter_trigger_list_splash()
  local result1 = featureFountainProfileEffViewPidList or {}
  local result2 = colossusRetrievalTrigger or {}
  local hate_list = featureUserHateList or {}
  if skip_fountain_eff_view_filter_splash == 0 then
    result1 = filter(result1, hate_list)
  end
  if skip_fountain_colossus_filter_splash == 0 then
    result2 = filter(result2, hate_list)
  end
  return result1, result2
end

function merge_table(table_a, table_b)
  local res = table_a or {}
  local start = #res
  table_b = table_b or {}
  index = 1
  for i = 1, #table_b do
    if table_b[i] > 0 then
      res[start + index] = table_b[i]
      index = index + 1
    end
  end
  return res
end

function shuffle_interaction_list()
  local res = featureUserProfileV1LikePidList or {}
  res = merge_table(res, featureUserProfileV1FollowPidList)
  res = merge_table(res, featureUserProfileV1ForwardPidList)
  res = merge_table(res, featureUserProfileV1CommentPidList)
  res = merge_table(res, featureUserProfileV1ProfileEnterPidListLite)
  if add_fountain_profile_for_interaction_list > 0 then
    res = merge_table(res, fountainActionTriggers)
  end
  res = shuffle(res, #res)
  return res
end

