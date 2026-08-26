function add_trigger(cand_list, trigger_list, trigger_set, limit)
  local cnt = 0
  if cand_list ~= nil and #cand_list > 0 then
    for i, pid in ipairs(cand_list) do
      if cnt >= limit then
        break
      end
      if not trigger_set[pid] then
        trigger_set[pid] = true
        table.insert(trigger_list, pid)
        cnt = cnt + 1
      end
    end
  end
end

function gen_trigger()
  local trigger_list = {}
  local trigger_set = {}
  add_trigger(click_list, trigger_list, trigger_set, click_limit)
  add_trigger(like_list, trigger_list, trigger_set, like_limit)
  add_trigger(follow_list, trigger_list, trigger_set, follow_limit)
  add_trigger(forward_list, trigger_list, trigger_set, forward_limit)
  add_trigger(profile_enter_list, trigger_list, trigger_set, profile_enter_limit)
  add_trigger(download_list, trigger_list, trigger_set, download_limit)
  add_trigger(collect_list, trigger_list, trigger_set, collect_limit)
  if trigger_limit > #trigger_list then
    add_trigger(click_list, trigger_list, trigger_set, trigger_limit - #trigger_list)
  end
  local hate_set = {}
  if hate_list ~= nil and #hate_list > 0 then
    for i, pid in ipairs(hate_list) do
      hate_set[pid] = true
    end
  end
  local final_trigger_list = {}
  for i, pid in ipairs(trigger_list) do
    if not hate_set[pid] then
      table.insert(final_trigger_list, pid)
    end
  end
  return final_trigger_list
end

function filter_trigger_list()
  local result3 = featureFountainProfileLongViewPidList or {}
  local hate_list = featureUserHateList or {}
  result3 = filter(result3, hate_list)
  return result3
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