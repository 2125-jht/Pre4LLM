function fill_result(attr, attr_ts, filter_set, result, result_set, ts_ths) 
  if attr ~= nil and #attr > 0 then
    for i, v in ipairs(attr) do
      if not filter_set[v] and not result_set[v] and (ts_ths <= 0 or attr_ts[i] ~= nil and attr_ts[i] > ts_ths) then
        table.insert(result, v)
        result_set[v] = true
      end
    end
  end
end

function calculate()
  local result = {}
  local result_set = {}
  local friend_aid_set = {}
  if (skip_filter_bifollow == nil or skip_filter_bifollow == 0) and friendAids ~= nil and #friendAids > 0 then
    for i, v in ipairs(friendAids) do
      friend_aid_set[v] = true
    end
  end
  local ts_ths = 0
  if max_history_s ~= nil and max_history_s > 0 then
    ts_ths = util.GetTimestamp() / 1000 - max_history_s * 1000
  end
  fill_result(followAids, followTimestamps, friend_aid_set, result, result_set, ts_ths)
  return result
end

function generate_valid_top_author()
  local author_list = {}
  local weight_list = {}
  if top_follow_author_list == nil or #top_follow_author_list == 0 or top_follow_author_weight_list == nil or #top_follow_author_weight_list ~= #top_follow_author_list then
    return author_list, weight_list
  end
  for i, aid in ipairs(top_follow_author_list) do
    if aid > 0 then
      table.insert(author_list, aid)
      table.insert(weight_list, top_follow_author_weight_list[i])
    end
  end
  return author_list, weight_list
end