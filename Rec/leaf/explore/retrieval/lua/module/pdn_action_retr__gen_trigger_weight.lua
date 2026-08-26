function generate_raw_trigger()
  local trigger_index_list = {}
  local trigger_list = {}
  local trigger_weight_dict = {}
  local action_weights = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
  if action_weight_list ~= nil and #action_weight_list == 7 then
    action_weights = action_weight_list
  end
  fill_trigger_info(like_list, trigger_list, trigger_index_list, trigger_weight_dict, action_weights[1])
  fill_trigger_info(follow_list, trigger_list, trigger_index_list, trigger_weight_dict, action_weights[2])
  fill_trigger_info(forward_list, trigger_list, trigger_index_list, trigger_weight_dict, action_weights[3])
  fill_trigger_info(comment_list, trigger_list, trigger_index_list, trigger_weight_dict, action_weights[4])
  fill_trigger_info(collect_list, trigger_list, trigger_index_list, trigger_weight_dict, action_weights[5])
  fill_trigger_info(download_list, trigger_list, trigger_index_list, trigger_weight_dict, action_weights[6])
  fill_trigger_info(search_click_list, trigger_list, trigger_index_list, trigger_weight_dict, action_weights[7])
  local trigger_weight_list = {}
  for i, pid in ipairs(trigger_list) do
    table.insert(trigger_weight_list, trigger_weight_dict[pid])
  end
  return trigger_index_list, trigger_list, trigger_weight_list
end


function fill_trigger_info(cand_list, trigger_list, trigger_index_list, trigger_weight_dict, action_weight)
  if cand_list == nil or #cand_list == 0 then
    return
  end
  local cnt = #trigger_list
  for i, pid in ipairs(cand_list) do
    table.insert(trigger_list, pid)
    cnt = cnt + 1
    table.insert(trigger_index_list, cnt)
    if trigger_weight_dict[pid] == nil then
      trigger_weight_dict[pid] = action_weight
    else
      trigger_weight_dict[pid] = trigger_weight_dict[pid] + action_weight
    end
  end
end

function generate_final_trigger()
  if trigger_list == nil or trigger_num >= #trigger_list then
    return trigger_list, trigger_weight_list
  end
  local final_trigger_list = {}
  local final_trigger_weight_list = {}
  for i = 1, trigger_num do
    table.insert(final_trigger_list, trigger_list[trigger_index_list[i]])
    table.insert(final_trigger_weight_list, trigger_weight_list[trigger_index_list[i]])
  end
  return final_trigger_list, final_trigger_weight_list
end