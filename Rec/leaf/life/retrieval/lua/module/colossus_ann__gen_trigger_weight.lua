function calculate()
  local trigger_weight_dict = {}
  local result_trigger_list = {}
  if trigger_list ~= nil then
    for i, v in ipairs(trigger_list) do
      trigger_weight_dict[v] = 1
      table.insert(result_trigger_list, v)
    end
    if colossus_user_info__trigger_id_list ~= nil and colossus_user_info__trigger_weight_list ~= nil then
      for i, v in ipairs(colossus_user_info__trigger_id_list) do
        if trigger_weight_dict[v] ~= nil and i <= #colossus_user_info__trigger_weight_list then
          trigger_weight_dict[v] = colossus_user_info__trigger_weight_list[i]
        end
      end
    end
  end
  if playlist ~= nil then
    for i, v in ipairs(playlist) do
      if trigger_weight_dict[v] == nil then
        trigger_weight_dict[v] = 1000000
        table.insert(result_trigger_list, v)
      end
    end
  end
  if enable_interaction_trigger > 0 and interaction_trigger_list ~= nil then
    for i, v in ipairs(interaction_trigger_list) do
      if trigger_weight_dict[v] == nil then
        trigger_weight_dict[v] = 1000000
        table.insert(result_trigger_list, v)
      end
    end
  end
  local result_trigger_weight_list = {}
  for i, v in ipairs(result_trigger_list) do
    table.insert(result_trigger_weight_list, trigger_weight_dict[v])
  end
  return result_trigger_list, result_trigger_weight_list
end

function calculate_realtime_list()
  local nums = enable_life_realtime_action_sensitive_browse_nums or 30
  local photo_id_list = colossus_photo_id_list_new or {}
  local play_time_list = colossus_play_time_list_new or {}
  local timestamp_list = colossus_timestamp_list_new or {}
  local browse_set = browse_screen__pid_list or {}
  local realtime_photo_id_list = {}
  local is_realtime_boost = 0
  
  if #photo_id_list == #play_time_list and #play_time_list == #timestamp_list then
    for i = #photo_id_list, 1, -1 do
        if play_time_list[i] > 15 and #realtime_photo_id_list < 20 then
          table.insert(realtime_photo_id_list, photo_id_list[i])
        end
    end
  end
  
  for i = #browse_set, math.max(#browse_set-nums, 1), -1 do
    for j = 1, #realtime_photo_id_list do
        if realtime_photo_id_list[j] == browse_set[i] then
          is_realtime_boost = 1
        end
    end
  end

return realtime_photo_id_list, is_realtime_boost
end