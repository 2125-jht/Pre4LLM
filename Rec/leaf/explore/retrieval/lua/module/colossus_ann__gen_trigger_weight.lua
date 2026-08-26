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