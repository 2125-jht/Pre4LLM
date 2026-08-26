function add_knowledge_trigger()
  if trigger_list == nil or #trigger_list == 0 or knowledge_trigger_max_num == 0 then
    return
  end
  local result = {}
  local result_set = {}
  for i, pid in ipairs(trigger_list) do
    table.insert(result, pid)
    result_set[pid] = true
  end
  local cnt = 0
  for i, pid in ipairs(colossus_user_info__knowledge_trigger_set) do
    if cnt >= knowledge_trigger_max_num then
      break
    end
    if not result_set[pid] then
      table.insert(result, pid)
    end
    cnt = cnt + 1
  end
  return result
end