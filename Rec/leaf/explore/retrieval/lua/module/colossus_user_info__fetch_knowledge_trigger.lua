function extract_knowledge_trigger()
  local result = {}
  if knowledge_hetu_set == nil or #knowledge_hetu_set == 0 or colossus_user_info__trigger_tag_list == nil 
     or #colossus_user_info__trigger_tag_list == 0 or colossus_user_info__trigger_id_list == nil 
     or #colossus_user_info__trigger_id_list ~= #colossus_user_info__trigger_tag_list
     or colossus_user_info__trigger_weight_list == nil or #colossus_user_info__trigger_weight_list ~= #colossus_user_info__trigger_id_list then
    return result
  end
  local result_set = {}
  local hetu_set = {}
  for i, tag_id in ipairs(knowledge_hetu_set) do
    hetu_set[tag_id] = true
  end
  for i, pid in ipairs(colossus_user_info__trigger_id_list) do 
    if hetu_set[colossus_user_info__trigger_tag_list[i]] ~= nil and not result_set[pid] and colossus_user_info__trigger_weight_list[i] > knowledge_trigger_play_time_ths then
      table.insert(result, pid)
      result_set[pid] = true
    end
  end
  return result
end