function calculate()
  local mc_emb_dim = 128
  if trigger_list == nil or colossus_trigger_embedding == nil or trigger_weight_list == nil or #trigger_list == 0 or #colossus_trigger_embedding ~= (#trigger_list * mc_emb_dim) or #trigger_list ~= #trigger_weight_list then
    return trigger_list, trigger_weight_list, colossus_trigger_embedding
  end
  local result_trigger_list = {}
  local result_trigger_weight_list = {}
  local result_trigger_embedding = {}
  for i, v in ipairs(trigger_list) do
    local non_zero = false
    local begin_ind = (i - 1) * mc_emb_dim + 1
    local end_ind = i * mc_emb_dim
    for j = begin_ind, end_ind do 
      if math.abs(colossus_trigger_embedding[j]) >= 1e-6 then
        non_zero = true
        break
      end
    end
    if non_zero then
      table.insert(result_trigger_list, v)
      table.insert(result_trigger_weight_list, trigger_weight_list[i])
      for j = begin_ind, end_ind do
        table.insert(result_trigger_embedding, colossus_trigger_embedding[j])
      end
    end
  end
  return result_trigger_list, result_trigger_weight_list, result_trigger_embedding
end