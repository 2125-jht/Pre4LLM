function fill_result(attr, filter_set, result, result_set) 
  if attr ~= nil and #attr > 0 then
    for i, v in ipairs(attr) do
      if not filter_set[v] and not result_set[v] then
        table.insert(result, v)
        result_set[v] = true
      end
    end
  end
end

function calculate()
  local result = {}
  local result_set = {}
  local final_result = {}
  local hate_aid_set = {}
  if hateAids ~= nil and #hateAids > 0 then
    for i, v in ipairs(hateAids) do
      hate_aid_set[v] = true
    end
  end
  fill_result(history_author_triggers, hate_aid_set, result, result_set)
  for i, v in ipairs(result) do
    if i > author_max_num then
      break
    end
    table.insert(final_result, v)
  end
  return final_result
end