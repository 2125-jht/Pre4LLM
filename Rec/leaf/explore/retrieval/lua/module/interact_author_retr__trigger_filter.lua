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
  local hate_aid_set = {}
  if hateAids ~= nil and #hateAids > 0 then
    for i, v in ipairs(hateAids) do
      hate_aid_set[v] = true
    end
  end
  fill_result(downloadAids, hate_aid_set, result, result_set)
  fill_result(searchClickAids, hate_aid_set, result, result_set)
  fill_result(dupClickAids, hate_aid_set, result, result_set)
  fill_result(longViewAids, hate_aid_set, result, result_set)
  fill_result(profileEnterAids, hate_aid_set, result, result_set)
  fill_result(likeAids, hate_aid_set, result, result_set)
  fill_result(forwardAids, hate_aid_set, result, result_set)
  fill_result(commentAids, hate_aid_set, result, result_set)
  return result
end