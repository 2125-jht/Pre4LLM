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