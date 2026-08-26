function calculate_ann_dist()
  if ann_dist_list == nil or #ann_dist_list == 0 then
    return ann_dist_threshold - 1
  end
  local final_dist = ann_dist_threshold - 1
  for i, dist in ipairs(ann_dist_list) do
    if dist >= ann_dist_threshold then
      final_dist = dist
      break
    end
  end
  return final_dist
end
