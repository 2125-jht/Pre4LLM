function calculate()
  local score = 0.0
  if ann_dist_list ~= nil and #ann_dist_list > 0 then
    score = math.max(ann_dist_list)
  end
  return score
end