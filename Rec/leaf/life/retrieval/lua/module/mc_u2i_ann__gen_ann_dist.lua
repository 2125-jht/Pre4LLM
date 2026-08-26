function calculate()
  local score = 0.0
  if ann_dist_list ~= nil and #ann_dist_list > 0 then
    score = ann_dist_list[1]
  end
  return score
end