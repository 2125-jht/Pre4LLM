local trigger_map

function calc_trigger_map()
  trigger_map = {}
  for ind, pid in ipairs(trigger_list) do
    trigger_map[pid] = trigger_weight_list[ind]
  end
end

function calculate()
  local score = -1.0
  local src_weight = 1
  if src_id_list ~= nil and #src_id_list > 0 and src_dist_list ~= nil and #src_dist_list > 0 then
    for i, v in ipairs(src_id_list) do
      if i <= #src_dist_list and src_dist_list[i] >= ann_dist_threshold then
        if trigger_map[v] then
          src_weight = trigger_map[v]
        end
        score = src_weight * src_dist_list[i]
        break
      end
    end
  end
  return score
end