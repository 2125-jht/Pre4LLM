-- split mc_pxtr_value
-- function split(seq, item_key, reason, score)
--   if seq < #mc_pxtr_value then
--     return mc_pxtr_value[seq+1]
--   else 
--     return 0.0
--   end
-- end
function split(seq, item_key, reason, score)
  local fc_mc_label_num = 3
  if mc_pxtr_value == nil then
    return cascade_pctr, cascade_plvtr, cascade_psvtr
  elseif (seq + 1) * fc_mc_label_num <= #mc_pxtr_value then
    return mc_pxtr_value[seq * fc_mc_label_num + 1], mc_pxtr_value[seq * fc_mc_label_num + 2], mc_pxtr_value[seq * fc_mc_label_num + 3]
  else 
    return 0.0, 0.0, 0.0
  end
end
