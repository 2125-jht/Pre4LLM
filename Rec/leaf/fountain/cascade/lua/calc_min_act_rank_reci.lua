function calc_min_act_rank_reci()
  local min_act_rank = 500
  local act_weights_tb = fountain_mc_min_act_rank_weights and fountain_mc_min_act_rank_weights or {}
  local act_rank_tb = {cascade_pltr_rank, cascade_pwtr_rank, cascade_pftr_rank, cascade_ptr_rank, 
                        cascade_pcestr_rank, cascade_pcmtr_rank, cascade_pepstr_rank, cascade_pcltr_rank}
  if (#act_weights_tb == #act_rank_tb) then
      for i = 1, #act_rank_tb do
          if (act_weights_tb[i] ~= nil and act_weights_tb[i] > 0) then
              min_act_rank = math.min(min_act_rank, act_rank_tb[i])
          end
      end
  end
  min_act_rank = 1.0 / (min_act_rank + 10.0)
  return min_act_rank
end