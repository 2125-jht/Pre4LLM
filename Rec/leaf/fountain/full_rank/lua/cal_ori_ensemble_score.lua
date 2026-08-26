function cal_ori_ensemble_score()
  local watchtime_score = fullrank_ori_ensemble_watchtime_score or 1.0
  local interact_score = fullrank_ori_ensemble_interact_score or 1.0
  local neg_score = fullrank_ori_ensemble_neg_score or 1.0
  local vv_score = fullrank_ori_ensemble_vv_score or 1.0
  local ensemble_score = vv_score ^ fullrank_ori_ensemble_vv_score_power_weight * watchtime_score ^ fullrank_ori_ensemble_watchtime_score_power_weight * interact_score ^ fullrank_ori_ensemble_interact_score_power_weight * neg_score ^ fullrank_ori_ensemble_neg_score_power_weight
  return ensemble_score
end