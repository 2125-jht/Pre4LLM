-- 粗排阶段 rrr discount 计算
function calc_rrr_discount()
  local report_smoth = fountain_cascade_rrr_discount_report_smooth and fountain_cascade_rrr_discount_report_smooth or 1.0
  local show_smoth = fountain_cascade_rrr_discount_show_smooth and fountain_cascade_rrr_discount_show_smooth or 1.0
  local report_count = _G['explore_stat__report_detail__total_report_count'] and _G['explore_stat__report_detail__total_report_count'] or 0
  local show_count = _G['explore_stat__real_show_count'] and _G['explore_stat__real_show_count'] or 0
  if (report_count < 1) or (show_count < 1) or (show_count < report_count) then
      return cascade_ensemble_score, 1.0
  end
  local param_n = fountain_cascade_rrr_discount_param_n and fountain_cascade_rrr_discount_param_n or 1.0
  local param_o = fountain_cascade_rrr_discount_param_o and fountain_cascade_rrr_discount_param_o or 1.0
  local rrr_factor = (report_count + report_smoth) / (show_count + show_smoth)
  local discount = 1.0 - 1.0 / math.exp((param_n - rrr_factor) / param_o)
  local calibration = 1 - 1 / (1 + math.exp(param_n / param_o));
  rrr_factor = discount / calibration
  return (cascade_ensemble_score and cascade_ensemble_score or 1.0) * rrr_factor, rrr_factor
end