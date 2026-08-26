-- 粗排阶段 search_retri_boost 计算
function calc_search_retri_boost()
  if reason ~= nil and reason == 424 then
    return (cascade_ensemble_score and cascade_ensemble_score or 0.0) * cascade_search_retri_boost_factor
  else
    return cascade_ensemble_score
  end
end