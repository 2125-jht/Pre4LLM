function calc_duration_boost_factor()
  local duration_max = fountain_duration_boost_duration_max and fountain_duration_boost_duration_max or 300
  local duration_factor = fountain_duration_boost_factor and fountain_duration_boost_factor or 0.0
  local duration = (duration_ms and duration_ms or 0.0) / 1000.0
  duration = math.min(duration, duration_max) / duration_max
  local score = fullrank_ensemble_score and fullrank_ensemble_score or 0.0
  duration_factor = 1.0 + duration * duration_factor
  score = score * duration_factor
  return score
end