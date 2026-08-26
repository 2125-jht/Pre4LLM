function fullrank_service_control()
  local enableFountainFullrankExp = enableFountainFullrankExp and enableFountainFullrankExp or 0
  local fullrank_service = fountain_fullrank_sim_predict_kess_service and fountain_fullrank_sim_predict_kess_service or ""
  local fullrank_request_type = fountain_fullrank_sim_predict_request_type and fountain_fullrank_sim_predict_request_type or ""
  local local fullrank_service = fountain_fullrank_sim_predict_kess_service and fountain_fullrank_sim_predict_kess_service or ""
  if (enableFountainFullrankExp > 0) then
    fullrank_service = fountain_fullrank_sim_predict_kess_service_exp
    fullrank_request_type = fountain_fullrank_sim_predict_request_type_exp
  end
  return fullrank_service, fullrank_request_type
end