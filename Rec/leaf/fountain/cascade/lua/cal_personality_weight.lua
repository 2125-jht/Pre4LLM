function cal_cascade_adaptive_weights()
    local ratio_min = fountain_cascade_ensemble_power_weight_adjust_ratio_min or 1.0
    local ratio_max = fountain_cascade_ensemble_power_weight_adjust_ratio_max or 1.0
    local like_weight = fountain_fast_ensemble_weight_cascade_like_score * calc_xtr_weight_adaptive_ratio(
            userExpLtr, fountain_cascade_ensemble_power_weight_cascade_like_emp, ratio_min, ratio_max)
    local follow_weight = fountain_fast_ensemble_weight_cascade_follow_score * calc_xtr_weight_adaptive_ratio(
            userExpWtr, fountain_cascade_ensemble_power_weight_cascade_follow_emp, ratio_min, ratio_max)
    local comment_weight = fountain_fast_ensemble_weight_cascade_comment_score * calc_xtr_weight_adaptive_ratio(
            userExpCmtr, fountain_cascade_ensemble_power_weight_cascade_comment_emp, ratio_min, ratio_max)
    local profile_weight = fountain_fast_ensemble_weight_cascade_profile_score * calc_xtr_weight_adaptive_ratio(
            userExpPtr, fountain_cascade_ensemble_power_weight_cascade_profile_emp, ratio_min, ratio_max)
    local forward_weight = fountain_fast_ensemble_weight_cascade_forward_score * calc_xtr_weight_adaptive_ratio(
            userExpFtr, fountain_cascade_ensemble_power_weight_cascade_forward_emp, ratio_min, ratio_max)
    local eps_weight = fountain_fast_ensemble_weight_cascade_epstr_score * calc_xtr_weight_adaptive_ratio(
            userExpEptr, fountain_cascade_ensemble_power_weight_cascade_eps_emp, ratio_min, ratio_max)
    return like_weight, follow_weight, comment_weight, profile_weight, forward_weight, eps_weight
end

function cal_cascade_adaptive_watch_time_weights()
    local cluster_sort_use_personal_pwatch_time_weight = fountain_cascade_fast_cluster_sort_use_personal_pwatch_time_weight or 0
    local ensemble_sort_use_personal_pwatch_time_weight = fountain_cascade_fast_ensemble_sort_use_personal_pwatch_time_weight or 0
    local cluster_sort_use_personal_pwtd_weight = fountain_cascade_fast_cluster_sort_use_personal_pwtd_weight or 0
    local ensemble_sort_use_personal_pwtd_weight = fountain_cascade_fast_ensemble_sort_use_personal_pwtd_weight or 0
    local use_colossus_res = fountain_cascade_watch_time_reweight_use_colossus_res or 0
    local user_avg_watch_time = userAvgEffectiveWatchTime
    if (use_colossus_res > 0) then
      user_avg_watch_time = user_emp_watch_time
    end
    local adaptive_ratio_power = cascade_watch_time_weight_adaptive_ratio_power or 1.0
    local adaptive_ratio_offset = cascade_watch_time_weight_adaptive_ratio_offset or 0.0
    local pwatch_time_ratio_min = fountain_cascade_ensemble_power_weight_adjust_min_ratio_pwatch_time or 1.0
    local pwatch_time_ratio_max = fountain_cascade_ensemble_power_weight_adjust_max_ratio_pwatch_time or 1.0
    local cluster_sort_weight_pwatch_time = fountain_variant_cluster_sort_weight_cascade_pwatch_time
    local ensemble_sort_weight_pwatch_time = fountain_fast_ensemble_power_weight_cascade_pwatch_time
    local cluster_sort_weight_pwtd = fountain_variant_cluster_sort_weight_cascade_pwtd
    local ensemble_sort_weight_pwtd = fountain_fast_ensemble_weight_cascade_pwtd
    if cluster_sort_use_personal_pwatch_time_weight > 0 or ensemble_sort_use_personal_pwatch_time_weight > 0 or cluster_sort_use_personal_pwtd_weight > 0 or ensemble_sort_use_personal_pwtd_weight > 0 then
        local pwatch_time_ratio = calc_watch_time_weight_adaptive_ratio(user_avg_watch_time, fountain_cascade_ensemble_power_weight_cascade_watch_time_emp, pwatch_time_ratio_min, pwatch_time_ratio_max, adaptive_ratio_power, adaptive_ratio_offset)
        if cluster_sort_use_personal_pwatch_time_weight > 0 then
            cluster_sort_weight_pwatch_time = cluster_sort_weight_pwatch_time * pwatch_time_ratio
        end
        if ensemble_sort_use_personal_pwatch_time_weight > 0 then
          ensemble_sort_weight_pwatch_time = ensemble_sort_weight_pwatch_time * pwatch_time_ratio
        end
        if cluster_sort_use_personal_pwtd_weight > 0 then
          cluster_sort_weight_pwtd = cluster_sort_weight_pwtd * pwatch_time_ratio
        end
        if ensemble_sort_use_personal_pwtd_weight > 0 then
          ensemble_sort_weight_pwtd = ensemble_sort_weight_pwtd * pwatch_time_ratio
        end
    end
    return cluster_sort_weight_pwatch_time, ensemble_sort_weight_pwatch_time, cluster_sort_weight_pwtd, ensemble_sort_weight_pwtd
end

function cal_cascade_adaptive_splash_weights()
    local ratio_min = fountain_cascade_ensemble_power_weight_adjust_ratio_min or 1.0
    local ratio_max = fountain_cascade_ensemble_power_weight_adjust_ratio_max or 1.0
    local like_weight = fountain_ensemble_weight_cascade_like_score * calc_xtr_weight_adaptive_ratio(
            userExpLtr, fountain_cascade_ensemble_power_weight_cascade_like_emp, ratio_min, ratio_max)
    local follow_weight = fountain_ensemble_weight_cascade_follow_score * calc_xtr_weight_adaptive_ratio(
            userExpWtr, fountain_cascade_ensemble_power_weight_cascade_follow_emp, ratio_min, ratio_max)
    local comment_weight = fountain_ensemble_weight_cascade_comment_score * calc_xtr_weight_adaptive_ratio(
            userExpCmtr, fountain_cascade_ensemble_power_weight_cascade_comment_emp, ratio_min, ratio_max)
    local profile_weight = fountain_ensemble_weight_cascade_profile_score * calc_xtr_weight_adaptive_ratio(
            userExpPtr, fountain_cascade_ensemble_power_weight_cascade_profile_emp, ratio_min, ratio_max)
    local forward_weight = fountain_ensemble_weight_cascade_forward_score * calc_xtr_weight_adaptive_ratio(
            userExpFtr, fountain_cascade_ensemble_power_weight_cascade_forward_emp, ratio_min, ratio_max)
    local eps_weight = fountain_ensemble_weight_cascade_epstr_score * calc_xtr_weight_adaptive_ratio(
            userExpEptr, fountain_cascade_ensemble_power_weight_cascade_eps_emp, ratio_min, ratio_max)
    return like_weight, follow_weight, comment_weight, profile_weight, forward_weight, eps_weight
end

-- 返回 ratio
function calc_xtr_weight_adaptive_ratio(user_xtr, emp_xtr, min_ratio, max_ratio)
    if emp_xtr ~= nil and emp_xtr > 0.0 then
      local user_emp_xtr = user_xtr or emp_xtr
      return math.min(math.max(user_emp_xtr / emp_xtr, min_ratio), max_ratio)
    else
      return 1.0
    end
end

function calc_watch_time_weight_adaptive_ratio(user_xtr, emp_xtr, min_ratio, max_ratio, ratio_power, ratio_offset)
  local pwatch_time_ratio = calc_xtr_weight_adaptive_ratio(user_xtr, emp_xtr, min_ratio, max_ratio)
  -- 进一步调控ratio, eg : 对低时长消费用户, 放大时长权重
  pwatch_time_ratio = pwatch_time_ratio ^ ratio_power + ratio_offset
  return pwatch_time_ratio
end