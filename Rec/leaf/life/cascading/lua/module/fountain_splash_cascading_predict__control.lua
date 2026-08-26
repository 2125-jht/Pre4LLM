-- 粗排参数控制
function split(input, delimiter)
  local t={}
  for str in string.gmatch(input, "([^"..delimiter.."]+)") do
      table.insert(t, str)
  end
  return t
end

function parse_increase_factor(factor_list, index, expect_len)
  local factor = 1.0
  if (index <= 0 or index > expect_len) then
    return factor
  end
  if (factor_list == nil or factor_list == "" or string.len(factor_list) == 0) then
    return factor
  end
  local factors = split(factor_list, ",")
  if (expect_len == #factors) then
    factor = factors[index]
  end
  return factor
end

function cascade_control_fast()

  local enableFountainFullrankExp = enableFountainFullrankExp and enableFountainFullrankExp or 0
  local increase_quota_factor_list = fountain_mc_increase_quota_factor_list and fountain_mc_increase_quota_factor_list or ""
  local increase_quota_current_index = increase_quota_current_index and increase_quota_current_index or 0
  local increase_quota_window_len = increase_quota_window_len and increase_quota_window_len or 0
  local increase_quota_factor = parse_increase_factor(increase_quota_factor_list, increase_quota_current_index, increase_quota_window_len)
  local increase_quota_status = increase_quota_status and increase_quota_status or 0
  if (increase_quota_status > 0) then
    fullrank_fast_before_variant_mc_limit_size = math.floor(fullrank_fast_before_variant_mc_limit_size * increase_quota_factor)
  end
  return fullrank_fast_before_variant_mc_limit_size
end


function cascade_feature_trans_splash()
  local long_term_interest_ee_score_default = 1.0
  return long_term_interest_ee_score_default
end

function getTagIdFromHetuTag(hetuTag)
  local tagId = ((hetuTag >> 8) & 0xffffff)
  return tagId
end

function cascade_hetu_list_size()
  local hetu_level_one_v2_index_cascade_list = _G['hetu_level_one_v2_index_cascade_list'] and _G['hetu_level_one_v2_index_cascade_list'] or {}
  return  #hetu_level_one_v2_index_cascade_list
  end

function cascade_explore_list_size()
  local similar_user_colossus_hetu_list = _G['similar_user_colossus_hetu_list'] and _G['similar_user_colossus_hetu_list'] or {}
  local explore_hetu_list = _G['explore_hetu_list'] and _G['explore_hetu_list'] or {}
  local input_explore_interest_hetu_list = _G['input_explore_interest_hetu_list'] and _G['input_explore_interest_hetu_list'] or {}
  return  #similar_user_colossus_hetu_list, #explore_hetu_list , #input_explore_interest_hetu_list
  end

function cascade_feature_trans()
  local hetu_level_one_v2_original = _G['hetu_tag_level_info_v2__hetu_level_one'] and _G['hetu_tag_level_info_v2__hetu_level_one'] or {}
  local hetu_level_one_v2 = {}
  local index = 1
  for i = 1, #hetu_level_one_v2_original do
    local tagId = getTagIdFromHetuTag(hetu_level_one_v2_original[i])
    table.insert(hetu_level_one_v2, index, tagId)
    index = index + 1
  end
  local hetu_level_one_v2_index = (#hetu_level_one_v2 > 0) and hetu_level_one_v2[1] or -1
  local hetu_level_one_list = _G["hetu_tag_level_info__hetu_level_one"] and _G["hetu_tag_level_info__hetu_level_one"] or {}
  local hetu_level_one_index = (#hetu_level_one_list > 0) and hetu_level_one_list[1] or -1
  local duration_s = math.min(math.floor(duration_ms/1000 or 0),300)

  local hetu_level_one = (#hetu_level_one_list > 0) and hetu_level_one_list[1] or nil
  local hetu_level_two_list = _G["hetu_tag_level_info__hetu_level_two"] and _G["hetu_tag_level_info__hetu_level_two"] or {}
  local hetu_level_two = (#hetu_level_two_list > 0) and hetu_level_two_list[1] or nil

  local show_count = _G['explore_stat__show_count'] or 0
  local negative_count = _G['explore_stat__negative_count'] or 0
  local empirical_htr = (negative_count / (show_count + 100))

  local duration_perf_id = calc_perf_duration_id(duration_ms)

  return hetu_level_one_index,hetu_level_one_v2_index,duration_s,hetu_level_one, hetu_level_two,empirical_htr,duration_perf_id
end


function cascade_control_splash()
  local enableFountainFullrankExp = enableFountainFullrankExp and enableFountainFullrankExp or 0
  local fountain_splash_mc_increase_quota_factor = fountain_splash_mc_increase_quota_factor and fountain_splash_mc_increase_quota_factor or 1.0
  local increase_quota_status = increase_quota_status and increase_quota_status or 0
  if (increase_quota_status > 0) then
    fountain_pre_fullrank_size_limit_v2_splash = math.floor(fountain_pre_fullrank_size_limit_v2_splash * fountain_splash_mc_increase_quota_factor)
  end
  return fountain_pre_fullrank_size_limit_v2_splash
end

function cascade_control_model()
  local common_request_type = common_request_type or ""
  if (common_request_type == "fountain_fast_v1_life" or common_request_type == "fountain_fast_life_pic_inside") then
    fountain_casade_is_fast = 1
  else
    fountain_casade_is_fast = 0
  end
  return fountain_casade_is_fast
end

function cascade_score_for_fullrank_splash()
  local cascade_ensemble_score = cascade_ensemble_score and cascade_ensemble_score or 0
  return cascade_ensemble_score
end

function cascade_score_for_fullrank()
  local cascade_mc_score = cascade_mc_score and cascade_mc_score or 0
  return cascade_mc_score
end

function  cascade_ensemble_score_discount_calc()
  local cascade_ensemble_score = cascade_ensemble_score and cascade_ensemble_score or 0.0
  local cascade_discount_ratio = cascade_discount_ratio and cascade_discount_ratio or 0
  local cascade_neg_feedback_discount_score = cascade_ensemble_score
  if (cascade_discount_ratio > 0) then
    cascade_neg_feedback_discount_score = cascade_neg_feedback_discount_score * cascade_discount_ratio
  end

  return cascade_neg_feedback_discount_score
end

function calc_cascade_cluster_type()
  local cascade_cluster_id = cascade_cluster_id or 0
  local cascade_cluster_type = cascade_cluster_id - cascade_cluster_id%10000
  return cascade_cluster_type
end

function calc_perf_duration_id(duration_ms)
  local duration_sec = duration_ms/1000 or 0
  if (duration_sec >= 0 and duration_sec < 7.0) then
    cluster_id = 1
  elseif (duration_sec >= 7.0 and  duration_sec < 9.0) then
    cluster_id = 2
  elseif (duration_sec >= 9.0 and duration_sec < 12.0) then
    cluster_id = 3
  elseif (duration_sec >= 12.0 and duration_sec < 17.0) then
    cluster_id = 4
  elseif (duration_sec >= 17.0 and duration_sec < 20.0) then
    cluster_id = 5
  elseif (duration_sec >= 20.0 and duration_sec < 30.0) then
    cluster_id = 6
  elseif (duration_sec >= 30.0 and duration_sec < 40.0) then
    cluster_id = 7
  elseif (duration_sec >= 40.0 and duration_sec < 50.0) then
    cluster_id = 8
  elseif (duration_sec >= 50.0 and duration_sec < 58.0) then
    cluster_id = 9
  elseif (duration_sec >= 58.0 and duration_sec < 120.0) then
    cluster_id = 10
  else
    cluster_id = 11
  end
  return cluster_id
end

function emp_xtr_change()
  return user_emp_ltr, user_emp_wtr, user_emp_cmtr, user_emp_ftr, user_emp_eptr
end
