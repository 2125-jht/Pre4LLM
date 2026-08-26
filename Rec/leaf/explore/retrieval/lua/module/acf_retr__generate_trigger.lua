function add_triggers(candidate_list, limit_num, trigger_set)
  if candidate_list == nil or #candidate_list == 0 or limit_num <= 0 then
    return
  end
  local cnt = 0
  for i, aid in ipairs(candidate_list) do
    if cnt >= limit_num then
      break
    end
    if not trigger_set[aid] then
      trigger_set[aid] = true
      cnt = cnt + 1
    end
  end
end

function add_play_triggers(candidate_list, playtime_list, duration_list, limit_num, trigger_set)
  if limit_num <= 0 or candidate_list == nil or playtime_list == nil or duration_list == nil or #candidate_list ~= #playtime_list or #playtime_list ~= #duration_list then
    return
  end
  local cnt = 0
  for i, aid in ipairs(candidate_list) do
    if cnt >= limit_num then
      break
    end
    if not trigger_set[aid] and (playtime_list[i] >= 7000 and (playtime_list[i] > duration_list[i] or playtime_list[i] > 18000)) then
      trigger_set[aid] = true
      cnt = cnt + 1
    end
  end
end

function generate_triggers()
  local trigger_set = {}
  local source_aids = {}
  local hate_set = {}
  add_triggers(followList, follow_source_num, trigger_set)
  local follow_cnt = #trigger_set
  add_triggers(profileAidList, profile_num, trigger_set)
  add_play_triggers(videoPlayRawAids, videoPlayTime, videoDurations, click_num, trigger_set)
  add_triggers(likeAidList, like_num, trigger_set)
  add_triggers(forwardAidList, forward_num, trigger_set)
  add_triggers(downloadAidList, download_num, trigger_set)
  add_triggers(collectAidList, collect_num, trigger_set)
  add_triggers(clickAidList, recent_source_num + follow_cnt - #trigger_set, trigger_set)
  if hateAidList ~= nil and #hateAidList > 0 then
    for i, aid in ipairs(hateAidList) do
      hate_set[aid] = true
    end
  end
  for aid, v in pairs(trigger_set) do
    if not hate_set[aid] then
      table.insert(source_aids, aid)
    end
  end
  return source_aids
end