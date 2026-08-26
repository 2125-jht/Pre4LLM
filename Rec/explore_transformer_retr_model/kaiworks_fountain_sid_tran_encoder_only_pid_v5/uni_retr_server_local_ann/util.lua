function extract_pids()
  local uLongViewPidsGlobal = {}
  local uLongViewHetu2Global = {}
  local uLongViewAidsGlobal = {}
  local uLongViewTimesGlobal = {}
  for i=1, #playstat_pids do
    local pid = playstat_pids[i]
    local play_time = playstat_playtimes[i]
    local duration = playstat_durations[i]
    local hetu2 = playstat_hetu2s[i]
    local aids = playstat_aids[i]
    if is_long_view(play_time, duration) and pid > 0 then
      table.insert(uLongViewPidsGlobal, pid)
      table.insert(uLongViewHetu2Global, hetu2)
      table.insert(uLongViewAidsGlobal, aids)
      table.insert(uLongViewTimesGlobal, play_time)
    end
  end
  return uLongViewPidsGlobal, uLongViewHetu2Global, uLongViewAidsGlobal, uLongViewTimesGlobal
end

function extract_ft_pids()
  local uLongViewPidsFountain = {}
  local uLongViewHetu2Fountain = {}
  local uLongViewAidsFountain = {}
  local uLongViewTimesFountain = {}
  for i=1, #user_fountain_play_id_list do
    local pid = user_fountain_play_id_list[i]
    local play_time = user_fountain_play_time_list[i]
    local duration = user_fountain_play_duration_list[i]
    local hetu2 = user_fountain_play_hetu_l2_top1_list[i]
    local aids = user_fountain_play_aid_list[i]
    if is_long_view(play_time, duration) and pid > 0 then
      table.insert(uLongViewPidsFountain, pid)
      table.insert(uLongViewHetu2Fountain, hetu2)
      table.insert(uLongViewAidsFountain, aids)
      table.insert(uLongViewTimesFountain, play_time)
    end
  end
  return uLongViewPidsFountain, uLongViewHetu2Fountain, uLongViewAidsFountain, uLongViewTimesFountain
end

function extract_eff_pids()
  local uValidViewPidsHot = {}
  local uValidViewHetu2Hot = {}
  local uValidViewAidsHot = {}
  local uValidViewTimesHot = {}
  for i=1, #playstat_pids do
    local pid = playstat_pids[i]
    local play_time = playstat_playtimes[i]
    local duration = playstat_durations[i]
    local hetu2 = playstat_hetu2s[i]
    local aids = playstat_aids[i]
    local page_id = playstat_pages[i]
    if is_eff_view(play_time, duration) and page_id == 1 and pid > 0 then
      table.insert(uValidViewPidsHot, pid)
      table.insert(uValidViewHetu2Hot, hetu2)
      table.insert(uValidViewAidsHot, aids)
      table.insert(uValidViewTimesHot, play_time)
    end
  end
  return uValidViewPidsHot, uValidViewHetu2Hot, uValidViewAidsHot, uValidViewTimesHot
end

function extract_ft_eff_pids()
  local uValidViewPidsFountain = {}
  local uValidViewHetu2Fountain = {}
  local uValidViewAidsFountain = {}
  local uValidViewTimesFountain = {}
  for i=1, #user_fountain_play_id_list do
    local pid = user_fountain_play_id_list[i]
    local play_time = user_fountain_play_time_list[i]
    local duration = user_fountain_play_duration_list[i]
    local hetu2 = user_fountain_play_hetu_l2_top1_list[i]
    local aids = user_fountain_play_aid_list[i]
    if is_eff_view(play_time, duration) and pid > 0 then
      table.insert(uValidViewPidsFountain, pid)
      table.insert(uValidViewHetu2Fountain, hetu2)
      table.insert(uValidViewAidsFountain, aids)
      table.insert(uValidViewTimesFountain, play_time)
    end
  end
  return uValidViewPidsFountain, uValidViewHetu2Fountain, uValidViewAidsFountain, uValidViewTimesFountain
end

function is_long_view(play_time, duration)
  if duration > 7000 then
    if play_time >= duration then
      return true
    end
    if duration > 36000 then
      return play_time > 36000
    end
    return play_time >= 18000
  end
    return false
end

function is_eff_view(play_time, duration)
  if duration < 1000 then
    return play_time >= 7000
  end
  if duration < 7000 then
    return play_time >= duration
  end
  return play_time >= 7000
end

function extract_hetu_feature()
  local uLongTermHetuLevel1topN = select_top_hetus(featrueUserLongTermHetu1Id, featrueUserLongTermHetu1Score, 5)
  local uLongTermHetuLevel2topN = select_top_hetus(featrueUserLongTermHetu2Id, featrueUserLongTermHetu2Score, 5)
  local uLongTermHetuLevel3topN = select_top_hetus(featrueUserLongTermHetu3Id, featrueUserLongTermHetu3Score, 5)
  return uLongTermHetuLevel1topN, uLongTermHetuLevel2topN, uLongTermHetuLevel3topN
end

function select_top_hetus(hetus, scores, k)
  local top_hetus = {}
  local hetuScore = {}
  for i, hetu in pairs(hetus) do
    local score = scores[i]
    hetuScore[hetu] = score
  end
  table.sort(hetus, function(a, b)
    return hetuScore[a] > hetuScore[b]
  end)
  for i=1, k do
    if i <= #hetus then
      table.insert(top_hetus, hetus[i])
    end
  end
  return top_hetus
end

function add_noise()
  length = #user_emb
  user_emb_with_noise
  for i=1, length do
    noise = (util.Random() - 0.5) * 0.2
    table.insert(user_emb_with_noise, user_emb[i] + noise)
  end
  return user_emb_with_noise
end