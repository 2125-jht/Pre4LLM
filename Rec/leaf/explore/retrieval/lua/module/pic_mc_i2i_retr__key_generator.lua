function calculate()
  local result = {}
  local result_map = {}
  local clk_list_length = (realtimeClickList and #realtimeClickList) or 0
  local play_list_lenth = (videoPlayingPid and #videoPlayingPid) or 0 
  local trigger_num = trigger_num or 0
  local num_click = clk_list_length > trigger_num and trigger_num or clk_list_length
  local num_playing = play_list_lenth > trigger_num  and trigger_num or play_list_lenth
  if enable_nouse_clicklist_trigger ~= 0 then
    for i = 1, num_playing do
      local stat = videoPlayingPid[i]
      table.insert(result, stat);
      result_map[stat] = videoPlayingDuration[i];
    end
  else
    for i = 1 ,num_click do
      table.insert(result, realtimeClickList[i]);
      if enable_duration_more_trigger ~= 0 then
        result = {};
        for  i = 1, num_playing do
          table.insert(result, videoPlayingPid[i]);
        end
      end
    end
  end
  if enable_search_more_trigger ~= 0 then
    local last_num = trigger_num - #result
    if last_num > 0  and searchList ~= nil and #searchList>=last_num then
      for i=1, last_num do
        table.insert(result, searchList[i])
      end 
    end
  end
  return result,result_map
end