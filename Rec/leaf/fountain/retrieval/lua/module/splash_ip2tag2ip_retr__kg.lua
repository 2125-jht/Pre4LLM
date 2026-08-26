function gen_kg_tag_key()
  local kg_tag_key = {}
  local kg_tag_first_priority = {}
  local kg_tag_second_priority = {}
  local kg_tag_third_priority = {}
  local kg_tag_fourth_priority = {}
  local origin_list = kg_tag or {}
  local kg_tag = ""
  local kg_tag_type = ""
  local kg_tag2ip_prefix = "kgtag:tag2ip:"
  if #origin_list <= 0 then
    return kg_tag_key
  else
    if fountain_enable_ip2tag2ip_retr_opt == 1 then 
      kg_tag2ip_prefix = "kgtag:tag2ipv2:"
    end
    for i, v in ipairs(origin_list) do
      j = 1
      i = 0
      if_break = 0
      while true do
        i = string.find(v, ",", j)
        if i == nil then
          kg_tag_info = string.sub(v, j, -1)
          if_break = 1
        else
          kg_tag_info = string.sub(v, j, i-1)
        end
        type_pos = string.find(kg_tag_info, "-", 1)
        if type_pos == nil then
          kg_tag_type = ""
        else
          kg_tag_type = string.sub(kg_tag_info, type_pos+1, -1)
        end
        if kg_tag_type == "系列大IP标签" then
          table.insert(kg_tag_first_priority, kg_tag2ip_prefix .. kg_tag_info)
        elseif kg_tag_type == "特色题材标签" or kg_tag_type == "特色风格标签" or kg_tag_type == "时代背景标签" or kg_tag_type == "人物标签" then
          if fountain_ip2tag2ip_retr_kg_tag_priority_level >= 2 then
            table.insert(kg_tag_second_priority, kg_tag2ip_prefix .. kg_tag_info)
          end
        elseif kg_tag_type == "形式来源标签" then
          if fountain_ip2tag2ip_retr_kg_tag_priority_level >= 3 then
            table.insert(kg_tag_third_priority, kg_tag2ip_prefix .. kg_tag_info)
          end
        else
          if fountain_ip2tag2ip_retr_kg_tag_priority_level >= 4 then
            table.insert(kg_tag_fourth_priority, kg_tag2ip_prefix .. kg_tag_info)
          end
        end
        if if_break > 0 then
          break
        end
        j = i + 1
      end
    end
    for i, v in ipairs(kg_tag_first_priority) do
      table.insert(kg_tag_key, v)
    end
    for i, v in ipairs(kg_tag_second_priority) do
      table.insert(kg_tag_key, v)
    end
    for i, v in ipairs(kg_tag_third_priority) do
      table.insert(kg_tag_key, v)
    end
    for i, v in ipairs(kg_tag_fourth_priority) do
      table.insert(kg_tag_key, v)
    end
    return kg_tag_key
  end
end

function fill_ip_list_with_order(result, origin_list, total_cnt)
  local all_table = {}
  local max_len = 0
  if #origin_list > 0 and total_cnt > 0 then
    for i, v in ipairs(origin_list) do
      tmp_table = {}
      for tmp in string.gmatch(v, "%w+") do
        table.insert(tmp_table, tonumber(tmp))
      end
      if #tmp_table > max_len then 
        max_len = #tmp_table
      end
      table.insert(all_table, tmp_table)
    end
    ind = 1
    cnt = 0
    while cnt < total_cnt and ind <= max_len do
      for i, v in ipairs(all_table) do
        if cnt >= total_cnt then 
          break
        end
        if ind <= #v then
          table.insert(result, v[ind])
          cnt = cnt + 1
        end
      end
      ind = ind + 1
    end
  end
end

function parse_kg_ips()
  local kg_extend_ips = {}
  local kg_ip2ip_extend_ips = {}
  local origin_list = kg_extend_ips_str or {}
  local origin_list2 = kg_ip2ip_extend_ip_str or {}
  local origin_key_list = kg_tag_key or {}
  local origin_key_list2 = source_movie_ip_extends_key or {}
  local series_lists = {}
  local theme_lists = {}
  local style_lists = {}
  local person_lists = {}
  local background_lists = {}
  if fountain_enable_ip2tag2ip_retr_opt == 0 then
    if #origin_list > 0 then
      for i, v in ipairs(origin_list) do
        for tmp in string.gmatch(v, "%w+") do
          table.insert(kg_extend_ips, tonumber(tmp))
        end
      end
    end
    if #origin_list2 > 0 then
      for i, v in ipairs(origin_list2) do 
        for tmp in string.gmatch(v, "%w+") do
          table.insert(kg_ip2ip_extend_ips, tonumber(tmp))
        end
      end
    end
  else
    for i, v in ipairs(origin_list2) do 
      if i <= #origin_key_list2 and string.find(origin_key_list2[i], "kgtag:movie2movie:series:") ~= nil then
        cnt = 0
        for tmp in string.gmatch(v, "%w+") do 
          if cnt >= 10 then 
            break
          end
          table.insert(kg_extend_ips, tonumber(tmp))
          cnt = cnt + 1
        end
      else
        for tmp in string.gmatch(v, "%w+") do
          table.insert(kg_ip2ip_extend_ips, tonumber(tmp))
        end
      end
    end
    for i, v in ipairs(origin_list) do
      if i <= #origin_key_list and string.find(origin_key_list[i], "系列大IP标签") then 
        table.insert(series_lists, v)
      elseif i <= #origin_key_list and string.find(origin_key_list[i], "特色题材标签") then
        table.insert(theme_lists, v)
      elseif i <= #origin_key_list and string.find(origin_key_list[i], "特色风格标签") then
        table.insert(style_lists, v)
      elseif i <= #origin_key_list and string.find(origin_key_list[i], "人物标签") then
        table.insert(person_lists, v)
      elseif i <= #origin_key_list and string.find(origin_key_list[i], "时代背景标签") then
        table.insert(background_lists, v)
      end
    end
    fill_ip_list_with_order(kg_extend_ips, series_lists, 5)
    fill_ip_list_with_order(kg_extend_ips, theme_lists, 30)
    fill_ip_list_with_order(kg_extend_ips, style_lists, 30)
    fill_ip_list_with_order(kg_extend_ips, person_lists, 15)
    fill_ip_list_with_order(kg_extend_ips, background_lists, 10)
  end
  return kg_extend_ips, kg_ip2ip_extend_ips
end