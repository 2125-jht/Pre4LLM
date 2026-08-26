function intersection(interest_hetu_tag_list, item_hetu_tag_list, hetu_tag_base_id)
    if (#interest_hetu_tag_list > 0 and #item_hetu_tag_list > 0) then
        for i, item_hetu_tag in pairs(item_hetu_tag_list) do
            for j, interest_hetu_tag in pairs(interest_hetu_tag_list) do
                if (interest_hetu_tag == item_hetu_tag) then
                    return interest_hetu_tag + hetu_tag_base_id
                end
            end
        end
    end
    return 0
  end

-- 用于计算平均完播率的时长分桶
function calc_cascade_duration_cluster_id()
  local duration_sec = math.floor(duration_ms / 1000 or 0)
  local cluster_id = 0
  if (duration_sec < 60) then
      cluster_id = duration_sec
  elseif (duration_sec < 120) then
      cluster_id = math.floor((duration_sec - 60) / 2) + 60
  elseif (duration_sec < 180) then
      cluster_id = math.floor((duration_sec - 120) / 3) + 90
  elseif (duration_sec < 300) then
      cluster_id = math.floor((duration_sec - 180) / 5) + 110
  elseif (duration_sec < 600) then
      cluster_id = math.floor((duration_sec - 300) / 10) + 134
  else
      cluster_id = 164
  end
  return "c"..cluster_id
end

-- 非首屏根据短期兴趣、互动兴趣、长期兴趣和时长, 生成分桶 ID
function calc_cascade_interest_cluster_id()
    local duration_sec = math.floor(duration_ms / 1000 or 0)
    local cluster_id = 0
    local short_interest_list = short_interest and short_interest or {}
    local action_interest_list = action_interest and action_interest or {}
    local long_interest_list = long_interest and long_interest or {}
    local short_interest_hetu_tag_base_id = 0
    local default_hetu_tag_base_id = 10000
    local action_interest_hetu_tag_base_id = 20000
    local long_interest_hetu_tag_base_id = 30000

    local item_hetu_tag_list = {}
    if (fountain_cascade_interest_use_level_one and fountain_cascade_interest_use_level_one > 0) then
        item_hetu_tag_list = _G['hetu_tag_level_info__hetu_level_one'] and _G['hetu_tag_level_info__hetu_level_one'] or {}
    else
        item_hetu_tag_list = _G['hetu_tag_level_info__hetu_level_two'] and _G['hetu_tag_level_info__hetu_level_two'] or {}
    end

    -- 首先是用户兴趣分桶
    if (#item_hetu_tag_list > 0) then
        if (#short_interest_list > 0) then
            cluster_id = intersection(short_interest_list, item_hetu_tag_list, short_interest_hetu_tag_base_id)
        end
        if (cluster_id == 0 and #action_interest_list > 0) then
            cluster_id = intersection(action_interest_list, item_hetu_tag_list, action_interest_hetu_tag_base_id)
        end
        if (cluster_id == 0 and #long_interest_list > 0) then
            cluster_id = intersection(long_interest_list, item_hetu_tag_list, long_interest_hetu_tag_base_id)
        end
    end

    -- 进入默认的分桶: 直播、图片、时长
    if (cluster_id == 0) then
        local is_living = _G['live_photo_info__is_living'] or 0
        local is_picture = is_picture or 0
        if (is_living > 0) then
            cluster_id = 0
        elseif (is_picture == 1) then
            cluster_id = 1
        elseif (duration_sec <= 3) then
            cluster_id = 2
        elseif (duration_sec <= 7) then
            cluster_id = 3
        elseif (duration_sec <= 11) then
            cluster_id = 4
        elseif (duration_sec < 18) then
            cluster_id = 5
        elseif (duration_sec < 58) then
            cluster_id = math.floor((duration_sec - 18) / 4) + 6
        elseif (duration_sec < 120) then
            cluster_id = math.floor((duration_sec - 58) / 10) + 16
        else
            cluster_id = 23
        end
        cluster_id = cluster_id + default_hetu_tag_base_id
    end
    return cluster_id
end