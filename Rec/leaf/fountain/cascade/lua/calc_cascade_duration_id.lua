function getTagIdFromHetuTag(hetuTag)
  local tagId = ((hetuTag >> 8) & 0xffffff)
  return tagId
end

-- 输入 source photo 的河图和候选 item 的河图, 返回命中的
-- 候选 item 的河图还未转化, 需转化对齐
function cacl_cascade_source_hetu_cluster(source_hetu_tag_list, item_hetu_tag_list_original, splash_variant_source_hetu_cluster_base_id)
  if (#source_hetu_tag_list > 0 and #item_hetu_tag_list_original > 0) then
    for i, item_tag_original in pairs(item_hetu_tag_list_original) do
        for j, source_tag in pairs(source_hetu_tag_list) do
            if (getTagIdFromHetuTag(item_tag_original) == source_tag) then
                return source_tag + splash_variant_source_hetu_cluster_base_id
            end
        end
    end
  end
  return 0
end

-- 首屏生成时长分桶 ID
function calc_cascade_duration_id()
  local duration_sec = duration_ms/1000 or 0
  local cluster_id = 0

  local skip_splash_variant_source_hetu_cluster_sort = skip_splash_variant_source_hetu_cluster_sort or 1

  -- 分桶添加 source photo 的 河图类别
  if (skip_splash_variant_source_hetu_cluster_sort == 0) then
    local splash_variant_source_hetu_cluster_base_id = splash_variant_source_hetu_cluster_base_id or 10000
    local skip_splash_variant_source_hetu_cluster_sort_level_three = skip_splash_variant_source_hetu_cluster_sort_level_three or 1
    local skip_splash_variant_source_hetu_cluster_sort_level_two = skip_splash_variant_source_hetu_cluster_sort_level_two or 1
    local skip_splash_variant_source_hetu_cluster_sort_level_one = skip_splash_variant_source_hetu_cluster_sort_level_one or 1

    -- 使用河图三级
    if (skip_splash_variant_source_hetu_cluster_sort_level_three == 0) then
      local source_hetu_level_three_v2 = source_hetu_level_three_v2 and source_hetu_level_three_v2 or {}
      local item_hetu_level_three_v2_original = _G['hetu_tag_level_info_v2__hetu_level_three'] and _G['hetu_tag_level_info_v2__hetu_level_three'] or {}
      cluster_id = cacl_cascade_source_hetu_cluster(source_hetu_level_three_v2, item_hetu_level_three_v2_original, splash_variant_source_hetu_cluster_base_id)
    end

    -- 使用河图二级
    if (cluster_id == 0 and skip_splash_variant_source_hetu_cluster_sort_level_two == 0) then
      local source_hetu_level_two_v2 = source_hetu_level_two_v2 and source_hetu_level_two_v2 or {}
      local item_hetu_level_two_v2_original = _G['hetu_tag_level_info_v2__hetu_level_two'] and _G['hetu_tag_level_info_v2__hetu_level_two'] or {}
      cluster_id = cacl_cascade_source_hetu_cluster(source_hetu_level_two_v2, item_hetu_level_two_v2_original, splash_variant_source_hetu_cluster_base_id)
    end

    -- 使用河图一级
    if (cluster_id == 0 and skip_splash_variant_source_hetu_cluster_sort_level_one == 0) then
      local source_hetu_level_one_v2 = source_hetu_level_one_v2 and source_hetu_level_one_v2 or {}
      local item_hetu_level_one_v2_original = _G['hetu_tag_level_info_v2__hetu_level_one'] and _G['hetu_tag_level_info_v2__hetu_level_one'] or {}
      cluster_id = cacl_cascade_source_hetu_cluster(source_hetu_level_one_v2, item_hetu_level_one_v2_original, splash_variant_source_hetu_cluster_base_id)
    end
  end

  if (cluster_id == 0) then
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
    elseif (duration_sec >= 20.0 and duration_sec < 58.0) then
      cluster_id = 6
    elseif (duration_sec >= 58.0 and duration_sec < 120.0) then
      cluster_id = 7
    else
      cluster_id = 8
    end
  end
  return cluster_id
end