function gen_variant_attr()
  -- 奥运字段生成
  local hetu_tag_v2 = hetu_tag_v2 and hetu_tag_v2 or {}
  local target_olympic_tags = {792552, 791455}
  local is_olympic_photo = nil
  for _, tgt in ipairs(target_olympic_tags) do
    for _, src in ipairs(hetu_tag_v2) do
      if (tgt == src) then
        is_olympic_photo = 1
        break
      end
    end
    if (is_olympic_photo == 1) then
      break
    end
  end

  -- 河图tag去除时效性tag
  local hetu_tag_v2_move_time = {}
  local time_tags = {4000006, 4000007, 4000008}
  for _, src in ipairs(hetu_tag_v2) do
    if (src ~= 4000006 and src ~= 4000007 and src ~= 4000008) then
      table.insert(hetu_tag_v2_move_time, src)
    end
  end

  -- 媒体号字段
  local similar_event_variant_attr = nil
  if (similar_event_id ~= nil and similar_event_id > 0) then
    similar_event_variant_attr = similar_event_id
  end
  local current_time_ms = currentTimeMs and currentTimeMs or 0
  local upload_time = upload_time and upload_time or 0
  local is_overdue_photo = 1
  local is_gr_account = _G['author__is_gr_account']
  local is_pr_account = _G['author__is_pr_account']
  local is_gr_account = is_gr_account and is_gr_account or 0
  local is_pr_account = is_pr_account and is_pr_account or 0
  local pr_gr_account_photo_id = photo_id and photo_id or 0
  if (skip_variant_v9_gr_pr <= 0 and (is_gr_account > 0 or is_pr_account > 0)) then
      pr_gr_account_photo_id = 0
  end
  if (current_time_ms - upload_time > fountain_variant_upload_days_threshold * 24 * 3600 * 1000) then
      is_overdue_photo = 0
  else
      is_overdue_photo = photo_id
  end
  -- pgc 打散
  local is_pgc_hetu_level_two = photo_id
  local is_pgc_hetu_level_three = photo_id
  local hetu_level_two_original = _G['hetu_tag_level_info__hetu_level_two'] and _G['hetu_tag_level_info__hetu_level_two'] or {}
  local hetu_level_three_original = _G['hetu_tag_level_info__hetu_level_three'] and _G['hetu_tag_level_info__hetu_level_three'] or {}
  local pgc_hetu_level_two = {[364] = 0, [323] = 0, [324] = 0, [327] = 0, [545] = 0, [548] = 0, [549] = 0, [311] = 0, [342] = 0, [350] = 0, [641] = 0, [223] = 0, [224] = 0, [402] = 0, [347] = 0, [352] = 0, [421] = 0}
  local pgc_hetu_level_three = {[1260] = 0, [2137] = 0, [1871] = 0, [1872] = 0, [1873] = 0, [1803] = 0, [1820] = 0}
  if (#hetu_level_two_original > 0) then
    for k,v in ipairs(hetu_level_two_original) do
      if (pgc_hetu_level_two[v] ~= nil) then
        is_pgc_hetu_level_two = -2
        break
      end
    end
  end
  if (#hetu_level_three_original > 0) then
    for k,v in ipairs(hetu_level_three_original) do
      if (pgc_hetu_level_three[v] ~= nil) then
        is_pgc_hetu_level_three = -2
        break
      end
    end
  end
  return is_olympic_photo, hetu_tag_v2_move_time, similar_event_variant_attr, pr_gr_account_photo_id, is_overdue_photo, is_pgc_hetu_level_two, is_pgc_hetu_level_three
end
