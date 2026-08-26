function calculate()
  local hetu_tag_raw = hetu_tag_v2 and hetu_tag_v2 or {}
  local hetu_tag_v2_theme = {}
  local hetu_tag_v2_style = {}
  local hetu_tag_v2_normal = {}
  for i = 1, #hetu_tag_raw do
    local tag_ = hetu_tag_raw[i]
    if (tag_ ~= nil) then
      if (tag_ >= 500006 and tag_ <= 3999999) then
        table.insert(hetu_tag_v2_theme, tag_)
      elseif ((tag_ >= 4000000 and tag_ <= 4000005) or tag_ == 4000009) then
        table.insert(hetu_tag_v2_style, tag_)
      else 
        table.insert(hetu_tag_v2_normal, tag_)
      end 
    end
    i = i + 1
  end
  -- 没有hetu_tag要置为空，否则打散时会对空值做打散
  if (#hetu_tag_v2_theme < 1) then
    hetu_tag_v2_theme = nil
  end
  if (#hetu_tag_v2_style < 1) then
    hetu_tag_v2_style = nil
  end
  if (#hetu_tag_v2_normal < 1) then
    hetu_tag_v2_normal = nil
  end
  local picture_variant_attr_adjust = nil
  if (picture_variant_attr ~= nil and picture_variant_attr > 0) then
    picture_variant_attr_adjust = 1
  end
  return hetu_tag_v2_theme, hetu_tag_v2_style, hetu_tag_v2_normal, picture_variant_attr_adjust
end
