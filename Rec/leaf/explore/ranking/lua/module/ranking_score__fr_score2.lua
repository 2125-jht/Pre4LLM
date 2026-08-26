function fr_score2_change()
  local fr_score2 = fr_score2 or 0.0
  if fr_score2 > 1.0 or fr_score2 < 0.0 then
    fr_score2 = 0.0
  elseif fr_score2 == 1.0 then
    fr_score2 = 1024.0
  else
    fr_score2 = fr_score2 / (1.0 - fr_score2)
  end

  return fr_score2
end
