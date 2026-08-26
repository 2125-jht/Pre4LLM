function handle(seq)
  local pctr
  if mc_pxtr_value then
    pctr = mc_pxtr_value[seq + 1]
  end

  return pctr
end
