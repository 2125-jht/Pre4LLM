-- 精排将reason整型转string，并加上h字母开头
function trans_reason_to_str()
  local reason_str = 'h' .. tostring(reason)
  return reason_str
end