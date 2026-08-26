function calculate()
  local num_per_key = redis_retrieval_num // trigger_num
  if triggerPids ~= nil and #triggerPids < trigger_num and #triggerPids > 0 then
    num_per_key = redis_retrieval_num // (#triggerPids)
  end
  return num_per_key
end