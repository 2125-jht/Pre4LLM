function split(input, delimiter)
  local t={}
  for str in string.gmatch(input, "([^"..delimiter.."]+)") do
      table.insert(t, str)
  end
  return t
end

function get_hour()
  return os.date("*t").hour
end

function get_quota_status(window)
  local increase_quota_status = 0
  local time_index = 0
  local window_length = 0
  if (window == nil or window == "" or string.len(window) == 0) then
    return increase_quota_status, time_index, window_length
  end
  local hour = get_hour()
  local from_to_pairs = split(window, ",")
  window_length = #from_to_pairs
  for index, pair in ipairs(from_to_pairs) do
    for from, to in string.gmatch(pair, "(%d+):(%d+)") do
      if (hour >= tonumber(from) and hour < tonumber(to)) then
        increase_quota_status = 1
        time_index = index
        break
      end
    end
  end
  return increase_quota_status, time_index, window_length
end

function increase_quota_status()
  local window = fountain_increase_quota_time_window
  return get_quota_status(window)
end

function splash_increase_quota_status()
  local window = fountain_splash_increase_quota_time_window
  local increase_quota_status, _, _ = get_quota_status(window)
  return increase_quota_status
end
