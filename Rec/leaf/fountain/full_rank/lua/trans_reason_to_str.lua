-- 精排将reason整型转string，并加上h字母开头
function trans_reason_to_str()
  local reason_str = 'h' .. tostring(reason)
  return reason_str
end

function gen_redis_key_for_sphinx()
  local user_stat_redis_key = '' .. tostring(featureUId)
  local user_app_redis_key = 'UserAppCart_' .. tostring(featureUId)
  return user_stat_redis_key, user_app_redis_key
end

function gen_similar_users_redis_key()
  local redis_key = fountain_similar_user_list_redis_prefix .. tostring(featureUId)
  return redis_key
end