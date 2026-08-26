function calculate()
  local mind_emb_uid_list = {}
  if k_user_vec_ ~= nil and #k_user_vec_ > 0 and (#k_user_vec_ % 64 == 0) then
    interest_num = #k_user_vec_ / 64
    for i = 1, interest_num do
      table.insert(mind_emb_uid_list, i, _USER_ID_ * 100 + i - 1)
    end
  end
  return mind_emb_uid_list
end