function content_id_adjust_func()
  local new_dup_cluster_id = -1 * item_id
  if (dup_cluster_id ~= nil and dup_cluster_id > 0) then
      new_dup_cluster_id = dup_cluster_id
  end
  local new_sim_remove_dup_id = -1 * item_id
  if (sim_remove_dup_id ~= nil and sim_remove_dup_id > 0) then
      new_sim_remove_dup_id = sim_remove_dup_id
  end
  local new_pic_and_selfdup_id = -1 * item_id
  if (pic_and_selfdup_id ~= nil and pic_and_selfdup_id > 0) then
      new_pic_and_selfdup_id = pic_and_selfdup_id
  end
  return new_dup_cluster_id, new_sim_remove_dup_id, new_pic_and_selfdup_id
end
