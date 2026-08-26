#!/usr/bin/env python3
# coding=utf-8

import sys
# 这里需要改成自己开发机的 svn 路径
sys.path.append("/home/maxiaojian/code/ks/ks/common_reco/ann_retrieve/dragonfly/")
from ann_retrieve_flow import AnnRetrieveFlow

filter_by_la = dict(
  import_item_attr=[
    "explore_stat__click_count",
    "thanos_stats__click_count",
    "nebula_stats__click_count",
    "upload_time",
    "author__fans_count",
    "picture_type",
    "audit_b_second_tag",
    "content_safety_level_with_namespace__level_hot_online"
  ],
  item_remove_check_func="CheckItemShouldRemove",
  lua_script="""
    function CheckItemShouldRemove()
        local cur = os.time()
        local upload = upload_time or 0
        if upload == 0 then
          return true
        end
        local time_diff = cur - upload
        local explore_click_count = explore_stat__click_count or 0
        local thanos_click_count = thanos_stats__click_count or 0
        local nebula_click_count = nebula_stats__click_count or 0
        local click_count_all = explore_click_count + thanos_click_count + nebula_click_count
        if author__fans_count ~= nil and author__fans_count > 1000000 then
          return false
        end
        if time_diff > 86400000 then
          return false
        end
        if click_count_all <= 100 then
          return true
        end
        if picture_type == nil or picture_type == 1 then
          return true
        end
        if picture_type == 2 or picture_type == 3 then
          if audit_b_second_tag == 2000866 or audit_b_second_tag == 2019671 or audit_b_second_tag == 2019672 or audit_b_second_tag == 2110332 or audit_b_second_tag == 2110333 or audit_b_second_tag == 2147753 or audit_b_second_tag == 2022202 or audit_b_second_tag == 2022203 or audit_b_second_tag == 2000865 then
            return false
          end
          return true
        end
        return false
    end
  """,
  remove_if_lua_fail=True,
)

def mio_data_scann_index_u2i():
  # 从 btq 消费数据
  datas = AnnRetrieveFlow() \
    .register_kess(kess_name="grpc_KaiWorksExploreLongSequenceTransformerRetrANN") \
    .consume_data_from_btq(queue_names=["kaiworks_explore_long_sequense_transformer_retr0"], thread_num=8)

  # 解析数据并存入 kv
  user = datas.parse_data_in_mio(
      data_name="photo",
      slot_id=4103, 
      begin_bit=0, 
      end_bit=64,
      max_item_num=100000000, 
      dim=64, 
      kv_expire_second=1800)

  bucket_user = user.build_scann_index(
    bucket_name="photo", 
    space="ip",
    final_neighbors_num=400, 
    leaves_num=1000, 
    leaves_to_search=30,
    pre_reorder_neighbors_num=400,
    training_sample_size=500000, 
    anisotropic_quantization_threshold=0.2).filter_index_by_lua_script(**filter_by_la)

  retr1 = user.retrieve_from(
    dest_bucket=bucket_user, 
    enable_precision_eval=True, 
    enable_auto_calc=False
  )
  return retr1

if __name__ == '__main__':
  mio_data_scann_index_u2i()