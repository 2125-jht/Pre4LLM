#!/usr/bin/env python3
# coding=utf-8

import sys
# 这里需要改成自己开发机的 svn 路径
sys.path.append("/home/maxiaojian/code/ks/ks/common_reco/ann_retrieve/dragonfly/")
from ann_retrieve_flow import AnnRetrieveFlow

filter_by_la = dict(
  import_item_attr=[
    "upload_time",
    "explore_stat__real_show_count",
    "thanos_stats__real_show_count"
  ],
  item_remove_check_func="CheckItemShouldRemove",
  lua_script="""
    function CheckItemShouldRemove()
        local cur = os.time()
        if upload_time == nil then
            return true
        end
        local upload = upload_time or 0
        local random = upload_time / 1000 % 100
        local time_diff = cur - upload / 1000
        local day_diff = time_diff / 86400
        local explore_count = explore_stat__real_show_count or 0
        local thanos_count = thanos_stats__real_show_count or 0
        local cur = os.time()
        if explore_count / (explore_count + thanos_count + 1.0) * 100 > 99 and random < 30 then
            return true
        end
        if day_diff < 7 then
            return false
        end
        if day_diff < 14 then
            if random < 40 then
                return true
            else 
                return false
            end
        end
        if day_diff < 30 then
            if random < 50 then
                return true
            else 
                return false
            end
        end
        if day_diff < 90 then
            if random < 73 then
                return true
            else 
                return false
            end
        end
        if day_diff >= 90 then
            if random < 80 then
                return true
            else 
                return false
            end
        end
        return true
    end
  """,
  remove_if_lua_fail=True,
)

def mio_data_scann_index_u2i():
  # 从 btq 消费数据
  datas = AnnRetrieveFlow() \
    .register_kess(kess_name="grpc_KaiWorksExploreLongSequenceTransformerRetrANN") \
    .consume_data_from_btq(queue_names=["kaiworks_explore_lst_v40"], thread_num=8)

  # 解析数据并存入 kv
  user = datas.parse_data_in_mio(
    data_name="photo",
    slot_id=4103, 
    begin_bit=0, 
    end_bit=64,
    max_item_num=100000000, 
    dim=64, 
    kv_expire_second=1800)

  photo_quality = datas.parse_data_in_mio(
    data_name="photo_quality",
    slot_id=4104, 
    begin_bit=0, 
    end_bit=2,
    max_item_num=100000000, 
    dim=2, 
    kv_expire_second=1800)

  bucket_photo_quality = photo_quality.build_scann_index(
    bucket_name="photo_quality", 
    space="ip",
    final_neighbors_num=450, 
    leaves_num=1000, 
    leaves_to_search=30,
    pre_reorder_neighbors_num=450,
    training_sample_size=500000, 
    anisotropic_quantization_threshold=0.2).filter_index_by_lua_script(**filter_by_la)
  

  bucket_user = user.build_scann_index(
    bucket_name="photo", 
    space="ip",
    final_neighbors_num=450, 
    leaves_num=1000, 
    leaves_to_search=30,
    pre_reorder_neighbors_num=450,
    training_sample_size=500000, 
    anisotropic_quantization_threshold=0.2).filter_index_by_lua_script(**filter_by_la)

  retr1 = user.retrieve_from(
    dest_bucket=bucket_user, 
    enable_precision_eval=True, 
    enable_auto_calc=False
  )

  retr2 = photo_quality.retrieve_from(
    dest_bucket=bucket_photo_quality, 
    enable_precision_eval=True, 
    enable_auto_calc=False
  )

  return retr1.merge_config(retr2)

if __name__ == '__main__':
  mio_data_scann_index_u2i()