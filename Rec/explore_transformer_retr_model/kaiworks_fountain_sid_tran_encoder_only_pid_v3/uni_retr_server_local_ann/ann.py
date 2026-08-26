#!/usr/bin/env python3
# coding=utf-8

import sys
# 这里需要改成自己开发机的 svn 路径
sys.path.append("/home/maxiaojian/code/ks/ks/common_reco/ann_retrieve/dragonfly/")
from ann_retrieve_flow import AnnRetrieveFlow

filter_by_la = dict(
  import_item_attr=[
    "author__id",
  ],
  item_remove_check_func="CheckItemShouldRemove",
  lua_script="""
    function CheckItemShouldRemove()
        if author__id == nil then
          return true
        end
    end
  """,
  remove_if_lua_fail=True,
)

def mio_data_scann_index_u2i():
  # 从 btq 消费数据
  datas = AnnRetrieveFlow() \
    .register_kess(kess_name="grpc_KaiWorksExploreLongSequenceTransformerRetrANN") \
    .consume_data_from_btq(queue_names=["kaiworks_fountain_lst_v10"], thread_num=8)

  # 解析数据并存入 kv
  user = datas.parse_data_in_mio(
    data_name="photo",
    slot_id=4103, 
    begin_bit=0, 
    end_bit=64,
    max_item_num=100000000, 
    dim=64, 
    kv_expire_second=43200)

  photo_quality = datas.parse_data_in_mio(
    data_name="photo_quality",
    slot_id=4104, 
    begin_bit=0, 
    end_bit=2,
    max_item_num=100000000, 
    dim=2, 
    kv_expire_second=43200)

  bucket_photo_quality = photo_quality.build_scann_index(
    bucket_name="photo_quality", 
    space="ip",
    final_neighbors_num=250, 
    leaves_num=1000, 
    leaves_to_search=30,
    pre_reorder_neighbors_num=250,
    training_sample_size=500000, 
    anisotropic_quantization_threshold=0.2).filter_index_by_lua_script(**filter_by_la)
  

  bucket_user = user.build_scann_index(
    bucket_name="photo", 
    space="ip",
    final_neighbors_num=250, 
    leaves_num=1000, 
    leaves_to_search=30,
    pre_reorder_neighbors_num=250,
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