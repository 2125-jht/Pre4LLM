#!/usr/bin/env python3
# coding=utf-8

import sys
# 这里需要改成自己开发机的 svn 路径
sys.path.append("/home/maxiaojian/code/ks/ks/common_reco/ann_retrieve/dragonfly/")
from ann_retrieve_flow import AnnRetrieveFlow

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
    anisotropic_quantization_threshold=0.2)

  retr1 = user.retrieve_from(
    dest_bucket=bucket_user, 
    enable_precision_eval=True, 
    enable_auto_calc=False
  )
  return retr1

if __name__ == '__main__':
  mio_data_scann_index_u2i()