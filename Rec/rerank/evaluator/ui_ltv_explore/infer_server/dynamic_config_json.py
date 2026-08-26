#!/usr/bin/python3
#coding: UTF-8
from backbone import *


predict_server_name = "grpc_RerankGenNextModelLtrInfer"
model_queue_prefix = "explore_rerank_revisit_model2"
embedding_kess_name = "grpc_RerankGenNextModelLtrEmb"
shard_num = 2
fullrank_infer_config = load_mio_tf_model("models/")

predict_with_kai = PredictServerFlow(name="predict_" + predict_server_name) \
  .predict_with_mio_model(
    model_config=fullrank_infer_config,
    embedding_kess_name=embedding_kess_name,
    receive_dnn_model_as_macro_block=True,
    rowmajor=True,
    shards = shard_num,
    queue_prefix=model_queue_prefix) \
  .copy_user_meta_info(save_result_size_to_attr="__inner_sample_length__") \
  .copy_user_meta_info(save_result_size_to_attr="item_num") \

service.CHECK_COMMON_LOGIC = False
service.CHECK_UNUSED_ATTR = False
service.ENABLE_ATTR_CHECK = False
service.add_leaf_flows(leaf_flows=[predict_with_kai], request_type="default", as_default=True)


if __name__ == '__main__':
  service.build(output_file=os.path.join(current_dir, "dynamic_config_json.json"))



