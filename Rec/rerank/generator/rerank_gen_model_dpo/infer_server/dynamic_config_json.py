#!/usr/bin/python3
#coding: UTF-8

from backbone import *

predict_limit = 1200
predict_server_name = "grpc_fountain_gen_multiple"
model_queue_prefix = "fountain_rerank_wsy_12_layer"
embedding_kess_name = "grpc_fountain_rerank_multiple"

fullrank_infer_config = load_mio_tf_model("models/")
embedding_shard_num = 2

predict_pepper_with_kai = PredictServerFlow(name="predict_" + predict_server_name) \
  .predict_with_mio_model(
    model_config=fullrank_infer_config,
    embedding_kess_name=embedding_kess_name,
    receive_dnn_model_as_macro_block=True,
    rowmajor=True,
    queue_prefix=model_queue_prefix,
    shards=embedding_shard_num
  )

service.CHECK_COMMON_LOGIC = False
service.CHECK_UNUSED_ATTR = False
service.ENABLE_ATTR_CHECK = False

service.add_leaf_flows(leaf_flows = [predict_pepper_with_kai], request_type="default", as_default=True)

if __name__ == '__main__':
  service.build(output_file=os.path.join(current_dir, "dynamic_config_json.json"))


