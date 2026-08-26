#!/usr/bin/python3
#coding: UTF-8

from backbone import *

predict_limit = 1200
predict_server_name = "grpc_fountain_rerank_model_flash_evaluator_server_v1"
model_queue_prefix = "fountain_rerank_model_flash_evaluator_listwise"
colossusdb_embd_model_name = "fountain_rerank_model_flash_evaluator_v1"
colossusdb_embd_table_name = "fountain_rerank_model_flash_evaluator_embv1"

fullrank_infer_config = load_mio_tf_model("infer/")
embedding_shard_num = 2

predict_pepper_with_kai = PredictServerFlow(name="predict_" + predict_server_name) \
  .predict_with_mio_model(
    predict_server_name = predict_server_name,
    model_config=fullrank_infer_config,
    colossusdb_embd_model_name=colossusdb_embd_model_name,
    colossusdb_embd_table_name=colossusdb_embd_table_name,
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
  service.build(output_file=os.path.join(current_dir, "infer/dynamic_config_json.json"))


