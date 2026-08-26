#!/usr/bin/python3
#coding: UTF-8

from backbone import *

predict_limit = 1200
predict_server_name = "grpc_fountain_rerank_eval_listwise_server"
# v3 使用独立模型队列，避免部署时误拉取 v2 参数。
model_queue_prefix = "fountain_rerank_eval_listwise_v3"
colossusdb_embd_model_name = "rerank_gen_model_nar_glat_v2" # 暂时用的别人已有的服务，改不了名字
colossusdb_embd_table_name = "rerank_gen_uniform_lists_model_emb"

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

