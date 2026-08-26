from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.uni_predict.uni_predict_api_mixin import UniPredictApiMixin
from dragonfly.common_leaf_dsl import LeafService, LeafFlow
from dragonfly.ext.retrieval.retrieval_api_mixin import RetrievalApiMixin
from dragonfly.ext.explore_model.explore_model_api_mixin import ExploreModelApiMixin
from dragonfly.ext.gsu.gsu_api_mixin import GsuApiMixin
import os
import sys
import json
import yaml
import base64
import collections
from ann import *
current_dir = os.path.dirname(__file__)
sys.path.append('/home/liyunhao/data/ks/dragon/tools/pypi/')

# global param
predict_server_name = "grpc_KaiWorksExploreLongSequenceTransformerRetrInferIndices"
embedding_kess_name = "grpc_KaiWorksExploreLongSequenceTransformerEmbedding"
model_queue_prefix="kaiworks_explore_long_sequense_transformer_retr"
model_dir = "predict/conf"
param_json = "parameter_config.json"
PXTRS = ['user_emb', 'indices', 'scores']
sign_format = "mio"
shard_offset = 0
queue_shard_num = 1
EMBEDDING_SHARDS = 1

class ModelRetrFlow(LeafFlow, MioApiMixin, OfflineApiMixin, UniPredictApiMixin, ExploreModelApiMixin, RetrievalApiMixin, GsuApiMixin):

    def predict_with_mio_model(self, **kwargs):
        model_config = kwargs.pop("model_config")
        queue_prefix = kwargs.pop("queue_prefix")
        embedding_kess_name = kwargs.pop("embedding_kess_name")
        key = kwargs.pop("key", queue_prefix)
        receive_dnn_model_as_macro_block = kwargs.pop(
            "receive_dnn_model_as_macro_block", False
        )
        extra_inputs = kwargs.pop("extra_inputs", [])
        
        ## 处理模型 inputs
        inputs = []
        for c in extra_inputs:
          tmp = {
            "attr_name": c['name'],
            "tensor_name": c['name'],
            "dim": c['dim'],
            "common": True
          }
          inputs.append(tmp)
        for c in model_config.slots_config:
          # if not c.get("common", False):
          #   continue
          tmp = {
            "attr_name": c['input_name'],
            "tensor_name": c['input_name'],
            "dim": len(str(c['slots']).split(' ')) * c['dim'] * c.get('expand', 1) + (1 if c.get('sized', False) else 0)
          }
          if c.get("compress_group", None) and c.get("compress_group") == "USER":
            tmp["compress_group"] = c.get("compress_group")
          else:
            tmp["common"] = c.get("common", False)
          inputs.append(tmp)

        ## 处理 embedding fetcher 的 slots_config
        slots_config = []
        for c in model_config.slots_config:
          if 'dtype' in c:
            # dtype 默认为 mio_int16
            # 支持 mio_int16, scale_int8, scale_int16, float16, float32 需要根据自己的 embedding server 确定
            c['dtype'] = 'mio_int16'
          slots_config.append(c)
        # For debug only
        # print(model_config.common_parameter_config)
        # print(model_config.slots_config)
        # print("debug!!!!")
        # print([dict(attr_name = attr_name, tensor_name=tensor_name) for attr_name, tensor_name in model_config.outputs if attr_name in PXTRS])
        print("inputs", inputs)
        print("model_config.outputs", model_config.outputs)
        return (
            self.uni_predict_fused(
              debug_tensor = True,
              ## embedding 拉取相关配置，对应 fetch_mio_embedding 部分
               embedding_fetchers = [dict(
                fetcher_type = "BtEmbeddingServerFetcher",
                shards = EMBEDDING_SHARDS,
                kess_service = embedding_kess_name,
                slots_inputs = ["item_slots"],
                parameters_inputs = ["item_signs"],
                common_slots_inputs = ["common_slots"],
                common_parameters_inputs = ["common_signs"],
                timeout_ms = 50,
                slots_config = [config for config in slots_config if config["common"]],
                max_signs_per_request = 4096,
                client_side_shard = True
              )],
              embedding_manager_type="parallel_fetch",
              ## graph 相关配置
              graph = model_config.graph,
              key = key,
              inputs = inputs,
              outputs = [dict(attr_name = attr_name, tensor_name=tensor_name, common=True) for attr_name, tensor_name in model_config.outputs if attr_name in PXTRS],
              param = model_config.param,
              queue_prefix = queue_prefix,
              ## 模型加载相关配置
              model_loader_config = dict(
                rowmajor = True,
                type = "MioTFExecutedByTensorFlowModelLoader",												# 使用 TensorFlow 预估
                implicit_batch = True,																								# 是否使用 implicit_batch，开启 batch 的情况下推荐使用 explicit_batch
                receive_dnn_model_as_macro_block = receive_dnn_model_as_macro_block,
              ),
              ## batching 相关配置
              batching_config = dict(
                batch_timeout_micros = 0,													                    # 如果开启 batching，则配置 batching 的 timeout
                max_batch_size = 1,
                max_enqueued_batches = 1,
                batch_task_type = "BatchTensorflowTask",
              ),
              ## executor_config 相关配置
              executor_config = dict(
                intra_op_parallelism_threads_num = 32,
                inter_op_parallelism_threads_num = 32,
              ),
            )
            .log_debug_info(
              common_attrs = [config["input_name"] for config in slots_config if config["common"]],
              for_debug_request_only = False
            )
        )

    def prepare_params_transform(self):
      num_interest = 4
      cluster_num = 10000
      dim = 64
      user_embedding_size = num_interest * dim
      return self.enrich_attr_by_lua(
        import_common_attr=["uId", "diversity_boost"],
        function_for_common="Transform",
        export_common_attr=["retr_src", "cluster_num",  "lambda"],
        lua_script=f"""
                      function Transform()
                        local retr_src = {{}}
                        local cluster_num = {cluster_num} or 10000
                        user_embedding_size = {user_embedding_size} or {dim}
                        local lambda = diversity_boost or 0.5
                        
                        for i = 1, {num_interest} do
                          table.insert(retr_src, uId * 10 + i)
                        end
                        
                        return retr_src, cluster_num, lambda 
                      end
                    """
      )

    def prepare(self):
      return self.copy_user_meta_info(
        save_user_id_to_attr="uId",
        save_device_id_to_attr="dId",
        save_request_num_to_attr="request_num",
        save_current_time_ms_to_attr="request_time"
        ) \
        .prepare_params_transform() \
        .parse_protobuf_from_string(
        is_common_attr=True,
        input_attr="user",
        output_attr="user_info",
        class_name="ks::reco::UserInfo") \
        .enrich_with_protobuf(
        from_extra_var="user_info",
        is_common_attr=True,
        attrs=[
          dict(path="device_id", name="dId"),
          dict(path="id", name="featureUId"),
          dict(path="browsed_photo_ids", name="browsed_photo_ids"),
          dict(path="slide_browsed_photo_ids", name="slide_browsed_photo_ids"),
        ]) \
        .reverse_list_attr(
          common_attrs = [
            "colossus_photo_id_list",
            "colossus_author_id_list",
            "colossus_channel_list",
            "colossus_play_time_list",
            "colossus_duration_list",
            "colossus_label_list"
          ]
        ) \
        .cast_attr_type(
          attr_type_cast_configs=[
            {
              "to_type": "double",
              "from_common_attr": "colossus_play_time_list",
              "to_common_attr": "colossus_play_time_list",
            },
            {
              "to_type": "double",
              "from_common_attr": "colossus_duration_list",
              "to_common_attr": "colossus_duration_list",
            },
            {
              "to_type": "double",
              "from_common_attr": "colossus_label_list",
              "to_common_attr": "colossus_label_list",
            }
          ]
        ) \
        .explore_extract_universal_feature(
          kconf_key = "reco.explore.rpc_sample_models",
          mode = "infer",
          models = [
            "grpc_KaiWorksExploreComiRecInferV2",
          ],
          save_common_slots_to_attr = "common_slots",
          save_common_signs_to_attr = "common_signs",
          save_item_slots_to_attr = "item_slots",
          save_item_signs_to_attr = "item_signs",
          user_info_attrs = {
            "user_id_attr": "uId",
            "user_colossus_pid_list_attr": "colossus_photo_id_list",
            "user_colossus_aid_list_attr": "colossus_author_id_list",
            "user_colossus_channel_list_attr": "colossus_channel_list"
          },
          photo_info_attrs = {
          },
          context_info_common_attrs = {
          },
          context_info_item_attrs = {
          },
        )
    
    def retrieve(self):
      return self.retrieve_by_local_ann(
        reason=1,
        src_data_type="photo",
        dest_bucket="photo",
        src_items_attr="retr_src",
        top_k="{{request_num}}",
        src_embedding_list_attr="user_emb",
        save_distance_to_attr="distance") \
      .enrich_attr_by_lua(
          import_common_attr = ["colossus_photo_id_list"],
          # function_for_item 的值也可用 "{{}}" 格式指定为某个 common_attr
          function_for_common = "calculate",
          # 将 calculate 函数的返回值依次存入 export_item_attr 指定的 3 个 item_attr 中
          export_common_attr = ["colossus_photo_id_list_length"],
          lua_script = """
            function calculate()
              length = #colossus_photo_id_list
              return length
            end
          """
      ) \
      .get_item_attr_by_local_attr_index(
        attrs = [
          "hetu_tag_level_info__hetu_cluster_id",
          "author__id",
          "click_count",
          "explore_stat__click_count",
          "thanos_stats__click_count",
          "nebula_stats__click_count"
        ]
      ) \
      .copy_attr(
        attrs=[{
          "from_item": "hetu_tag_level_info__hetu_cluster_id",
          "to_item": "photo_dnn_cluster_id"
        }]
      ) \
      .copy_item_meta_info(save_item_id_to_attr = "item_id") \
      .log_debug_info(
        log_tag='afterRetrieve',
        for_debug_request_only = True,
        common_attrs = [
          "user_emb",
          "request_num",
          "dId",
          "lambda",
        ],
        item_attrs = [
          "author__id",
          "photo_dnn_cluster_id",
          "click_count",
          "explore_stat__click_count",
          "thanos_stats__click_count",
          "nebula_stats__click_count"
        ]
      ) \
      

    def post_process(self):
      return self.deduplicate() \
        .filter_by_common_attr(common_attr=["browsed_photo_ids", "slide_browsed_photo_ids"], on_item_attr="item_id") \
        .sort(score_from_attr="distance") \
        .deduplicate(on_item_attr="author__id") \
        .pandora_multi_interest_retrieval_post_process(limit="{{request_num}}", diversity_boost="{{lambda}}", cluster_num="{{cluster_num}}") \
        .pack_item_attr(
            item_source = {
            "reco_results": True,
          },
          mappings = [{
            "from_item_attr": "item_id",
            "to_common_attr": "retrieve_result",
          }]
        ) \
        .log_debug_info(
          log_tag = "mxjResult",
          common_attrs = [
            "retrieve_result"
          ],
          for_debug_request_only = False,
        ) \
        .limit(1) \
        .write_to_csv(
          attrs=[
            "uId", 
            "request_time", 
            "indices", 
            "scores",
            "colossus_photo_id_list",
            "colossus_photo_id_list_length",
            "colossus_play_time_list",
            "colossus_duration_list",
            "colossus_channel_list",
            "colossus_label_list",
            "retrieve_result"
          ],
          has_header=True,
          path_prefix="/worker/krp_cpu_uni_predict_server/csv/common_leaf_log_",
        ) \


# load Resources
ModelConfig = collections.namedtuple(
    "ModelConfig",
    [
        "graph",
        "outputs",
        "slots_config",
        "param",
        "common_slots",
        "non_common_slots",
        "common_parameter_config",
        "non_common_parameter_config",
    ],
)
all_attrs = set()
extra_inputs = []

def load_mio_tf_model():
    global model_dir

    with open(os.path.join(model_dir, "dnn_model.yaml")) as f:
        dnn_model = yaml.load(f, Loader=yaml.SafeLoader)

    with open(os.path.join(model_dir, "graph.pb"), "rb") as f:
        base64_graph = base64.b64encode(f.read()).decode("ascii")
        graph = "base64://" + base64_graph

    with open(os.path.join(model_dir, param_json)) as f:
        parameter_config = json.load(f)

    graph_tensor_mapping = dnn_model["graph_tensor_mapping"]
    extra_preds = dnn_model["extra_preds"].split(" ")
    q_names = dnn_model["q_names"].split(" ")
    assert len(extra_preds) == len(q_names)
    outputs = [
        (attr_name, graph_tensor_mapping[q_name])
        for attr_name, q_name in zip(extra_preds, q_names)
    ]
    param = dnn_model["param"]

    slots_config = dnn_model["embedding"]["slots_config"]
    common_slots = set()
    non_common_slots = set()
    for c in slots_config:
        slots = map(int, str(c["slots"]).split(" "))
        if c.get("common", False):
            common_slots.update(slots)
        else:
            non_common_slots.update(slots)

    global all_attrs
    common_parameter_config = dict()
    non_common_parameter_config = dict()
    for name, c in parameter_config.items():
        attrs = c["attrs"]
        assert len(attrs) == 1
        if attrs[0]["converter"] == "combine":
            converter_args = attrs[0]["converter_args"]
            attr = [*converter_args["left"].keys(), *converter_args["right"].keys()]
        else:
            attr = attrs[0]["attr"]

        slot_id = attrs[0]["mio_slot_key_type"]

        if slot_id in common_slots:
            common_parameter_config[name] = c
            all_attrs.update(attr)
        # elif slot_id in non_common_slots:
        #     non_common_parameter_config[name] = c
        #     all_attrs.update(attr)
    global extra_inputs
    if "vec_input" in dnn_model and dnn_model["vec_input"] is not None:
      extra_inputs = dnn_model["vec_input"]
      for c in extra_inputs:
        all_attrs.update([c['name']])

    return ModelConfig(
        graph,
        outputs,
        slots_config,
        param,
        common_slots,
        non_common_slots,
        common_parameter_config,
        non_common_parameter_config,
    )

def generate_embedding_table_config():
    embedding_config = dict()
    # 过期时间，30天
    embedding_config["force_expire_timet"] = 25920000000
    embedding_config["sign_format"] = sign_format
    embedding_config["shard_offset"] = shard_offset
    embedding_config["thread_num"] = 65
    embedding_config["queue_prefix"] = model_queue_prefix
    embedding_config["queue_shard_num"] = queue_shard_num
    embedding_config["queue_suffixes"] = dict()
    embedding_config["read_slots"] = ""
    embedding_config["kess_service_config"] = dict()
    embedding_config["kess_service_config"]["default"] = predict_server_name
    embedding_config["kess_cluster_config"] = dict()
    embedding_config["kess_cluster_config"]["default"] = "PRODUCTION" 
    embedding_config["kess_weight"] = dict()
    embedding_config["kess_weight"]["default"] = 1000
    return embedding_config


embedding_table_config = generate_embedding_table_config()

# load model config first so that we have all_attrs
model_config = load_mio_tf_model()

all_attrs_list = list(sorted(all_attrs))

pipeline = ModelRetrFlow(name="default") \
    .prepare() \
   .predict_with_mio_model(
      model_config=model_config,
      embedding_kess_name=embedding_kess_name,
      queue_prefix=model_queue_prefix,
      extra_inputs=extra_inputs,
      receive_dnn_model_as_macro_block=True
    ) \
    .retrieve() \
    .post_process() \
    .log_debug_info(
      common_attrs = [
        "featureUId",
        "diversity_boost",
        "colossus_photo_id_list",
        "colossus_author_id_list",
      ],
      for_debug_request_only = True,
    ) \
    .log_debug_info(
      common_attrs = [
        "colossus_channel_list",
        "colossus_duration_list"
      ],
      for_debug_request_only = True,
    ) \
    .log_debug_info(
      common_attrs = [
        "user",
        "browsed_photo_ids",
        "slide_browsed_photo_ids",
        "colossus_label_list",
        "colossus_play_time_list",
      ],
      for_debug_request_only = True,
    )


returned_item_attrs = []

ann_conf_data = mio_data_scann_index_u2i()

service = LeafService(
  kess_name=predict_server_name,
  common_attrs_from_request = ["user", "diversity_boost", "colossus_photo_id_list", "colossus_author_id_list", "colossus_channel_list",
  "colossus_play_time_list", "colossus_duration_list", "colossus_label_list"],
  item_attrs_from_request=[],
  ann_config=ann_conf_data.get_config())

service.AUTO_INJECT_ITEM_ATTR = False
service.AUTO_INJECT_SAMPLE_LIST_USER_ATTR = False
service.return_item_attrs(attrs = returned_item_attrs)
service.return_common_attrs(attrs = ["user_emb"])
service.add_leaf_flows(leaf_flows = [pipeline], request_type = "default")
service.build(output_file=os.path.join(model_dir, "dynamic_json_config.json"))

print(all_attrs)