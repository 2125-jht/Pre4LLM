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
from ann_quality_filter_only import *
current_dir = os.path.dirname(__file__)
sys.path.append('/home/liyunhao/data/ks/dragon/tools/pypi/')

# global param
predict_server_name = "grpc_KaiWorksExploreLSTRetr"
model_queue_prefix="kaiworks_explore_lst_v4"
model_dir = "predict/conf"
param_json = "parameter_config.json"
PXTRS = ['user_emb']
sign_format = "mio"
shard_offset = 0
queue_shard_num = 1
EMBEDDING_SHARDS = 1

from dragonfly.matx.dragonfly_context import *
class FunctionSet:
  def __init__(self) -> None:
    pass

  def get_score(self, ctx: DragonflyContext) -> None:
    # Common Attr 处理逻辑
    video_quality_score_list_getter = ctx.ItemAttrGetter(b"video_quality_score_list")
    video_quality_score_setter = ctx.ItemAttrSetter(b"video_quality_score")
    video_cover_score_setter = ctx.ItemAttrSetter(b"video_cover_score")

    result_size = ctx.GetItemNum()
    for i in range(result_size):
      video_quality_score_list = video_quality_score_list_getter.GetDoubleList(i)
      video_quality_score_setter.SetDouble(i, video_quality_score_list[0])
      video_cover_score_setter.SetDouble(i, video_quality_score_list[1])

class ModelRetrFlow(LeafFlow, MioApiMixin, OfflineApiMixin, UniPredictApiMixin, ExploreModelApiMixin, RetrievalApiMixin, GsuApiMixin):

    def predict_with_mio_model(self, **kwargs):
        model_config = kwargs.pop("model_config")
        queue_prefix = kwargs.pop("queue_prefix")
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
                fetcher_type="ColossusdbEmbeddingServerFetcher", 
                colossusdb_embd_model_name="kaiworks_explore_ls_transformer_v2",
                colossusdb_embd_table_name="lst_v4emb",
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
              for_debug_request_only = True
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
      return self.get_abtest_params(
          biz_name = "RECO_RPC",
          ab_params = [
              ("explore_h3453_video_quality_score_threshold", 0.1),
              ("explore_h3453_video_random_int_threshold", 100),
              ("explore_h3453_video_cover_score_threshold", 0.7),
              ("explore_h3453_video_cover_level_threshold", 100)
          ]
        ) \
        .copy_user_meta_info(
          save_user_id_to_attr="uId",
          save_device_id_to_attr="dId",
          save_request_num_to_attr="request_num",
        ) \
        .prepare_params_transform() \
        .parse_protobuf_from_string(
          is_common_attr=True,
          input_attr="user",
          output_attr="user_info",
          class_name="ks::reco::UserInfo"
        ) \
        .enrich_with_protobuf(
          from_extra_var="user_info",
          is_common_attr=True,
          attrs=[
            dict(path="device_id", name="dId"),
            dict(path="id", name="featureUId"),
            dict(path="browsed_photo_ids", name="browsed_photo_ids"),
            dict(path="slide_browsed_photo_ids", name="slide_browsed_photo_ids"),
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

      # .enrich_attr_by_lua(
      #   import_item_attr = ["video_quality_score_list"],
      #   function_for_item = "calculate",
      #   export_item_attr = ["video_quality_score", "video_cover_score"],
      #   lua_script = """
      #     function calculate()
      #       return video_quality_score_list[1], video_quality_score_list[2]
      #     end
      #   """
      # ) \

    def filter_by_video_quality_score(self):
      return self.get_local_ann_embedding(
        src_data_type = "photo_quality",
        dim = 2,
        embedding_item_attr = "video_quality_score_list",
      ) \
      .enrich_attr_by_lua(
        import_item_attr = ["video_quality_score_list"],
        function_for_item = "calculate",
        export_item_attr = ["video_quality_score", "video_cover_score"],
        lua_script = """
          function calculate()
            return video_quality_score_list[1], video_quality_score_list[2]
          end
        """
      ) \
      .set_attr_default_value(
        item_attrs = [
          {
            "name": "content_safety_level_with_namespace__level_hot_online",
            "type": "int",
            "value": -1
          },
        ]
      ) \
      .count_reco_result(
        save_count_to="item_num_before_filter",
      ) \
      .gen_random_item_attr(
        attr_name = "random_int",
        attr_type = "int",
      ) \
      .filter_by_rule(
        rule = {
          "join": "and",
          "filters": [
            {
              "attr_name": "video_quality_score",
              "remove_if": "<=",
              "compare_to": "{{explore_h3453_video_quality_score_threshold}}"
            },
            {
              "attr_name": "content_safety_level_with_namespace__level_hot_online",
              "remove_if": "<",
              "compare_to": 3
            },
            {
              "attr_name": "random_int",
              "remove_if": "<=",
              "compare_to": "{{explore_h3453_video_random_int_threshold}}"
            }
          ]
        }
      ) \
      .count_reco_result(
        save_count_to="item_num_after_filter",
      ) \
      .filter_by_rule(
        rule = {
          "join": "and",
          "filters": [
            {
              "join": "or",
              "filters": [
                {
                  "attr_name": "audit_hot_cover_level",
                  "compare_to": 2023742,
                  "remove_if": "=="
                }, 
                {
                  "attr_name": "audit_hot_cover_level",
                  "compare_to": 2023743,
                  "remove_if": "=="
                },
                {
                  "attr_name": "audit_hot_cover_level",
                  "compare_to": 2023744,
                  "remove_if": "=="
                },
                {
                  "attr_name": "audit_hot_cover_level",
                  "compare_to": 2023745,
                  "remove_if": "=="
                },
                {
                  "attr_name": "audit_hot_cover_level",
                  "compare_to": 2023746,
                  "remove_if": "=="
                },
                {
                  "attr_name": "audit_hot_cover_level",
                  "compare_to": 2231037,
                  "remove_if": "=="
                },
              ]
            }, 
            {
              "attr_name": "random_int",
              "remove_if": "<=",
              "compare_to": 100
            }
          ]
        }
      ) \
      .count_reco_result(
        save_count_to="item_before_cover_filter",
      ) \
      .filter_by_rule(
        rule = {
          "join": "and",
          "filters": [
            {
              "attr_name": "video_cover_score",
              "remove_if": "<=",
              "compare_to": "{{explore_h3453_video_cover_score_threshold}}"
            },
            {
              "attr_name": "audit_hot_cover_level",
              "remove_if": "==",
              "compare_to": 0
            },
            {
              "attr_name": "random_int",
              "remove_if": "<=",
              "compare_to": "{{explore_h3453_video_cover_level_threshold}}"
            }
          ]
        }
      ) \
      .count_reco_result(
        save_count_to="item_after_cover_filter",
      ) \
      .log_debug_info(
        for_debug_request_only = True,
        log_tag = "mxj_filter",
        common_attrs = [
          "explore_h3453_video_quality_score_threshold"
        ],
        item_attrs = [
            "video_quality_score"
        ]
      ) \
      .perflog_attr_value(
        check_point="filter",
        common_attrs=
        [
          "item_num_before_filter", 
          "item_num_after_filter", 
          "explore_h3453_video_quality_score_threshold",
          "item_before_cover_filter",
          "item_after_cover_filter",
        ],
      ) \
  
    def retrieve(self):
      return self.retrieve_by_local_ann(
        reason=1,
        src_data_type="photo",
        dest_bucket="photo",
        src_items_attr="retr_src",
        top_k="{{request_num}}",
        src_embedding_list_attr="user_emb",
        save_distance_to_attr="distance") \
      .get_item_attr_by_local_attr_index(
        attrs = [
          "hetu_tag_level_info__hetu_cluster_id",
          "author__id",
          "content_safety_level_with_namespace__level_hot_online",
          "audit_hot_cover_level"
        ]
      ) \
      .perflog_attr_value(
        check_point="item",
        common_attrs = [
          "diversity_boost",
          "lambda"
        ],
        item_attrs = [
          "distance"
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
          "photo_dnn_cluster_id"
        ]
      ) \
      

    def post_process(self):
      return self.deduplicate() \
        .filter_by_common_attr(common_attr=["browsed_photo_ids", "slide_browsed_photo_ids"], on_item_attr="item_id") \
        .sort(score_from_attr="distance") \
        .deduplicate(on_item_attr="author__id") \
        .filter_by_video_quality_score() \
        .pandora_multi_interest_retrieval_post_process(limit="{{request_num}}", diversity_boost="{{lambda}}", cluster_num="{{cluster_num}}") \


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
service.build(output_file=os.path.join(model_dir, "dynamic_json_config_flow_filter_only.json"))

print(all_attrs)