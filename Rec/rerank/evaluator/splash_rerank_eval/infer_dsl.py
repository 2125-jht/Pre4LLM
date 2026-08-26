import os
import sys
import json
import yaml
import argparse
import base64
import collections

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, '../../../../../../cpp/dragon/tools/pypi/'))

from dragonfly.common_leaf_dsl import LeafService, LeafFlow
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.uni_predict.uni_predict_api_mixin import UniPredictApiMixin
from dragonfly.ext.explore_offline.explore_offline_api_mixin import ExploreOfflineApiMixin
from dragonfly.ext.common.common_api_mixin import CommonApiMixin


use_kuiba_predict_server=True
predict_use_batching = False
predict_use_gpu = False
predict_limit = 1200
predict_server_name = "grpc_FountainSplashRerankEvalModelInfer"
model_queue_prefix="splash_rerank_model_evaluator_mtl_v1"
colossusdb_embd_service_name = "emb_server"
colossusdb_embd_model_name = "fountain_splash_rerank_evaluator_model"
colossusdb_embd_table_name = "fountain_splash_rerank_evaluator_model_emb_v1"
# PXTRS = ["pos0", "pos1", "pos2", "pos3", "pos4", "pos5", "next0", "next1","next2", "next3", "next4", "next5", "play0", "play1","play2", "play3", "play4", "play5"]
PXTRS = ["pos0", "pos1", "next0", "next1", "play0", "play1"]
#PXTRS = ["point_pos","point_next"]
PXTRS_ATTRS_MAPPING = {
}
for key in PXTRS:
  if not (key in PXTRS_ATTRS_MAPPING.keys()):
    PXTRS_ATTRS_MAPPING[key] = key
output_pxtrs_as_embedding = False
return_item_attr = PXTRS
LIST_SIZE = 2

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--run', default=None, help='run the pipeline directly')
args = parser.parse_args()

all_attrs = set()
all_features = {}

sampling_lua_positive_rate = {}
sampling_lua_negative_rate = {}

# 线上特征为了避免和 attr 冲突, 统一加上 "KAI_" 前缀

# 将 kuiba feature name 到 slot id 的映射重命名
def LoadSlotMappingRename():
  filename = "kai_kuiba_config.json"
  slots_mapping=json.load(open(filename))["sign_feature_slot"]
  slots_mapping_rename = {}
  for name, slot_id in slots_mapping.items():
    if name.find('_weight') != -1 or name.find('_flag') != -1 or name.find('Weight') != -1:
      continue
    if name.find('_idx') != -1:
      n = (int)(name.split('_idx')[-1])
      if n >= 6 : continue
    slots_mapping_rename["KAI_" + name] = slot_id
  return slots_mapping_rename

slots_mapping_rename = LoadSlotMappingRename()

def LoadKuibaSamplingRate(filename):
  loss_functions = json.load(open(filename))["krp_ps_server"]["network"]["loss_functions"]
  for loss_name in loss_functions:
    if loss_name == 'default_feature_score':
      continue
    loss_function = loss_functions[loss_name]
    positive_rate = 1.0
    negative_rate = 1.0
    if ('positive_rate' in loss_function.keys()):
      positive_rate = loss_function['positive_rate']
    if ('negative_rate' in loss_function.keys()):
      negative_rate = loss_function['negative_rate']
    if (positive_rate != 1.0 or negative_rate != 1.0):
      sampling_lua_positive_rate[loss_name] = positive_rate
      sampling_lua_negative_rate[loss_name] = negative_rate

# 将 kuiba 抽特征的 config 线上需要的部分提取出来
# 需要的签名由 features_set 决定, features_set 是分析 dnn_model.yaml 倒推出来的
def LoadKuibaParameterConfigRename(filename, features_set):
  global slots_mapping_rename
  parameter_config = json.load(open(filename))["krp_ps_server"]["network"]["parameters"]
  parameter_config_rename = {}
  for name, feature_config in parameter_config.items():
    if name.find('_weight') != -1 or  name.find('_flag') != -1 or name.find('Weight') != -1:
      continue
    if name.find('_idx') != -1:
      n = (int)(name.split('_idx')[-1])
      if n >= 6 : continue
    if isinstance(feature_config, dict):
      #if name in user_features:
      #    feature_config["attrs"][0]["is_common"] = True
      #    feature_config["use_common_attr_only"] = True
      #if "use_common_attr_only" in feature_config:
      #  if feature_config["use_common_attr_only"] == True:
      #    feature_config["attrs"][0]["is_common"] = True
      kai_name = "KAI_" + name
      if kai_name in slots_mapping_rename.keys():
        slot = slots_mapping_rename[kai_name]
        for converter in feature_config["attrs"]:
          converter["mio_slot_key_type"] = slot
        if (kai_name in features_set.keys()):
          parameter_config_rename[kai_name] = feature_config
      else:
        parameter_config_rename[kai_name] = feature_config
        # 非 slot 类型的特征也需要抽取(dense特征)
        slots_mapping_rename[kai_name] = -1
    else:
      parameter_config_rename[name] = feature_config
  return parameter_config_rename

print("--------------------------------------------------")
print(slots_mapping_rename)
print("--------------------------------------------------")

# load Mixins
# load Resources
ModelConfig = collections.namedtuple('ModelConfig', ['graph', 'outputs', 'slots_config', 'vec_config', 'param', 'parameter_config_rename', 'common_parameter_config_rename', 'non_common_parameter_config_rename'])

def load_mio_tf_model():
  global model_dir

  # with open(os.path.join(model_dir, 'predict/dnn_model.yaml')) as f:
  with open('predict/dnn_model.yaml') as f:
    dnn_model = yaml.load(f, Loader=yaml.SafeLoader)

  # with open(os.path.join(model_dir, 'predict/graph.pb'), 'rb') as f:
  with open('predict/graph.pb', 'rb') as f:
    base64_graph = base64.b64encode(f.read()).decode('ascii')
    graph = 'base64://' + base64_graph


  graph_tensor_mapping = dnn_model['graph_tensor_mapping']
  extra_preds = dnn_model['extra_preds'].split(' ')
  q_names = dnn_model['q_names'].split(' ')
  assert len(extra_preds) == len(q_names)
  outputs = [(attr_name, graph_tensor_mapping[q_name]) for attr_name, q_name in zip(extra_preds, q_names)]
  param = dnn_model['param']

  slots_config = dnn_model['embedding']['slots_config']
  vec_config = dnn_model['vec_input'] if 'vec_input' in dnn_model else []

  global all_attrs
  global all_features
  new_slot_config = []
  for slot in slots_config:
    input_name = slot["input_name"]
    if input_name.find('_weight') != -1 or  input_name.find('_flag') != -1 or input_name.find('Weight') != -1:
      continue
    if input_name.find('_idx') != -1:
      n = (int)(input_name.split('_idx')[-1])
      if n >= 6 : continue
    all_features[input_name] = 1;
    new_slot_config.append(slot)
  slots_config = new_slot_config
  new_vec_config = []
  for vec in vec_config:
    input_name = vec["name"]
    if input_name.find('_weight') != -1 or input_name.find('_flag') != -1 or input_name.find('Weight') != -1:
      continue
    if input_name.find('_idx') != -1:
      n = (int)(input_name.split('_idx')[-1])
      if n >= 6 : continue
    all_features[input_name] = 1;
    new_vec_config.append(vec)
  vec_config = new_vec_config
  common_parameter_config_rename = {}
  non_common_parameter_config_rename = {}

  # parameter_config_rename = LoadKuibaParameterConfigRename(os.path.join(model_dir, 'dynamic_json_config.json'), all_features)
  parameter_config_rename = LoadKuibaParameterConfigRename('dynamic_json_config.json', all_features)
  print (parameter_config_rename)

  # LoadKuibaSamplingRate(os.path.join(model_dir, 'dynamic_json_config.json'))
  LoadKuibaSamplingRate('dynamic_json_config.json')

  # 筛选出线上用的 slot
  slots_mapping_for_predict = {}
  global slots_mapping_rename
  for name,slot_id in slots_mapping_rename.items():
    if name.find('_weight') != -1 or  name.find('_flag') != -1 or name.find('Weight') != -1:
      continue
    if name.find('_idx') != -1:
      n = (int)(name.split('_idx')[-1])
      if n >= LIST_SIZE : continue
    if name in parameter_config_rename.keys():
      slots_mapping_for_predict[name] = slot_id
      print(name, slot_id, parameter_config_rename[name])
  slots_mapping_rename = slots_mapping_for_predict

  for name, feature_config in parameter_config_rename.items():
    if name.find('_weight') != -1 or  name.find('_flag') != -1 or name.find('Weight') != -1:
      continue
    if name.find('_idx') != -1:
      n = (int)(name.split('_idx')[-1])
      if n >= 6 : continue
    if not isinstance(feature_config, dict):
      continue
    #if ('use_common_attr_only' in feature_config.keys()) and (feature_config['use_common_attr_only']):
    #  common_parameter_config_rename[name] = feature_config
    #else:
    non_common_parameter_config_rename[name] = feature_config
    extrators = feature_config['attrs']
    for extrator in extrators:
      if extrator['converter'] == 'combine':
        converter_args = extrator['converter_args']
        attr = [*converter_args['left'].keys(), *converter_args['right'].keys()]
      else:
        attr = extrator['attr']
      all_attrs.update(attr)

  for slot_config in slots_config:
    slot_config["common"] = False
    #if slot_config["input_name"] in common_parameter_config_rename.keys():
    #  if predict_use_batching:
    #    slot_config["common"] = True
    #  print (slot_config)

  return ModelConfig(graph, outputs, slots_config, vec_config, param, parameter_config_rename, common_parameter_config_rename, non_common_parameter_config_rename);

# load model config first so that we have all_attrs
model_config = load_mio_tf_model();
all_attrs_list = list(sorted(all_attrs))
# define the pipeline

class PredictServerFlow(LeafFlow, KuibaApiMixin, MioApiMixin, OfflineApiMixin, UniPredictApiMixin, CommonApiMixin):
  def prepare(self):
    return self \
      .deduplicate() \
      .limit(predict_limit)
  
  def cal_ensemble_score(self, **kwargs):
    return self \
      .get_kconf_params(
        kconf_configs = [
          {
            "kconf_key": "{{fountain_rr_play_watch_table_kconf}}",
            "value_type": "json",
            "json_path": "durationQuantile",
            "export_common_attr": "rerank_duration_buckets",
          },
          {
            "kconf_key": "{{fountain_rr_play_watch_table_kconf}}",
            "value_type": "json",
            "json_path": "playQ0",
            "export_common_attr": "rerank_wtd_table_0",
          },
          {
            "kconf_key": "{{fountain_rr_play_watch_table_kconf}}",
            "value_type": "json",
            "json_path": "playQ1",
            "export_common_attr": "rerank_wtd_table_1",
          },
          {
            "kconf_key": "{{fountain_rr_play_watch_table_kconf}}",
            "value_type": "json",
            "json_path": "playQ2",
            "export_common_attr": "rerank_wtd_table_2",
          },
          {
            "kconf_key": "{{fountain_rr_play_watch_table_kconf}}",
            "value_type": "json",
            "json_path": "playQ3",
            "export_common_attr": "rerank_wtd_table_3",
          },
          {
            "kconf_key": "{{fountain_rr_play_watch_table_kconf}}",
            "value_type": "json",
            "json_path": "playQ4",
            "export_common_attr": "rerank_wtd_table_4",
          },
          {
            "kconf_key": "{{fountain_rr_play_watch_table_kconf}}",
            "value_type": "json",
            "json_path": "playQ5",
            "export_common_attr": "rerank_wtd_table_5",
          },
          {
            "kconf_key": "{{fountain_rr_play_watch_table_kconf}}",
            "value_type": "json",
            "json_path": "playQ6",
            "export_common_attr": "rerank_wtd_table_6",
          },
          {
            "kconf_key": "{{fountain_rr_play_watch_table_kconf}}",
            "value_type": "json",
            "json_path": "playQ7",
            "export_common_attr": "rerank_wtd_table_7",
          },
          {
            "kconf_key": "{{fountain_rr_play_watch_table_kconf}}",
            "value_type": "json",
            "json_path": "playQ8",
            "export_common_attr": "rerank_wtd_table_8",
          },
        ],
      ) \
      .set_attr_value(
        no_overwrite = False,
        common_attrs = [
          {
            "name": "rerank_origin_score_attrs",
            "type": "string_list",
            "value": ["play" + str(i) for i in range(LIST_SIZE)],
          },
          {
            "name": "rerank_duration_attrs",
            "type": "string_list",
            "value": ["pDurationMs_idx" + str(i) for i in range(LIST_SIZE)],
          },
          {
            "name": "rerank_wtd_table_atts",
            "type": "string_list",
            "value": ["rerank_wtd_table_" + str(i) for i in range(9)],
          },
          {
            "name": "rerank_wtd_score_attrs",
            "type": "string_list",
            "value": ["wtd" + str(i) for i in range(LIST_SIZE)],
          },
          {
            "name": "rerank_list_es_score_attrs",
            "type": "string_list",
            "value": ["pos_score", "next_score", "wtd_score", "finish_score"],
          },
          {
            "name": "rerank_list_es_weight_attrs",
            "type": "string_list",
            "value": ["rerank_list_pos_weight", "rerank_list_next_weight", "rerank_list_wtd_weight", "rerank_list_finish_weight"],
          },
        ],
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          { "name": "rerank_duration_buckets", "as": "duration_buckets" },
          { "name": "rerank_origin_score_attrs", "as": "origin_score_attrs" },
          { "name": "rerank_duration_attrs", "as": "duration_attrs" },
          { "name": "rerank_wtd_table_atts", "as": "trans_score_list_attrs" },
          { "name": "rerank_wtd_score_attrs", "as": "wtd_score_attrs" },
        ] + ["rerank_wtd_table_" + str(i) for i in range(9)],
        import_item_attr = ["play" + str(i) for i in range(LIST_SIZE)] + ["pDurationMs_idx" + str(i) for i in range(LIST_SIZE)],
        export_item_attr = ["wtd" + str(i) for i in range(LIST_SIZE)],
        function_name = "CalcRerankWtdScore",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .calc_weighted_sum(
        # channels = [{ "name": "pos" + str(i), "weight": 1.0 } for i in range(LIST_SIZE)],
        channels = [
          {"name": "pos0", "weight": "{{fountain_splash_rerank_eval_ltr0_weight}}"},
          {"name": "pos1", "weight": "{{fountain_splash_rerank_eval_ltr1_weight}}"},
        ],
        output_item_attr = "pos_score",
      ) \
      .calc_weighted_sum(
        # channels = [{ "name": "next" + str(i), "weight": 1.0 } for i in range(LIST_SIZE)],
        channels = [
          {"name": "next0", "weight": "{{fountain_splash_rerank_eval_next0_weight}}"},
          {"name": "next1", "weight": "{{fountain_splash_rerank_eval_next1_weight}}"},
        ],
        output_item_attr = "next_score",
      ) \
      .calc_weighted_sum(
        # channels = [{ "name": "wtd" + str(i), "weight": 1.0 } for i in range(LIST_SIZE)],
        channels = [
          {"name": "wtd0", "weight": "{{fountain_splash_rerank_eval_wtd0_weight}}"},
          {"name": "wtd1", "weight": "{{fountain_splash_rerank_eval_wtd1_weight}}"},
        ],
        output_item_attr = "wtd_score",
      ) \
      .calc_weighted_sum(
        channels = [
          {"name": "play0", "weight": "{{fountain_splash_rerank_eval_finish0_weight}}"},
          {"name": "play1", "weight": "{{fountain_splash_rerank_eval_finish1_weight}}"},
        ],
        output_item_attr = "finish_score",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          { "name": "rerank_list_es_score_attrs", "as": "score_attrs" },
          { "name": "rerank_list_es_weight_attrs", "as": "weight_attrs" },
          "rerank_list_pos_weight",
          "rerank_list_next_weight",
          "rerank_list_wtd_weight",
          "rerank_list_finish_weight",
        ],
        import_item_attr = [
          "pos_score",
          "next_score",
          "wtd_score",
          "finish_score",
        ],
        export_item_attr = [
          "es_score",
        ],
        function_name = "CalcRerankListESScore",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .set_attr_value(
        item_attrs=[
          {
            "name": "pos1",
            "type": "double",
            "value": 0.0
          },
          # {
          #   "name": "pos2",
          #   "type": "double",
          #   "value": 0.0
          # },
          # {
          #   "name": "pos3",
          #   "type": "double",
          #   "value": 0.0
          # },
          # {
          #   "name": "pos4",
          #   "type": "double",
          #   "value": 0.0
          # },
          # {
          #   "name": "pos5",
          #   "type": "double",
          #   "value": 0.0
          # },
          {
            "name": "next1",
            "type": "double",
            "value": 0.0
          },
          # {
          #   "name": "next2",
          #   "type": "double",
          #   "value": 0.0
          # },
          # {
          #   "name": "next3",
          #   "type": "double",
          #   "value": 0.0
          # },
          # {
          #   "name": "next4",
          #   "type": "double",
          #   "value": 0.0
          # },
          # {
          #   "name": "next5",
          #   "type": "double",
          #   "value": 0.0
          # },
          {
            "name": "play1",
            "type": "double",
            "value": 0.0
          },
          # {
          #   "name": "play2",
          #   "type": "double",
          #   "value": 0.0
          # },
          # {
          #   "name": "play3",
          #   "type": "double",
          #   "value": 0.0
          # },
          # {
          #   "name": "play4",
          #   "type": "double",
          #   "value": 0.0
          # },
          # {
          #   "name": "play5",
          #   "type": "double",
          #   "value": 0.0
          # },
        ]
      ) \
      .copy_attr(
        attrs=[
          {"from_item": "es_score", "to_item": "pos0"},
          {"from_item": "es_score", "to_item": "next0"},
          {"from_item": "es_score", "to_item": "play0"},
        ]
      ) \
      .log_debug_info(
        for_debug_request_only = True,
        # respect_sample_logging = False,
        item_num_limit = 3,
        common_attrs = [
          "enable_return_ensemble_score",
          "rerank_duration_buckets",
          "rerank_list_pos_weight",
          "rerank_list_next_weight",
          "rerank_list_wtd_weight",
        ],
        item_attrs = [
          "es_score",
          "pos0",
          "next0",
          "play0",
          "pos1",
          "next1",
          "play1",
        ],
      )

  def predict_with_mio_model(self, **kwargs):
    model_config = kwargs.pop('model_config')
    print("colossusdb_embd_model_name:", colossusdb_embd_model_name)
    queue_prefix = kwargs.pop('queue_prefix')
    key = kwargs.pop('key', queue_prefix)
    receive_dnn_model_as_macro_block = kwargs.pop('receive_dnn_model_as_macro_block', False)
    extra_inputs = kwargs.pop('extra_inputs', [])
    batch_sizes = kwargs.pop('batch_sizes', [])
    implicit_batch = kwargs.pop('implicit_batch', True)
    
    slots_config = []
    for c in model_config.slots_config:
      if 'dtype' in c:
        # 支持 mio_int16, scale_int8, scale_int16, float16, float32 需要根据自己的 embedding server 确定
        c['dtype'] = 'mio_int16'
      slots_config.append(c)
    ## inputs 配置
    inputs = []
    for c in model_config.slots_config:
      input_config = dict(
        attr_name=c['input_name'],tensor_name=c['input_name'],
        dim=len(str(c['slots']).split(' ')) * c['dim'] * c.get('expand', 1) + (1 if c.get('sized', False) else 0),
      )
      if c.get('compress_group', None) and c.get('compress_group') == "USER":
        input_config['compress_group'] = c.get('compress_group')
      else:
        input_config['common'] = c.get('common', False)
      inputs.append(input_config)
    for c in model_config.vec_config:
      input_config = dict(
        attr_name=c['name'],tensor_name=c['name'],
        dim=c['dim'] * c.get('expand', 1) + (1 if c.get('sized', False) else 0),
      )
      if c.get('compress_group', None) and c.get('compress_group') == "USER":
        input_config['compress_group'] = c.get('compress_group')
      else:
        input_config['common'] = c.get('common', False)
      inputs.append(input_config)

    prepare_pipeline = self.prepare()
    if use_kuiba_predict_server:
      prepare_pipeline.retrieve_from_kuiba_predict_request(
        from_extra_var="KuibaCompressCommonPredictRequest",
        attr_list=list(sorted(all_attrs)),
      )

    pipeline_final = prepare_pipeline\
      .kai_extract_kuiba_feature(common_slots="common_slots", common_signs="common_parameters",
                                 item_slots="slots", item_signs="parameters",
                                 slots_mapping=slots_mapping_rename,
                                 config=model_config.parameter_config_rename) \
      .uni_predict_fused(
        embedding_fetchers = [dict(
          fetcher_type="ColossusdbEmbeddingServerFetcher",
          colossusdb_embd_model_name=colossusdb_embd_model_name,
          colossusdb_embd_service_name=colossusdb_embd_service_name,
          colossusdb_embd_table_name=colossusdb_embd_table_name,
          slots_inputs=["slots"],
          parameters_inputs=["parameters"],
          common_slots_inputs=["common_slots"],
          common_parameters_inputs=["common_parameters"],
          timeout_ms=50,
          slots_config=slots_config,
          client_side_shard=True,
          max_signs_per_request=500,
        )],
        ## 图相关配置
        graph=model_config.graph,
        queue_prefix=queue_prefix,
        key=key,
        param = model_config.param,
        inputs = inputs + [dict(attr_name=attr_name,tensor_name=tensor_name,common=common,dim=dim) for attr_name, tensor_name, common, dim in extra_inputs],
        outputs = [dict(attr_name=attr_name,tensor_name=tensor_name) for attr_name, tensor_name in model_config.outputs if attr_name in PXTRS],
        ## batching 相关配置
        # cpu 预估也支持 batching，但是由于 cpu 服务性能有限，开 batching 效果未必明显
        batching_config = dict(
          batch_timeout_micros=0,
          max_batch_size=max(batch_sizes),
          max_enqueued_batches=1,
          batch_task_type="BatchTensorflowTask",
        ),
        ## 模型加载相关的配置
        model_loader_config = dict(
          type="MioTFExecutedByTensorFlowModelLoader",
          executor_batchsizes=batch_sizes,
          implicit_batch=implicit_batch,	# cpu 预估推荐使用 implicit batch 预估
          rowmajor=True,
          receive_dnn_model_as_macro_block=receive_dnn_model_as_macro_block,
        ),
        ## 执行相关配置
        executor_config=dict(
          intra_op_parallelism_threads_num=32,
          inter_op_parallelism_threads_num=32,
        ),
      ) \
      .copy_attr(
        attrs=[
          {"from_common": "uId", "to_common": "user_id"},
          {"from_common": "dId", "to_common": "device_id"},
        ]
      ) \
      .get_abtest_params(
        biz_name = "KUAISHOU_APPS",
        user_id = "{{user_id}}",
        device_id = "{{device_id}}",
        ab_params = [
          {
            "param_name": "fountain_splash_rerank_eval_enable_return_ensemble_score",
            "param_type": "bool",
            "default_value": True,
            "attr_name": "enable_return_ensemble_score"
          },
          ("fountain_splash_rerank_eval_list_pos_weight", 0.6, "rerank_list_pos_weight"),
          ("fountain_splash_rerank_eval_list_next_weight", 0.6, "rerank_list_next_weight"),
          ("fountain_splash_rerank_eval_list_wtd_weight", 0.6, "rerank_list_wtd_weight"),
          ("fountain_splash_rerank_eval_list_finish_weight", 0.0, "rerank_list_finish_weight"),
          ("fountain_rr_play_watch_table_kconf", "reco.author.play_buck_duration"),
          ("fountain_splash_rerank_eval_wtd0_weight", 1.0),
          ("fountain_splash_rerank_eval_wtd1_weight", 1.0),
          ("fountain_splash_rerank_eval_ltr0_weight", 1.0),
          ("fountain_splash_rerank_eval_ltr1_weight", 1.0),
          ("fountain_splash_rerank_eval_next0_weight", 1.0),
          ("fountain_splash_rerank_eval_next1_weight", 1.0),
          ("fountain_splash_rerank_eval_finish0_weight", 0.0),
          ("fountain_splash_rerank_eval_finish1_weight", 0.0),
        ]
      ) \
      .if_("enable_return_ensemble_score == 1") \
        .cal_ensemble_score() \
      .end_() \
      .perflog_attr_value(check_point="find.splash.evaluator",item_attrs=PXTRS,)

    return pipeline_final
  
model_config = load_mio_tf_model()
all_attrs_list = list(sorted(all_attrs))
predict_flow = PredictServerFlow(name = "default").predict_with_mio_model(
   model_config=model_config,
   queue_prefix=model_queue_prefix,
   receive_dnn_model_as_macro_block=True,
   batch_sizes=[512],
   implicit_batch=True,
)

kuiba_predict_request_attr = []
if use_kuiba_predict_server:
  kuiba_predict_request_attr.append("KuibaCompressCommonPredictRequest")

service = LeafService(kess_name=predict_server_name,
                      item_attrs_from_request=all_attrs_list,
                      common_attrs_from_request=all_attrs_list + kuiba_predict_request_attr)
# service.ENABLE_ATTR_CHECK = False
service.AUTO_INJECT_ITEM_ATTR = False
service.AUTO_INJECT_SAMPLE_LIST_USER_ATTR = False
service.return_item_attrs([key for key in return_item_attr])

service.add_leaf_flows(leaf_flows = [predict_flow], request_type = "default")
# service.build(output_file=os.path.join(model_dir, "predict_config_use_kai_extractor_kuiba.json"))
service.build(output_file="predict_config_use_kai_extractor_kuiba.json")
json.dump(model_config.parameter_config_rename, open('parameter_config.json','w'), sort_keys=True, indent=2)