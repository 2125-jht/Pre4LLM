import os
import sys
import json
import yaml
import argparse
import base64
import collections
import uuid

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, '../../../../dragon/tools/pypi/'))

from dragonfly.common_leaf_dsl import LeafService, LeafFlow
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin


use_kuiba_predict_server=True
model_name = "splash_rerank"
predict_use_batching = False
predict_use_gpu = False
predict_limit = 1200
predict_server_name = "grpc_fountainKaiSplashMixRerankPredictServer";
model_queue_prefix="fountain_rerank_mix";
embedding_shard_num=4
embedding_kess_name = model_queue_prefix + "_embedding_server"
embedding_kess_name = "grpc_fountainKaiMixRerankEmbeddingServer"
PXTRS = ["splash_pos" + str(i) for i in range(2)]
PXTRS_ATTRS_MAPPING = {
}
for key in PXTRS:
  if not (key in PXTRS_ATTRS_MAPPING.keys()):
    PXTRS_ATTRS_MAPPING[key] = key
output_pxtrs_as_embedding = False
return_item_attr = PXTRS

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--run', default=None, help='run the pipeline directly')
args = parser.parse_args()

model_dir = os.path.join(current_dir, 'models', model_name)
all_attrs = set()
all_features = {}

sampling_lua_positive_rate = {}
sampling_lua_negative_rate = {}

# 线上特征为了避免和 attr 冲突, 统一加上 "KAI_" 前缀

# 将 kuiba feature name 到 slot id 的映射重命名
def LoadSlotMappingRename():
  global model_name
  filename = "./models/" + model_name + "/kai_kuiba_config.json"
  slots_mapping=json.load(open(filename))["sign_feature_slot"]
  slots_mapping_rename = {}
  for name, slot_id in slots_mapping.items():
    if name.find('_idx') != -1:
      n = (int)(name.split('_idx')[-1])
      if n >= 2 : continue
    slots_mapping_rename["KAI_" + name] = slot_id
  return slots_mapping_rename

slots_mapping_rename = LoadSlotMappingRename()

def LoadKuibaSamplingRate(filename):
  loss_functions = json.load(open(filename))["krp_ps_server"]["network"]["loss_functions"]
  for loss_name in loss_functions:
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
    if name.find('_idx') != -1:
      n = (int)(name.split('_idx')[-1])
      if n >= 2 : continue
    if isinstance(feature_config, dict):
      if "use_common_attr_only" in feature_config:
        if feature_config["use_common_attr_only"] == True:
          feature_config["attrs"][0]["is_common"] = True
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
class PredictServerFlow(LeafFlow, KuibaApiMixin, MioApiMixin, OfflineApiMixin):
  def prepare(self):
    return self \
      .deduplicate() \
      .limit(predict_limit) \

  def predict_with_mio_model(self, **kwargs):
    model_config = kwargs.pop('model_config')
    embedding_kess_name = kwargs.pop('embedding_kess_name')
    queue_prefix = kwargs.pop('queue_prefix')
    key = kwargs.pop('key', queue_prefix)
    receive_dnn_model_as_macro_block = kwargs.pop('receive_dnn_model_as_macro_block', False)
    extra_inputs = kwargs.pop('extra_inputs', [])

    prepare_pipeline = self.prepare();
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
      .fetch_mio_embedding(kess_service=embedding_kess_name,  # 
                           shards=embedding_shard_num,
                           timeout_ms=50,
                           slots_inputs=["slots"],
                           parameters_inputs=["parameters"],
                           common_slots_inputs=["common_slots"],
                           common_parameters_inputs=["common_parameters"],
                           slots_config=model_config.slots_config,
                           client_side_shard=True,
                           max_signs_per_request=4096,
                           direct_write=not predict_use_batching,
                           direct_copy_to_pinned_memory=predict_use_gpu,
                           save_result_as_tensor_output=True) \
      .mio_predict(graph=model_config.graph,
                   tensorflow_use_batching=predict_use_batching,
                   queue_prefix=queue_prefix,
                   key=key,
                   inputs=[dict(
                     attr_name=c['name'],
                     tensor_name=c['name'],
                     common=c.get('common', False),
                     dim=c['dim'],
                   ) for c in model_config.vec_config]
                   + [dict(
                     attr_name=c['input_name'],
                     tensor_name=c['input_name'],
                     common=c.get('common', False),
                     dim=len(str(c['slots']).split(' ')) * c['dim'] * c.get('expand', 1) + (1 if c.get('sized', False) else 0),
                   ) for c in model_config.slots_config]
                   + [dict(
                     attr_name=attr_name,
                     tensor_name=tensor_name,
                     common=common,
                     dim=dim,
                   ) for attr_name, tensor_name, common, dim in extra_inputs],
                   outputs = [dict(
                     attr_name=attr_name,
                     tensor_name=tensor_name,
                   ) for attr_name, tensor_name in model_config.outputs if attr_name in PXTRS],
                   flatten_outputs=not output_pxtrs_as_embedding,
                   param = model_config.param,
                   receive_dnn_model_as_macro_block=receive_dnn_model_as_macro_block,
                   rowmajor=True,
                   read_input_from_extra_var=True) \

    for loss_name in sampling_lua_positive_rate:
      positive_rate = float(sampling_lua_positive_rate[loss_name])
      negative_rate = float(sampling_lua_negative_rate[loss_name])
      lua_script="function calculate() \n"
      lua_script += "  return (%f * %s)/(%f * %s + (1 - %s) * %f)" %(negative_rate, loss_name, positive_rate, loss_name, loss_name, negative_rate)
      lua_script += "\nend"
      print (lua_script)
      pipeline_final = pipeline_final\
          .enrich_attr_by_lua(
            import_item_attr=[loss_name],
            function_for_item='calculate',
            export_item_attr=[loss_name],
            lua_script=lua_script
          )
    pipeline_final = pipeline_final\
      .perflog_attr_value(check_point="nearby.live.fullrank",
                          item_attrs=[key for key in PXTRS])
    return pipeline_final

# load Resources
ModelConfig = collections.namedtuple('ModelConfig', ['graph', 'outputs', 'slots_config', 'vec_config', 'param', 'parameter_config_rename', 'common_parameter_config_rename', 'non_common_parameter_config_rename'])

def load_mio_tf_model():
  global model_name
  global model_dir

  with open(os.path.join(model_dir, 'dnn_model.yaml')) as f:
    dnn_model = yaml.load(f, Loader=yaml.SafeLoader)

  with open(os.path.join(model_dir, 'graph.pb'), 'rb') as f:
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
    if input_name.find('_idx') != -1:
      n = (int)(input_name.split('_idx')[-1])
      if n >= 2 : continue
    all_features[input_name] = 1;
    new_slot_config.append(slot)
  slots_config = new_slot_config
  new_vec_config = []
  for vec in vec_config:
    input_name = vec["name"]
    if input_name.find('_idx') != -1:
      n = (int)(input_name.split('_idx')[-1])
      if n >= 2 : continue
    all_features[input_name] = 1;
    new_vec_config.append(vec)
  vec_config = new_vec_config
  common_parameter_config_rename = {}
  non_common_parameter_config_rename = {}

  parameter_config_rename = LoadKuibaParameterConfigRename(os.path.join(model_dir, 'dynamic_json_config.json'), all_features)
  print (parameter_config_rename)

  LoadKuibaSamplingRate(os.path.join(model_dir, 'dynamic_json_config.json'))

  # 筛选出线上用的 slot
  slots_mapping_for_predict = {}
  global slots_mapping_rename
  for name,slot_id in slots_mapping_rename.items():
    if name.find('_idx') != -1:
      n = (int)(name.split('_idx')[-1])
      if n >= 2 : continue
    if name in parameter_config_rename.keys():
      slots_mapping_for_predict[name] = slot_id
      print(name, slot_id, parameter_config_rename[name])
  slots_mapping_rename = slots_mapping_for_predict

  for name, feature_config in parameter_config_rename.items():
    if name.find('_idx') != -1:
      n = (int)(name.split('_idx')[-1])
      if n >= 2 : continue
    if not isinstance(feature_config, dict):
      continue
    if ('use_common_attr_only' in feature_config.keys()) and (feature_config['use_common_attr_only']):
      common_parameter_config_rename[name] = feature_config
    else:
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
    if slot_config["input_name"] in common_parameter_config_rename.keys():
      if predict_use_batching:
        slot_config["common"] = True
      print (slot_config)

  return ModelConfig(graph, outputs, slots_config, vec_config, param, parameter_config_rename, common_parameter_config_rename, non_common_parameter_config_rename);

# load model config first so that we have all_attrs
model_config = load_mio_tf_model();
all_attrs_list = list(sorted(all_attrs))
# define the pipeline

predict_pipeline = PredictServerFlow(name = "default") \
  .predict_with_mio_model(
    model_config=model_config,
    embedding_kess_name=embedding_kess_name,
    queue_prefix=model_queue_prefix,
    receive_dnn_model_as_macro_block=True)

kuiba_predict_request_attr = []
if use_kuiba_predict_server:
  kuiba_predict_request_attr.append("KuibaCompressCommonPredictRequest")
service = LeafService(kess_name=predict_server_name,
                      item_attrs_from_request=all_attrs_list,
                      common_attrs_from_request=all_attrs_list + kuiba_predict_request_attr)

service.AUTO_INJECT_ITEM_ATTR = False
service.AUTO_INJECT_SAMPLE_LIST_USER_ATTR = False
service.return_item_attrs([key for key in return_item_attr])

service.add_leaf_flows(leaf_flows = [predict_pipeline], request_type = "default")
service.build(output_file=os.path.join(model_dir, "predict_config_use_kai_extractor.json"))
json.dump(model_config.parameter_config_rename, open('parameter_config.json','w'), sort_keys=True, indent=2)
