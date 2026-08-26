import os
import sys
import json
import yaml
import argparse
import base64
import re

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, '../../../../../dragon/tools/pypi/'))

from dragonfly.common_leaf_dsl import LeafFlow, OfflineRunner
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin

parser = argparse.ArgumentParser()
parser.add_argument('--run', dest="run", default=False, action='store_true')
parser.add_argument('--from_kuiba_conf', default='./dynamic_json_config.json')
parser.add_argument('--from_kai_kuiba_conf', default='./kai_kuiba_config.json')
parser.add_argument('-s', '--src', type=str, default='hdfs-kafka', help='sample source: hdfs/kafka/btq/local')
parser.add_argument('-o', '--output', type=str, default='./conf/', help='directory of generated pipeline.json')
parser.add_argument('-m', '--mode', type=str, default='train', help='train/eval')
kafka_topic="reco_hot_context_rank_joint_log"
#hdfs_path="viewfs:///home/reco_5/mpi/panshangkao/gamora_tab_v4/sample_batch_data/2021-09-01/15"
hdfs_path=""
args = parser.parse_args()

kai_kuiba_config_filename = args.from_kai_kuiba_conf
kuiba_config_filename = args.from_kuiba_conf

# 这个最好 kml 做渲染
# kaiworks 中需要在配置首页配置
reader_name="mio_kai_splash_rerank_evaluator_mtl_v1"
# reader_name="mio_fountain_rerank_mix"

# 这个 attr 在 dragonfly 里面是必须的, 用来给 item 一个 id，起到一个过滤功能
item_id_attr_name=""
# 用于特征算分里面，表示哪个算click
label_atttr_name="slide_l2r_6_sample_flag"

current_dir = os.path.dirname(__file__)

# 将 kuiba feature name 到 slot id 的映射重命名
def LoadSlotMappingRename(filename):
  slots_mapping=json.load(open(filename))["sign_feature_slot"]
  slots_mapping_rename = {}
  for name, slot_id in slots_mapping.items():
    slots_mapping_rename["KAI_" + name] = slot_id
  return slots_mapping_rename

slots_mapping_rename = LoadSlotMappingRename(kai_kuiba_config_filename)
print("--------------------------------------------------")
print(slots_mapping_rename)
print("--------------------------------------------------")

# 将 kuiba 抽特征的 config 线上需要的部分提取出来
# 需要的签名由 features_set 决定, features_set 是分析 dnn_model.yaml 倒推出来的
def LoadKuibaParameterConfigRename(filename, dense_attrs_list):
  global slots_mapping_rename
  parameter_config = json.load(open(filename))["krp_ps_server"]["network"]["parameters"]
  parameter_config_rename = {}
  for name, feature_config in parameter_config.items():
    if isinstance(feature_config, dict):
      kai_name = "KAI_" + name
      if kai_name in slots_mapping_rename.keys():
        slot = slots_mapping_rename[kai_name]
        for converter in feature_config["attrs"]:
          converter["mio_slot_key_type"] = slot
        parameter_config_rename[kai_name] = feature_config
      else:
        parameter_config_rename[kai_name] = feature_config
        dense_attrs_list.append(kai_name)
        # 非 slot 类型的特征也需要抽取(dense特征)
        slots_mapping_rename[kai_name] = -1
    else:
      parameter_config_rename[name] = feature_config
  return parameter_config_rename

# parse dynamic_json_config.json
auc_uid = ""           # "userId"
labels = []            # [dict(label="China,push_mix_all,show", attr="show")]
loss_attrs = []        # ctr_sample_flag, ctr_label, ctr_label_value
loss_weight_attrs = [] # 用于给样本配置权重的 attr
kuiba_conf = json.load(open(kuiba_config_filename, 'r'))
loss_functions = kuiba_conf["krp_ps_server"]["network"]["loss_functions"]
kuiba_loss_functions = {}
kuiba_loss_function_sample_filter = {}
labels_mapping={}
for loss_name in loss_functions:
  loss_attrs.append(loss_name + "_sample_flag")
  loss_attrs.append(loss_name + "_label")
  loss_attrs.append(loss_name + "_label_value")
  loss_function = loss_functions[loss_name]
  if not isinstance(loss_function, dict):
    continue
  if "weight" in loss_function.keys():
    if loss_function["weight"] != "":
      loss_weight_attrs.append(loss_function["weight"])
  # auc_uid
  if "auc_uid" in loss_function:
    if auc_uid == "":
      auc_uid = loss_function["auc_uid"]
    else:
      assert(auc_uid == loss_function["auc_uid"])  # auc_uid must be the same
  loss_labels = loss_function["labels"]
  kuiba_loss_functions[loss_name] = {}
  for loss_label in loss_labels:
    kuiba_loss_functions[loss_name][loss_label] = loss_labels[loss_label]
    label_key = "LABEL_ATTR_" + re.sub(r'[^a-zA-Z0-9]', "_", loss_label);
    labels_mapping[loss_label] = label_key
  if "filter" in loss_function.keys():
    kuiba_loss_function_sample_filter[loss_name] = loss_function["filter"];

for loss_label,label_key in labels_mapping.items():
  labels.append(dict(label=loss_label, attr=label_key))

dense_attrs_list = []    # item_attr from const parameter
parameter_config_rename = LoadKuibaParameterConfigRename(kuiba_config_filename, dense_attrs_list)
print("----------dense_attrs_list------------------------")
print(dense_attrs_list)
print("--------------------------------------------------")


# load plugins
class DataReaderFlow(LeafFlow, MioApiMixin, KuibaApiMixin, OfflineApiMixin):
  def clean_all(self, reason):
    return self.limit(0, name="clean_all_for_" + reason)

# define the pipeline
logical_flow = DataReaderFlow(name = "default")
logical_flow \
  .fetch_message(
    hdfs_path=hdfs_path,
    group_id=reader_name,
    kafka_topic=kafka_topic,
    output_attr="raw_sample_package_str") \
  .parse_protobuf_from_string(
    input_attr="raw_sample_package_str",
    output_attr="raw_sample_package",
    class_name="kuiba::RawSamplePackage") \
  .retrieve_from_raw_sample_package(
    use_sub_biz=True,
    from_extra_var="raw_sample_package",
    labels=labels,
    compatible_labels=[],
    ignore_label_and_value_unmatch=True,
    kuiba_loss_functions=kuiba_loss_functions,
    kuiba_loss_function_sample_filter=kuiba_loss_function_sample_filter,
    save_locale_to="",
    save_channel_to="",
    save_common_attr_names_to="common_attrs",
    save_item_attr_names_to="item_attrs",
    device_id_attr_name="dId",
    pid_attr_name=item_id_attr_name,
    uid_attr_name="uId") \
  .kai_extract_kuiba_feature(common_slots="common_slots", common_signs="common_signs",
                             item_slots="item_slots", item_signs="item_signs",
                             slots_mapping=slots_mapping_rename,
                             config=parameter_config_rename) 

runner = OfflineRunner("push_feature_pipeline_demo")
if args.run:
  runner.ENABLE_ATTR_CHECK = False
else:
  logical_flow.send_to_mio_learner(
    attrs = loss_weight_attrs + loss_attrs + dense_attrs_list,
    user_hash_attr=auc_uid,
    pid_attr=item_id_attr_name,
    label_attr=label_atttr_name,
    slots_attrs=["common_slots", "item_slots"],
    signs_attrs=["common_signs", "item_signs"]
  )

runner.IGNORE_UNUSED_ATTR = ["common_attrs", "item_attrs", "item_slots", "item_signs", "common_slots", "common_signs"]
runner.add_leaf_flows(leaf_flows = [logical_flow])

print(args.run)
if args.run:
  print("begin")
  exe = runner.executor()
  for i in range(1000):
    exe.reset()
    exe.run("default")

    if not exe.items:
      continue

    print ("common_slots", exe["common_slots"])
    print ("common_signs", exe["common_signs"])
    for item in exe.items:
      print("item_key: ", item.item_key)
      for key in (["item_slots", "item_signs"] + loss_attrs + dense_attrs_list + loss_weight_attrs):
        print(key, item[key])
else:
  output_file_name = "pipeline.json" if args.mode == 'train' else "eval_pipeline.json"
  output_file = os.path.join(args.output, output_file_name)
  with open(output_file, "w") as f:
      runner.build(output_file=output_file)
