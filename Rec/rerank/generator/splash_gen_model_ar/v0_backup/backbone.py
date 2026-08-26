import base64
import collections
import json
import os
import sys

import yaml

current_dir = os.path.dirname(__file__)
#sys.path.append(os.path.join(current_dir, '../../../../../../../ks/common_reco/leaf/tools/pypi/'))
#sys.path.append(os.path.join(current_dir, '../../../../../../../ks/common_reco/leaf/tools/pypi/'))

# from tensorrt_optimizer import Optimize
from dragonfly.common_leaf_dsl import LeafService, LeafFlow
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.gsu.gsu_api_mixin import GsuApiMixin
from dragonfly.ext.cofea.cofea_api_mixin import CofeaApiMixin
from dragonfly.ext.uni_predict.uni_predict_api_mixin import UniPredictApiMixin
from dragonfly.ext.common.common_api_mixin import CommonApiMixin
from dragonfly.ext.explore_model.explore_model_api_mixin import ExploreModelApiMixin

kess_name = "grpc_splash_rerank_model_gen_ar_server"
all_model_preds = [
  "rerank_gen_score_"+str(i) for i in range(10)
]
all_model_preds.append("photo_id_emb")
all_model_preds.append("context_cascade_pctr_emb")
all_model_preds.append("logits_0")
all_model_preds.append("logits_1")
all_model_preds.append("logits_2")
all_model_preds.append("logits_3")

# load Mixins
class PredictServerFlow(LeafFlow, KuibaApiMixin, MioApiMixin, OfflineApiMixin, GsuApiMixin, CofeaApiMixin, UniPredictApiMixin, CommonApiMixin, ExploreModelApiMixin):
    def predict_with_mio_model(self, **kwargs):
        predict_server_name = kwargs.pop('predict_server_name')
        model_config = kwargs.pop('model_config')
        colossusdb_embd_model_name = kwargs.pop('colossusdb_embd_model_name')
        colossusdb_embd_table_name = kwargs.pop('colossusdb_embd_table_name')
        queue_prefix = kwargs.pop('queue_prefix')
        key = kwargs.pop('key', queue_prefix)
        receive_dnn_model_as_macro_block = kwargs.pop('receive_dnn_model_as_macro_block', True)
        # extra_inputs = kwargs.pop('extra_inputs', [])
        shards = kwargs.pop('shards', 1)
        rowmajor = kwargs.pop('rowmajor', True)
        extra_signs = kwargs.pop('extra_signs', [])
        extra_slots = kwargs.pop('extra_slots', [])
        batch_size = kwargs.pop("batch_size", [30, 60])
        slots_config, inputs, extra_inputs = [], [], []
        for c in model_config.slots_config:
            if 'dtype' in c:
                c['dtype'] = 'mio_int16'
            slots_config.append(c)
            inp = dict(attr_name=c['input_name'],
                       tensor_name=c['input_name'],
                       dim=len(str(c['slots']).split(' ')) * c['dim'] * c.get('expand', 1) + (1 if c.get('sized', False) else 0)
                       )
            if c.get('compress_group', None) and c.get('compress_group') == 'USER':
                inp['compress_group'] = c.get('compress_group')
            else:
                inp['common'] = c.get('common', False)
            inputs.append(inp)
        for c in model_config.vec_input:
            inp = dict(attr_name=c['name'],
                       tensor_name=c['name'],
                       common=c.get('common', False),
                       dim=c['dim']
                       )
            extra_inputs.append(inp)

        return self \
            .copy_item_meta_info(
              save_item_id_to_attr = "item_id",
            ) \
            .get_item_attr_by_distributed_flat_index(
              photo_store_kconf_key = "reco.distributedIndex.explorePhotoInfoCommon",
              use_dynamic_photo_store = True,
              item_id_attr = "item_id",
              attrs = [
                "photo_id",
                "duration_ms",
                "author_age_info__age_segment",
                "author__id",
                "author__gender",
                "location__province_id",
                "location__city_id",
                "hetu_tag_level_info__hetu_level_one",
                "hetu_tag_level_info__hetu_level_two",
                "hetu_tag_level_info__hetu_level_three",
                "hetu_tag_level_info__hetu_level_four",
                "hetu_tag_level_info__hetu_level_five",
                "explore_stat__click_count",
                "explore_stat__real_show_count",
                "explore_stat__like_count",
                "explore_stat__long_play_count",
                "explore_stat__short_play_count",
                "explore_stat__follow_count",
                "mod",
                "music",
                "upload_type",
              ],
            ) \
            .parse_protobuf_from_string(
              input_attr = "user_info_str",
              output_attr = "user_info",
              class_name = "ks::reco::UserInfo",
            ) \
            .enrich_with_protobuf(
              from_extra_var = "user_info",
              attrs = [
                dict(name = "user_info__id", path = "id"),
                dict(name = "user_info__active_days", path = "active_days"),
                dict(name = "user_info__basic_info__age_segment", path = "basic_info.age_segment"),
                dict(name = "user_info__location__city_id", path = "location.city_id"),
                dict(name = "user_info__location__region_type", path = "location.region_type"),
                dict(name = "user_info__client_id", path = "client_id"),
                dict(name = "user_info__device_id", path = "device_id"),
                dict(name = "user_info__gender", path = "gender"),
                dict(name = "user_info__request_location__poi_type", path = "request_location.poi_type"),
                dict(name = "user_info__request_location__province_id", path = "request_location.province_id"),
                dict(name = "user_info__request_location__city_id", path = "request_location.city_id"),
                dict(name = "user_info__user_profile__exp_stat__exp_click", path = "user_profile.exp_stat.exp_click"),
                dict(name = "user_info__user_profile__exp_stat__exp_like", path = "user_profile.exp_stat.exp_like"),
                dict(name = "user_info__user_profile__exp_stat__exp_follow", path = "user_profile.exp_stat.exp_follow"),
                dict(name = "user_info__user_profile__exp_stat__exp_realshow", path = "user_profile.exp_stat.exp_realshow"),
                dict(name = "user_info__user_profile__exp_stat__exp_long_view", path = "user_profile.exp_stat.exp_long_view"),
                dict(name = "user_info__user_profile__user_level", path = "user_profile.user_level"),
                dict(name = "user_info__realtime_click_list", path = "realtime_click_list"),
                dict(name = "user_info__realtime_follow_list", path = "realtime_follow_list"),
                dict(name = "user_info__realtime_forward_list", path = "realtime_forward_list"),
                dict(name = "user_info__realtime_like_list", path = "realtime_like_list"),
                dict(name = "user_info__fountain_reco_user_profile__like_list__author_id", path = "fountain_reco_user_profile.like_list.author_id"),
                dict(name = "user_info__fountain_reco_user_profile__like_list__photo_id", path = "fountain_reco_user_profile.like_list.photo_id"),
                dict(name = "user_info__fountain_reco_user_profile__video_play_stat__photo_id", path = "fountain_reco_user_profile.video_play_stat.photo_id"),
                dict(name = "user_info__fountain_reco_user_profile__video_play_stat__author_id", path = "fountain_reco_user_profile.video_play_stat.author_id"),
                dict(name = "user_info__fountain_reco_user_profile__video_play_stat__video_duration", path = "fountain_reco_user_profile.video_play_stat.video_duration"),
                dict(name = "user_info__fountain_reco_user_profile__video_play_stat__playing_time", path = "fountain_reco_user_profile.video_play_stat.playing_time"),
              ],
            ) \
            .explore_extract_universal_feature(
              kconf_key = "reco.explore.rpc_sample_models",
              mode = "infer",
              models = [
                "grpc_fountainFullrankDeepLtrInferKai2",
              ],
              save_common_slots_to_attr = "common_slots",
              save_common_signs_to_attr = "common_signs",
              save_item_slots_to_attr = "item_slots",
              save_item_signs_to_attr = "item_signs",
              user_info_attrs = {
                "user_id_attr": "user_info__id",
                "user_active_days_attr": "user_info__active_days",
                "user_age_segment_attr": "user_info__basic_info__age_segment",
                "user_city_id_attr": "user_info__location__city_id",
                "user_region_type_attr": "user_info__location__region_type",
                "user_client_id_attr": "user_info__client_id",
                "user_device_id_attr": "user_info__device_id",
                "user_gender_attr": "user_info__gender",
                "user_infer_gender_attr": "user_info__infer_gender",
                "user_true_gender_attr": "user_info__true_gender",
                "user_request_poi_type_attr": "user_info__request_location__poi_type",
                "user_request_province_id_attr": "user_info__request_location__province_id",
                "user_request_city_id_attr": "user_info__request_location__city_id",
                "user_visit_mod_attr": "user_info__visit_mod",
                "user_fountain_profile_like_aid_list_attr": "user_info__fountain_reco_user_profile__like_list__author_id",
                "user_fountain_profile_like_pid_list_attr": "user_info__fountain_reco_user_profile__like_list__photo_id",
                "user_fountain_profile_video_play_pid_list_attr": "user_info__fountain_reco_user_profile__video_play_stat__photo_id",
                "user_fountain_profile_video_play_aid_list_attr": "user_info__fountain_reco_user_profile__video_play_stat__author_id",
                "user_fountain_profile_video_play_duration_list_attr": "user_info__fountain_reco_user_profile__video_play_stat__video_duration",
                "user_fountain_profile_video_play_playing_time_list_attr": "user_info__fountain_reco_user_profile__video_play_stat__playing_time",
                "user_profile_exp_click_attr": "user_info__user_profile__exp_stat__exp_click",
                "user_profile_exp_like_attr": "user_info__user_profile__exp_stat__exp_like",
                "user_profile_exp_follow_attr": "user_info__user_profile__exp_stat__exp_follow",
                "user_profile_exp_realshow_attr": "user_info__user_profile__exp_stat__exp_realshow",
                "user_profile_exp_long_view_attr": "user_info__user_profile__exp_stat__exp_long_view",
                "user_profile_user_level_attr": "user_info__user_profile__user_level",
                "user_realtime_click_list_attr": "user_info__realtime_click_list",
                "user_realtime_follow_list_attr": "user_info__realtime_follow_list",
                "user_realtime_forward_list_attr": "user_info__realtime_forward_list",
                "user_realtime_like_list_attr": "user_info__realtime_like_list",
              },
              context_info_common_attrs = {
                "context_source_pid_attr": "source_pid",
                "context_source_aid_attr": "source_aid",
                "context_source_hetu_tag_level1_list_attr": "source_hetu_tag_level1_list",
                "context_source_hetu_tag_level2_list_attr": "source_hetu_tag_level2_list",
              },
              photo_info_attrs = {
                "photo_id_attr": "photo_id",
                "photo_duration_ms_attr": "duration_ms",
                "photo_author_age_segment_attr": "author_age_info__age_segment",
                "photo_author_id_attr": "author__id",
                "photo_author_gender_attr": "author__gender",
                "photo_province_id_attr": "location__province_id",
                "photo_city_id_attr": "location__city_id",
                "photo_hetu_tag_level1_list_attr": "hetu_tag_level_info__hetu_level_one",
                "photo_hetu_tag_level2_list_attr": "hetu_tag_level_info__hetu_level_two",
                "photo_hetu_tag_level5_list_attr": "hetu_tag_level_info__hetu_level_five",
                "photo_exp_click_attr": "explore_stat__click_count",
                "photo_exp_real_show_attr": "explore_stat__real_show_count",
                "photo_exp_like_attr": "explore_stat__like_count",
                "photo_exp_long_play_attr": "explore_stat__long_play_count",
                "photo_exp_short_play_attr": "explore_stat__short_play_count",
                "photo_exp_follow_attr": "explore_stat__follow_count",
                "photo_mod_attr": "mod",
                "photo_music_attr": "music",
                "photo_upload_type_attr": "upload_type",
              },
              context_info_item_attrs = {
                "context_cascade_pctr_attr": "cascade_pctr",
                "context_cascade_pltr_attr": "cascade_pltr",
                "context_cascade_plvtr_attr": "cascade_plvtr",
                "context_cascade_psvr_attr": "cascade_psvtr",
                "context_cascade_pwtr_attr": "cascade_pwtr",
                "context_pcmtr_attr": "fullrank_detail_pcmtr",
                "context_pctr_attr": "fullrank_detail_pctr",
                "context_pftr_attr": "fullrank_detail_pftr",
                "context_pltr_attr": "fullrank_detail_pltr",
                "context_plvtr_attr": "fullrank_detail_plvtr",
                "context_pptr_attr": "fullrank_detail_pptr",
                "context_psvtr_attr": "fullrank_detail_psvr",
                "context_pvtr_attr": "fullrank_detail_pvtr",
                "context_pwtd_attr": "fullrank_detail_pwtd",
                "context_pwtr_attr": "fullrank_detail_pwtr",
              },
            ) \
            .count_reco_result(save_count_to="item_num") \
            .if_("item_num >= 2") \
              .enrich_attr_by_lua(    # 转list供uni_predict使用
                  import_item_attr=[
                      "fullrank_detail_plvtr",
                      "fullrank_detail_pvtr",
                      "fullrank_detail_pwtd",
                      "fullrank_detail_pctr",
                      "fullrank_detail_pltr",
                      "fullrank_detail_pwtr",
                  ],
                  export_item_attr=[
                      "context_info__plvtr",
                      "context_info__pvtr",
                      "context_info__pwtd",
                      "context_info__pctr",
                      "context_info__pltr",
                      "context_info__pwtr",
                  ],
                  function_for_item="calculate",
                  lua_script="""
                      function calculate()
                        
                        return {fullrank_detail_plvtr or 0.0}, {fullrank_detail_pvtr or 0.0}, {fullrank_detail_pwtd or 0.0},
                              {fullrank_detail_pctr or 0.0}, {fullrank_detail_pltr or 0.0}, {fullrank_detail_pwtr or 0.0}
                      end
                    """,
              ) \
              .uni_predict_fused(
                  ## embedding 相关配置 https://docs.corp.kuaishou.com/k/home/VLKi3OBO2qik/fcAB1k2CwkZRcxQ09lTBYMRkH
                  embedding_fetchers=[dict(
                      fetcher_type="ColossusdbEmbeddingServerFetcher",
                      colossusdb_embd_model_name=colossusdb_embd_model_name,
                      colossusdb_embd_service_name=colossusdb_embd_model_name,
                      colossusdb_embd_table_name=colossusdb_embd_table_name,
                      client_side_shard=True,
                      slots_inputs=["item_slots"] + extra_slots,
                      parameters_inputs=["item_signs"] + extra_signs,
                      common_slots_inputs=["common_slots"],
                      common_parameters_inputs=["common_signs"],
                      timeout_ms=50,
                      slots_config=slots_config,
                      max_signs_per_request=600,
                  )],
                  embedding_manager_type="parallel_fetch",
                  ## 模型相关配置
                  graph=model_config.graph,
                  queue_prefix=queue_prefix,
                  key=key,
                  inputs=inputs + extra_inputs,
                  outputs=[dict(attr_name=attr_name, tensor_name=tensor_name) for attr_name, tensor_name in model_config.outputs if attr_name in all_model_preds],
                  param=model_config.param,
                  ## 模型加载设置
                  model_loader_config=dict(
                      type="MioTFExecutedByTensorFlowModelLoader",  # 使用 TF 加载模型
                      executor_batchsizes=batch_size,
                      rowmajor=rowmajor,
                      implicit_batch=True,
                      receive_dnn_model_as_macro_block=receive_dnn_model_as_macro_block,
                  ),
                  ## batching 设置
                  batching_config=dict(
                      batch_timeout_micros=0,
                      max_batch_size=max(batch_size),
                      max_enqueued_batches=1, # 关闭 batching
                      batch_task_type="BatchTensorflowTask",
                  ),
                  ## executor_config
                  executor_config=dict(
                      inter_op_parallelism_threads_num=32,
                      intra_op_parallelism_threads_num=32,
                  ),
              ) \
            .else_() \
              .log_debug_info(
                  for_debug_request_only=False,
                  # respect_sample_logging = False,
                  item_num_limit = 1,
                  item_attrs = [
                      "photo_id",
                  ]
              ) \
            .end_if_() \
            .pack_item_attr_to_item_attr(
                from_item_attrs = ["rerank_gen_score_"+str(i) for i in range(10)], #! from score
                to_item_attr = "rerank_gen_score", #! return to def rerank_gen_model_beam
                default_val = 0.0
            )\
            .log_debug_info(
                for_debug_request_only=False,
                item_num_limit = 10,
                item_attrs = [
                    # "rerank_gen_score",
                    "rerank_gen_score_"+str(i) for i in range(10)
                ] + ["logits_0", "logits_1", "logits_2", "logits_3"]
            ) \
            .log_debug_info(
                for_debug_request_only=False,
                item_num_limit = 2,
                item_attrs = [
                    "photo_id_emb",
                    "context_cascade_pctr_emb",
                    "preward",
                ]
            )

# load Resources
ModelConfig = collections.namedtuple(
    'ModelConfig',
    ['graph', 'outputs', 'slots_config', 'vec_input', 'param', 'common_parameter_config_rename', 'non_common_parameter_config_rename']
)
all_attrs = set()
all_features = {}


def load_mio_tf_model(model_dir):
    with open(os.path.join(model_dir, 'dnn_model.yaml')) as f:
        dnn_model = yaml.load(f, Loader=yaml.SafeLoader)

    with open(os.path.join(model_dir, 'graph.pb'), 'rb') as f:
        base64_graph = base64.b64encode(f.read()).decode('ascii')
        graph = 'base64://' + base64_graph

    graph_tensor_mapping = dnn_model['graph_tensor_mapping']
    extra_preds = dnn_model['extra_preds'].split(' ')
    q_names = dnn_model['q_names'].split(' ')
    assert len(extra_preds) == len(q_names)
    outputs = [(extra_pred, graph_tensor_mapping[q_name]) for extra_pred, q_name in zip(extra_preds, q_names)]
    param = [param for param in dnn_model['param'] if param.get('send_to_online', True)]

    slots_config = dnn_model['embedding']['slots_config']
    vec_input = dnn_model['vec_input']

    global all_attrs
    global all_features

    common_parameter_config_rename = dict()
    non_common_parameter_config_rename = dict()

    for slot_config in slots_config:
        input_name = slot_config["input_name"]
        all_features[input_name] = 1
        print(slot_config)

    return ModelConfig(graph, outputs, slots_config, vec_input, param, common_parameter_config_rename, non_common_parameter_config_rename)


all_attrs_list = list(sorted(all_attrs))
service = LeafService(kess_name=kess_name)

service.AUTO_INJECT_ITEM_ATTR = False
service.AUTO_INJECT_SAMPLE_LIST_USER_ATTR = False
service.return_item_attrs(all_model_preds)
