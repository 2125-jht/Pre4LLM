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

kess_name = "grpc_fountain_rerank_model_flash_evaluator_server"
all_model_preds = [
  "pctr",
  "pwtd",
  "pvtr",
  "pltr",
  # "pslide",
  "context_pctr",
  "context_pwtd",
  "context_pltr",
  "context_pcmtr",
  "context_pwtr",
  "context_plvtr",
  "context_psvtr",
  "context_pftr",
  "duration_s",
  # "pwtd_gather_list_print",
  "pltr_gather_list_print",
  "rerank_list_item_idx_flat_list_print",
  # "context_lv_count",
  # "context_sv_count",
]
return_common_attrs = [
  "rerank_list_item_idx_flat_list_padded",
  "eval_list_scores",
]

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
        batch_size = kwargs.pop("batch_size", [60])
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
            .copy_user_meta_info(
                save_request_time_to_attr = "request_time",   # ms int
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
                "explore_stat__view_length_sum",
                "fountain_stats__real_show_count",
                "fountain_stats__like_count",
                "fountain_stats__long_play_count",
                "fountain_stats__short_play_count",
                "fountain_stats__follow_count",
                "fountain_stats__view_length_sum",
                "mod",
                "tag",
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
                dict(name = "user_info__infer_gender", path = "infer_gender"),
                dict(name = "user_info__true_gender", path = "true_gender"),
                dict(name = "user_info__visit_mod", path = "visit_mod"),
                dict(name = "user_info__follow_count", path = "follow_count", skip_unset_field = True),
                dict(name = "user_info__upload_count", path = "upload_count", skip_unset_field = True),
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
                dict(name = "user_info__fountain_reco_user_profile__click_list__author_id", path = "fountain_reco_user_profile.click_list.author_id"),
                dict(name = "user_info__fountain_reco_user_profile__click_list__photo_id", path = "fountain_reco_user_profile.click_list.photo_id"),
                dict(name = "user_info__fountain_reco_user_profile__comment_list__author_id", path = "fountain_reco_user_profile.comment_list.author_id"),
                dict(name = "user_info__fountain_reco_user_profile__comment_list__photo_id", path = "fountain_reco_user_profile.comment_list.photo_id"),
                dict(name = "user_info__fountain_reco_user_profile__follow_list__author_id", path = "fountain_reco_user_profile.follow_list.author_id"),
                dict(name = "user_info__fountain_reco_user_profile__follow_list__photo_id", path = "fountain_reco_user_profile.follow_list.photo_id"),
                dict(name = "user_info__fountain_reco_user_profile__like_list__author_id", path = "fountain_reco_user_profile.like_list.author_id"),
                dict(name = "user_info__fountain_reco_user_profile__like_list__photo_id", path = "fountain_reco_user_profile.like_list.photo_id"),
                dict(name = "user_info__fountain_reco_user_profile__video_play_stat__photo_id", path = "fountain_reco_user_profile.video_play_stat.photo_id"),
                dict(name = "user_info__fountain_reco_user_profile__video_play_stat__author_id", path = "fountain_reco_user_profile.video_play_stat.author_id"),
                dict(name = "user_info__fountain_reco_user_profile__video_play_stat__video_duration", path = "fountain_reco_user_profile.video_play_stat.video_duration"),
                dict(name = "user_info__fountain_reco_user_profile__video_play_stat__playing_time", path = "fountain_reco_user_profile.video_play_stat.playing_time"),
                dict(name = "user_info__fountain_reco_user_profile__video_play_stat__client_timestamp", path = "fountain_reco_user_profile.video_play_stat.client_timestamp"),
                dict(name = "user_info__user_profile_v1__click_list__author_id", path = "user_profile_v1.click_list.author_id"),
                dict(name = "user_info__user_profile_v1__click_list__photo_id", path = "user_profile_v1.click_list.photo_id"),
                dict(name = "user_info__user_profile_v1__follow_list__author_id", path = "user_profile_v1.follow_list.author_id"),
                dict(name = "user_info__user_profile_v1__follow_list__photo_id", path = "user_profile_v1.follow_list.photo_id"),
                dict(name = "user_info__user_profile_v1__like_list__author_id", path = "user_profile_v1.like_list.author_id"),
                dict(name = "user_info__user_profile_v1__like_list__photo_id", path = "user_profile_v1.like_list.photo_id"),
                dict(name = "user_info__user_profile_v1__video_playing_stat__playing_time", path = "user_profile_v1.video_playing_stat.playing_time"),
                dict(name = "user_info__user_profile_v1__video_playing_stat__author_id", path = "user_profile_v1.video_playing_stat.author_id"),
                dict(name = "user_info__user_profile_v1__video_playing_stat__photo_id", path = "user_profile_v1.video_playing_stat.photo_id"),
                # timestamp在样本流中未加入 如果有特征依赖本字段，需要样本流也增加
                dict(name = "user_info__user_profile_v1__video_playing_stat__client_timestamp", path = "user_profile_v1.video_playing_stat.client_timestamp"),
              ],
            ) \
            .copy_attr(
              attrs=[
                {"from_common": "user_info__id", "to_common": "user_id"},
                {"from_common": "user_info__device_id", "to_common": "device_id"},
                {"from_common": "origin_rerank_list_item_idx_flat_list", "to_common": "rerank_list_item_idx_flat_list"},
              ]
            ) \
            .get_abtest_params(
              biz_name = "KUAISHOU_APPS",
              user_id = "{{user_id}}",
              device_id = "{{device_id}}",
              ab_params = [
                ("fountain_rerank_eval_list_enable_position_decay", True, "enable_position_decay"),
                ("fountain_rerank_eval_list_enable_use_multiply", False, "use_multiply"),
                ("fountain_rerank_eval_list_pctr_weight", 1.0, "pctr_weight"),
                ("fountain_rerank_eval_list_pwtd_weight", 1.0, "pwtd_weight"),
                ("fountain_rerank_eval_list_pvtr_weight", 1.0, "pvtr_weight"),
                ("fountain_rerank_eval_list_pltr_weight", 1.0, "pltr_weight"),
                ("fountain_rerank_eval_list_pslide_weight", 0.0, "pslide_weight"),
                ("fountain_rerank_eval_list_context_pctr_weight", 4.5, "context_pctr_weight"),
                ("fountain_rerank_eval_list_context_plvtr_weight", 0.0, "context_plvtr_weight"),
                ("fountain_rerank_eval_list_context_psvtr_weight", 0.0, "context_psvtr_weight"),
                ("fountain_rerank_eval_list_context_pwtd_weight", 0.0, "context_pwtd_weight"),
                ("fountain_rerank_eval_list_context_pltr_weight", 0.7, "context_pltr_weight"),
                ("fountain_rerank_eval_list_context_pwtr_weight", 1.0, "context_pwtr_weight"),
                ("fountain_rerank_eval_list_context_pcmtr_weight", 0.7, "context_pcmtr_weight"),
                ("fountain_rerank_eval_list_context_pftr_weight", 0.0, "context_pftr_weight"),
              ]
            ) \
            .pack_item_attr_to_item_attr(
              from_item_attrs=["fullrank_detail_new_pevtr_v2"],
              to_item_attr="context_info__pevtr_v2",
              default_val=0.0
            ) \
            .pack_item_attr_to_item_attr(
              from_item_attrs=["fullrank_detail_pwtd"],
              to_item_attr="context_info__pwtd",
              default_val=0.0
            ) \
            .set_attr_value(
              common_attrs=[
                {
                  "name": "rerank_list_item_idx_flat_list_max_size",
                  "type": "int",
                  "value": 180
                },
                {
                  "name": "rerank_list_item_size",
                  "type": "int",
                  "value": 6
                },
              ]
            ) \
            .enrich_attr_by_lua(
              name='rerank_list_padding_default_value',
              import_common_attr=["rerank_list_item_idx_flat_list_max_size", "rerank_list_item_idx_flat_list", "rerank_list_item_size"],
              function_for_common="calculate",
              export_common_attr=["rerank_list_item_idx_flat_list_padded", "rerank_list_true_num"],
              lua_script="""
                function calculate()
                  if rerank_list_item_idx_flat_list == nil or #rerank_list_item_idx_flat_list % rerank_list_item_size ~= 0 then
                    error("rerank_list_item_idx_flat_list为空或者长度不能整除 rerank_list_item_size")
                  end
                  local rerank_list_true_num = #rerank_list_item_idx_flat_list // rerank_list_item_size
                  local padded = {}
                  for i, v in ipairs(rerank_list_item_idx_flat_list) do
                    padded[i] = v
                  end
                  for i = #rerank_list_item_idx_flat_list + 1, rerank_list_item_idx_flat_list_max_size do
                    padded[i] = -1
                  end
                  return padded, rerank_list_true_num
                end
              """
            ) \
            .cast_attr_type(
              attr_type_cast_configs=[
                {
                  "to_type": "double",
                  "from_item_attr": "duration_ms",
                  "to_item_attr": "duration_ms_double"
                },
                {
                  "to_type": "double",
                  "from_common_attr": "rerank_list_item_idx_flat_list_padded",
                  "to_common_attr": "rerank_list_item_idx_flat_list_double"
                },
                {
                  "to_type": "double",
                  "from_item_attr": "explore_stat__click_count",
                  "to_item_attr": "explore_stat__click_count_double"
                },
                {
                  "to_type": "double",
                  "from_item_attr": "explore_stat__real_show_count",
                  "to_item_attr": "explore_stat__real_show_count_double"
                },
                {
                  "to_type": "double",
                  "from_item_attr": "explore_stat__like_count",
                  "to_item_attr": "explore_stat__like_count_double"
                },
                {
                  "to_type": "double",
                  "from_item_attr": "explore_stat__long_play_count",
                  "to_item_attr": "explore_stat__long_play_count_double"
                },
                {
                  "to_type": "double",
                  "from_item_attr": "explore_stat__short_play_count",
                  "to_item_attr": "explore_stat__short_play_count_double"
                },
                {
                  "to_type": "double",
                  "from_item_attr": "explore_stat__follow_count",
                  "to_item_attr": "explore_stat__follow_count_double"
                },
                {
                  "to_type": "double",
                  "from_item_attr": "explore_stat__view_length_sum",
                  "to_item_attr": "explore_stat__view_length_sum_double"
                },
                {
                  "to_type": "double",
                  "from_item_attr": "fountain_stats__real_show_count",
                  "to_item_attr": "fountain_stats__real_show_count_double"
                },
                {
                  "to_type": "double",
                  "from_item_attr": "fountain_stats__like_count",
                  "to_item_attr": "fountain_stats__like_count_double"
                },
                {
                  "to_type": "double",
                  "from_item_attr": "fountain_stats__long_play_count",
                  "to_item_attr": "fountain_stats__long_play_count_double"
                },
                {
                  "to_type": "double",
                  "from_item_attr": "fountain_stats__short_play_count",
                  "to_item_attr": "fountain_stats__short_play_count_double"
                },
                {
                  "to_type": "double",
                  "from_item_attr": "fountain_stats__follow_count",
                  "to_item_attr": "fountain_stats__follow_count_double"
                },
                {
                  "to_type": "double",
                  "from_item_attr": "fountain_stats__view_length_sum",
                  "to_item_attr": "fountain_stats__view_length_sum_double"
                },
              ]
            ) \
            .explore_extract_universal_feature(
              kconf_key = "reco.explore.fl_sample_models",
              mode = "infer",
              models = [
                "grpc_splash_rerank_model_gen_ar_server",
              ],
              # slot_dup_config = [
              #   { "slot": 1016, "dup_to": [4000] },
              # ],
              slot_share_config = [
                [61, 64],
                [34, 40, 55, 59],
                [33, 39, 56, 60]
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
                "user_follow_count_attr": "user_info__follow_count",
                "user_upload_count_attr": "user_info__upload_count",
                "user_profile_exp_click_attr": "user_info__user_profile__exp_stat__exp_click",
                "user_profile_exp_like_attr": "user_info__user_profile__exp_stat__exp_like",
                "user_profile_exp_follow_attr": "user_info__user_profile__exp_stat__exp_follow",
                "user_profile_exp_realshow_attr": "user_info__user_profile__exp_stat__exp_realshow",
                "user_profile_exp_long_view_attr": "user_info__user_profile__exp_stat__exp_long_view",
                "user_profile_user_level_attr": "user_info__user_profile__user_level",
                "user_fountain_profile_click_aid_list_attr": "user_info__fountain_reco_user_profile__click_list__author_id",
                "user_fountain_profile_click_pid_list_attr": "user_info__fountain_reco_user_profile__click_list__photo_id",
                "user_fountain_profile_comment_aid_list_attr": "user_info__fountain_reco_user_profile__comment_list__author_id",
                "user_fountain_profile_comment_pid_list_attr": "user_info__fountain_reco_user_profile__comment_list__photo_id",
                "user_fountain_profile_follow_aid_list_attr": "user_info__fountain_reco_user_profile__follow_list__author_id",
                "user_fountain_profile_follow_pid_list_attr": "user_info__fountain_reco_user_profile__follow_list__photo_id",
                "user_fountain_profile_like_aid_list_attr": "user_info__fountain_reco_user_profile__like_list__author_id",
                "user_fountain_profile_like_pid_list_attr": "user_info__fountain_reco_user_profile__like_list__photo_id",
                "user_fountain_profile_video_play_pid_list_attr": "user_info__fountain_reco_user_profile__video_play_stat__photo_id",
                "user_fountain_profile_video_play_aid_list_attr": "user_info__fountain_reco_user_profile__video_play_stat__author_id",
                "user_fountain_profile_video_play_duration_list_attr": "user_info__fountain_reco_user_profile__video_play_stat__video_duration",
                "user_fountain_profile_video_play_playing_time_list_attr": "user_info__fountain_reco_user_profile__video_play_stat__playing_time",
                "user_profile_v1_click_aid_list_attr": "user_info__user_profile_v1__click_list__author_id",
                "user_profile_v1_click_pid_list_attr": "user_info__user_profile_v1__click_list__photo_id",
                "user_profile_v1_follow_aid_list_attr": "user_info__user_profile_v1__follow_list__author_id",
                "user_profile_v1_follow_pid_list_attr": "user_info__user_profile_v1__follow_list__photo_id",
                "user_profile_v1_like_aid_list_attr": "user_info__user_profile_v1__like_list__author_id",
                "user_profile_v1_like_pid_list_attr": "user_info__user_profile_v1__like_list__photo_id",
                "user_profile_v1_video_playing_time_list_attr": "user_info__user_profile_v1__video_playing_stat__playing_time",
                "user_profile_v1_video_playing_aid_list_attr": "user_info__user_profile_v1__video_playing_stat__author_id",
                "user_profile_v1_video_playing_pid_list_attr": "user_info__user_profile_v1__video_playing_stat__photo_id",
                "user_realtime_click_list_attr": "user_info__realtime_click_list",
                "user_realtime_follow_list_attr": "user_info__realtime_follow_list",
                "user_realtime_forward_list_attr": "user_info__realtime_forward_list",
                "user_realtime_like_list_attr": "user_info__realtime_like_list",
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
                "photo_hetu_tag_level3_list_attr": "hetu_tag_level_info__hetu_level_three",
                "photo_hetu_tag_level4_list_attr": "hetu_tag_level_info__hetu_level_four",
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
                "context_cascade_plvtr_attr": "cascade_plvtr",
                "context_cascade_pwtr_attr": "cascade_pwtr",
                "context_cascade_pltr_attr": "cascade_pltr",
                "context_cascade_pftr_attr": "cascade_pftr",
                "context_cascade_pptr_attr": "cascade_ptr",
                "context_cascade_pcmtr_attr": "cascade_pcmtr",
                "context_pcmtr_attr": "fullrank_detail_pcmtr",
                "context_pctr_attr": "fullrank_detail_pctr",
                "context_pftr_attr": "fullrank_detail_pftr",
                "context_pltr_attr": "fullrank_detail_pltr",
                "context_plvtr_attr": "fullrank_detail_plvtr",
                "context_pptr_attr": "fullrank_detail_pptr",
                "context_pvtr_attr": "fullrank_detail_pvtr",
                "context_pwtr_attr": "fullrank_detail_pwtr",
                "context_pwtd_attr": "fullrank_detail_pwtd",
                "context_pepstr_attr": "fullrank_sim_pepstr",
                "context_pcpr_attr": "fullrank_sim_pcpr",
                "context_pcltr_attr": "fullrank_sim_pcltr",
                "context_psvr_attr": "fullrank_sim_psvr",
                "context_fullrank_ltr_score_attr": "fullrank_ltr_score",
                "context_fullrank_act_wtd_attr": "fullrank_act_wtd",
                "context_fullrank_ltr_v4_fountain_next_attr": "fullrank_ltr_v4_fountain_next",
                "context_fountain_related_score_attr": "fountain_related_score_v2",
              },
              context_info_common_attrs = {
                "context_source_pid_attr": "source_pid",
                "context_source_aid_attr": "source_aid",
                "context_source_duration_ms_attr": "source_duration_ms",
                "context_source_tag_attr": "source_tag",
                "context_source_hetu_tag_level1_list_attr": "source_hetu_tag_level1_list",
                "context_source_hetu_tag_level2_list_attr": "source_hetu_tag_level2_list",
                "context_page_attr": "page",
                "context_similar_user_list_attr": "similar_user_list",
                "context_time_attr": "request_time",
              },
            ) \
            .pack_item_attr_to_item_attr(
              from_item_attrs=["duration_ms_double"],
              to_item_attr="duration_ms_infer",
              default_val=0.0
            ) \
            .pack_item_attr_to_item_attr(
              from_item_attrs=["fullrank_detail_pwtd"],
              to_item_attr="context_info__pwtd_infer",
              default_val=0.0
            ) \
            .pack_item_attr_to_item_attr(
              from_item_attrs=["fullrank_detail_pctr"],
              to_item_attr="context_info__pctr_infer",
              default_val=0.0
            ) \
            .pack_item_attr_to_item_attr(
              from_item_attrs=["fullrank_detail_pltr"],
              to_item_attr="context_info__pltr_infer",
              default_val=0.0
            ) \
            .pack_item_attr_to_item_attr(
              from_item_attrs=["fullrank_detail_pcmtr"],
              to_item_attr="context_info__pcmtr_infer",
              default_val=0.0
            ) \
            .pack_item_attr_to_item_attr(
              from_item_attrs=["fullrank_detail_pwtr"],
              to_item_attr="context_info__pwtr_infer",
              default_val=0.0
            ) \
            .pack_item_attr_to_item_attr(
              from_item_attrs=["fullrank_detail_pftr"],
              to_item_attr="context_info__pftr_infer",
              default_val=0.0
            ) \
            .pack_item_attr_to_item_attr(
              from_item_attrs=["fullrank_detail_plvtr"],
              to_item_attr="context_info__plvtr_infer",
              default_val=0.0
            ) \
            .pack_item_attr_to_item_attr(
              from_item_attrs=["fullrank_sim_psvr"],
              to_item_attr="context_info__psvtr_infer",
              default_val=0.0
            ) \
            .pack_item_attr_to_item_attr(
              from_item_attrs=["explore_stat__click_count_double"],
              to_item_attr="photo_info__explore_stat__click_count_list",
              default_val=0.0
            ) \
            .pack_item_attr_to_item_attr(
              from_item_attrs=["explore_stat__real_show_count_double"],
              to_item_attr="photo_info__explore_stat__real_show_count_list",
              default_val=0.0
            ) \
            .pack_item_attr_to_item_attr(
              from_item_attrs=["explore_stat__like_count_double"],
              to_item_attr="photo_info__explore_stat__like_count_list",
              default_val=0.0
            ) \
            .pack_item_attr_to_item_attr(
              from_item_attrs=["explore_stat__long_play_count_double"],
              to_item_attr="photo_info__explore_stat__long_play_count_list",
              default_val=0.0
            ) \
            .pack_item_attr_to_item_attr(
              from_item_attrs=["explore_stat__short_play_count_double"],
              to_item_attr="photo_info__explore_stat__short_play_count_list",
              default_val=0.0
            ) \
            .pack_item_attr_to_item_attr(
              from_item_attrs=["explore_stat__follow_count_double"],
              to_item_attr="photo_info__explore_stat__follow_count_list",
              default_val=0.0
            ) \
            .pack_item_attr_to_item_attr(
              from_item_attrs=["explore_stat__view_length_sum_double"],
              to_item_attr="photo_info__explore_stat__view_length_sum_list",
              default_val=0.0
            ) \
            .pack_item_attr_to_item_attr(
              from_item_attrs=["fountain_stats__real_show_count_double"],
              to_item_attr="photo_info__fountain_stats__real_show_count_list",
              default_val=0.0
            ) \
            .pack_item_attr_to_item_attr(
              from_item_attrs=["fountain_stats__like_count_double"],
              to_item_attr="photo_info__fountain_stats__like_count_list",
              default_val=0.0
            ) \
            .pack_item_attr_to_item_attr(
              from_item_attrs=["fountain_stats__follow_count_double"],
              to_item_attr="photo_info__fountain_stats__follow_count_list",
              default_val=0.0
            ) \
            .pack_item_attr_to_item_attr(
              from_item_attrs=["fountain_stats__long_play_count_double"],
              to_item_attr="photo_info__fountain_stats__long_play_count_list",
              default_val=0.0
            ) \
            .pack_item_attr_to_item_attr(
              from_item_attrs=["fountain_stats__short_play_count_double"],
              to_item_attr="photo_info__fountain_stats__short_play_count_list",
              default_val=0.0
            ) \
            .pack_item_attr_to_item_attr(
              from_item_attrs=["fountain_stats__view_length_sum_double"],
              to_item_attr="photo_info__fountain_stats__view_length_sum_list",
              default_val=0.0
            ) \
            .count_reco_result(save_count_to="item_num") \
            .if_("item_num >= 60") \
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
                outputs=[dict(attr_name=attr_name, tensor_name=tensor_name, common=True) for attr_name, tensor_name in model_config.outputs if attr_name in all_model_preds],
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
              .enrich_attr_by_lua(
                name='calculate_list_scores',
                import_common_attr=["pctr", "pwtd", "pvtr", "pltr", "pslide", "context_pctr", "context_pwtd", "context_pltr", "context_pcmtr", "context_pwtr", "context_plvtr", "context_psvtr", "context_pftr", "pctr_weight", "pwtd_weight", "pvtr_weight", "pltr_weight", "pslide_weight", "context_pctr_weight", "context_pwtd_weight", "context_pltr_weight", "context_pwtr_weight", "context_pcmtr_weight", "context_plvtr_weight", "context_psvtr_weight", "context_pftr_weight", "rerank_list_item_idx_flat_list_max_size", "rerank_list_item_size", "rerank_list_true_num", "enable_position_decay"],
                function_for_common="calculate_list_scores",
                export_common_attr=["eval_list_scores"],
                lua_script="""
                  function calculate_list_scores()
                    local n = #pvtr
                    if n ~= rerank_list_item_idx_flat_list_max_size or n ~= #pvtr or n ~= #pltr or n % rerank_list_item_size ~= 0 then
                      error("输入列表长度不一致或不能整除 rerank_list_item_size")
                    end
                    local list_num = n // rerank_list_item_size
                    local scores = {}
                    for i = 1, list_num do
                      local start_idx = (i - 1) * rerank_list_item_size
                      local es_i = 0.0
                      if i < rerank_list_true_num + 1 then
                        for j = 1, rerank_list_item_size do
                          local context_pwtd_reward = context_pwtd_weight * context_pwtd[start_idx + j]
                          local context_pctr_reward = context_pctr_weight * context_pctr[start_idx + j] * 10.0
                          local context_plvtr_reward = context_plvtr_weight * context_plvtr[start_idx + j] * 10.0
                          local context_psvtr_reward = context_psvtr_weight * (1 - context_psvtr[start_idx + j]) * 10.0
                          local context_pltr_reward = context_pltr_weight * context_pltr[start_idx + j] * 100.0
                          local context_pcmtr_reward = context_pcmtr_weight * context_pcmtr[start_idx + j] * 100.0
                          local context_pwtr_reward = context_pwtr_weight * context_pwtr[start_idx + j] * 100.0
                          local context_pftr_reward = context_pftr_weight * context_pftr[start_idx + j] * 100.0
                          local pvtr_reward = pvtr_weight * pvtr[start_idx + j]
                          local pwtd_reward = pwtd_weight * pwtd[start_idx + j]
                          local pltr_reward = pltr_weight * pltr[start_idx + j] * 10.0
                          local pctr_reward = pctr_weight * pctr[start_idx + j] * 10.0

                          local element_score = 0.0
                          if use_multiply then
                            -- 乘法公式：只有权重 > 0 的目标才参与乘法，权重作为幂指数
                            -- 底数加 1.0 保证单调性：(1 + reward_base)^weight，避免底数<1时权重越大值越小
                            local multiply_score = 1.0
                            if pvtr_weight > 0 then
                              multiply_score = multiply_score * ((1.0 + pvtr[start_idx + j] / 10.0) ^ pvtr_weight)
                            end
                            if pwtd_weight > 0 then
                              multiply_score = multiply_score * ((1.0 + pwtd[start_idx + j] / 10.0) ^ pwtd_weight)
                            end
                            if pltr_weight > 0 then
                              multiply_score = multiply_score * ((1.0 + pltr[start_idx + j]) ^ pltr_weight)
                            end
                            if pctr_weight > 0 then
                              multiply_score = multiply_score * ((1.0 + pctr[start_idx + j]) ^ pctr_weight)
                            end
                            if context_pctr_weight > 0 then
                              multiply_score = multiply_score * ((1.0 + context_pctr[start_idx + j]) ^ context_pctr_weight)
                            end
                            if context_pltr_weight > 0 then
                              multiply_score = multiply_score * ((1.0 + context_pltr[start_idx + j]) ^ context_pltr_weight)
                            end
                            if context_pcmtr_weight > 0 then
                              multiply_score = multiply_score * ((1.0 + context_pcmtr[start_idx + j]) ^ context_pcmtr_weight)
                            end
                            if context_pwtr_weight > 0 then
                              multiply_score = multiply_score * ((1.0 + context_pwtr[start_idx + j]) ^ context_pwtr_weight)
                            end
                            element_score = multiply_score
                          else
                            element_score = context_pwtd_reward + context_pctr_reward + context_plvtr_reward + context_psvtr_reward + context_pltr_reward + context_pcmtr_reward + context_pwtr_reward + context_pftr_reward + pvtr_reward + pltr_reward + pctr_reward + pwtd_reward
                          end
                          if enable_position_decay then
                            es_i = es_i + element_score * (1 / (0.3 + j^0.6))
                          else
                            es_i = es_i + element_score
                          end
                        end
                      end
                      scores[i] = es_i
                    end
                    return scores
                  end
                """
              ) \
            .end_() \
            .perflog_attr_value(
              check_point="predict_score",
              common_attrs=[
                "eval_list_scores",
                "pctr",
                "pwtd",
                "pvtr",
                "pltr",
                "context_pctr",
                "context_pwtd",
                "context_pltr",
              ],
              item_attrs=[
              ]
             ) \
            .log_debug_info(
              for_debug_request_only=True,
              item_num_limit = 10,
              common_attrs = [
                "user_id",
                "device_id",
                "rerank_list_true_num",
                "rerank_list_item_idx_flat_list_double",
                "rerank_list_item_idx_flat_list_print",
                "pctr",
                "pslide",
                "pwtd",
                "pvtr",
                "pltr",
                "context_pwtd",
                "context_pctr",
                "eval_list_scores",
                # "context_lv_count",
                # "context_sv_count",
              ],
              item_attrs = [
              ],
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
# service.return_item_attrs(all_model_preds)
service.return_common_attrs(return_common_attrs)
