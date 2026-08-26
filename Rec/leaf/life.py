"""
需要将 dragon 目录加入到下面环境变量中（可以添加到类似 ~/.bashrc 中每次启动自动配置环境变量）
export PYTHONPATH={GIT_ROOT_PATH}:$PYTHONPATH
生活页 leaf
"""
import sys
sys.path.append("./life")

from dragonfly.common_leaf_dsl import LeafService
from life import RetrievalFlow
from life import FilterFlow
from life import CascadingFlow
from life import RankingFlow
from life import RerankFlow
from life import PreparingFlow
from life import PostprocFlow
from life import FountainFilterFlow
from life import FountainPostprocFlow
from life import return_attrs

preparing_default = PreparingFlow("preparing_default")
retrieval_default = RetrievalFlow("retrieval_default")
retrieval_fountain = RetrievalFlow("retrieval_fountain")
filter_default = FilterFlow("filter_default")
cascading_default = CascadingFlow("cascading_default")
ranking_default = RankingFlow("ranking_default")
rerank_default = RerankFlow("rerank_default")
fountain_splash_cascading_default = CascadingFlow("fountain_splash_cascading_default")
fountain_fast_cascading_default = CascadingFlow("fountain_fast_cascading_default")
fountain_splash_ranking_default = RankingFlow("fountain_splash_ranking_default")
fountain_fast_ranking_default = RankingFlow("fountain_fast_ranking_default")
postproc_default = PostprocFlow("postproc_default", return_attrs.DEFAULT_COMMON_ATTRS, return_attrs.DEFAULT_ITEM_ATTRS, return_attrs.TRACEBACK_ITEM_ATTRS)
postproc_backup = PostprocFlow("postproc_backup", return_attrs.BACKUP_COMMON_ATTRS, return_attrs.BACKUP_ITEM_ATTRS, return_attrs.TRACEBACK_ITEM_ATTRS)

fountain_splash_filter_default = FountainFilterFlow("fountain_splash_filter_default")
fountain_fast_filter_default = FountainFilterFlow("fountain_fast_filter_default")
fountain_postproc_default = FountainPostprocFlow("fountain_postproc_default",  return_attrs.DEFAULT_COMMON_ATTRS, return_attrs.DEFAULT_ITEM_ATTRS, return_attrs.TRACEBACK_ITEM_ATTRS)

service = LeafService(kess_name = "grpc_exploreRecoLeafLifeDryrun")
service.CHECK_DYNAMIC_PARAM_FORMAT = False
INVALID_UNUSED_COMMON_ATTR = [
  'pic_comment_aid_list', 'pic_comment_list', 'pic_follow_list', 'pic_hetu_l1_cnt', 'pic_like_list', 'pic_play_list',
  'longview_0', 'longview_1', 'longview_10', 'longview_11', 'longview_12', 'longview_13', 'longview_14', 'longview_15', 'longview_16', 'longview_17',
  'longview_18', 'longview_19', 'longview_2', 'longview_20', 'longview_21', 'longview_22', 'longview_23', 'longview_24', 'longview_25', 'longview_26',
  'longview_27', 'longview_28', 'longview_29', 'longview_3', 'longview_4', 'longview_5', 'longview_6', 'longview_7', 'longview_8', 'longview_9',
  'longview_aid_0', 'longview_aid_1', 'longview_aid_10', 'longview_aid_11', 'longview_aid_12', 'longview_aid_13', 'longview_aid_14', 'longview_aid_15',
  'longview_aid_16', 'longview_aid_17', 'longview_aid_18', 'longview_aid_19', 'longview_aid_2', 'longview_aid_20', 'longview_aid_21', 'longview_aid_22',
  'longview_aid_23', 'longview_aid_24', 'longview_aid_25', 'longview_aid_26', 'longview_aid_27', 'longview_aid_28', 'longview_aid_29', 'longview_aid_3',
  'longview_aid_4', 'longview_aid_5', 'longview_aid_6', 'longview_aid_7', 'longview_aid_8', 'longview_aid_9', 'longview_play_0', 'longview_play_1',
  'longview_play_10', 'longview_play_11', 'longview_play_12', 'longview_play_13', 'longview_play_14', 'longview_play_15', 'longview_play_16', 'longview_play_17',
  'longview_play_18', 'longview_play_19', 'longview_play_2', 'longview_play_20', 'longview_play_21', 'longview_play_22', 'longview_play_23', 'longview_play_24',
  'longview_play_25', 'longview_play_26', 'longview_play_27', 'longview_play_28', 'longview_play_29', 'longview_play_3', 'longview_play_4', 'longview_play_5',
  'longview_play_6', 'longview_play_7', 'longview_play_8', 'longview_play_9', 'longview_tag_0', 'longview_tag_1', 'longview_tag_10', 'longview_tag_11',
  'longview_tag_12', 'longview_tag_13', 'longview_tag_14', 'longview_tag_15', 'longview_tag_16', 'longview_tag_17', 'longview_tag_18', 'longview_tag_19',
  'longview_tag_2', 'longview_tag_20', 'longview_tag_21', 'longview_tag_22', 'longview_tag_23', 'longview_tag_24', 'longview_tag_25', 'longview_tag_26',
  'longview_tag_27', 'longview_tag_28', 'longview_tag_29', 'longview_tag_3', 'longview_tag_4', 'longview_tag_5', 'longview_tag_6', 'longview_tag_7', 'longview_tag_8',
  'longview_tag_9', 'retr_space', 'backexplore_to_current_interval', 'enable_search_back_explore_cal', 'search_to_current_interval', 'ann_space', 'service_timeout_ms',
  'uCityLevelNew', 'uInferGender', 'uIsDouyin', 'filter_future_seconds', 'enable_rand_source', 'ann_dist_threshold', 'enable_extract_common_user_feature', 'enable_infer_user_emb',
  'predict_service_name', 'use_cached_user_emb', 'click_limit', 'collect_limit', 'download_limit', 'follow_limit', 'forward_limit', 'like_limit', 'profile_enter_limit',
  'trigger_list_weight', 'enable_shuffle_result', 'scatter_each_limit', 'hetu_level2_tags', 'hetu_low_level_tags', 'colossus_trigger_tag_list', 'hetu_tag_retrieval_num',
  'click_weight', 'embedding_service_name', 'fountain_short_view_threshold', 'hate_weight', 'max_click_num', 'max_hate_num', 'max_not_click_num', 'max_report_num', 'max_short_view_num',
  'not_click_limit_hour', 'not_click_weight', 'play_stat_limit_hour', 'report_weight', 'short_view_threshold', 'short_view_weight', 'valid_user_list',
  'enable_pdn_action_fountain_trigger', 'explore_fountain_comment_limit', 'explore_fountain_follow_limit', 'explore_fountain_forward_limit', 'explore_fountain_like_limit',
  'explore_pdn_action_comment_limit', 'explore_pdn_action_follow_limit', 'explore_pdn_action_forward_limit', 'explore_pdn_action_like_limit', 'i2i_retr__trigger_pid@result_item_id',
  'trigger_weight_list', 'life_mc_i2i_seed_photo_size', 'trigger_num', 'colossus_pic_trigger_tag_list', 'enable_shuffle_retr_result', 'colossus_trigger_limit',
  'mc_enable_opt_card_trigger', 'enable_interaction_trigger_shuffle', 'enable_emb_server', 'enable_emb_server', 'enable_emb_server',
  'fountain_comment_list', 'fountain_follow_list', 'fountain_forward_list', 'fountain_like_list', 'dId', 'uId', 'ann_dist', 'open_user_person'
]

service.IGNORE_UNUSED_ATTR = [
  "life_reco_leaf_retrieval_default_ts",
  "life_reco_leaf_retrieval_default_cpu_cost_ts",
  "life_reco_leaf_filter_default_ts",
  "life_reco_leaf_filter_default_cpu_cost_ts",
  "life_reco_leaf_cascading_default_ts",
  "life_reco_leaf_cascading_default_cpu_cost_ts",
  "life_reco_leaf_ranking_default_ts",
  "life_reco_leaf_ranking_default_cpu_cost_ts",
  "life_reco_leaf_rerank_default_ts",
  "life_reco_leaf_rerank_default_cpu_cost_ts",
  "life_reco_leaf_retrieval_fountain_ts",
  "life_reco_leaf_retrieval_fountain_cpu_cost_ts",
  "life_reco_leaf_fountain_splash_filter_default_ts",
  "life_reco_leaf_fountain_splash_filter_default_cpu_cost_ts",
  "life_reco_leaf_fountain_splash_cascading_default_ts",
  "life_reco_leaf_fountain_splash_cascading_default_cpu_cost_ts",
  "life_reco_leaf_fountain_splash_ranking_default_ts",
  "life_reco_leaf_fountain_splash_ranking_default_cpu_cost_ts",
  "life_reco_leaf_fountain_fast_filter_default_ts",
  "life_reco_leaf_fountain_fast_filter_default_cpu_cost_ts",
  "life_reco_leaf_fountain_fast_cascading_default_ts",
  "life_reco_leaf_fountain_fast_cascading_default_cpu_cost_ts",
  "life_reco_leaf_fountain_fast_ranking_default_ts",
  "life_reco_leaf_fountain_fast_ranking_default_cpu_cost_ts",
  "pic_xtr_fractile_score_attr_from_redis_ptr",
  "explore_pic_xtr_cluster_emp_map_ptr",
] + INVALID_UNUSED_COMMON_ATTR
service.add_leaf_flows(
    leaf_flows = [preparing_default, retrieval_default, filter_default, cascading_default, ranking_default, rerank_default, postproc_default],
    request_type = "life",
  ) \
  .add_leaf_flows(
    leaf_flows=[preparing_default, retrieval_fountain, fountain_splash_filter_default, fountain_splash_cascading_default, fountain_splash_ranking_default, fountain_postproc_default],
    request_type = ["fountain_splash_life", "fountain_splash_life_pic_inside"],
  ) \
  .add_leaf_flows(
    leaf_flows = [preparing_default, retrieval_fountain, fountain_fast_filter_default, fountain_fast_cascading_default, fountain_fast_ranking_default, fountain_postproc_default],
    request_type = ["fountain_fast_v1_life", "fountain_fast_life_pic_inside"],
  ) \
  .add_leaf_flows(
    leaf_flows = [preparing_default, retrieval_default, filter_default, cascading_default, ranking_default, postproc_backup],
    request_type = "backup",
  ) \
  .build(
    output_file = "dynamic_json_config.json",
  )
