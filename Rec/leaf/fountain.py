"""
需要将 dragon 目录加入到下面环境变量中（可以添加到类似 ~/.bashrc 中每次启动自动配置环境变量）
export PYTHONPATH={GIT_ROOT_PATH}:$PYTHONPATH
"""
import sys
sys.path.append("./fountain")

from dragonfly.common_leaf_dsl import LeafService
from fountain import PreparingFlow

from fountain import RetrievalFlow

# 老filter
from fountain import FilterV12Flow
from fountain import FilterSplashV2Flow

# cascade
from fountain import CascadeV12Flow
from fountain import CascadeSplashV13Flow

# full_rank
from fountain import FullRankV44Flow
from fountain import FullRankV46Flow

# rerank
from fountain import RerankFastV1Flow
from fountain import RerankSplashBaseFlow

# post_process
from fountain import PostProcessV1Flow
from fountain import PostProcessSplashExp3Flow

preparing_default = PreparingFlow("preparing_default")

retrieval_default = RetrievalFlow("retrieval_default")
retrieval_splash = RetrievalFlow("retrieval_splash")

# filter_default = FilterFlow("filter_default")
# filter_splash = FilterFlow("filter_splash")

filter_default = FilterV12Flow()
filter_splash = FilterSplashV2Flow()

cascading_default = CascadeV12Flow()
cascading_splash = CascadeSplashV13Flow()

ranking_default = FullRankV44Flow()
ranking_splash = FullRankV46Flow()

post_process_default = PostProcessV1Flow()
post_process_splash = PostProcessSplashExp3Flow()

rerank_default = RerankFastV1Flow()
rerank_splash = RerankSplashBaseFlow()

service = LeafService(kess_name = "grpc_fountainRecoLeafDryrun")
service.IGNORE_UNUSED_ATTR = [
  "splash_fullrank_ltr_user_relavance_intention_score", # 首屏使用，非首屏不使用
  "colossus_tag_list", # 首屏使用，非首屏暂时不使用
  "fullrank_ltr_intn_rate", # 打分非首屏用到，首屏不使用
  "cascade_variant_sort_score", # 粗排分桶排序score, 供先知回查使用
  "fullrank_min_act_rank_reci_score", # rerank 首屏没用到，非首屏使用
  "fullrank_ori_pswptr_score", # rerank 首屏没用到，非首屏用到
  "hetu_tag_level_info_v2__hetu_level_four",
  "fullrank_ensemble_absolute_score", #精排es绝对分，供先知排查
  "fullrank_ensemble_ori_score", #精排es原始分，供先知排查
  "fullrank_ensemble_fractile_score", #精排es分位数，供先知排查
  "reason", # rerank gen生成算法标识，供先知排查和占比统计使用,
  "list_reason", # rerank gen pid 生成算法标识，供先知排查和占比统计使用,
  "eval_pred_itemkey", # rerank eval 预测值落地，计算wauc
  "fountain_full_link_reco_log_message", # 全链路样本流, 首屏统一产出, 使用处不统一导致首屏产出后形式上未使用
  "pctr_index", # pxtr index 暂未用到，离线训练使用，后续ltr使用后删掉
  "plvtr_index", # pxtr index 暂未用到，离线训练使用，后续ltr使用后删掉
  "pvtr_index", # pxtr index 暂未用到，离线训练使用，后续ltr使用后删掉
  "pltr_index", # pxtr index 暂未用到，离线训练使用，后续ltr使用后删掉
  "pftr_index", # pxtr index 暂未用到，离线训练使用，后续ltr使用后删掉
  "pwtr_index", # pxtr index 暂未用到，离线训练使用，后续ltr使用后删掉
  "pfintr_index", # pxtr index 暂未用到，离线训练使用，后续ltr使用后删掉
  "pcmtr_index", # pxtr index 暂未用到，离线训练使用，后续ltr使用后删掉
  "pcltr_index", # pxtr index 暂未用到，离线训练使用，后续ltr使用后删掉
  "pesptr_index", # pxtr index 暂未用到，离线训练使用，后续ltr使用后删掉
  "psvr_index", # pxtr index 暂未用到，离线训练使用，后续ltr使用后删掉
  "cascade_click_comment_button", # 粗排首屏没用到, 非首屏用到
  "cascade_debias_wtd_mix", # 粗排首屏没用到, 非首屏用到
  "cascade_wtd_fintr", # 粗排首屏没用到, 非首屏用到
  "sorted_item_pxtrs_attr",  # 精排gsu没用到
  "sorted_item_pxtrs_res",    # 精排gsu没用到
  "dup_cluster_id_duplicate_count",
  "sim_remove_dup_id_duplicate_count",
  "pic_and_selfdup_id_duplicate_count",
  # 迁移入口获取索引，无用的attr待删除
  "duration_ms@featureSourcePId", "hetu_tag_level_info_v2__hetu_level_one@featureSourcePId", "photo_id@featureSourcePId",
  "source_hetu_face_ids", "source_hetu_level_one", 'source_hetu_level_five', 'source_hetu_level_three', 'source_hetu_level_two',
  "skip_explore_subdivision_nn_retrieval_v2", "skip_fountain_colossus_retr", "skip_fountain_colossus_retr_emb_fetch_new", "skip_fountain_colossus_retr_emb_fetch_old", "skip_fountain_mid_photo_gnn_i2i_retr", "skip_top_subdivision_nn_retrieval_v2", "source_hetu_face_id_v2", "source_hetu_level_four_v2", "source_hetu_level_three_v2", "source_hetu_level_two_v2", "source_hetu_tag_v2",
  # dragon更新后新检测出来的无用attr待删除
  "explore_rank_pos_photo_id_retrieval_list",
  "top_follow_author_weight_list",
  "redis_score",
  "colossus_user_info__trigger_weight_list",
  "colossus_user_info__redis_val",
  "high_photo_count_author_map_ptr", # 非首屏降权使用，首屏未使用
  "fountain_save_user_hetu1_distribution_map",
  "is_minority_photo",
  'act_res', # 粗精排实时act pre cache 请求返回标识位，后续使用不到
  'fs_res', # feature server 请求返回标识位
]

service \
  .return_common_attrs([
    "commonRetrievalPhotos",
    "sourcePidSecondLevelCategory",
    "sourcePidDnnCluster",
    "sourcePidTagId",
    "sourcePidMmuImgClusterV3",
    "sourcePidMmuContentId",
    "sourcePidMmuVerticalTagId",
    "sim_one_tags",
    "short_interest",
    "action_interest",
    "long_interest",
    "sourcePidConfigId",
    "featureSourcePId",
    "sourcePhotoExpTag",
    "user_emp_ltr",
    "user_emp_wtr",
    "user_emp_ftr",
    "user_emp_htr",
    "user_emp_cmtr",
    "user_emp_eptr",
    "mc_s1_cluster_id", # 先知使用
    "mc_s1_cluster_cnt", # 先知使用
    "fr_carm_pid_list",  # 精排主模型 carm 特征回流
    ]) \
  .return_item_attrs([
    "fullrank_final_pctr",
    "fullrank_final_pltr",
    "fullrank_final_pwtr",
    "fullrank_final_psvr",
    "fullrank_final_plvtr",
    "fullrank_final_pvtr",
    "fullrank_final_pcmef",
    "fullrank_final_lstr",
    "fullrank_final_pwatchtime_no_bias",
    "fullrank_final_pcmtr",
    "fullrank_final_longview_no_bias",
    "fullrank_final_phtr",
    "fullrank_final_pepstr",
    "fullrank_ensemble_score",
    "fullrank_ltr_score",
    "upload_type",
    "empirical_ctr",
    "empirical_ltr",
    "empirical_wtr",
    "empirical_ftr",
    "empirical_ptr",
    "empirical_cmtr",
    "empirical_htr",
    "empirical_watchtime",
    "fullrank_ftime",
    "cascade_pctr",
    "cascade_pltr",
    "cascade_pwtr",
    "cascade_pftr",
    "cascade_plvtr",
    "cascade_psvtr",
    "cascade_pcmtr",
    "cascade_pptr",
    "cascade_pcestr",
    "cascade_pepstr",
    "cascade_pwatch_time",
    "cascade_ensemble_score",
    "explore_stat__real_show_count",
    "collect_count",
    "author__id",
    "comment_ltr",
    "hetu_tag_level_info__hetu_level_one",
    "hetu_tag_level_info__hetu_level_two",
    "hetu_tag_level_info__hetu_level_three",
    "hetu_tag_level_info__hetu_level_four",
    "hetu_tag_level_info__hetu_level_five",
    "hetu_tag_level_info__hetu_tag",
    "mmu_img_cluster_v1",
    "mmu_img_cluster_v3",
    "mmu_content_id",
    "photo_dnn_cluster_id",
    "userExpLtr",
    "userExpWtr",
    "userExpCmtr",
    "userExpPtr",
    "cascade_cluster_id",
    "cascade_pctr_debias",
    "cascade_pltr_debias",
    "cascade_pwtr_debias",
    "cascade_longview_score_debias",
    "cascade_psvtr_debias",
    "relative_photo_original_id",
    "disable_plc"]) \
  .add_leaf_flows(
    leaf_flows = [preparing_default, retrieval_default, filter_default, cascading_default, ranking_default, post_process_default, rerank_default],
    request_type = ["fountain_fast_v1", "fountain_fast_v1_life", "fountain_fast_pic_inside", "fountain_fast_life_pic_inside"],
    as_default = True) \
  .add_leaf_flows(
    leaf_flows = [preparing_default, retrieval_splash, filter_splash, cascading_splash, ranking_splash, post_process_splash, rerank_splash],
    request_type = ["fountain_splash", "fountain_splash_life", "fountain_splash_pic_inside", "fountain_splash_life_pic_inside", "fountain_splash_vane"]) \
  .build(
    output_file = "dynamic_json_config.json",
  )
