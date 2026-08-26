#!/usr/bin/env python3
# coding=utf-8

from dragonfly.common_leaf_dsl import LeafFlow
from dragonfly.ext.subdivision.subdivision_api_mixin import subdivisionApiMixin
from cascade.cascade_base_flow import CascadeBaseFlow
from cascade.ab_params import cascade_common_params, cascade_splash_params, cascade_common_param_abhit, cascade_splash_params_abhit
from cascade.cascade_utils import cascade_ltr_common_feature, ltr_opt_item_features, cascade_fc_sim3_feature, cascade_distill_precict_item_features
from util import enrich_ab_param

fountain_variant_cluster_sort_queue = [
  {
    "name": "cascade_score",
    "weight": 0.3,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_score",
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_score",
  },
  {
    "name": "cascade_pctr",
    "weight": 0.65,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_click_score",
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_click_score",
    "raw_weight_attr": "fountain_splash_cascade_variant_cluster_sort_click_raw_weight",
    "raw_pow_weight_attr": "fountain_splash_cascade_variant_cluster_sort_click_raw_pow_weight",
  },
  {
    "name": "cascade_pltr",
    "weight": 0.05,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_like_score",
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_like_score",
    "raw_weight_attr": "fountain_splash_cascade_variant_cluster_sort_like_raw_weight",
    "raw_pow_weight_attr": "fountain_splash_cascade_variant_cluster_sort_like_raw_pow_weight",
  },
  {
    "name": "cascade_pwtr",
    "weight": 0.06,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_follow_score",
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_follow_score",
    "raw_weight_attr": "fountain_splash_cascade_variant_cluster_sort_follow_raw_weight",
    "raw_pow_weight_attr": "fountain_splash_cascade_variant_cluster_sort_follow_raw_pow_weight",
  },
  {
    "name": "cascade_pftr",
    "weight": 0.07,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_forward_score",
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_forward_score",
    "raw_weight_attr": "fountain_splash_cascade_variant_cluster_sort_forward_raw_weight",
    "raw_pow_weight_attr": "fountain_splash_cascade_variant_cluster_sort_forward_raw_pow_weight",
  },
  {
    "name": "cascade_longview_score",
    "weight": 0.3,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_longview_score",
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_longview_score",
  },
  {
    "name": "cascade_psvtr",
    "weight": 0.4,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_shortview_score",
    "reverse_order": True,
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_shortview_score",
  },
  {
    "name": "cascade_shortview_score2",
    "weight": 0.75,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_shortview_score2",
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_shortview_score2",
    "raw_weight_attr": "fountain_splash_cascade_variant_cluster_sort_shortview_score2_raw_weight",
    "raw_pow_weight_attr": "fountain_splash_cascade_variant_cluster_sort_shortview_score2_raw_pow_weight",
  },
  {
    "name": "cascade_ptr",
    "weight": 0.2,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_profile_score",
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_profile_score",
    "raw_weight_attr": "fountain_splash_cascade_variant_cluster_sort_profile_raw_weight",
    "raw_pow_weight_attr": "fountain_splash_cascade_variant_cluster_sort_profile_raw_pow_weight",
  },
  {
    "name": "cascade_pcmtr",
    "weight": 0.0,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_comment_score",
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_comment_score",
    "raw_weight_attr": "fountain_splash_cascade_variant_cluster_sort_comment_raw_weight",
    "raw_pow_weight_attr": "fountain_splash_cascade_variant_cluster_sort_comment_raw_pow_weight",
  },
  {
    "name": "cascade_pcestr",
    "weight": 0.0,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_cestr_score",
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_cestr_score",
    "raw_weight_attr": "fountain_splash_cascade_variant_cluster_sort_cestr_raw_weight",
    "raw_pow_weight_attr": "fountain_splash_cascade_variant_cluster_sort_cestr_raw_pow_weight",
  },
  {
    "name": "cascade_pepstr",
    "weight": 0.0,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_epstr_score",
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_epstr_score",
    "raw_weight_attr": "fountain_splash_cascade_variant_cluster_sort_epstr_raw_weight",
    "raw_pow_weight_attr": "fountain_splash_cascade_variant_cluster_sort_epstr_raw_pow_weight",
  },
  {
    "name": "cascade_pwatch_time",
    "weight": 0.0,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_pwatch_time",
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_pwatch_time",
    "raw_weight_attr": "fountain_splash_cascade_variant_cluster_sort_pwatch_time_raw_weight",
    "raw_pow_weight_attr": "fountain_splash_cascade_variant_cluster_sort_pwatch_time_raw_pow_weight",
  },
  {
    "name": "cascade_pcltr",
    "weight": 0.0,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_pcltr",
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_pcltr",
    "raw_weight_attr": "fountain_splash_cascade_variant_cluster_sort_collect_raw_weight",
    "raw_pow_weight_attr": "fountain_splash_cascade_variant_cluster_sort_collect_raw_pow_weight",
  },
  {
    "name": "cascade_pwtd",
    "weight": 0.0,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_pwtd",
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_pwtd",
    "raw_weight_attr": "fountain_splash_cascade_variant_cluster_sort_pwtd_raw_weight",
    "raw_pow_weight_attr": "fountain_splash_cascade_variant_cluster_sort_pwtd_raw_pow_weight",
  },
  {
    "name": "cascade_phtr",
    "weight": 0.0,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_phtr",
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_phtr",
    "raw_weight_attr": "fountain_splash_cascade_variant_cluster_sort_hate_raw_weight",
    "raw_pow_weight_attr": "fountain_splash_cascade_variant_cluster_sort_hate_raw_pow_weight",
  },
  {
    "name": "cascade_ftr_kai",
    "weight": 0.0,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_ftr_kai",
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_ftr_kai",
  },
  {
    "name": "cascade_ftr_kai_duration",
    "weight": 0.0,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_ftr_kai_duration",
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_ftr_kai_duration",
  },
  {
    "name": "cascade_ipw_opt_ftr",
    "weight": 0.0,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_ipw_opt_ftr",
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_ipw_opt_ftr",
  },
  {
    "name": "cascade_pcotr",
    "weight": 0.0,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_pcotr",
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_pcotr",
  },
  {
    "name": "cascade_fc_pvtr2",
    "weight": 0.0,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_fc_pvtr2_score",
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_fc_pvtr2_score",
    "raw_weight_attr": "fountain_splash_cascade_variant_cluster_sort_fc_pvtr2_raw_weight",
    "raw_pow_weight_attr": "fountain_splash_cascade_variant_cluster_sort_fc_pvtr2_raw_pow_weight",
  },
  {
    "name": "cascade_fc_pwtd2_inverse",
    "weight": 0.0,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_fc_pwtd2_score",
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_fc_pwtd2_score",
    "raw_weight_attr": "fountain_splash_cascade_variant_cluster_sort_fc_pwtd2_raw_weight",
    "raw_pow_weight_attr": "fountain_splash_cascade_variant_cluster_sort_fc_pwtd2_raw_pow_weight",
  },
  {
    "name": "cascade_slide_kai",
    "weight": 0.0,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_slide_kai",
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_slide_kai",
  },
  {
    "name": "cascade_wtd_kai_mix",
    "weight": 0.0,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_wtd_kai_mix",
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_wtd_kai_mix",
  },
  {
    "name": "cascade_wtd_kai",
    "weight": 0.0,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_wtd_kai",
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_wtd_kai",
  },
  {
    "name": "cascade_act_kai",
    "weight": 0.0,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_act_kai",
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_act_kai",
  },
  {
    "name": "cascade_action_once_interact_score",
    "weight": 0.0,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_action_once_interact_score",
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_action_once_interact_score",
  },
  {
    "name": "cascade_action_once_watchtime_score",
    "weight": 0.0,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_action_once_watchtime_score",
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_action_once_watchtime_score",
  },
  {
    "name": "cascade_wtd_percent",
    "weight": 0.0,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_wtd_percent",
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_wtd_percent",
  },
  {
    "name": "cascade_wtd_duration_mix",
    "weight": 0.0,
    "weight_attr": "fountain_variant_mc_splash_weight_cascade_wtd_duration_mix",
    "temperature_attr": "fountain_variant_mc_splash_temperature",
    "power_weight_attr": "fountain_variant_mc_splash_weight_cascade_wtd_duration_mix",
  },
]

related_rank_score_config_queues = [
  {
    "enable": True,
    "weight": "{{source_related_rank_weight_sim_cluster_id}}",
    "attr_type": "int",
    "source_attr": "source_hetu_sim_cluster_id",
    "item_attr": "hetu_sim_cluster_id",
  },
  {
    "enable": True,
    "weight": "{{source_related_rank_weight_hetu_cluster_id_v2}}",
    "attr_type": "int",
    "source_attr": "source_hetu_cluster_id_v2",
    "item_attr": "hetu_tag_level_info_v2__hetu_cluster_id",
  },
  {
    "enable": True,
    "weight": "{{source_related_rank_weight_mmu_img_cluster_v3}}",
    "attr_type": "int",
    "source_attr": "sourcePidMmuImgClusterV3",
    "item_attr": "mmu_img_cluster_v3",
  },
  {
    "enable": True,
    "weight": "{{source_related_rank_weight_mmu_text_cluster}}",
    "attr_type": "int",
    "source_attr": "sourcePidMmuTextCluster",
    "item_attr": "mmu_text_cluster",
  },
  {
    "enable": True,
    "weight": "{{source_related_rank_weight_author__id}}",
    "attr_type": "int",
    "source_attr": "sourcePidAuthorId",
    "item_attr": "author__id",
  },
  {
    "enable": "{{enable_source_related_rank_use_author_cate_one}}",
    "weight": "{{source_related_rank_weight_author_cate_one}}",
    "attr_type": "int",
    "source_attr": "sourcePidFirstLevelCategory",
    "item_attr": "author__category_detail__first_level_id",
  },
  {
    "enable": "{{enable_source_related_rank_use_author_cate_two}}",
    "weight": "{{source_related_rank_weight_author_cate_two}}",
    "attr_type": "int",
    "source_attr": "sourcePidSecondLevelCategory",
    "item_attr": "author__category_detail__second_level_id",
  },
  {
    "enable": "{{enable_source_related_rank_use_author_cate_three}}",
    "weight": "{{source_related_rank_weight_author_cate_three}}",
    "attr_type": "int",
    "source_attr": "sourcePidThirdLevelCategory",
    "item_attr": "author__category_detail__third_level_id",
  },
  {
    "enable": True,
    "weight": "{{source_related_rank_weight_tag}}",
    "attr_type": "int",
    "source_attr": "sourcePidTagId",
    "item_attr": "tag",
  },
  {
    "enable": "{{enable_source_related_rank_use_upload_type}}",
    "weight": "{{source_related_rank_weight_upload_type}}",
    "attr_type": "int",
    "source_attr": "sourcePidUploadType",
    "item_attr": "upload_type",
  },
  {
    "enable": "{{enable_source_related_rank_use_author_circle_v2}}",
    "weight": "{{source_related_rank_weight_author_circle_v2}}",
    "attr_type": "int",
    "source_attr": "source_author_circle_v2",
    "item_attr": "author_circle_v2",
  },

  # list attr
  {
    "enable": True,
    "weight": "{{source_related_rank_weight_hetu_level_one_v2}}",
    "attr_type": "int_list",
    "source_attr": "source_hetu_level_one_v2",
    "item_attr": "hetu_level_one_v2",
  },
  {
    "enable": True,
    "weight": "{{source_related_rank_weight_hetu_level_two_v2}}",
    "attr_type": "int_list",
    "source_attr": "source_hetu_level_two_v2",
    "item_attr": "hetu_level_two_v2",
  },
  {
    "enable": True,
    "weight": "{{source_related_rank_weight_hetu_level_three_v2}}",
    "attr_type": "int_list",
    "source_attr": "source_hetu_level_three_v2",
    "item_attr": "hetu_level_three_v2",
  },
  {
    "enable": True,
    "weight": "{{source_related_rank_weight_hetu_level_four_v2}}",
    "attr_type": "int_list",
    "source_attr": "source_hetu_level_four_v2",
    "item_attr": "hetu_level_four_v2",
  },
  {
    "enable": True,
    "weight": "{{source_related_rank_weight_hetu_tag_v2}}",
    "attr_type": "int_list",
    "source_attr": "related_source_hetu_tag_v2",
    "item_attr": "hetu_tag_v2",
  },
  {
    "enable": True,
    "weight": "{{source_related_rank_weight_hetu_face_id_v2}}",
    "attr_type": "int_list",
    "source_attr": "source_hetu_face_id_v2",
    "item_attr": "hetu_face_id_v2",
  },
  {
    "enable": "{{enable_source_related_rank_use_user_hash_tag_id}}",
    "weight": "{{source_related_rank_weight_user_hash_tag_id}}",
    "attr_type": "int_list",
    "source_attr": "source_user_hash_tag_id",
    "item_attr": "user_hash_tag_id",
  },
]

class CascadeSplashV13Flow(CascadeBaseFlow, subdivisionApiMixin):
  def __init__(self):
    LeafFlow.__init__(self, "cascade_splash_v13")
    self \
      .namespace_(ns = "cascade_splash_v13", nest = True) \
      ._timestamp_begin("cascade_splash") \
      ._rank() \
      ._timestamp_end("cascade_splash") \
      ._count_stage_cpu_cost("cascade_splash") \
      .namespace_()

  def _rank(self):
    self \
    .count_reco_result(save_count_to = "cascade_enter") \
    .get_abtest_params(
      biz_name = "RECO_RPC",
      ab_params = enrich_ab_param(cascade_common_params + cascade_splash_params),
      prioritized_suffix = "{{_ABTEST_SUFFIX_LIST_}}",
    ) \
    .get_abtest_params(
      biz_name = "RECO_RPC",
      ab_params = cascade_common_param_abhit + cascade_splash_params_abhit,
      prioritized_suffix = "{{_ABTEST_SUFFIX_LIST_}}",
    ) \
    .if_("enable_replace_colossus_v2_from_mc == 1") \
      .copy_attr(
        attrs = [
          {
            "from_common": "colossus_v2_resp",
            "to_common": "colossus_resp_v2",
          },
        ],
      ) \
    .end_() \
    .if_("fountain_splash_cascade_get_emp_xtr == 1") \
      ._get_emp_xtr() \
    .end_() \
    .if_("fountain_enable_personal_weight == 1") \
      ._interactive_emp_xtr_change() \
    .end_() \
    ._calc_true_living() \
    .enrich_attr_by_lua(
      import_common_attr = [
        "enableFountainFullrankExp",
        "increase_quota_status",
        "fountain_splash_mc_increase_quota_factor"
      ],
      export_common_attr = [
        "fountain_pre_fullrank_size_limit_v2_splash"
      ],
      export_item_attr = ["cascade_long_term_interest_ee_score",],
      function_for_common = "cascade_control_splash",
      function_for_item = "cascade_feature_trans_splash",
      lua_script_file = "fountain/cascade/lua/cascade_control.lua",
    ) \
    .if_("enable_fountain_splash_cascade_enter_size_truncate == 1") \
      .shuffle() \
      .truncate(
        size_limit = "{{cascade_model_limit_size_limit_splash}}",
      ) \
    .end_() \
    .perflog_reason_count(
      check_point = "post_cascade_prerank",
    ) \
    .count_reco_result(save_count_to = "cascade_first_truncate") \
    ._enrich_cascade_score_splash() \
    .if_("enable_splash_source_related_rank_score_v3 == 1") \
      .if_("enable_splash_calc_related_rank_score == 1") \
        .explore_related_rank_score(
          queues = related_rank_score_config_queues,
          save_score_to_attr = "source_related_score",
        ) \
      .else_() \
        .fountain_calc_related_score_v2(
          enable_cal_photo_sim_by_intersect = "{{enable_splash_source_related_rank_score_v3}}",
          diversity_dim_weight = "{{source_related_dim_weight_v3}}",
          save_score_to_attr = "source_related_score",
          int_source_attrs = [
            "source_hetu_sim_cluster_id", "source_hetu_cluster_id_v2",
            "sourcePidMmuImgClusterV3", "sourcePidMmuTextCluster", 
            "sourcePidAuthorId", "sourcePidFirstLevelCategory",
            "sourcePidSecondLevelCategory", "sourcePidThirdLevelCategory",
            "sourcePidTagId", "sourcePidUploadType",
          ],
          int_list_source_attrs = [
            "source_hetu_level_one_v2", "source_hetu_level_two_v2",
            "source_hetu_level_three_v2", "source_hetu_level_four_v2",
            "source_hetu_tag_v2", "source_hetu_face_id_v2"
          ],
          int_item_attrs = [
            "hetu_sim_cluster_id", "hetu_tag_level_info_v2__hetu_cluster_id",
            "mmu_img_cluster_v3", "mmu_text_cluster",
            "author__id", "author__category_detail__first_level_id",
            "author__category_detail__second_level_id", "author__category_detail__third_level_id",
            "tag", "upload_type",
          ],
          int_list_item_attrs = [
            "hetu_level_one_v2", "hetu_level_two_v2",
            "hetu_level_three_v2", "hetu_level_four_v2",
            "hetu_tag_v2", "hetu_face_id_v2",
          ],
        ) \
      .end_() \
    .else_() \
      .fountain_calc_related_score_v2(
        diversity_dim_weight = "{{source_related_dim_weight}}",
        save_score_to_attr = "source_related_score",
        int_source_attrs = [
          "sourcePidHetu0", "sourcePidHetuLevelTwo0", "sourcePidAuthorId",
          "sourcePidFirstLevelCategory", "sourcePidSecondLevelCategory",
          "sourcePidThirdLevelCategory",
          "sourcePidDnnCluster", "sourcePidMmuImgClusterV3",
          "sourcePidGEClusterId",
          "sourcePidTagId", "sourcePidMmuTextLdaTopic",
          "sourcePidMmuTextCluster", "sourcePidUploadType",
        ],
        int_list_source_attrs = [
          "sourcePidHetuFaceId", "sourcePidOnlineLdaTopicId"
        ],
        int_item_attrs = [
          "hetu_level_one", "hetu_level_two", "author__id",
          "author__category_detail__first_level_id",
          "author__category_detail__second_level_id",
          "author__category_detail__third_level_id",
          "photo_dnn_cluster_id", "mmu_img_cluster_v3", "GE_cluster_id",
          "tag", "mmu_text_lda_topic", "mmu_text_cluster", "upload_type",
        ],
        int_list_item_attrs = [
          "hetu_tag_level_info__hetu_face_id", "online_lda_topic__ids",
        ],
      ) \
    .end_() \
    ._prerank_v2() \
    ._cascade_fc_model_in_splash() \
    .fountain_perflog_attr_fractile(
      skip = "{{fountain_skip_cascade_perflog_attr_fractile}}",
      item_attrs = [
        "cascade_pctr",
        "cascade_pltr",
        "cascade_pwtr",
        "cascade_pftr",
        "cascade_ptr",
        "cascade_plvtr",
        "cascade_psvtr",
        "cascade_pwatch_time",
        "cascade_longview_score",
        "cascade_shortview_score2",
        "source_related_score",
      ],
      fractile_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
      check_point = "cascade"
    ) \
    .fountain_environment_perf_log(
      skip = "{{fountain_skip_cascade_perf_upload_day_and_miss_pxtr}}",
      enable_pxtr_miss_perf = True,
      enable_upload_day_perf = True,
      upload_time_attr = "upload_time",
      item_pxtrs = [
        "cascade_pctr",
      ],
      upload_day_divide = "0-1-2-3-4-5-6-30-60-120-180",
      check_point = "fountain.cascade"
    ) \
    .enrich_attr_by_lua(
      skip = "{{skip_cascade_duration_cluster_calc}}",
      import_common_attr = [
        "skip_splash_variant_source_hetu_cluster_sort",
        "skip_splash_variant_source_hetu_cluster_sort_level_three",
        "skip_splash_variant_source_hetu_cluster_sort_level_two",
        "skip_splash_variant_source_hetu_cluster_sort_level_one",
        "splash_variant_source_hetu_cluster_base_id",
        "source_hetu_level_one_v2",
        "source_hetu_level_two_v2",
        "source_hetu_level_three_v2",
      ],
      import_item_attr = [
        "duration_ms",
        "hetu_tag_level_info_v2__hetu_level_three",
        "hetu_tag_level_info_v2__hetu_level_two",
        "hetu_tag_level_info_v2__hetu_level_one",
        ],
      export_item_attr = ["cascade_duration_cluster_id"],
      function_for_item = "calc_cascade_duration_id",
      lua_script_file = "fountain/cascade/lua/calc_cascade_duration_id.lua",
    ) \
    .if_("skip_variant_duration_cluster_sort == 0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .fountain_variant_cluster_sort_v2(
        cluster_sort_list_attr_name = "cascade_duration_cluster_id",
        cluster_config = "{{fountain_cascade_variant_duration_cluster_config}}",
        global_cut_ratio = "{{fountain_variant_mc_splash_global_cut_ratio}}",
        min_survival = "{{fountain_variant_mc_splash_min_survival}}",
        enable_proportional = "{{fountain_variant_mc_splash_enable_proportional}}",
        size_limit = "{{fountain_variant_mc_splash_size_limit}}",
        use_power_calc = "{{fountain_variant_mc_splash_use_power_calc}}",
        queues = fountain_variant_cluster_sort_queue
      ) \
    .end_if_() \
    .if_("skip_variant_duration_cluster_sort_new == 0") \
      .explore_cluster_variant_sort_v3(
        check_point = "cascade",
        cluster_sort_list_attr_name = "cascade_duration_cluster_id",
        cluster_config = "{{fountain_cascade_variant_duration_cluster_config}}",
        global_cut_ratio = "{{fountain_variant_mc_splash_global_cut_ratio}}",
        min_survival = "{{fountain_variant_mc_splash_min_survival}}",
        enable_proportional = "{{fountain_variant_mc_splash_enable_proportional}}",
        size_limit = "{{fountain_variant_mc_splash_size_limit}}",
        use_power_calc = "{{fountain_variant_mc_splash_use_power_calc}}",
        use_power_calc_v2 = "{{fountain_splash_cascade_variant_cluster_sort_use_power_calc_v2}}",
        rank_smooth = "{{fountain_splash_cascade_variant_cluster_sort_rank_smooth}}",
        user_info_ptr_attr = "userInfoPb",
        action_day = "{{fountain_mc_variant_weight_action_day_num}}",
        time_cluster_base_id = "{{fountain_variant_mc_time_cluster_base_id_splash}}",
        queues = fountain_variant_cluster_sort_queue
      ) \
    .end_if_() \
    .perflog_reason_count(
      check_point = "post_cascade_stage1",
    ) \
    .count_reco_result(
      name = "fountain_splash_mc_stage1",
      traceback = True,
      save_count_to = "cascade_variant"
    ) \
    .fountain_perflog_attr_fractile(
      skip = "{{fountain_skip_cascade_perflog_attr_fractile}}",
      item_attrs = [
        "cascade_pctr",
        "cascade_pltr",
        "cascade_pwtr",
        "cascade_pftr",
        "cascade_ptr",
        "cascade_plvtr",
        "cascade_psvtr",
        "cascade_pwatch_time",
        "cascade_longview_score",
        "cascade_shortview_score2",
        "source_related_score",
      ],
      fractile_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
      check_point = "cascade_stage2"
    ) \
    .enrich_attr_by_lua(
      skip = "{{skip_cascade_user_adaptive_weight_cal_splash}}",
      import_common_attr = [
        "fountain_ensemble_weight_cascade_like_score",
        "fountain_ensemble_weight_cascade_follow_score",
        "fountain_ensemble_weight_cascade_comment_score",
        "fountain_ensemble_weight_cascade_profile_score",
        "fountain_ensemble_weight_cascade_forward_score",
        "fountain_ensemble_weight_cascade_epstr_score",
        "fountain_cascade_ensemble_power_weight_adjust_ratio_min",
        "fountain_cascade_ensemble_power_weight_adjust_ratio_max",
        "fountain_cascade_ensemble_power_weight_cascade_like_emp",
        "fountain_cascade_ensemble_power_weight_cascade_follow_emp",
        "fountain_cascade_ensemble_power_weight_cascade_comment_emp",
        "fountain_cascade_ensemble_power_weight_cascade_profile_emp",
        "fountain_cascade_ensemble_power_weight_cascade_forward_emp",
        "fountain_cascade_ensemble_power_weight_cascade_eps_emp"
        "userExpLtr",
        "userExpWtr",
        "userExpCmtr",
        "userExpPtr",
        "userExpFtr",
        "userExpEptr"
      ],
      export_common_attr = [
        "fountain_ensemble_weight_cascade_like_score",
        "fountain_ensemble_weight_cascade_follow_score",
        "fountain_ensemble_weight_cascade_comment_score",
        "fountain_ensemble_weight_cascade_profile_score",
        "fountain_ensemble_weight_cascade_forward_score",
        "fountain_ensemble_weight_cascade_epstr_score"
      ],
      function_for_common = "cal_cascade_adaptive_splash_weights",
      lua_script_file = "fountain/cascade/lua/cal_personality_weight.lua",
    ) \
    .fountain_calc_ensemble_score(
      skip = "{{fountain_cascade_skip_ensemble}}",
      range_end = "{{fountain_cascade_ensemble_range_end}}",
      user_new_proportion = "{{fountain_cascade_ensemble_sort_enable_proportion}}",
      user_power_calc = "{{fountain_cascade_ensemble_sort_enable_power_calc}}",
      user_power_calc_v2 = "{{fountain_cascade_ensemble_sort_enable_power_calc_v2}}",
      use_xtr_raw_score =  "{{fountain_cascade_ensemble_sort_use_xtr_raw_score}}",
      enable_time_cost_opt = "{{fountain_cascade_enable_time_cost_opt}}",
      save_score_to_attr = "cascade_ensemble_score",
      user_info_ptr_attr = "userInfoPb",
      action_day = "{{fountain_mc_variant_weight_action_day_num}}",
      rank_smooth = "{{fountain_splash_cascade_ensemble_rank_smooth}}",
      queues = [
        {
          "name": "cascade_score",
          "weight": 0.5,
          "weight_attr": "fountain_ensemble_weight_cascade_score",
          "power_weight_attr": "fountain_ensemble_weight_cascade_score",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
        },
        {
          "name": "cascade_pctr",
          "weight": 0.2,
          "weight_attr": "fountain_ensemble_weight_cascade_click_score",
          "power_weight_attr": "fountain_ensemble_weight_cascade_click_score",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
          "raw_weight_attr": "fountain_splash_cascade_ensemble_sort_click_raw_weight",
          "raw_pow_weight_attr": "fountain_splash_cascade_ensemble_sort_click_raw_pow_weight",
        },
        {
          "name": "cascade_pltr",
          "weight": 0.25,
          "weight_attr": "fountain_ensemble_weight_cascade_like_score",
          "power_weight_attr": "fountain_ensemble_weight_cascade_like_score",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
          "raw_weight_attr": "fountain_splash_cascade_ensemble_sort_like_raw_weight",
          "raw_pow_weight_attr": "fountain_splash_cascade_ensemble_sort_like_raw_pow_weight",
        },
        {
          "name": "cascade_pwtr",
          "weight": 0.18,
          "weight_attr": "fountain_ensemble_weight_cascade_follow_score",
          "power_weight_attr": "fountain_ensemble_weight_cascade_follow_score",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
          "raw_weight_attr": "fountain_splash_cascade_ensemble_sort_follow_raw_weight",
          "raw_pow_weight_attr": "fountain_splash_cascade_ensemble_sort_follow_raw_pow_weight",
        },
        {
          "name": "cascade_pftr",
          "weight": 0.0,
          "weight_attr": "fountain_ensemble_weight_cascade_forward_score",
          "power_weight_attr": "fountain_ensemble_weight_cascade_forward_score",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
          "raw_weight_attr": "fountain_splash_cascade_ensemble_sort_forward_raw_weight",
          "raw_pow_weight_attr": "fountain_splash_cascade_ensemble_sort_forward_raw_pow_weight",
        },
        {
          "name": "cascade_longview_score",
          "weight": 0.1,
          "weight_attr": "fountain_ensemble_weight_cascade_longview_score",
          "power_weight_attr": "fountain_ensemble_weight_cascade_longview_score",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
        },
        {
          "name": "cascade_psvtr",
          "weight": 0.0,
          "weight_attr": "fountain_ensemble_weight_cascade_shortview_score",
          "power_weight_attr": "fountain_ensemble_weight_cascade_shortview_score",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
          "reverse_order": True,
        },
        {
          "name": "cascade_shortview_score2",
          "weight": 0.0,
          "weight_attr": "fountain_ensemble_weight_cascade_shortview_score2",
          "power_weight_attr": "fountain_ensemble_weight_cascade_shortview_score2",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
          "raw_weight_attr": "fountain_splash_cascade_ensemble_sort_shortview_score2_raw_weight",
          "raw_pow_weight_attr": "fountain_splash_cascade_ensemble_sort_shortview_score2_raw_pow_weight",
        },
        {
          "name": "cascade_ptr",
          "weight": 0.0,
          "weight_attr": "fountain_ensemble_weight_cascade_profile_score",
          "power_weight_attr": "fountain_ensemble_weight_cascade_profile_score",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
          "raw_weight_attr": "fountain_splash_cascade_ensemble_sort_profile_raw_weight",
          "raw_pow_weight_attr": "fountain_splash_cascade_ensemble_sort_profile_raw_pow_weight",
        },
        {
          "name": "cascade_pcmtr",
          "weight": 0.0,
          "weight_attr": "fountain_ensemble_weight_cascade_comment_score",
          "power_weight_attr": "fountain_ensemble_weight_cascade_comment_score",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
          "raw_weight_attr": "fountain_splash_cascade_ensemble_sort_comment_raw_weight",
          "raw_pow_weight_attr": "fountain_splash_cascade_ensemble_sort_comment_raw_pow_weight",
        },
        {
          "name": "cascade_pcestr",
          "weight": 0.0,
          "weight_attr": "fountain_ensemble_weight_cascade_cestr_score",
          "power_weight_attr": "fountain_ensemble_weight_cascade_cestr_score",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
          "raw_weight_attr": "fountain_splash_cascade_ensemble_sort_cestr_raw_weight",
          "raw_pow_weight_attr": "fountain_splash_cascade_ensemble_sort_cestr_raw_pow_weight",
        },
        {
          "name": "cascade_pepstr",
          "weight": 0.0,
          "weight_attr": "fountain_ensemble_weight_cascade_epstr_score",
          "power_weight_attr": "fountain_ensemble_weight_cascade_epstr_score",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
          "raw_weight_attr": "fountain_splash_cascade_ensemble_sort_epstr_raw_weight",
          "raw_pow_weight_attr": "fountain_splash_cascade_ensemble_sort_epstr_raw_pow_weight",
        },
        {
          "name": "cascade_pwatch_time",
          "weight": 0.0,
          "weight_attr": "fountain_ensemble_weight_cascade_pwatch_time",
          "power_weight_attr": "fountain_ensemble_weight_cascade_pwatch_time",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
          "raw_weight_attr": "fountain_splash_cascade_ensemble_sort_pwatch_time_raw_weight",
          "raw_pow_weight_attr": "fountain_splash_cascade_ensemble_sort_pwatch_time_raw_pow_weight",
        },
        {
          "name": "cascade_pcltr",
          "weight": 0.0,
          "weight_attr": "fountain_ensemble_weight_cascade_pcltr",
          "power_weight_attr": "fountain_ensemble_weight_cascade_pcltr",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
          "variant_weight":"fountain_ensemble_variant_weight_cascade_pcltr",
          "raw_weight_attr": "fountain_splash_cascade_ensemble_sort_collect_raw_weight",
          "raw_pow_weight_attr": "fountain_splash_cascade_ensemble_sort_collect_raw_pow_weight",
        },
        {
          "name": "cascade_pwtd",
          "weight": 0.0,
          "weight_attr": "fountain_ensemble_weight_cascade_pwtd",
          "power_weight_attr": "fountain_ensemble_weight_cascade_pwtd",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
          "raw_weight_attr": "fountain_splash_cascade_ensemble_sort_pwtd_raw_weight",
          "raw_pow_weight_attr": "fountain_splash_cascade_ensemble_sort_pwtd_raw_pow_weight",
        },
        {
          "name": "cascade_phtr",
          "weight": 0.0,
          "weight_attr": "fountain_ensemble_weight_cascade_phtr",
          "power_weight_attr": "fountain_ensemble_weight_cascade_phtr",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
          "raw_weight_attr": "fountain_splash_cascade_ensemble_sort_hate_raw_weight",
          "raw_pow_weight_attr": "fountain_splash_cascade_ensemble_sort_hate_raw_pow_weight",
        },
        {
          "name": "cascade_ftr_kai",
          "weight": 0.0,
          "weight_attr": "fountain_ensemble_weight_cascade_ftr_kai",
          "power_weight_attr": "fountain_ensemble_weight_cascade_ftr_kai",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
        },
        {
          "name": "cascade_ftr_kai_duration",
          "weight": 0.0,
          "weight_attr": "fountain_ensemble_weight_cascade_ftr_kai_duration",
          "power_weight_attr": "fountain_ensemble_weight_cascade_ftr_kai_duration",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
        },
        {
          "name": "cascade_ipw_opt_ftr",
          "weight": 0.0,
          "weight_attr": "fountain_ensemble_weight_cascade_ipw_opt_ftr",
          "power_weight_attr": "fountain_ensemble_weight_cascade_ipw_opt_ftr",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
        },
        {
          "name": "cascade_pcotr",
          "weight": 0.0,
          "weight_attr": "fountain_ensemble_weight_cascade_pcotr",
          "power_weight_attr": "fountain_ensemble_weight_cascade_pcotr",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
        },
        {
          "name": "cascading_watch_comment_score",
          "weight": 0.0,
          "weight_attr": "fountain_ensemble_weight_cascade_watch_comment_score_weight",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
          "power_weight_attr": "fountain_ensemble_weight_cascade_watch_comment_score_weight",
          "raw_weight_attr": "fountain_cascade_ensemble_sort_cascade_watch_comment_score_raw_weight",
          "raw_pow_weight_attr": "fountain_cascade_ensemble_sort_cascade_watch_comment_score_raw_pow_weight",
        },
        {
          "name": "cascading_comment_like_score",
          "weight": 0.0,
          "weight_attr": "fountain_ensemble_weight_cascade_comment_like_score_weight",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
          "power_weight_attr": "fountain_ensemble_weight_cascade_comment_like_score_weight",
          "raw_weight_attr": "fountain_cascade_ensemble_sort_cascade_comment_like_score_raw_weight",
          "raw_pow_weight_attr": "fountain_cascade_ensemble_sort_cascade_comment_like_score_raw_pow_weight",
        },
        {
          "name": "cascading_comment_time_score",
          "weight": 0.0,
          "weight_attr": "fountain_ensemble_weight_cascade_comment_time_score_weight",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
          "power_weight_attr": "fountain_ensemble_weight_cascade_comment_time_score_weight",
          "raw_weight_attr": "fountain_cascade_ensemble_sort_cascade_comment_time_score_raw_weight",
          "raw_pow_weight_attr": "fountain_cascade_ensemble_sort_cascade_comment_time_score_raw_pow_weight",
        },
        {
          "name": "cascading_valid_play_score",
          "weight": 0.0,
          "weight_attr": "fountain_ensemble_weight_cascade_valid_play_score_weight",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
          "power_weight_attr": "fountain_ensemble_weight_cascade_valid_play_score_weight",
          "raw_weight_attr": "fountain_cascade_ensemble_sort_cascade_valid_play_score_raw_weight",
          "raw_pow_weight_attr": "fountain_cascade_ensemble_sort_cascade_valid_play_score_raw_pow_weight",
        },
        {
          "name": "cascade_fc_pvtr2",
          "weight": 0.0,
          "weight_attr": "fountain_ensemble_weight_cascade_fc_pvtr2_score",
          "power_weight_attr": "fountain_ensemble_weight_cascade_fc_pvtr2_score",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
          "raw_weight_attr": "fountain_splash_cascade_ensemble_sort_fc_pvtr2_raw_weight",
          "raw_pow_weight_attr": "fountain_splash_cascade_ensemble_sort_fc_pvtr2_raw_pow_weight",
        },
        {
          "name": "cascade_fc_pwtd2_inverse",
          "weight": 0.0,
          "weight_attr": "fountain_ensemble_weight_cascade_fc_pwtd2_score",
          "power_weight_attr": "fountain_ensemble_weight_cascade_fc_pwtd2_score",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
          "raw_weight_attr": "fountain_splash_cascade_ensemble_sort_fc_pwtd2_raw_weight",
          "raw_pow_weight_attr": "fountain_splash_cascade_ensemble_sort_fc_pwtd2_raw_pow_weight",
        },
        {
          "name": "cascade_slide_kai",
          "weight": 0.0,
          "weight_attr": "fountain_ensemble_weight_cascade_slide_kai",
          "power_weight_attr": "fountain_ensemble_weight_cascade_slide_kai",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
        },
        {
          "name": "cascade_act_kai",
          "weight": 0.0,
          "weight_attr": "fountain_ensemble_weight_cascade_act_kai",
          "power_weight_attr": "fountain_ensemble_weight_cascade_act_kai",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
        },
        {
          "name": "cascade_wtd_kai",
          "weight": 0.0,
          "weight_attr": "fountain_ensemble_weight_cascade_wtd_kai",
          "power_weight_attr": "fountain_ensemble_weight_cascade_wtd_kai",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
        },
        {
          "name": "cascade_wtd_kai_mix",
          "weight": 0.0,
          "weight_attr": "fountain_ensemble_weight_cascade_wtd_kai_mix",
          "power_weight_attr": "fountain_ensemble_weight_cascade_wtd_kai_mix",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
        },
        {
          "name": "cascade_action_once_interact_score",
          "weight": 0.0,
          "weight_attr": "fountain_ensemble_weight_cascade_action_once_interact_score",
          "power_weight_attr": "fountain_ensemble_weight_cascade_action_once_interact_score",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
        },
        {
          "name": "cascade_action_once_watchtime_score",
          "weight": 0.0,
          "weight_attr": "fountain_ensemble_weight_cascade_action_once_watchtime_score",
          "power_weight_attr": "fountain_ensemble_weight_cascade_action_once_watchtime_score",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
        },
        {
          "name": "cascade_wtd_percent",
          "weight": 0.0,
          "weight_attr": "fountain_ensemble_weight_cascade_wtd_percent",
          "power_weight_attr": "fountain_ensemble_weight_cascade_wtd_percent",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
        },
        {
          "name": "cascade_wtd_duration_mix",
          "weight": 0.0,
          "weight_attr": "fountain_ensemble_weight_cascade_wtd_duration_mix",
          "power_weight_attr": "fountain_ensemble_weight_cascade_wtd_duration_mix",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
        },
        {
          "name": "emp_report_score",
          "weight": 0.0,
          "weight_attr": "fountain_fast_ensemble_weight_cascade_emp_report_score",
          "power_weight_attr": "fountain_fast_ensemble_weight_cascade_emp_report_score",
          "temperature_attr": "fountain_cascade_variant_cluster_sort_temperature",
          "raw_weight_attr": "fountain_fast_cascade_ensemble_report_raw_weight",
          "raw_pow_weight_attr": "fountain_fast_cascade_ensemble_report_raw_pow_weight",
        },
        {
          "name": "source_related_score",
          "weight": 0.0,
          "weight_attr": "fountain_splash_cascade_ensemble_power_weight_source_related_score",
          "power_weight_attr": "fountain_splash_cascade_ensemble_power_weight_source_related_score",
          "temperature_attr": "fountain_cascade_ensemble_temperature",
          "raw_weight_attr": "fountain_splash_cascade_source_related_score_raw_weight",
          "raw_pow_weight_attr": "fountain_splash_cascade_source_related_score_raw_pow_weight",
        },
      ],
    ) \
    ._dump_attr_to_kafka( # 混合排序之后, 将全部item的重要 item attr 落盘
      stage_name = "mc_s2_score",
      dump_item_attr_list = [
        "cascade_score",
        "cascade_pwatch_time",
        "cascade_pwtd",
        "cascade_pctr",
        "cascade_psvtr",
        "cascade_shortview_score2",
        "cascade_longview_score",
        "cascade_pltr",
        "cascade_pwtr",
        "cascade_pftr",
        "cascade_ptr",
        "cascade_pepstr",
        "cascade_pcmtr",
        "cascade_pcestr",
        "cascade_pcltr",
        "cascade_phtr",
        "cascade_ensemble_score",
        "fountain_related_score_v2_detail"
      ]
    ) \
    ._audit_adjust_score() \
    .if_("enable_fountain_mc_living_photo_adjust_splash == 1") \
      ._mc_living_photo_adjust_by_paying_type_splash() \
    .end_() \
    .if_("fountain_cascade_skip_ensemble == 0") \
      .if_("fountain_cascade_skip_questionnaire_boost==0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .enrich_attr_by_lua(
          import_common_attr = [
            "cascade_questionnaire_boost_ratio",
            "cascade_questionnaire_boost_threshold",
          ],
          import_item_attr = [
            "cascade_ensemble_score",
            "questionnaire_score",
          ],
          export_item_attr = [
            "cascade_ensemble_score",
          ],
          function_for_item = "calc_questionnaire_boost",
          lua_script_file = "fountain/cascade/lua/calc_pxtr.lua",
        ) \
      .end_if_() \
      .sort(
        range_end = "{{fountain_cascade_ensemble_range_end}}",
        score_from_attr = "cascade_ensemble_score",
      ) \
    .end_if_() \
    .if_("(enable_splash_cascading_personified_author_boost == 1) or (enable_splash_cascading_blacklist_author_boost == 1) or (enable_splash_cascading_personified_cart_boost == 1)") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "splash_cascading_personified_author_boost_coef", "as": "personified_author_coeff"},
          {"name": "splash_cascading_blacklist_author_boost_coef", "as": "blacklist_author_coeff"},
          {"name": "splash_cascading_merchant_cart_boost_coef", "as": "merchant_cart_coeff"},
          {"name": "fountain_personified_author_fans_thre_max", "as": "author_fans_thre_max"},
          {"name": "fountain_personified_author_fans_thre_min", "as": "author_fans_thre_min"},
        ],
        import_item_attr = [
          {"name": "author__fans_count", "as": "author_fans_count"},
          {"name": "eyeshot_source", "as": "eyeshot_source"},
          {"name": "merchant_photo_cart_relation", "as": "cart_relation"},
          {"name": "merchant_item_info__item_id_list", "as": "cart_itemlist"},
          {"name": "live_photo_info__is_living", "as": "is_living"},
          {"name": "cascade_ensemble_score", "as": "ensemble_score"},
        ],
        export_item_attr = [
          {"name": "ensemble_score", "as": "cascade_ensemble_score"},
        ],
        function_name = "PersonifiedAuthorBoost",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_if_() \
    .if_("enable_fountain_mc_hv_pic_boost == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      ._mc_high_value_pic_boost() \
    .end_() \
    .if_('skip_fountain_pre_fullrank_model_limit_v2_splash == 0') \
      .if_('enable_fountain_splash_cascade_quota_control == 1') \
        .explore_control_hetu_count_arranger(
          duration_ms_attr = "duration_ms",
          enable_duration_control_diversity = "{{enable_fountain_splash_cascade_duration_quota_control}}",
          duration_control_diversity_start = "{{fountain_splash_cascade_control_duration_diversity_start}}",
          keep_size = "{{fountain_pre_fullrank_size_limit_v2_splash}}",
          duration_0_7s_max_size = "{{fountain_splash_cascade_control_duration_0_7s_max_size}}",
          duration_7_9s_max_size = "{{fountain_splash_cascade_control_duration_7_9s_max_size}}",
          duration_9_12s_max_size = "{{fountain_splash_cascade_control_duration_9_12s_max_size}}",
          duration_12_17s_max_size = "{{fountain_splash_cascade_control_duration_12_17s_max_size}}",
          duration_17_20s_max_size = "{{fountain_splash_cascade_control_duration_17_20s_max_size}}",
          duration_58_120s_max_size = "{{fountain_splash_cascade_control_duration_58_120s_max_size}}"
        ) \
      .else_() \
        .truncate(
          size_limit = "{{fountain_pre_fullrank_size_limit_v2_splash}}",
        ) \
      .end_if_() \
    .end_if_() \
    .perflog_reason_count(
      check_point = "cascade_finish",
    ) \
    .count_reco_result(
      name = "fountain_splash_mc_stage2",
      traceback = True,
      save_count_to = "cascade_final"
    ) \
    .perflog_attr_value(check_point="splash_cascade_item_num",
      common_attrs=[
        "cascade_enter",
        "cascade_first_truncate",
        "cascade_pre_filter",
        "cascade_variant",
        "cascade_final"
      ]
    ) \
    .pack_item_attr(
      item_source = {"reco_results": True},
      mappings = [{
        "from_item_attr": "item_id",
        "to_common_attr": "cascade_output_item_id_list",
      }]
    )

    return self

  def _mc_living_photo_adjust_by_paying_type_splash(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "living_certain_aid_list", "as": "attr_list"},
      ],
      import_item_attr = [
        {"name": "author__id", "as": "attr"},
      ],
      export_item_attr = [
        {"name": "is_in_set", "as": "is_certain_ua"},
      ],
      function_name = "AttrIsInSet",
      class_name = "ExploreLightFunctionSetV2"
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "is_live_big_g_user", "as": "is_live_big_g_user"},
        {"name": "uUserKuaishouLivePayTag", "as": "user_live_paying_type"},
        {"name": "fountain_mc_living_photo_boost_coef_str_splash", "as": "paying_user_boost_coef_str"},
        {"name": "fountain_mc_living_photo_boost_coef_big_g_splash", "as": "boost_coef_big_g"},
      ],
      export_common_attr = [
        {"name": "living_boost_coef", "as": "mc_fountain_living_photo_coef_splash"}
      ],
      function_name = "LivingCalcBoostCoef",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "mc_fountain_living_photo_coef_splash", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "cascade_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "cascade_ensemble_score"},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        "is_true_living" : 1, "is_certain_ua" : 1
      }
    )

    return self

  def _cascade_fc_model_in_splash(self):
    """
    首屏fc_model
    """
    self \
    .if_("enable_fc_in_splash_flow == 1 and enable_fc_for_splash == 0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .extract_with_ks_sign_feature(
        extractor_kconf_path = "reco.hot.fountainLeafMcFeature",
        caller_model = "{{fountain_cascade_fc_predict_service}}",
        feature_list = cascade_fc_sim3_feature,
        update_ks_sign_feature_type = 1,
        update_interval_sec = 600,
        user_info_attr = "userInfoPb",
        common_slots_output = "user_feature_slots",
        common_parameters_output = "user_feature_signs",
      ) \
      .delegate_enrich(
        kess_service = "{{fountain_cascade_fc_predict_service}}",
        request_type = "{{fountain_cascade_fc_request_type}}",
        timeout_ms = 100,
        send_common_attrs = ["user_feature_slots", "user_feature_signs"],
        recv_item_attrs = [
          {"name":"fc_pctr_value", "as":"cascade_fc_pctr"},
          {"name":"fc_plvr_value", "as":"cascade_fc_plvtr"},
          {"name":"fc_psvr_value", "as":"cascade_fc_psvtr"},
          {"name":"fc_pvtr_value", "as":"cascade_fc_pvtr"},
        ],
        use_item_id_in_attr = "item_id",
        use_packed_item_attr = True,
      ) \
      .enrich_attr_by_light_function(
        import_item_attr = [
          "cascade_fc_pctr", "cascade_fc_plvtr", "cascade_fc_psvtr", "cascade_fc_pvtr",
        ],
        export_item_attr = ["cascade_pctr", "cascade_plvtr", "cascade_psvtr", "cascade_pwatch_time"],
        function_name = "ReplaceMcPxtr",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .fountain_enrich_cascade_score(
        pwatch_time_attr = "cascade_pwatch_time",
        pptr_attr = "cascade_ptr",
        pepstr_attr = "cascade_pepstr",
        pcestr_attr = "cascade_pcestr",
        pcmtr_attr = "cascade_pcmtr",
        pwtd_attr = "cascade_pwtd",
        pslide_attr = "cascade_slide_kai",
        svtr_coeff = "{{fountain_cascade_svtr_coeff}}",
        svtr_power = "{{fountain_cascade_svtr_power}}",
        short_play_discount_value = "{{fountain_cascade_short_play_discount_value}}",
        lvtr_use_predict_watch_time = "{{fountain_cascade_ensemble_lvtr_use_predict_watch_time}}",
        mid_photo_boost_coeff = "{{fountain_cascade_mid_photo_boost_coeff}}",
      ) \
      .log_debug_info(
        for_debug_request_only=True,
        item_attrs=[
          "cascade_fc_psvtr",
          "cascade_psvtr"
          ],
        common_attrs=["enable_fc_in_splash_flow"]
      ) \
    .end_()

    return self

  def _prerank_v2(self):
    """
    粗排预排序
    """
    self \
    .if_("enable_fountain_splash_prerank == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .sort(
        score_from_attr = "prerank_score",
        traceback = True,
      ) \
      .truncate(
        size_limit = "{{fountain_splash_prerank_keep_size}}",
      ) \
      .count_reco_result(save_count_to = "cascade_prerank_truncate") \
      .log_debug_info(
        for_debug_request_only=True,
        common_attrs=["cascade_prerank_truncate"],
        item_attrs=["prerank_score"],
      ) \
    .end_()

    return self

