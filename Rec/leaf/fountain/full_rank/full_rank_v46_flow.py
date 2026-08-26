#!/usr/bin/env python3
# coding=utf-8

from dragonfly.common_leaf_dsl import LeafFlow
from dragonfly.ext.subdivision.subdivision_api_mixin import subdivisionApiMixin
from full_rank.full_rank_base_flow import FullRankBaseFlow, item_features, fullrank_common_attrs, fullrank_splash_attrs, \
  photo_features, user_features, user_features_v2, photo_pxtr_features, fullrank_common_copy_attrs
from full_rank.ab_params import fullrank_common_params, fullrank_splash_params, fullrank_common_param_abhit, fullrank_splash_param_abhit
from full_rank.fullrank_base_features import *
from util import enrich_ab_param

class FullRankV46Flow(FullRankBaseFlow, subdivisionApiMixin):
  def __init__(self):
    LeafFlow.__init__(self, "full_rank_v46")
    self \
      .namespace_(ns = "full_rank_v46", nest = True) \
      ._timestamp_begin("rank_splash") \
      ._rank() \
      .namespace_()

  def fullrank_rrwtd_predict_splash(self):
    """
    仅在首屏生效的rerank wtd模型
    """
    self \
      .if_("enable_fountain_deep_rrwtd_splash_predict == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .delegate_enrich(
          kess_service = "{{fountain_fullrank_deep_rrwtd_kess_service}}",
          recv_item_attrs = [
            {"name": "point_pos", "as": "fullrank_rrwtd_score"},
            {"name": "point_next", "as": "fullrank_rrnext_score"},
          ],
          timeout_ms = 100,
          send_item_attrs = rr_photo_features,
          send_common_attrs = rr_user_feature,
          request_type = "{{fountain_deep_rrwtd_request_type}}",
          partition_size = "{{fountain_deep_rrwtd_partition_size}}",
        ) \
      .end_if_()
    return self

  def enrich_plc_rank_model_attr_splash(self):
    self \
    .if_("enable_fountain_plc_rank_model_predict_splash == 1") \
      .explore_custom_trim_user_info(
        user_info_attr = "userInfo",
        save_trimed_user_info_to_attr = "fullrank_plc_trimmed_user_info",
        trim_user_info = [
          "id",
          "device_id",
          "follow_count",
          "like_count",
          "upload_count",
          "basic_info.age_segment",
          "basic_info.gender",
          "gender",
          "true_year",
          "true_gender",
          "infer_gender",
          "app_version"
          "active_days",
          "location.city_id",
          "location.region_type",
          "client_id",
          "fans_count",
          "request_location.poi_type",
          "request_location.province_id",
          "request_location.city_id",
          "visit_mod",
          "user_profile_v1.click_list.author_id",
          "user_profile_v1.click_list.photo_id",
          "user_profile_v1.follow_list.author_id",
          "user_profile_v1.follow_list.photo_id",
          "user_profile_v1.like_list.author_id",
          "user_profile_v1.like_list.photo_id",
          "user_profile_v1.video_playing_stat.playing_time",
          "user_profile_v1.video_playing_stat.author_id",
          "user_profile_v1.video_playing_stat.photo_id",
        ],
      ) \
      .enrich_attr_by_light_function(
        export_item_attr = [
          "is_plc_item",
        ],
        import_common_attr = [
          "fountain_plc_business_type_predict_str",
        ],
        import_item_attr = [
          "plc_business_type",
        ],
        function_name = "IsPlcItemAttr",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .delegate_enrich(
        kess_service = "{{fountain_plc_rank_model_service}}",
        send_common_attrs = [
          { "name": "fullrank_plc_trimmed_user_info", "as": "user_info_str" }
        ],
        send_item_attrs = [
          { "name": "live_photo_info__is_living", "as": "living" },
          { "name": "fullrank_sim_pevtr", "as": "pctr"},
          { "name": "fullrank_sim_pltr", "as": "pltr"},
          { "name": "fullrank_sim_pwtr", "as": "pwtr"},
          { "name": "fullrank_sim_pftr", "as": "pftr"},
          { "name": "fullrank_sim_plvtr", "as": "plvtr"},
          { "name": "fullrank_sim_pvtr", "as": "pvtr"},
          { "name": "fullrank_sim_pptr", "as": "pptr"},
          { "name": "fullrank_sim_pcmtr", "as": "pcmtr"},
          "cascade_pctr",
          "cascade_pltr",
          "cascade_pwtr",
          "cascade_pftr",
          "cascade_plvtr",
          { "name": "cascade_psvtr", "as": "cascade_psvr"},
          "reason",
        ],
        recv_item_attrs = [
          {"name": "plc_click_predict", "as": "fountain_plc_ctr"},
        ],
        target_item = {"is_plc_item": 1},
        timeout_ms = 100,
        request_type = "{{fountain_plc_rank_model_request_type}}",
        partition_size = "{{fountain_plc_rank_model_partition_size}}",
      ) \
    .end_()
    return self

  def _rank(self):
    self \
    .get_abtest_params(
      biz_name = "RECO_RPC",
      ab_params = enrich_ab_param(fullrank_common_params + fullrank_splash_params),
      prioritized_suffix = "{{_ABTEST_SUFFIX_LIST_}}"
    ) \
    .get_abtest_params(
      biz_name = "RECO_RPC",
      ab_params = fullrank_common_param_abhit + fullrank_splash_param_abhit,
      prioritized_suffix = "{{_ABTEST_SUFFIX_LIST_}}"
    ) \
    .if_("enable_replace_colossus_v2_from_fr == 1") \
      .copy_attr(
        attrs = [
          {
            "from_common": "colossus_v2_resp",
            "to_common": "colossus_resp_v2",
          },
        ],
      ) \
    .end_() \
    .set_attr_value(
      common_attrs = [
        {
          "name" : "featureUserIsFountainSplash",
          "type" : "int",
          "value" : 1
        },
        {
          "name" : "featureUserIsFountainRequest",
          "type" : "int",
          "value" : 1
        }
      ]
    ) \
    .set_attr_default_value(
      item_attrs = [
        {
          "name" : "merchant_fr_photo_gmv_score",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "merchant_fr_living_ctcvr_score",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "merchant_fr_living_ctcvr_gmv_score",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "fr_living_ctr",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "fr_living_mix_score",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "fr_living_lwtr",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "fr_living_gtr",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "fountain_produce_rank_mtctr",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "fountain_produce_rank_twhtr",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "fountain_produce_rank_mfctr",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "fountain_produce_rank_mtcotr",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "fountain_produce_rank_mtjtr",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "fountain_produce_rank_mtm1",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "fountain_produce_rank_uploads",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "fountain_produce_rank_upload_sum_score",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "fountain_produce_rank_consuv_sum_score",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "fountain_produce_rank_new_user_clk_score",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "fountain_produce_rank_month_user_clk_score",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "fountain_produce_rank_weeks_user_clk_score",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "fountain_produce_rank_week_user_clk_score",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "fountain_produce_rank_new_user_ups_score",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "fountain_produce_rank_month_user_ups_score",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "fountain_produce_rank_weeks_user_ups_score",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "fountain_produce_rank_week_user_ups_score",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "fountain_plc_ctr_score",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "cascade_distill_show",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "hetu_retargeting_score",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "sidinfo_retargeting_discount_score",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "explore_valid_interest_score",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "fullrank_life_stage_cid_ipw_debias_plvtr",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "full_rank_rise_follow_boost_score",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "fullrank_age_gender_prof_cid_ipw_debias_plvtr",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "fullrank_age_gender_north_cid_ipw_debias_plvtr",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "fullrank_age_gender_cid_ipw_debias_plvtr",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "fullrank_reco_base_model_gen_time_wt",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "fullrank_reco_base_model_evtr",
          "type" : "double",
          "value" : 0.0
        },
        {
          "name" : "fullrank_reco_base_model_lvtr",
          "type" : "double",
          "value" : 0.0
        }
      ]
    ) \
    .if_("enable_fountain_transform_photo_proinc_type == 1") \
      .item_attr_operation(
        item_attr_a = "photo_proinc_type",
        common_attr_b = 8,
        operator = "&",
        output_attr = "userfulness_author_tag"
      ) \
      .cast_attr_type(
        attr_type_cast_configs = [
          {
            "to_type": "double",
            "from_item_attr": "userfulness_author_tag",
            "to_item_attr": "userfulness_author_score"
          }
        ]
      ) \
    .end_if_() \
    .disable_forward_social_queue() \
    .fetch_similar_user_list() \
    .fetch_duration_group_id() \
    .get_item_attr_by_distributed_flat_index(
      skip = "{{fountain_skip_get_fullrank_attrs_distributed}}",
      photo_store_kconf_key = "reco.distributedIndex.hotPhotoInfoCommonIndex",
      perf_log = "fullrank",
      photo_store_request_data_set_tags_attr = "fountain_request_data_set_tags",
      use_dynamic_photo_store = True,
      item_id_attr = "item_id",
      attrs = fullrank_common_attrs + fullrank_splash_attrs,
    ) \
    .copy_attr(
      attrs = fullrank_common_copy_attrs,
    ) \
    .if_("fountain_enable_rank_write_rank_neg_result_to_redis == 1", to_be_delete = "date=2024-05-29;committer=denghong") \
      .pack_item_attr(
        item_source = {
          "reco_results": True
        },
        mappings = [{
          "from_item_attr": "photo_id",
          "to_common_attr": "rank_candidate_photo_id_list",
        }]
      ) \
    .end_() \
    .if_("fountain_enable_get_pxtr_index == 1") \
      .get_cascade_index() \
    .end_() \
    .enrich_fullrank_features_by_lua() \
    .calculate_comment_ltr()\
    .calc_hate_list_similary_score() \
    .delegate_enrich(
      skip = "{{skip_fountain_splash_slide_predict}}",
      kess_service = "{{fountain_splash_slide_predict_kess_service}}",
      recv_item_attrs = [{"name":"slide", "as":"fountain_splash_slide"}],
      timeout_ms = 150,
      send_item_attrs = [feature["name"] for feature in photo_features if feature["name"] not in photo_pxtr_features],
      send_common_attrs = user_features_v2,
      request_type = "kai_predict",
      partition_size = "{{fountain_splash_slide_predict_partition_size}}",
    ) \
    .enrich_fullrank_score_attr_splash() \
    .request_feature_server() \
    .fullrank_cl_ltr_predict_pre() \
    .fetch_sim_gsu_feature() \
    .fullrank_cl_ltr_predict_post() \
    .if_("fountain_enable_get_ranking_pxtr_index == 1") \
      .get_ranking_index() \
    .end_() \
    .if_("fountain_rank_ensemble_filter_opt == 1") \
      .ensemble_filter("fountain_splash_fr_stage1") \
    .end_if_() \
    .trans_sim_pxtr_names() \
    .if_("fountain_enable_pxtr_calibration == 1") \
      .fountain_fullrank_pxtr_calibration() \
    .end_() \
    .fullrank_ltr_predict_splash() \
    .fullrank_rrwtd_predict_splash() \
    .enrich_plc_rank_model_attr_splash() \
    .calc_plc_rank_model_attr_score() \
    .enrich_fullrank_score_attr() \
    .calculate_xgb_ltr() \
    .if_("fountain_skip_user_ada_xtr_score==0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .cal_user_ada_xtr_score() \
    .end_if_() \
    .if_("fountain_splash_skip_user_rl_xtr_score==0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .cal_user_rl_xtr_score() \
    .end_if_() \
    .if_("fountain_splash_enable_calc_topk_mgs_expected_score == 1", to_be_delete = "date=2024-05-29;committer=liuhu") \
      .calc_topk_mgs_expected_score() \
    .end_if_() \
    .if_("fountain_enable_calc_interact_similarity_score == 1", to_be_delete = "date=2024-05-29;committer=lijinyu") \
      .calc_interact_similarity_score() \
    .end_if_() \
    .cal_opportunity_cost_score() \
    .cal_action_once_score() \
    .cal_cascade_linear_score() \
    .cal_value_multiply_score() \
    .if_("enable_fountain_fullrank_iput_score == 1") \
      .calc_fountain_fullrank_iput_score() \
    .end_() \
    .if_("fountain_enable_fullrank_get_hetu_behavior_score == 1") \
     .fullrank_get_hetu_behavior_score() \
    .end_() \
    .cal_satisfy_score() \
    .cal_fit_ptime_score() \
    .if_("fountain_splash_enable_rank_triplem_time_queue == 1") \
      .cal_triplem_time_score() \
    .end_if_() \
    .if_("fountain_splash_enable_rank_triplem_interaction_queue == 1") \
      .cal_triplem_interaction_score() \
    .end_if_() \
    .calc_debias_mix_score() \
    .calc_debias_score() \
    .calculate_debias_pxtr() \
    .cal_distill_fusion_score() \
    .if_("fountain_rank_ensemble_filter_opt == 0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .fullrank_ensemble_pre_filter() \
    .end_if_() \
    .calc_fr_cdf_mapping() \
    .enrich_attr_by_lua(
      import_common_attr = [
        "fountain_fullrank_fr_duration_factor_offset",
        "skip_fountain_finish_rate_adjust",
        "fountain_fullrank_finish_duration_factor_max_value",
        "fountain_fullrank_finish_duration_factor_pow_weight",
        "fountain_ensemble_power_weight_fullrank_ltr_v4_next",
        "page",
        "fountain_fullrank_next_score_debias_pow_weight",
        "skip_fountain_fullrank_ltr_v4_next_splash",
        "fountain_fullrank_enable_cdf_fr_smooth",
        "fountain_fullrank_cdf_fr_smooth_alpha",
        "fountain_fullrank_cdf_fr_smooth_beta",
        "skip_fountain_finish_rate_adjust_v3",
        "fountain_fullrank_not_svr_pow_weight_for_pfr",
        "fountain_skip_fr_pred_only_fast_v1",
       ],
      import_item_attr = [
        "fullrank_ltr_v4_fountain_finish_rate",
        "duration_ms",
        "fullrank_sim_psvr",
        "fullrank_dura_cdf_pfr",
       ],
      export_common_attr = [
        "fountain_ensemble_power_weight_fullrank_ltr_v4_next",
      ],
      export_item_attr = [
        "fullrank_ltr_v4_fountain_finish_rate",
        "long_term_interest_ee_score",
        "fullrank_dura_cdf_pfr",
      ],
      function_for_item = "unify_fullrank_pxtr",
      function_for_common = "unify_fullrank_common_attr",
      lua_script_file = "fountain/full_rank/lua/unify_fullrank_pxtr.lua",
    ) \
    .fountain_related_cluster_sort(
      skip = "{{skip_fountain_splash_fullrank_multi_cluster_sort}}",
      global_cut_ratio = "{{fountain_splash_fullrank_multi_cluster_global_cut_ratio}}",
      min_survival = "{{fountain_splash_fullrank_multi_cluster_min_survival}}",
      queues = [
        {
          "name": "cascade_score",
          "weight": 0.5,
          "weight_attr": "fountain_splash_fullrank_variant_weight_cascade_score",
          "power_weight_attr": "fountain_splash_fullrank_variant_weight_cascade_score"
        },
        {
          "name": "fullrank_sim_click_score",
          "weight": 0.2,
          "weight_attr": "fountain_splash_fullrank_variant_weight_click_score",
          "power_weight_attr": "fountain_splash_fullrank_variant_weight_click_score",
        },
        {
          "name": "fullrank_sim_like_score",
          "weight": 0.25,
          "weight_attr": "fountain_splash_fullrank_variant_weight_like_score",
          "power_weight_attr": "fountain_splash_fullrank_variant_weight_like_score",
        },
        {
          "name": "fullrank_sim_follow_score",
          "weight": 0.18,
          "weight_attr": "fountain_splash_fullrank_variant_weight_follow_score",
          "power_weight_attr": "fountain_splash_fullrank_variant_weight_follow_score",
        },
        {
          "name": "fullrank_sim_longview_score_no_bias",
          "weight": 0.1,
          "weight_attr": "fountain_splash_fullrank_variant_weight_longview_score",
          "power_weight_attr": "fountain_splash_fullrank_variant_weight_longview_score",
        },
        {
          "name": "fullrank_sim_out_pctr",
          "weight": 0.0,
          "weight_attr": "fountain_splash_fullrank_variant_weight_out_ctr_score",
          "power_weight_attr": "fountain_splash_fullrank_variant_weight_out_ctr_score",
        },
        {
          "name": "fullrank_sim_psvr",
          "weight": 0.0,
          "weight_attr": "fountain_splash_fullrank_variant_weight_shortview_score",
          "power_weight_attr": "fountain_splash_fullrank_variant_weight_shortview_score",
          "reverse_order": True,
        },
        {
          "name": "fullrank_sim_pwatchtime_no_bias",
          "weight": 0.0,
          "weight_attr": "fountain_splash_fullrank_variant_weight_watch_time",
          "power_weight_attr": "fountain_splash_fullrank_variant_weight_watch_time",
        },
      ],
      cluster_config = [
        {
          "item_attr": "is_photo_same_hetu_face_id",
          "ratio": 1.0,
          "importance": 5,
          "min_survival": 30
        },
        {
          "item_attr": "is_photo_same_hetu_level_four",
          "ratio": 1.0,
          "importance": 5,
          "min_survival": 30
        },
        {
          "item_attr": "is_photo_same_hetu_tag",
          "ratio": 1.0,
          "importance": 5,
          "min_survival": 20
        },
        {
          "item_attr": "is_photo_same_author_third_level_id",
          "ratio": 0.8,
          "importance": 3,
          "min_survival": 20
        },
        {
          "item_attr": "is_photo_same_hetu_level_three",
          "ratio": 0.8,
          "importance": 3,
          "min_survival": 20
        },
        {
          "item_attr": "is_photo_same_hetu_level_two",
          "ratio": 0.5,
          "importance": 2,
          "min_survival": 10
        },
        {
          "item_attr": "is_photo_same_hetu_level_one",
          "ratio": 0.3,
          "importance": 1,
          "min_survival": 5
        },
      ]
    ) \
    .perflog_reason_count(
      check_point = "post_fullrank_cluster_sort",
    ) \
    .if_("fullrank_splash_use_jarvis_auto_param == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .explore_sphinx_param() \
    .end_if_() \
    .pack_item_attr(
      skip = "{{fountain_fullrank_skip_calc_pxtr_avg}}",
      item_source = {
        "reco_results": True,
      },
      mappings = [
      {
        "aggregator": "avg",
        "from_item_attr": "fullrank_sim_click_score",
        "to_common_attr": "pctr_avg"
      },
      {
        "aggregator": "avg",
        "from_item_attr": "fullrank_sim_psvr",
        "to_common_attr": "psvr_avg"
      },
      {
        "aggregator": "avg",
        "from_item_attr": "fullrank_sim_pltr",
        "to_common_attr": "pltr_avg"
      },
      {
        "aggregator": "avg",
        "from_item_attr": "fullrank_sim_pwtr",
        "to_common_attr": "pwtr_avg"
      },
      {
        "aggregator": "avg",
        "from_item_attr": "fullrank_sim_pftr",
        "to_common_attr": "pftr_avg"
      },
      {
        "aggregator": "avg",
        "from_item_attr": "fullrank_sim_pcmtr",
        "to_common_attr": "pcmtr_avg"
      },
      {
        "aggregator": "avg",
        "from_item_attr": "fullrank_sim_pptr",
        "to_common_attr": "pptr_avg"
      },
      {
        "aggregator": "avg",
        "from_item_attr": "fullrank_sim_pevtr",
        "to_common_attr": "pevtr_avg"
      },
      {
        "aggregator": "avg",
        "from_item_attr": "fullrank_detail_new_pevtr_v2",
        "to_common_attr": "pevtr_v2_avg"
      },
      {
        "aggregator": "avg",
        "from_item_attr": "fullrank_sim_plvtr",
        "to_common_attr": "plvtr_avg"
      },
      {
        "aggregator": "avg",
        "from_item_attr": "fullrank_sim_pfintr",
        "to_common_attr": "pfintr_avg"
      },
      {
        "aggregator": "avg",
        "from_item_attr": "fullrank_sim_pwatchtime_no_bias",
        "to_common_attr": "pwatchtime_avg"
      },
      ]
      ) \
    .if_("enable_splash_use_request_based_pxtr_ada_weight == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .cal_request_pxtr_ada_weight() \
    .end_if_() \
    .if_("enable_fountain_fr_htr_weight_adjust_by_user_htr == 1 and fountain_recent_hate_count > fountain_fr_koc_htr_count_threshold") \
      ._fullrank_htr_weight_adjust_by_uv_htr() \
    .end_if_() \
    .if_("enable_cal_user_group_emp_xtr_in_rank == 1") \
      .cal_user_group_emp_xtr_all() \
    .end_if_() \
    .if_("enable_cal_user_group_dynamic_weight_in_rank == 1", to_be_delete = "date=2024-05-29;committer=xuwei09") \
      .cal_user_group_dynamic_weight() \
    .end_if_() \
    .if_("fountain_rank_enable_get_pxtr_boost_coef == 1") \
      .request_xtr_adap_weight() \
    .end_if_() \
    .if_("fountain_rank_enable_request_pxtr_judge == 1") \
      .cal_request_rank_weight_adjust() \
    .end_if_() \
    .if_("enable_cal_request_adaptive_score == 1") \
      .cal_request_adaptive_score() \
    .end_if_() \
    .if_("enable_fountain_share_pull_ftr_weight_adjust_coef == 1") \
      .share_pull_ftr_weight_adjust_coef() \
    .end_() \
    .if_("enable_fountain_cal_share_pull_ftr_full_rank == 1") \
      .cal_share_pull_ftr_full_rank() \
    .end_() \
    .related_score_weight_adjust_only_splash() \
    .replace_fullrank_ltr_score_weight_only_splash() \
    .calc_fullrank_ensemble_score() \
    .if_("skip_fountain_fullrank_rrr_discount == 0") \
      .rrr_discount() \
    .end_if_() \
    .fr_ensemble_score_multiply_gate() \
    .fr_discount_single_pic() \
    .if_("enable_splash_fullrank_personified_author_boost == 1 or enable_splash_fullrank_blacklist_author_boost == 1 or enable_splash_fullrank_personified_cart_boost == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "splash_fullrank_personified_author_boost_coef", "as": "personified_author_coeff"},
          {"name": "splash_fullrank_blacklist_author_boost_coef", "as": "blacklist_author_coeff"},
          {"name": "splash_fullrank_merchant_cart_boost_coef", "as": "merchant_cart_coeff"},
          {"name": "fountain_personified_author_fans_thre_max", "as": "author_fans_thre_max"},
          {"name": "fountain_personified_author_fans_thre_min", "as": "author_fans_thre_min"},
        ],
        import_item_attr = [
          {"name": "author__fans_count", "as": "author_fans_count"},
          {"name": "eyeshot_source", "as": "eyeshot_source"},
          {"name": "merchant_photo_cart_relation", "as": "cart_relation"},
          {"name": "merchant_item_info__item_id_list", "as": "cart_itemlist"},
          {"name": "live_photo_info__is_living", "as": "is_living"},
          {"name": "fullrank_ensemble_score", "as": "ensemble_score"},
        ],
        export_item_attr = [
          {"name": "ensemble_score", "as": "fullrank_ensemble_score"},
        ],
        function_name = "PersonifiedAuthorBoost",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_if_() \
    .if_("fountain_enable_fullrank_score_adjust_splash == 1") \
      .enrich_attr_by_lua(
        import_common_attr = [
          "enable_fountain_movie_ip_boost",
          "fountain_movie_ip_boost_ratio",
          "long_duration_boost",
          "long_duration_boost_min_plvtr",
          "fullrank_enable_questionnaire_boost",
          "fullrank_questionnaire_boost_ratio",
          "fullrank_questionnaire_boost_threshold",
        ],
        import_item_attr = [
          "fullrank_ensemble_score",
          "source_related_score",
          "duration_ms",
          "reason",
          "fullrank_sim_plvtr",
          "questionnaire_score"
        ],
        export_item_attr = [
          "fullrank_ensemble_score_after_adjust",
        ],
        function_for_item = "fullrank_score_adjust_splash",
        lua_script_file = "fountain/full_rank/lua/fullrank_score_adjust.lua",
      ) \
    .else_() \
      .copy_attr(
        attrs = [
          {
            "from_item": "fullrank_ensemble_score",
            "to_item": "fullrank_ensemble_score_after_adjust"
          }
        ]
      ) \
    .end_() \
    .if_("enable_fountain_fr_hv_pic_boost == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .fr_high_value_pic_boost() \
    .end_() \
    .if_("enable_fountain_fr_interact_similarity_score_boost == 1", to_be_delete = "date=2024-05-29;committer=lijinyu") \
      .fr_interact_similarity_score_boost() \
    .end_() \
    .if_("enable_fountain_hetu_level1_discount_splash == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .fullrank_hetu_level1_discount() \
    .end_() \
    .audit_adjust_score() \
    .sort(
      skip = "{{skip_fullrank_ensemble_score_adjust}}",
      score_from_attr = "fullrank_ensemble_score_after_adjust",
    ) \
    .if_("enable_fountain_splash_fr_variant == 1") \
      .variant(
        variant_config = {
          "default_decay_window_size": 20,
          "default_decay_occurrent_times": 2,
          "default_decay_rate": 0.2,
          "author__id": {
            "decay_window_size": 20,
            "decay_occurrent_times": 2,
            "decay_rate": 0.1,
          },
          "is_photo_author_followed": {
            "decay_window_size": "{{fountain_variant_follow_author_id_win_size_splash}}",
            "decay_occurrent_times": 2,
            "decay_rate": 0.1,
            "enabled": "{{enable_fountain_variant_follow_author_id_splash}}"
          },
        }
      ) \
    .end_() \
    .if_("fountain_splash_enable_i2i_reason_force_insert == 1 and morePage == 0") \
      .force_insert(
        reason = [
          10002, 10046, 10038, 10788, 10790, 11201, 11207, 10071, 10076, 10060, 10081, 10082, 10086, 10087, 10083,
          10085, 10131, 10132, 10133, 10097, 10072, 10134, 10003, 10308,
          10098, 10099, 10055, 10004, 10056, 10057, 10135, 10140, 10141, 10145, 10146, 10147, 10148, 10149, 10300,
          10150, 10310, 10311, 10312, 10400, 10317, 10401, 10402, 10403, 10405, 10406, 10318, 10324, 10325, 10326,
          10328, 10329, 10407, 10408, 10411, 10412, 10416
        ],
        position = 0,
        limit = "{{fountain_post_process_promotion_nn_limit}}",
      ) \
    .end_() \
    .if_("enable_fountain_splash_fr_related_promotion == 1") \
      .fountain_related_force_insert(
        out_page_promotion = "{{enable_out_page_promotion}}",
        out_page_promotion_num_page2 = "{{out_page_promotion_num_page2}}",
        out_page_promotion_num_page3 = "{{out_page_promotion_num_page3}}"
      ) \
    .end_() \
    .if_("enable_fountain_splash_fr_force_insert_movie_ip == 1") \
      .intermix(
        mix_on_attr = "reason",
        mix_pattern = [10305],
        num_limit = "{{fountain_force_insert_movie_ip_post_num}}",
      ) \
    .end_() \
    .if_("fountain_enable_rank_write_rank_neg_result_to_redis == 1", to_be_delete = "date=2024-05-29;committer=liuhao07") \
      .write_rank_neg_result_to_redis() \
    .end_() \
    .copy_item_meta_info(
      save_item_seq_to_attr = "rank_index_after_es"
    ) \
    .perflog_reason_count(
      check_point = "fullrank_finish",
    ) \
    .copy_user_meta_info(
      save_flow_cpu_cost_to_attr = "full_rank_splash_cpu_cost_ts",
    ) \

    return self

  def related_score_weight_adjust_only_splash(self):
    self \
    .if_("enable_fountain_splash_cacl_long_term_relevance_preference_weight == 1 and enable_fountain_splash_adjust_relavance_intention_by_user_preference_weight == 1") \
      .gen_common_attr_by_lua(
        attr_map={
          "fountain_fullrank_source_related_score_weight": "fountain_fullrank_source_related_score_weight * user_long_term_relevance_preference_weight",
        }
      ) \
    .end_() \
    .if_("enable_fountain_splash_adjust_relate_score_by_relavance_intention == 1") \
      .gen_common_attr_by_lua(
        attr_map={
          "fountain_fullrank_source_related_score_weight": "fountain_fullrank_source_related_score_weight * splash_fullrank_ltr_user_relavance_intention_score",
        }
      ) \
    .end_()
    return self

  def replace_fullrank_ltr_score_weight_only_splash(self):
    # 分离首屏非首屏队列参数，这些参数 combo、普通世界中已使用，为了不影响线上指标，用开关控制替换，相关 ltr 替换掉非首屏 ltr 后再做删除。
    self \
    .if_("enable_fountain_replace_fullrank_ltr_score_weight_only_splash == 1") \
      .copy_attr(
        attrs=[
          {
            "from_common": "fountain_ensemble_power_weight_fullrank_ltr_score_splash",
            "to_common": "fountain_ensemble_power_weight_fullrank_ltr_score"
          },
          {
            "from_common": "fountain_fullrank_ensemble_ltr_raw_weight_attr_splash",
            "to_common": "fountain_fullrank_ensemble_ltr_raw_weight_attr"
          },
          {
            "from_common": "fountain_ensemble_power_weight_fullrank_ltr_v4_fountain_finish_rate_splash",
            "to_common": "fountain_ensemble_power_weight_fullrank_ltr_v4_fountain_finish_rate"
          },
          {
            "from_common": "fountain_fullrank_ensemble_finish_rate_raw_weight_attr_splash",
            "to_common": "fountain_fullrank_ensemble_finish_rate_raw_weight_attr"
          },
          {
            "from_common": "fountain_ensemble_power_weight_fullrank_ltr_v4_next_splash",
            "to_common": "fountain_ensemble_power_weight_fullrank_ltr_v4_next"
          },
          {
            "from_common": "fountain_fullrank_ensemble_next_raw_weight_attr_splash",
            "to_common": "fountain_fullrank_ensemble_next_raw_weight_attr"
          },
          {
            "from_common": "fountain_ensemble_weight_fullrank_act_ctr_splash",
            "to_common": "fountain_ensemble_weight_fullrank_act_ctr"
          },
          {
            "from_common": "fountain_fullrank_ensemble_act_ctr_raw_weight_attr_splash",
            "to_common": "fountain_fullrank_ensemble_act_ctr_raw_weight_attr"
          },
          {
            "from_common": "fountain_ensemble_power_weight_fullrank_ltr_v4_reward_splash",
            "to_common": "fountain_ensemble_power_weight_fullrank_ltr_v4_reward"
          },
          {
            "from_common": "fountain_fullrank_ensemble_fullrank_ltr_v4_reward_raw_weight_attr_splash",
            "to_common": "fountain_fullrank_ensemble_fullrank_ltr_v4_reward_raw_weight_attr"
          },
          {
            "from_common": "fountain_ensemble_power_weight_fountain_ordinal_wtd_splash",
            "to_common": "fountain_ensemble_power_weight_fountain_ordinal_wtd"
          },
          {
            "from_common": "fountain_fullrank_ensemble_fountain_ordinal_wtd_raw_weight_attr_splash",
            "to_common": "fountain_fullrank_ensemble_fountain_ordinal_wtd_raw_weight_attr"
          },
          {
            "from_common": "fountain_ensemble_power_weight_comment_staytime_splash",
            "to_common": "fountain_ensemble_power_weight_comment_staytime"
          },
          {
            "from_common": "fountain_fullrank_ensemble_comment_staytime_raw_weight_attr_splash",
            "to_common": "fountain_fullrank_ensemble_comment_staytime_raw_weight_attr"
          },
          {
            "from_common": "fountain_ensemble_power_weight_d2co_playtime_splash",
            "to_common": "fountain_ensemble_power_weight_d2co_playtime"
          },
          {
            "from_common": "fountain_fullrank_ensemble_d2co_playtime_raw_weight_attr_splash",
            "to_common": "fountain_fullrank_ensemble_d2co_playtime_raw_weight_attr"
          },
          {
            "from_common": "fountain_ensemble_power_weight_ltr_lph_splash",
            "to_common": "fountain_ensemble_power_weight_ltr_lph"
          },
          {
            "from_common": "fountain_fullrank_ensemble_ltr_lph_raw_weight_attr_splash",
            "to_common": "fountain_fullrank_ensemble_ltr_lph_raw_weight_attr"
          },
          {
            "from_common": "fountain_ensemble_power_weight_ltr_dfvr_splash",
            "to_common": "fountain_ensemble_power_weight_ltr_dfvr"
          },
          {
            "from_common": "fountain_fullrank_ensemble_ltr_dfvr_raw_weight_attr_splash",
            "to_common": "fountain_fullrank_ensemble_ltr_dfvr_raw_weight_attr"
          },
          {
            "from_common": "fountain_ensemble_weight_fullrank_act_wtd_splash",
            "to_common": "fountain_ensemble_weight_fullrank_act_wtd"
          },
          {
            "from_common": "fountain_fullrank_ensemble_act_wtd_raw_weight_attr_splash",
            "to_common": "fountain_fullrank_ensemble_act_wtd_raw_weight_attr"
          },
        ]
      ) \
    .end_()
    return self