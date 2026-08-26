#!/usr/bin/env python3
# coding=utf-8

from dragonfly.common_leaf_dsl import LeafFlow
from dragonfly.ext.subdivision.subdivision_api_mixin import subdivisionApiMixin
from cascade.cascade_base_flow import CascadeBaseFlow
from cascade.ab_params import cascade_common_params, cascade_fast_params, cascade_common_param_abhit, cascade_fast_params_abhit
from cascade.cascade_utils import cascade_ltr_common_feature, cascade_fc_sim3_feature, cascade_slide_features, ltr_item_features, cascade_distill_item_features, cascade_distill_user_features, cascade_distill_precict_item_features, cascade_prerank_list_item_features
from cascade.cascade_fast_queues import *
from util import enrich_ab_param



class CascadeV12Flow(CascadeBaseFlow, subdivisionApiMixin):
  def __init__(self):
    LeafFlow.__init__(self, "cascade_v12")
    self \
      .namespace_(ns = "cascade_v12", nest = True) \
      ._timestamp_begin("cascade_fast") \
      ._prepare() \
      ._enrich_cascade_score_fast() \
      ._rank() \
      ._timestamp_end("cascade_fast") \
      ._count_stage_cpu_cost("cascade_fast") \
      .namespace_()

  def _prepare(self):
    self \
    .count_reco_result(save_count_to = "cascade_enter") \
    .get_abtest_params(
      biz_name = "RECO_RPC",
      ab_params = enrich_ab_param(cascade_common_params + cascade_fast_params),
      prioritized_suffix = "{{_ABTEST_SUFFIX_LIST_}}",
    ) \
    .get_abtest_params(
      biz_name = "RECO_RPC",
      ab_params = cascade_common_param_abhit + cascade_fast_params_abhit,
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
    .transform_item_attr(
      mappings = [{
        "check_attr_name": "author__id",
        "check_attr_type": "int",
        "output_attr_name": "is_photo_author_followed",
        "output_attr_type": "int",
        "rules": [{
          "check_values": [ "{{followAuthors}}" ],
          "output_value": 1
        }],
      }]) \
    .if_("fountain_enable_get_user_hetu_distribution == 1") \
      .if_("fountain_enable_daily_update_global_hetu_distribution == 1") \
        .explore_memory_data_enrich(
          data_key = "{{fountain_global_hetu_distribution_map}}",
          data_type = "string_int32_map",
          save_data_ptr_to_attr = "fountain_latest_global_hetu_distribution_map",
        ) \
      .end_if_() \
      .explore_photo_distribution_colossus_stat_enricher(
        enable_only_fountain_stat = "{{fountain_hetu_distribution_stat_only_fountain}}",
        enable_only_positive_stat = "{{fountain_hetu_distribution_stat_only_positive}}",
        colossus_resp_attr = "colossus_resp_v2",
        save_total_count = "colossus_hetu_distribution_total_count",
        save_user_hetu_stat_attr = "colossus_hetu_distribution_hetu_stat",
        enable_user_hetu1_distribution = "{{fountain_enable_user_hetu1_distribution}}",
        save_user_hetu1_distribution_attr = "fountain_save_user_hetu1_distribution_map",
        interest_stat_use_reward = "{{fountain_interest_stat_use_reward}}",
        interest_stat_vv_weight = "{{fountain_interest_stat_vv_weight}}",
        interest_stat_reward_weight = "{{fountain_interest_stat_reward_weight}}",
        interest_stat_avg_reward_smooth = "{{fountain_interest_stat_avg_reward_smooth}}",
        enable_interest_stat_avg_reward = "{{fountain_enable_interest_stat_avg_reward}}",
        minus_hate_stat_coeff = "{{fountain_interest_stat_minus_hate_coeff}}",
        minus_sv_stat_coeff = "{{fountain_interest_stat_minus_sv_coeff}}",
        enable_use_actual_reward = "{{fountain_colossus_enable_use_actual_reward}}",
        max_history_size = "{{fountain_colossus_actual_reward_max_history_size}}",
        save_actual_hetu_stat_attr = "colossus_actual_reward_hetu_stat",
        enable_interest_stat_use_true_feedback = "{{enable_fountain_mc_interest_stat_use_true_feedback}}",
        enable_avg_reward_coeff_hetu_stat = "{{fountain_enable_avg_reward_coeff_hetu_stat}}",
        avg_reward_base_idx_ratio_level1 = "{{fountain_avg_reward_base_idx_ratio_level1}}",
        avg_reward_base_idx_ratio_level2 = "{{fountain_avg_reward_base_idx_ratio_level2}}",
        avg_reward_coeff_pow_weight_level1 = "{{fountain_avg_reward_coeff_pow_weight_level1}}",
        avg_reward_coeff_pow_weight_level2 = "{{fountain_avg_reward_coeff_pow_weight_level2}}",
        avg_reward_coeff_upper_bound_level1 = "{{fountain_avg_reward_coeff_upper_bound_level1}}",
        avg_reward_coeff_upper_bound_level2 = "{{fountain_avg_reward_coeff_upper_bound_level2}}",
        avg_reward_coeff_lower_bound_level1 = "{{fountain_avg_reward_coeff_lower_bound_level1}}",
        avg_reward_coeff_lower_bound_level2 = "{{fountain_avg_reward_coeff_lower_bound_level2}}",
        avg_reward_coeff_count_pow_weight_level1 = "{{fountain_avg_reward_coeff_count_pow_weight_level1}}",
        avg_reward_coeff_count_pow_weight_level2 = "{{fountain_avg_reward_coeff_count_pow_weight_level2}}",
        avg_reward_coeff_count_thres_level1 = "{{fountain_avg_reward_coeff_count_thres_level1}}",
        avg_reward_coeff_count_thres_level2 = "{{fountain_avg_reward_coeff_count_thres_level2}}",
        save_avg_reward_coeff_hetu_stat_attr = "colossus_avg_reward_coeff_hetu_stat"
      ) \
      .if_("fountain_enable_save_user_mixed_hetu_stat == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .explore_mix_user_interest_stat_enricher(
          user_hetu_stat_attr = "colossus_hetu_distribution_hetu_stat",
          global_hetu_stat_attr = "fountain_latest_global_hetu_distribution_map",
          global_hetu_stat_redis_key_prefix = "{{fountain_global_hetu_stat_redis_key_prefix}}",
          enable_debias_with_global_stat = "{{fountain_enable_debias_with_global_stat}}",
          enable_debias_multipy_original_stat = "{{fountain_enable_debias_multipy_original_stat}}",
          global_fuse_corr = "{{fountain_user_interest_global_fuse_corr}}",
          save_user_mixed_hetu_stat_attr = "user_mixed_interest_stat"
        ) \
      .end_if_() \
    .end_() \
    .if_("fountain_enable_cal_interest_adjust_weight == 1 and (enable_fountain_intrest_adjust_location_filter == 0 or uCityId == 16842752)") \
      .explore_intrest_adjust_enricher(
        gamora_hetu_adjust_history_list_attr = "gamora_hetu_adjust_history_list",
        opt_card_like_list_attr = "opt_card_like_list",
        opt_card_dis_like_list_attr = "opt_card_dis_like_list",
        interest_adjust_decay_attr = "{{interest_adjust_decay}}",
        interest_adjust_immediate_adjust_thres_attr = "{{interest_adjust_immediate_adjust_thres}}",
        interest_adjust_immediate_adjust_weight_attr = "{{interest_adjust_immediate_adjust_weight}}",
        output_intrest_key_list_attr = "output_intrest_key_list",
        output_intrest_value_list_attr = "output_intrest_value_list"
      ) \
      .if_("fountain_enable_user_need_break_cocoon == 1") \
        .find_value(
          input = "{{output_intrest_key_list}}",
          value = 0,
          result = "user_need_break_cocoon_flag"
        ) \
      .end_() \
    .end_() \
    .if_("enable_replace_colossus_v2_from_mc_2 == 1") \
      .copy_attr(
        attrs = [
          {
            "from_common": "colossus_v2_resp",
            "to_common": "colossus_resp_v2",
          },
        ],
      ) \
    .end_() \
    .if_("fountain_fast_cascade_get_emp_xtr == 1") \
      ._get_emp_xtr() \
    .end_() \
    .if_("enable_replace_colossus_v2_from_mc_3 == 1") \
      .copy_attr(
        attrs = [
          {
            "from_common": "colossus_v2_resp",
            "to_common": "colossus_resp_v2",
          },
        ],
      ) \
    .end_() \
    .if_("enable_fountain_fast_cascade_get_recent_emp_xtr == 1") \
      ._get_recent_emp_xtr() \
    .end_() \
    .if_("enable_fountain_fast_cascade_get_duration_longview_adjust == 1") \
      ._his_cur_duration_longview_adjust() \
    .end_() \
    .if_("enable_replace_colossus_v2_from_mc_4 == 1") \
      .copy_attr(
        attrs = [
          {
            "from_common": "colossus_v2_resp",
            "to_common": "colossus_resp_v2",
          },
        ],
      ) \
    .end_() \
    .if_("fountain_top_sv_hetu_count > 0 or fountain_mc_enable_stat_interest_from_colossus > 0") \
      .if_("fountain_mc_enable_stat_interest_from_colossus > 0", to_be_delete = "date=2024-05-29;committer=gengxiao03") \
        .split_string(
          input_common_attr="fountain_mc_colossus_short_interest_reward_weights",
          output_common_attr="colossus_short_interest_reward_weights_list",
          delimiters=",",
          parse_to_double=True,
        ) \
        .split_string(
          input_common_attr="fountain_mc_colossus_long_interest_reward_weights",
          output_common_attr="colossus_long_interest_reward_weights_list",
          delimiters=",",
          parse_to_double=True,
        ) \
        .split_string(
          input_common_attr="fountain_mc_colossus_explore_interest_reward_weights",
          output_common_attr="colossus_explore_interest_reward_weights_list",
          delimiters=",",
          parse_to_double=True,
        ) \
      .end_() \
      .explore_colossus_top_svtr_hetu_enricher(
        colossus_resp_attr = "colossus_resp_v2",
        # 统计 top 短播河图
        save_user_top_sv_hetu_attr = "colossus_hetu_emp_svtr_stat",
        enable_top_sv_hetu2 = "{{fountain_enable_top_sv_hetu2}}",
        enable_stat_top_sv_only_fountain = "{{fountain_enable_stat_top_sv_only_fountain}}",
        top_sv_stat_max_show = "{{fountain_top_sv_stat_max_show}}",
        enable_top_sv_stat_use_rate = "{{enable_fountain_top_sv_stat_use_rate}}",
        top_sv_stat_default_svtr = "{{fountain_top_sv_stat_default_svtr}}",
        top_sv_stat_base_show = "{{fountain_top_sv_stat_base_show}}",
        # 统计长短期兴趣
        enable_stat_short_interest = "{{fountain_mc_enable_stat_short_interest}}",
        enable_stat_long_interest = "{{fountain_mc_enable_stat_long_interest}}",
        enable_stat_explore_interest = "{{fountain_mc_enable_stat_explore_interest}}",
        get_short_interest_attr = "short_interest",
        save_short_interest_attr = "short_interest",
        save_long_interest_attr = "long_interest",
        save_explore_interest_attr = "random_explore_interest",
        short_interest_reward_weights_attr = "colossus_short_interest_reward_weights_list",
        long_interest_reward_weights_attr = "colossus_long_interest_reward_weights_list",
        explore_interest_reward_weights_attr = "colossus_explore_interest_reward_weights_list",
        enable_stat_short_interest_only_explore_fountain = "{{fountain_mc_enable_stat_short_interest_only_explore_fountain}}",
        short_interest_max_hours = "{{fountain_cascade_short_interest_limit_hour}}",
        enable_interest_use_hetu1 = "{{fountain_cascade_interest_use_level_one}}",
        play_time_slope = "{{fountain_mc_interest_reward_play_time_slope}}",
        play_time_max = "{{fountain_mc_interest_reward_play_time_max}}",
        enable_stat_long_interest_only_explore_fountain = "{{fountain_mc_enable_stat_long_interest_only_explore_fountain}}",
        long_interest_max_days = "{{fountain_cascade_longterm_interest_max_history_days}}",
        long_interest_min_days = "{{fountain_cascade_longterm_interest_min_history_days}}",
        long_interest_min_play_time = "{{fountain_cascade_interest_min_play_second}}",
        short_interest_reward_lower_bound = "{{fountain_mc_short_interest_reward_lower_bound}}",
        short_interest_num = "{{fountain_colossus_short_interest_max_num}}",
        long_interest_reward_lower_bound = "{{fountain_mc_long_interest_reward_lower_bound}}",
        long_interest_num = "{{fountain_colossus_longterm_interest_max_num}}",
        explore_interest_reward_lower_bound = "{{fountain_mc_explore_interest_reward_lower_bound}}",
        explore_interest_num = "{{fountain_colossus_explore_interest_max_num}}",
        enable_interest_reward_use_rate = "{{fountain_cascade_interest_calc_use_percent}}",
      ) \
    .end_() \
    .if_("fountain_enable_personal_weight == 1") \
      ._interactive_emp_xtr_change() \
    .end_() \
    .enrich_attr_by_lua(
      import_common_attr = [
        "enableFountainFullrankExp",
        "fullrank_fast_before_variant_mc_limit_size",
        "fullrank_fast_before_variant_mc_limit_size_exp",
        "increase_quota_status",
        "fountain_mc_increase_quota_factor_list",
        "increase_quota_window_len",
        "increase_quota_current_index"
      ],
      import_item_attr = [
        "hetu_tag_level_info__hetu_level_one",
        "hetu_tag_level_info__hetu_level_two",
        "hetu_tag_level_info_v2__hetu_level_one",
        "duration_ms",
        "explore_stat__show_count",
        "explore_stat__negative_count",
      ],
      export_item_attr = [
        "hetu_level_one_index",
        "hetu_level_one_v2_index_cascade",
        "duration_s",
        "hetu_level_one",
        "hetu_level_two",
        "emp_htr",
        "duration_perf_id"
      ],
      export_common_attr = [
        "fullrank_fast_before_variant_mc_limit_size",
      ],
      function_for_common = "cascade_control_fast",
      function_for_item = "cascade_feature_trans",
      lua_script_file = "fountain/cascade/lua/cascade_control.lua",
    ) \
    .if_("disable_merchant_explore_all_photo_optimize == 0 and enable_fountain_merchant_photo_calc_type == 1") \
      .explore_memory_data_enrich(
        data_key = "merchant_live_authors_set",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "merchant_live_authors_set__memory_data",
      ) \
      .enrich_attr_by_light_function(
        import_item_attr = [
          {"name": "author__id", "as": "author__id"},
          {"name": "merchant_item_info__item_id_list", "as": "cart_itemlist"},
          {"name": "merchant_photo_cart_relation", "as": "cart_relation"},
        ],
        import_common_attr = [
          "merchant_live_authors_set__memory_data",
        ],
        export_item_attr = [
          "is_merchant_cart",
          "is_merchant_living",
          "merchant_author_in_living"
        ],
        function_name = "MerchantGetAuthorInLiving",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    ._calc_true_living() \
    .if_("enable_fountain_cascade_produce_photo_predict == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          { "name": "enable_fountain_cascade_produce_need_divuser", "as": "enable_explore_rank_produce_need_divuser" },
          { "name": "fountain_cascade_produce_real_show_photo_recent_hours", "as": "explore_ranking_produce_real_show_photo_recent_hours" },
          { "name": "fountain_cascade_produce_his_zhongcao_threholds", "as": "explore_ranking_produce_his_zhongcao_threholds" },
          { "name": "fountain_cascade_produce_his_magic_face_threholds", "as": "explore_ranking_produce_his_magic_face_threholds" },
          { "name": "userInfoPb", "as": "user_info_ptr" },
          { "name": "enable_fountain_cascade_produce_need_divuser_v2", "as": "rank_produce_need_divuser_v2" },
          { "name": "uGamoraUploadDayNum30d", "as": "gamora_upload_day_num_30d" },
          { "name": "uNebulaUploadDayNum30d", "as": "nebula_upload_day_num_30d" }
        ],
        export_common_attr = [
          { "name": "ranking_need_produce_flag", "as": "fountain_cascade_need_produce_model_flag" }
        ],
        function_name = "JudgeNeedProduceModel",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .if_("enable_fountain_calc_duration_bucket == 1") \
      .enrich_attr_by_light_function(
        import_item_attr = [
          "duration_ms"
        ],
        export_item_attr = [
          "duration_0_7s",
          "duration_7_9s",
          "duration_9_12s",
          "duration_12_17s",
          "duration_17_20s",
          "duration_20_58s",
          "duration_58_120s",
          "duration_gt_120s"
        ],
        function_name = "CalDurationBucket",
        class_name = "ExploreLightFunctionSetV2"
      ) \
    .end_() \
    .if_("enable_fountain_calc_photo_type == 1") \
      .enrich_attr_by_light_function(
        import_item_attr = [
          "eyeshot_source"
        ],
        export_item_attr = [
          "is_personified_author",
          "is_hot_content",
          "is_authority_content"
        ],
        function_name = "CalcPhotoType",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .transform_item_attr(
        mappings = [{
          "check_attr_name": "collection_type",
          "check_attr_type": "int",
          "output_attr_name": "is_collection",
          "output_attr_type": "int",
          "output_default_value": 0,
          "rules": [{
            "check_range": {
              "lower_bound": 1, # 包含，可缺省
            },
            "output_value": 1,
          }]
        }]
      ) \
    .end_() \
    .if_("enable_fountain_mc_calc_bid_author == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
            {"name": "friendAids", "as": "attr_list"},
        ],
        import_item_attr = [
            {"name": "author__id", "as": "attr"},
        ],
        export_item_attr = [
            {"name": "is_in_set", "as": "is_bid_follow_author"},
        ],
        function_name = "AttrIsInSet",
        class_name = "ExploreLightFunctionSetV2"
      ) \
    .end_() \
    .if_("enable_fountain_mc_calc_high_share_photo == 1") \
      .enrich_attr_by_light_function(
        import_item_attr = [
          "if_title_share",
          "upload_time"
        ],
        export_item_attr = [
          {"name": "is_high_share", "as": "is_high_share_photo"}
        ],
        function_name = "ItemIsHighShare",
        class_name = "ExploreLightFunctionSetV2"
      ) \
    .end_() \
    .if_("fountain_rerank_enable_adjust_marketing_compensation_photo == 1") \
      .gen_is_marketing_compensation_photo() \
    .end_() \
    .if_("fountain_gen_is_low_cost_photo == 1") \
      .gen_is_low_cost_photo() \
    .end_() \
    .if_("enable_fountain_fast_gen_minority_photo == 1") \
      .gen_is_minority_photo() \
    .end_() \
    .if_("enable_fountain_gen_is_sexy_induce_photo == 1")\
      .gen_is_sexy_induce_photo()\
    .end_() \
    .if_("enable_fountain_pack_fountain_mc_cascade == 1") \
      .pack_fountain_mc_cascade() \
    .end_() \
    .if_("enable_fountain_get_hetu_info_for_llm == 1") \
      .extract_hetu_info_tag_for_llm() \
    .end_() \
    .if_("enable_fountain_calc_photo_cluster_id_632 == 1") \
      .cal_fountain_photo_cluster_id_632() \
    .end_() \
    .if_("enable_fountain_cal_is_import_explore_interest_user == 1") \
      .cal_is_import_explore_interest_user() \
    .end_()
    return self

  def _rank(self):
    self \
    ._perf_cascade_before() \
    .if_("enable_fountain_personilize_issue_score == 1") \
      ._enrich_personilize_issue_score() \
    .end_() \
    ._enrich_debias_score() \
    ._adjust_forward_social_params() \
    ._debias_by_user_personal_weight() \
    ._boost_low_follow_user_follow_weight() \
    ._cascade_calc_pure_value() \
    ._disable_forward_social_queue() \
    ._disable_forward_dur_social_queue() \
    .if_("enable_fountain_cal_cascade_update_xtr_fix_mc_s1_score == 1") \
      .fountain_cal_update_xtr_score_mc_s1() \
    .end_() \
    .if_("enable_replace_colossus_v2_from_mc_5 == 1") \
      .copy_attr(
        attrs = [
          {
            "from_common": "colossus_v2_resp",
            "to_common": "colossus_resp_v2",
          },
        ],
      ) \
    .end_() \
    .if_("enable_fountain_pack_fountain_positive_trigger == 1") \
      .pack_fountain_positive_trigger() \
    .end_() \
    .if_("enable_get_fountain_trigger_embbedding == 1") \
      .get_fountain_trigger_embbedding() \
    .end_() \
    .if_("enable_cal_fountain_ecology_positive_score == 1") \
      .cal_fountain_positive_triggers_to_ecology_score() \
    .end_() \
    .if_("enable_fountain_cal_hetu_retargeting_score == 1") \
      .fountain_cal_hetu_retargeting_score() \
    .end_() \
    .if_("enable_fountain_cal_sidinfo_retargeting_score == 1") \
      .fountain_cal_sidinfo_retargeting_score() \
    .end_() \
    .if_("enable_fountain_cal_fountain_import_explore_valid_interest_score == 1 and is_import_explore_interest_user == 1") \
      .cal_fountain_import_explore_valid_interest_score() \
    .end_() \
    .if_("enable_fountain_cal_fountain_import_gamora_interest_score == 1 and is_import_explore_interest_user == 1") \
      .cal_fountain_import_gamora_interest_score() \
    .end_() \
    .if_("enable_fountain_cal_dynamic_i2i_score_mc_s1 == 1") \
      .fountain_cal_dynamic_i2i_score_mc_s1() \
    .end_() \
    .if_("enable_fountain_cal_user_group_interest_tagnex_tgi == 1") \
      .fountain_cal_user_interest_tagnex_tgi("group", "uMultiDimensionGroupKV", "uMultiDimensionGroupDetailKV") \
    .end_() \
    .if_("enable_fountain_cal_user_career_interest_tagnex_tgi == 1") \
      .fountain_cal_user_interest_tagnex_tgi("career", "uJobIdLv1KV", "uJobIdLv2KV") \
    .end_() \
    .if_("enable_fountain_mc_calc_adjust_coeff == 1") \
      ._mc_calc_adjust_coeff() \
    .end_() \
    ._cascade_cluster_sort() \
    ._cascade_cluster_score_boost_discount() \
    ._perf_cascade_mid() \
    ._gen_min_act_rank_reci() \
    ._cascade_calc_opportunity_score() \
    .if_("enable_fountain_mc_s2_sort == 1") \
      ._cascade_ensemble_sort() \
    .end_() \
    .if_("enable_fountain_mc_s2_sort_v2 == 1") \
      ._cascade_ensemble_sort_v2() \
    .end_() \
    .if_("fountain_enable_mc_s2_cascade_distill_full_link_sample == 1") \
      ._full_link_distill_sample() \
    .end_if_() \
    ._final_truncate() \
    ._perf_cascade_final() \
    ._cascade_stage2_count_distribution() \
    .copy_item_meta_info(
      save_item_seq_to_attr = "mc_final_rank_index"
    ) \
    .pack_item_attr(
      item_source = {"reco_results": True},
      mappings = [{
        "from_item_attr": "item_id",
        "to_common_attr": "cascade_output_item_id_list",
      }]
    )

    return self

  def _cascade_stage2_count_distribution(self):
    """
    粗排 stage2 之后统计视频分布
    """
    self \
    .count_reco_result(
      save_count_to = "fountain_mc_stage2_hot_content_count",
      target_item = {"is_hot_content": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_mc_stage2_authority_content_count",
      target_item = {"is_authority_content": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_mc_stage2_personified_author_count",
      target_item = {"is_personified_author": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_mc_stage2_duration_0_7s_count",
      target_item = {"duration_0_7s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_mc_stage2_duration_7_9s_count",
      target_item = {"duration_7_9s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_mc_stage2_duration_9_12s_count",
      target_item = {"duration_9_12s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_mc_stage2_duration_12_17s_count",
      target_item = {"duration_12_17s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_mc_stage2_duration_17_20s_count",
      target_item = {"duration_17_20s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_mc_stage2_duration_20_58s_count",
      target_item = {"duration_20_58s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_mc_stage2_duration_58_120s_count",
      target_item = {"duration_58_120s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_mc_stage2_duration_gt_120s_count",
      target_item = {"duration_gt_120s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_mc_stage2_collection_count",
      target_item = {"is_collection": 1}
    ) \
    .count_reco_result(
      name = "fountain_mc_stage2",
      traceback = True,
      save_count_to = "cascade_final"
    ) \

    return self


  def _gen_min_act_rank_reci(self):
    """
    粗排生成最小互动rank
    """
    self \
    .if_("skip_cascade_fast_gen_min_act_rank_reci==0") \
      .sort(
        score_from_attr = "cascade_pltr",
      ) \
      .copy_item_meta_info(
        save_item_seq_to_attr = "cascade_pltr_rank",
      ) \
      .sort(
        score_from_attr = "cascade_pwtr",
      ) \
      .copy_item_meta_info(
        save_item_seq_to_attr = "cascade_pwtr_rank",
      ) \
      .sort(
        score_from_attr = "cascade_pftr",
      ) \
      .copy_item_meta_info(
        save_item_seq_to_attr = "cascade_pftr_rank",
      ) \
      .sort(
        score_from_attr = "cascade_ptr",
      ) \
      .copy_item_meta_info(
        save_item_seq_to_attr = "cascade_ptr_rank",
      ) \
      .sort(
        score_from_attr = "cascade_pcestr",
      ) \
      .copy_item_meta_info(
        save_item_seq_to_attr = "cascade_pcestr_rank",
      ) \
      .sort(
        score_from_attr = "cascade_pcmtr",
      ) \
      .copy_item_meta_info(
        save_item_seq_to_attr = "cascade_pcmtr_rank",
      ) \
      .sort(
        score_from_attr = "cascade_pepstr",
      ) \
      .copy_item_meta_info(
        save_item_seq_to_attr = "cascade_pepstr_rank",
      ) \
      .sort(
        score_from_attr = "cascade_pcltr",
      ) \
      .copy_item_meta_info(
        save_item_seq_to_attr = "cascade_pcltr_rank",
      ) \
      .split_string(
        input_common_attr = "fountain_mc_min_act_rank_weights_str",
        output_common_attr = "fountain_mc_min_act_rank_weights",
        delimiters = ":",
        parse_to_int = True,
      ) \
      .enrich_attr_by_lua(
        import_common_attr = [
          "fountain_mc_min_act_rank_weights",
        ],
        import_item_attr = [
          "cascade_pltr_rank",
          "cascade_pwtr_rank",
          "cascade_pftr_rank",
          "cascade_ptr_rank",
          "cascade_pcestr_rank",
          "cascade_pcmtr_rank",
          "cascade_pepstr_rank",
          "cascade_pcltr_rank",
        ],
        export_item_attr = [
          "cascade_min_act_rank_reci"
        ],
        function_for_item = "calc_min_act_rank_reci",
        lua_script_file = "fountain/cascade/lua/calc_min_act_rank_reci.lua",
      ) \
    .end_if_() \

    return self

  def _cascade_htr_weight_adjust(self):
    self \
      .enrich_with_protobuf(
        skip = "{{fountain_mc_skip_htr_weight_adjust}}",
        from_extra_var = "userInfoPb",
        attrs = [
          dict(name="find_user_active_degree", path="find_user_active_degree"),
        ]
      ) \
      .enrich_attr_by_light_function(
      skip = "{{fountain_mc_skip_htr_weight_adjust}}",
      import_common_attr = [
        {"name": "fountain_fast_ensemble_weight_cascade_phtr", "as": "raw_htr_weight"},
        "find_user_active_degree",
        {"name": "fountain_boost_htr_weight_by_mid_degree", "as": "boost_htr_weight_by_mid_degree"},
        {"name": "fountain_boost_htr_weight_by_high_degree", "as": "boost_htr_weight_by_high_degree"},
        {"name": "fountain_boost_htr_weight_by_full_degree", "as": "boost_htr_weight_by_full_degree"},
      ],
      export_common_attr = [
        {"name": "raw_htr_weight", "as": "fountain_fast_ensemble_weight_cascade_phtr"},
      ],
      function_name = "HtrWeightAdjust",
      class_name = "ExploreLightFunctionSetV2",
    )

    return self

  def _cascade_s1_get_hetu_distribution_adjust_coeff(self):
    self \
      .if_("fountain_mc_s1_enable_hetu_cluster_adjust_cut_ratio == 1 or fountain_mc_s1_enable_duration_cluster_adjust_hetu_score == 1") \
        .explore_photo_distribution_adjust_enricher(
          colossus_total_count_attr = "colossus_hetu_distribution_total_count",
          user_hetu_stat_attr = "colossus_hetu_distribution_hetu_stat",
          colossus_total_count_threshold = "{{fountain_mc_hetu_distribution_colossus_total_count_threshold}}",
          max_count = "{{fountain_mc_s1_hetu_distribution_stat_max_count}}",
          global_fuse_corr = "{{fountain_mc_s1_hetu_distribution_global_fuse_corr}}",
          enable_daily_update_global_distribution = "{{fountain_enable_daily_update_global_hetu_distribution}}",
          latest_global_hetu_distribution_attr = "fountain_latest_global_hetu_distribution_map",
          global_hetu_stat_redis_key_prefix = "{{fountain_global_hetu_stat_redis_key_prefix}}",
          global_hetu_distribution_use_fountain_flag = True,
          hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
          candidate_hetu_adjust_coeff_map_attr = "fountain_mc_s1_candidate_hetu_adjust_coeff_map" # 候选集调整系数
        ) \
      .end_() \
      .if_("fountain_mc_s1_enable_hetu_cluster_adjust_cut_ratio == 1") \
        .split_string(
          input_common_attr="fountain_mc_s1_hetu_cluster_hetu_adjust_paras",
          output_common_attr="mc_s1_hetu_cluster_hetu_adjust_para_list",
          delimiters=",",
          parse_to_double=True,
        ) \
      .end_() \
      .if_("fountain_mc_s1_enable_duration_cluster_adjust_hetu_score == 1") \
        .split_string(
          input_common_attr="fountain_mc_s1_duration_cluster_hetu_adjust_paras",
          output_common_attr="mc_s1_duration_cluster_hetu_adjust_para_list",
          delimiters=",",
          parse_to_double=True,
        ) \
      .end_()

    return self

  def _cascade_final_hetu_distribution_adjust(self):
    self \
      .if_("fountain_mc_enable_final_hetu_distribution_adjust == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .if_("fountain_mc_final_hetu_distribution_enable_max_count_opt == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
          .copy_attr(
            attrs=[{
                "from_common": "fullrank_fast_before_variant_mc_limit_size",
                "to_common": "fountain_mc_final_hetu_distribution_max_count"
            }]
          ) \
          .sort(
            score_from_attr = "cascade_ensemble_score",
          ) \
        .else_() \
          .copy_attr(
            attrs=[{
                "from_common": "fountain_mc_cluster_fixed_final_size",
                "to_common": "fountain_mc_final_hetu_distribution_max_count"
            }]
          ) \
        .end_if_() \
        .explore_photo_distribution_adjust_enricher(
          colossus_total_count_attr = "colossus_hetu_distribution_total_count",
          user_hetu_stat_attr = "colossus_hetu_distribution_hetu_stat",
          colossus_total_count_threshold = "{{fountain_mc_hetu_distribution_colossus_total_count_threshold}}",
          max_count = "{{fountain_mc_final_hetu_distribution_max_count}}",
          global_fuse_corr = "{{fountain_mc_final_hetu_distribution_global_fuse_corr}}",
          enable_daily_update_global_distribution = "{{fountain_enable_daily_update_global_hetu_distribution}}",
          latest_global_hetu_distribution_attr = "fountain_latest_global_hetu_distribution_map",
          global_hetu_stat_redis_key_prefix = "{{fountain_global_hetu_stat_redis_key_prefix}}",
          global_hetu_distribution_use_fountain_flag = True,
          hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
          candidate_hetu_adjust_coeff_map_attr = "candidate_hetu_adjust_coeff_map",
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_mc_final_hetu_distribution_hetu_coef_alpha", "as": "hetu_coef_alpha"},
            {"name": "fountain_mc_final_hetu_distribution_hetu_coef_beta", "as": "hetu_coef_beta"},
            {"name": "fountain_mc_final_hetu_distribution_hetu_discount_threshold", "as": "hetu_discount_threshold"},
            {"name": "fountain_mc_final_hetu_distribution_hetu_encourage_threshold", "as": "hetu_encourage_threshold"},
            {"name": "fountain_mc_final_hetu_distribution_hetu_coef_upper_bound", "as": "hetu_coef_upper_bound"},
            {"name": "fountain_mc_final_hetu_distribution_enable_unknown_hetu_adjust", "as": "enable_unknown_hetu_adjust"},
            "candidate_hetu_adjust_coeff_map",
          ],
          import_item_attr=[
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_level_one_list"},
            {"name": "cascade_ensemble_score", "as": "es_score"},
          ],
          export_item_attr = [
            {"name": "es_score", "as": "cascade_ensemble_score"},
          ],
          function_name = "AdjustScoreByHetuDistribution",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()

    return self

  def _cascade_calc_pure_value(self):
    self \
    .explore_ensemble_score_calc_pure_value_enricher(
        save_score_to_attr = "cascade_pure_value_score",
        user_power_calc = "{{fountain_cascade_pure_value_use_pow_calc}}",
        queues = cascade_pure_value_queue,
        skip = "{{fountain_skip_cascade_pure_value_enricher}}"
      ) \
    .log_debug_info(
      common_attrs = ["fountain_skip_cascade_pure_value_enricher"],
      item_attrs = ["cascade_pure_value_score"],
      for_debug_request_only = True,
    ) \

    return self

  def _cascade_cluster_sort(self):
    """
    粗排分桶排序
    """
    self \
    .sort(
      # TODO(wangpenglin) 这轮排序没必要，待删除
      skip = "{{skip_cascade_score_sorter_v1}}",
      score_from_attr = "cascade_score",
    ) \
    .if_("fountain_skip_cascade_cluster_id_calc == 0") \
      ._enrich_is_picture() \
      .switch_("fountain_fast_mc_cluster_method") \
        .case_("duration_quantile", to_be_delete = "date=2024-05-29;committer=huzengyi") \
          .if_("fountain_variant_mc_enable_gen_living_cluster == 1", to_be_delete = "date=2024-05-29;committer=gengxiao03") \
            .set_attr_value(
              no_overwrite=True,
              item_attrs=[
                {
                  "name": "cascade_cluster_id",
                  "type": "int",
                  "value": 0
                }
              ],
              target_item = {"live_photo_info__is_living": 1}
            ) \
          .end_() \
          .if_("fountain_variant_mc_enable_gen_pic_cluster == 1", to_be_delete = "date=2024-05-29;committer=gengxiao03") \
            .set_attr_value(
              no_overwrite=True,
              item_attrs=[
                {
                  "name": "cascade_cluster_id",
                  "type": "int",
                  "value": 1
                }
              ],
              target_item = {"is_picture": 1}
            ) \
          .end_() \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "fountain_variant_mc_enable_gen_living_cluster", "as": "enable_gen_living_cluster"},
              {"name": "fountain_variant_mc_enable_gen_pic_cluster", "as": "enable_gen_pic_cluster"},
            ],
            import_item_attr = [
              {"name": "live_photo_info__is_living", "as": "is_living_attr"},
              {"name": "is_picture", "as": "is_picture_attr"},
            ],
            export_item_attr = [
              "photo_cluster_flag", # 除了living和pic之外(可选)的photo置1, 粗排分桶专用
            ],
            function_name = "SetPhotoClusterFlag",
            class_name = "ExploreLightFunctionSetV2",
          ) \
          .sort(
            score_from_attr = "duration_ms",
            desc = False,
            target_item = {"photo_cluster_flag": 1}
          ) \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "fountain_mc_time_cluster_num", "as": "explore_mc_time_cluster_num"},
              {"name": "fountain_variant_mc_time_cluster_base_id", "as": "mc_time_cluster_base_id"}
            ],
            export_item_attr = [
              "cascade_cluster_id", # 根据duration将候选等频分桶
            ],
            function_name = "EqualSizeCluster",
            class_name = "ExploreLightFunctionSetV2",
            target_item = {"photo_cluster_flag": 1}
          ) \
        .case_("random_cluster") \
          .shuffle() \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "fountain_mc_random_cluster_num", "as": "cluster_num"},
              {"name": "fountain_mc_random_cluster_base_id", "as": "cluster_base_id"},
              {"name": "fountain_mc_random_cluster_enable_diversity_cluster", "as": "enable_diversity_cluster"},
              {"name": "fountain_mc_random_cluster_diversity_ratio", "as": "diversity_ratio"},
              {"name": "fountain_mc_random_cluster_enable_diversity_cluster_hetu_level_two", "as": "enable_diversity_cluster_hetu_level_two"},
            ],
            import_item_attr = [
              {"name": "hetu_tag_level_info_v2__hetu_level_one", "as": "hetu_level_one"},
              {"name": "hetu_tag_level_info_v2__hetu_level_two", "as": "hetu_level_two"},
            ],
            export_item_attr = [
              "cascade_cluster_id",
            ],
            function_name = "RandomCluster", # 按顺序蛇形分到cluster_num桶
            class_name = "ExploreLightFunctionSetV2"
          ) \
        .default_() \
          ._cascade_cluster_id_calc() \
      .end_() \
      ._cascade_longterm_interest_ee() \
      ._cascade_s1_get_hetu_distribution_adjust_coeff() \
      .explore_cluster_variant_sort_enricher(
        check_point = "cascade",
        cluster_sort_list_attr_name = "cascade_cluster_id",
        cluster_config = "{{fountain_combine_variant_cluster_sort_config}}",
        global_cut_ratio = "{{fountain_cascade_variant_cluster_global_cut_ratio}}",
        min_survival = "{{fountain_cascade_variant_cluster_min_survival}}",
        enable_proportional = "{{fountain_cascade_variant_cluster_sort_enable_proportional}}",
        size_limit = "{{fountain_cascade_variant_cluster_sort_size_limit}}",
        use_power_calc = "{{fountain_cascade_variant_cluster_sort_use_power_calc}}",
        use_power_calc_v2 = "{{fountain_cascade_variant_cluster_sort_use_power_calc_v2}}",
        rank_value_fusion_type = "{{fountain_cascade_variant_cluster_sort_rank_value_fusion_type}}",
        rank_smooth = "{{fountain_cascade_variant_cluster_sort_rank_smooth}}",
        use_reciprocal = "{{fountain_cascade_variant_cluster_sort_use_reciprocal}}",
        user_info_ptr_attr = "userInfoPb",
        time_cluster_base_id = "{{fountain_variant_mc_time_cluster_base_id}}",
        queues = explore_cluster_sort_queues,
        fixed_final_size="{{fountain_mc_cluster_fixed_final_size}}",
        enable_dynamic_cut_ratio="{{fountain_mc_cluster_enable_dynamic_cut_ratio}}",
        save_score_to_attr = "cascade_variant_sort_score",
        save_adjust_score_to_attr = "cascade_variant_sort_adjust_score",
        save_filter_flag_to_attr = "cascade_s1_filter_flag",  # 用于标记是否截断 = 1 时要被截断, 缺省或其他值不截断
        # 根据兴趣分布调整兴趣分桶截断比例和时长分桶内 score
        enable_hetu_cluster_adjust_cut_ratio  = "{{fountain_mc_s1_enable_hetu_cluster_adjust_cut_ratio}}",
        enable_duration_cluster_adjust_hetu_score  = "{{fountain_mc_s1_enable_duration_cluster_adjust_hetu_score}}",
        hetu_cluster_hetu_adjust_paras_attr = "mc_s1_hetu_cluster_hetu_adjust_para_list",
        duration_cluster_hetu_adjust_para_attr = "mc_s1_duration_cluster_hetu_adjust_para_list",
        duration_cluster_enable_unknown_hetu_adjust = "{{fountain_mc_s1_duration_cluster_enable_unknown_hetu_adjust}}",
        candidate_hetu_adjust_coeff_map_attr = "fountain_mc_s1_candidate_hetu_adjust_coeff_map",
        hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
        enable_ceil_keep_size = "{{fountain_mc_s1_enable_ceil_keep_size}}",
        enable_hetu_adjust_fixed_size = "{{fountain_mc_s1_enable_hetu_adjust_fixed_size}}",
        save_cluster_id_common_attr = "mc_s1_cluster_id",
        save_cluster_cnt_common_attr = "mc_s1_cluster_cnt",
        vv_min_rank_pow_weight = "{{fountain_mc_vv_min_rank_pow_weight}}",
        act_min_rank_pow_weight = "{{fountain_mc_act_min_rank_pow_weight}}",
        play_min_rank_pow_weight = "{{fountain_mc_play_min_rank_pow_weight}}",
        combine_min_rank_pow_weight = "{{fountain_mc_combine_min_rank_pow_weight}}",
        enable_vv_min_rank = "{{fountain_mc_enable_vv_min_rank}}",
        enable_act_min_rank = "{{fountain_mc_enable_act_min_rank}}",
        enable_play_min_rank = "{{fountain_mc_enable_play_min_rank}}",
        enable_combine_min_rank = "{{fountain_mc_enable_combine_min_rank}}",
        vv_min_rank_weight = "{{fountain_mc_vv_min_rank_weight}}",
        act_min_rank_weight = "{{fountain_mc_act_min_rank_weight}}",
        play_min_rank_weight = "{{fountain_mc_play_min_rank_weight}}",
        enable_score_adjust = "{{enable_fountain_mc_score_adjust}}",
        adjust_coeff_final_attr = "mc_adjust_coeff_final",
        enable_es_score_normal = "{{enable_fountain_mc_es_score_normal}}",
        es_score_dist_temperature = "{{fountain_mc_es_score_dist_temperature}}",
        es_score_dist_smooth = "{{fountain_mc_es_score_dist_smooth}}"
      ) \
      ._dump_attr_to_kafka( # dump 粗排 重要的item attr
        stage_name = "mc_score",
        dump_item_attr_list = [
          "cascade_cluster_id",
          "cascade_score",
          "cascade_pwatch_time",
          "cascade_pwtd",
          "cascade_ipw_opt_ftr",
          "cascade_ftr_kai_duration",
          "cascade_wtd_kai_mix",
          "cascade_pctr",
          "cascade_psvtr",
          "cascade_shortview_score2",
          "cascade_slide_kai",
          "cascade_longview_score",
          "cascade_pcotr",
          "cascade_pltr",
          "cascade_pwtr",
          "cascade_pftr",
          "cascade_ptr",
          "cascade_pepstr",
          "cascade_pcmtr",
          "cascade_pcestr",
          "cascade_pcltr",
          "cascade_phtr",
          "cascade_action_once_interact_score",
          "cascade_action_once_watchtime_score",
          "cascade_pure_value_score",
          "comirec_rank_score",
          "cascade_variant_sort_score"
        ],
        dump_common_attr_list = [
          "user_is_low_interact"
        ]
      ) \
      .if_("fountain_enable_cascade_distill_full_link_sample == 1") \
        ._full_link_cascade_s1_distill_sample() \
      .end_if_() \
      .filter_by_attr(
        attr_name = "cascade_s1_filter_flag",
        remove_if = "==",
        compare_to = 1,
        remove_if_attr_missing = False,
      ) \
      ._cascade_stage1_count_distribution() \
    .end_if_() \

    return self

  def _cascade_stage1_count_distribution(self):
    """
    粗排 stage1 之后统计视频分布
    """
    self \
    .count_reco_result(
      save_count_to = "fountain_mc_stage1_hot_content_count",
      target_item = {"is_hot_content": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_mc_stage1_authority_content_count",
      target_item = {"is_authority_content": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_mc_stage1_personified_author_count",
      target_item = {"is_personified_author": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_mc_stage1_duration_0_7s_count",
      target_item = {"duration_0_7s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_mc_stage1_duration_7_9s_count",
      target_item = {"duration_7_9s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_mc_stage1_duration_9_12s_count",
      target_item = {"duration_9_12s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_mc_stage1_duration_12_17s_count",
      target_item = {"duration_12_17s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_mc_stage1_duration_17_20s_count",
      target_item = {"duration_17_20s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_mc_stage1_duration_20_58s_count",
      target_item = {"duration_20_58s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_mc_stage1_duration_58_120s_count",
      target_item = {"duration_58_120s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_mc_stage1_duration_gt_120s_count",
      target_item = {"duration_gt_120s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_mc_stage1_collection_count",
      target_item = {"is_collection": 1}
    ) \
    .count_reco_result(
      name = "fountain_mc_stage1",
      traceback = True,
      save_count_to = "cascade_combine_variant"
    ) \

    return self

  def _cascade_cluster_id_calc(self):
    """
    粗排分桶id计算
    """
    self \
    .pack_common_attr(
      input_common_attrs = ["similar_user_colossus_hetu_list","explore_hetu_list"],
      output_common_attr = "input_explore_interest_hetu_list",
    ) \
    .if_("enable_user_explore_interest_cluster_from_list==1", to_be_delete = "date=2024-05-29;committer=gengxiao03") \
      ._perf_explore_sizes() \
    .end_if_() \
    .explore_rule_cluster_enricher(
      save_cluster_id_to_attr = "cascade_cluster_id",
      check_point = "cascade_v2",
      enable_time_cluster = "{{fountain_mc_enable_time_cluster}}",
      enable_living_cluster = "{{fountain_mc_enable_living_cluster}}",
      enable_user_interest_level_one_cluster = "{{fountain_cascade_interest_use_level_one}}",
      enable_hetu_cluster = "{{fountain_mc_enable_hetu_cluster}}",
      enable_merge_interact_cluster = "{{fountain_mc_enable_merge_interact_cluster_v2}}",
      enable_interact_cluster = "{{fountain_mc_enable_interact_cluster}}",
      input_short_interest_attr ="short_interest",
      input_action_interest_attr ="action_interest",
      input_long_interest_attr ="long_interest",
      input_random_explore_interest_attr ="random_explore_interest",
      enable_user_explore_interest_cluster = "{{enable_user_explore_interest_cluster}}",
      explore_interest_reason = "{{explore_interest_reason_str}}",
      enable_explore_use_hetu_level_one = "{{enable_explore_use_hetu_level_one}}",
      explore_interest_cnt = "{{explore_interest_cnt}}",
      duration_cluster_cfg_str = "{{fountain_cascade_duration_cluster_cfg}}",
      hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
      hetu_level_two_attr = "hetu_tag_level_info__hetu_level_two",
      enable_user_explore_interest_cluster_from_list = "{{enable_user_explore_interest_cluster_from_list}}",
      input_explore_interest_attr = "input_explore_interest_hetu_list",
      duration_ms_attr = "duration_ms",
      is_picture_attr = "is_picture",
      is_living_attr = "live_photo_info__is_living",
      enable_user_follow_author_cluster = "{{enable_user_follow_author_cluster}}",
      enable_user_follow_author_cluster_first = "{{enable_user_follow_author_cluster_first}}",
      is_follow_author_attr = "is_photo_author_followed",
      enable_hot_content_cluster = "{{enable_fountain_mc_hot_content_cluster}}",
      hot_content_exp_tag_attr = "hot_content_thompson_sampling_exp_tag_list",
      enable_duration_one_cluster = "{{enable_fountain_mc_duration_one_cluster}}",
    ) \
    .if_("enable_fountain_mc_duration_one_cluster == 1 and enable_fountain_mc_duration_equal_size_cluster == 1", to_be_delete = "date=2024-05-29;committer=gengxiao03") \
      .sort(
        score_from_attr = "duration_ms",
        desc = False,
        target_item = { "cascade_cluster_id": 10002}
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_mc_time_cluster_num", "as": "explore_mc_time_cluster_num"},
          {"name": "fountain_variant_mc_duration_cluster_begin_id", "as": "mc_time_cluster_base_id"}
        ],
        export_item_attr = [
          "cascade_cluster_id",
        ],
        function_name = "EqualSizeCluster",
        class_name = "ExploreLightFunctionSetV2",
        target_item = { "cascade_cluster_id": 10002}
      ) \
    .end_if_()
    return self

  def _cascade_longterm_interest_ee(self):
    """
    垂类兴趣探索：统计大盘兴趣和粗排候选集兴趣的差集，计算 ee_score
    """
    self \
    .if_("fountain_skip_long_term_interest_ee_cascade==0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .calc_long_term_interest_ee_score(
        user_info_pb_name = "userInfoPb",
        hetu_attrs = "hetu_tag_level_info__hetu_level_one;hetu_tag_level_info__hetu_level_two;hetu_tag_level_info__hetu_level_three;hetu_tag_level_info__hetu_level_four;hetu_tag_level_info__hetu_face_id;hetu_tag_level_info__hetu_tag",
        enable_click_history = "{{fountain_mc_enable_click_history}}",
        enable_like_history = "{{fountain_mc_enable_like_history}}",
        enable_follow_history = "{{fountain_mc_enable_follow_history}}",
        enable_long_view_history = "{{fountain_mc_enable_long_view_history}}",
        long_view_threshold = "{{fountain_mc_long_view_threshold}}",
        export_item_attr = "cascade_long_term_interest_ee_score",
        enable_division_way = "{{fountain_mc_enable_division_way}}",
        photo_hetu_tag_level_info_type = "{{foutnain_mc_photo_hetu_tag_level_info_type}}",
        boost_threshold = "{{fountain_mc_long_term_interest_ee_boost_threshold}}",
      ) \
      .get_kconf_params(
        kconf_configs = [{
          "kconf_key": "reco.fountain.highValueHetuList",
          "value_type": "list_int64",
          "defult_value": [134, 120, 114, 189, 220, 316, 179, 199, 325, 161, 208, 203],
          "export_common_attr": "high_value_hetu_list"
        }]
      ) \
      .pack_item_attr(
        item_source = {
          "reco_results": True,
        },
        mappings = [{
          "from_item_attr": "hetu_level_one_v2_index_cascade",
          "to_common_attr": "hetu_level_one_v2_index_cascade_list_no_dedup",
          "dedup_to_common_attr": False,
        }],
      ) \
      .enrich_attr_by_lua(
        import_common_attr = [
          "fountain_mc_high_value_hetu_debias_coef",
          "fountain_mc_enable_only_longterm_debias",
          "high_value_hetu_list",
          "fountain_mc_enable_lt_weight_adjust",
          "hetu_level_one_v2_index_cascade_list_no_dedup",
          "fountain_fast_ensemble_weight_cascade_long_term_interest_ee_score",
          "fountain_mc_lt_weight_adjust_threshold",
          "fountain_mc_lt_weight_adjust_coef",
        ],
        import_item_attr = [
          "cascade_long_term_interest_ee_score",
          "hetu_tag_level_info_v2__hetu_level_one",
          ],
        export_item_attr = [
          "cascade_long_term_interest_ee_score",
        ],
        export_common_attr = [
          "fountain_fast_ensemble_weight_cascade_long_term_interest_ee_score"
        ],
        function_for_item = "calc_mc_high_value_hetu_debias",
        function_for_common = "calc_mc_max_hetu_one_rate",
        lua_script_file = "fountain/cascade/lua/high_value_hetu_debias.lua",
      ) \
      .log_debug_info(
        common_attrs = [
          "page",
          "high_value_hetu_list",
          "hetu_level_one_v2_index_cascade_list_no_dedup",
          "fountain_fast_ensemble_weight_cascade_long_term_interest_ee_score",
        ],
        item_attrs = [
          "photo_id",
          "cascade_long_term_interest_ee_score",
          "hetu_tag_level_info_v2__hetu_level_one",
        ],
        item_num_limit = 10,
        for_debug_request_only = True,
      ) \
    .end_if_()

    return self

  def _cascade_rrr_discount(self):
    self \
      .enrich_attr_by_lua(
        import_common_attr = [
          "fountain_cascade_rrr_discount_report_smooth",
          "fountain_cascade_rrr_discount_show_smooth",
          "fountain_cascade_rrr_discount_param_n",
          "fountain_cascade_rrr_discount_param_o",
        ],
        import_item_attr = [
          "cascade_ensemble_score",
          "explore_stat__report_detail__total_report_count",
          "explore_stat__real_show_count",
        ],
        export_item_attr = [
          "cascade_ensemble_score",
          "cascade_rrr_discount_factor",
        ],
        function_for_item = "calc_rrr_discount",
        lua_script_file = "fountain/cascade/lua/calc_rrr_discount.lua",
      ) \
      .log_debug_info(
        common_attrs = [
          "fountain_cascade_rrr_discount_report_smooth",
          "fountain_cascade_rrr_discount_show_smooth",
          "fountain_cascade_rrr_discount_param_n",
          "fountain_cascade_rrr_discount_param_o",
        ],
        item_attrs = [
          "cascade_ensemble_score",
          "explore_stat__report_detail__total_report_count",
          "explore_stat__real_show_count",
          "cascade_rrr_discount_factor",
        ],
        item_num_limit = 10,
        for_debug_request_only = True,
      )
    return self

  def _cascade_cluster_score_boost_discount(self):
    self \
      .if_("enable_fountain_cascade_s1_collection_type_boost == 1") \
        ._mc_s1_collection_type_boost() \
      .end_() \
      .if_("fountain_enable_gen_is_reason_top_photo == 1") \
        ._cascade_boost_top_photo_topk() \
      .end_()
    return self
  
  def _cascade_score_boost_discount(self):
    self \
      .if_("fountain_cascade_skip_questionnaire_boost == 0", to_be_delete = "date=2024-05-29;committer=gengxiao03") \
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
            "questionnaire_score",
          ],
          function_for_item = "calc_questionnaire_boost",
          lua_script_file = "fountain/cascade/lua/calc_pxtr.lua",
        ) \
      .end_if_() \
      .if_("enable_fountain_cascading_personified_author_boost == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_cascading_personified_author_boost_coef", "as": "personified_author_coeff"},
            {"name": "fountain_cascading_blacklist_author_boost_coef", "as": "blacklist_author_coeff"},
            {"name": "fountain_cascading_merchant_cart_boost_coef", "as": "merchant_cart_coeff"},
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
      .end_() \
      .if_("enable_fountain_mc_follow_aid_followtime_boost == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_mc_new_follow_aid_followtime_boost_coeff", "as": "new_follow_boost_discount_coeff"},
            {"name": "fountain_mc_media_follow_aid_followtime_boost_coeff", "as": "media_follow_boost_discount_coeff"},
            {"name": "user_fountain_follow_aid_list", "as": "follow_aid_list"},
            {"name": "follow_timestamps", "as": "follow_aid_time_list"}
          ],
          import_item_attr = [
            {"name": "cascade_ensemble_score", "as": "score"},
            {"name": "author__id", "as": "aid"}
          ],
          export_item_attr = [
            {"name": "score", "as": "cascade_ensemble_score"},
          ],
          function_name = "AidFollowTimeBoostOrDiscount",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_fountain_mc_follow_people_boost == 1", to_be_delete = "date=2024-05-29;committer=caoying03") \
        ._fountain_mc_follow_people_boost() \
      .end_() \
      .if_("enable_fountain_mc_hv_pic_boost == 1") \
        ._mc_high_value_pic_boost() \
      .end_() \
      .if_("enable_fountain_mc_operation_target_photo_boost == 1") \
        ._mc_s2_operation_target_photo_boost() \
      .end_() \
      .if_("enable_fountain_mc_hot_content_retr_boost == 1", to_be_delete = "date=2024-05-29;committer=lijinyu") \
        ._mc_hot_content_retr_boost() \
      .end_() \
      .if_("enable_fountain_mc_merchant_photo_boost == 1", to_be_delete = "date=2024-05-29;committer=zhanglinjiang") \
        ._mc_merchant_photo_boost_by_buyer_type() \
      .end_() \
      .if_("enable_fountain_mc_living_photo_boost == 1") \
        ._mc_living_photo_boost_by_paying_type() \
      .end_() \
      .if_("enable_fountain_mc_merchant_live_boost == 1", to_be_delete = "date=2024-05-29;committer=zhanglinjiang") \
        ._mc_merchant_live_boost_by_buyer_type() \
      .end_() \
      .if_("enable_fountain_mc_merchant_reduce_cart_show == 1 and merchant_buyer_type < 2", to_be_delete = "date=2024-05-29;committer=zhanglinjiang") \
        ._mc_merchant_reduce_cart_show() \
      .end_() \
      .if_("enable_fountain_mc_merchant_reduce_live_show == 1 and merchant_buyer_type < 2", to_be_delete = "date=2024-05-29;committer=zhanglinjiang") \
        ._mc_merchant_reduce_live_show() \
      .end_() \
      .if_("enable_fountain_produce_uploads_boost_switch == 1", to_be_delete = "date=2024-05-29;committer=liuyipeng03") \
        ._mc_produce_uploads_boost() \
      .end_() \
      .if_("enable_fountain_mc_produce_boost == 1", to_be_delete = "date=2024-05-29;committer=liuyipeng03") \
        ._mc_produce_item_boost() \
      .end_() \
      .if_("enable_fountain_mc_top_sv_hetu_discount == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_mc_top_sv_hetu_discount_coeff", "as": "discount_coeff"},
            {"name": "fountain_enable_top_sv_hetu2", "as": "enable_top_sv_hetu2"},
            {"name": "fountain_top_sv_hetu_count", "as": "top_sv_hetu_count"},
            {"name": "fountain_hetu_psvtr_mix_coeff", "as": "hetu_psvtr_mix_coeff"},
            {"name": "fountain_enable_dynamic_coeff", "as": "enable_dynamic_coeff"},
            {"name": "fountain_top_sv_stat_hetu_score_lower_bound", "as": "top_sv_stat_hetu_score_lower_bound"},
            "colossus_hetu_emp_svtr_stat"
          ],
          import_item_attr = [
            {"name": "cascade_ensemble_score", "as": "es_score"},
            {"name": "cascade_psvtr", "as": "psvtr"},
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_level_one_list"},
            {"name": "hetu_tag_level_info__hetu_level_two", "as": "hetu_level_two_list"},
          ],
          export_item_attr = [
            {"name": "es_score", "as": "cascade_ensemble_score"},
          ],
          function_name = "DiscountTopSvHetus",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_fountain_cascade_refinement_boost_personified_author == 1", to_be_delete = "date=2024-05-29;committer=xubaoquan") \
        ._refinement_boost_personified_author() \
      .end_() \
      .if_("enable_fountain_cascade_collection_type_boost == 1", to_be_delete = "date=2024-05-29;committer=wangyalong03") \
        ._mc_s2_collection_type_boost() \
      .end_() \
      .if_("enable_fountain_mc_s2_behaviour_hetu_diversity_boost == 1", to_be_delete = "date=2024-05-29;committer=gengxiao03") \
        ._mc_s2_behaviour_hetu_diversity_boost() \
      .end_() \
      .if_("enable_fountain_mc_s2_pos_neg_ratio_boost == 1", to_be_delete = "date=2024-05-29;committer=gengxiao03") \
        .if_("enable_fountain_mc_s1_calc_pos_neg_ratio_boost_coeff == 0", to_be_delete = "date=2024-05-29;committer=gengxiao03") \
          .calc_mc_pos_neg_ratio_boost_coeff( # 仅当S1没有计算系数时 S2才计算boost系数
          ) \
        .end_() \
        .enrich_attr_by_light_function(
          import_item_attr = [
            {"name": "fountain_mc_pos_neg_ratio_boost_coeff", "as": "boost_discount_coeff"},
            {"name": "cascade_ensemble_score", "as": "score"},
          ],
          export_item_attr = [
            {"name": "score", "as": "cascade_ensemble_score"},
          ],
          function_name = "BoostOrDiscountByItemCoeff",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_() \
      .if_("enable_fountain_mc_s2_watch_time_boost == 1", to_be_delete = "date=2024-05-29;committer=gengxiao03") \
        .if_("enable_fountain_mc_s1_calc_watch_time_boost_coeff == 0", to_be_delete = "date=2024-05-29;committer=gengxiao03") \
          .calc_mc_watch_time_boost_coeff( # 仅当S1没有计算系数时 S2才计算boost系数
          ) \
        .end_() \
        .enrich_attr_by_light_function(
          import_item_attr = [
            {"name": "mc_watch_time_boost_coeff", "as": "boost_discount_coeff"},
            {"name": "cascade_ensemble_score", "as": "score"},
          ],
          export_item_attr = [
            {"name": "score", "as": "cascade_ensemble_score"},
          ],
          function_name = "BoostOrDiscountByItemCoeff",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_() \

    return self

  def _cascade_count_hetu_before(self):
    self \
    .if_("skip_fountain_cascade_hetu_count == 0") \
      .pack_item_attr(
        item_source = {
          "reco_results": True,
        },
        mappings = [{
          "from_item_attr": "hetu_level_one_v2_index_cascade",
          "to_common_attr": "hetu_level_one_v2_index_cascade_list",
          "dedup_to_common_attr":True,
        }],
      ) \
      .enrich_attr_by_lua(
        import_common_attr = [
          "hetu_level_one_v2_index_cascade_list",
        ],
        export_common_attr = [
          "hetu_level_one_v2_index_cascade_list_size_before",
        ],
        function_for_common = "cascade_hetu_list_size",
        lua_script_file = "fountain/cascade/lua/cascade_control.lua",
      ) \
      .perflog_attr_value(
        check_point="{{before_cascade_hetu_count}}",
        common_attrs=["hetu_level_one_v2_index_cascade_list_size_before"],
        aggregator="avg",
      ) \
    .end_if_()

    return self

  def _perf_explore_sizes(self):
    self \
    .enrich_attr_by_lua(
        import_common_attr = [
          "similar_user_colossus_hetu_list",
          "explore_hetu_list",
          "input_explore_interest_hetu_list"
        ],
        export_common_attr = [
          "similar_user_list_size",
          "explore_history_hetu_list_size",
          "explore_hetu_list_all_size"
        ],
        function_for_common = "cascade_explore_list_size",
        lua_script_file = "fountain/cascade/lua/cascade_control.lua",
      ) \
    .perflog_attr_value(
      check_point="cascade_explore_list_size",
      common_attrs=["similar_user_list_size","explore_history_hetu_list_size","explore_hetu_list_all_size"],
      aggregator="avg",
    )
    return self


  def _enrich_debias_score(self):
    self \
    .get_common_attr_from_redis(
      cluster_name = "recoNewUserPhotos",
      skip = "{{fountain_skip_cascade_bias_enricher}}",
      timeout_ms = 50,
      cache_bits = 16,
      redis_params = [
        {
          "redis_key": "fountain_cascade_debias_hetu_level_one",
          "output_attr_name": "fountain_cascade_debias_hetu_level_one"
        },
        {
          "redis_key": "fountain_cascade_debias_hetu_level_two",
          "output_attr_name": "fountain_cascade_debias_hetu_level_two"
        },
      ]
    ) \
    .explore_cascade_debias_enricher(
      skip = "{{fountain_skip_cascade_bias_enricher}}",
      use_hetu_level_one = "{{fountain_cascade_bias_use_hetu_level_one}}",
      hetu_emp_xtr_level_one = "fountain_cascade_debias_hetu_level_one",
      hetu_emp_xtr_level_two = "fountain_cascade_debias_hetu_level_two",
      hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
      hetu_level_two_attr = "hetu_tag_level_info__hetu_level_two",
      queues=[
        {
          "original_name":"cascade_pctr",
          "debias_name":"cascade_pctr_debias",
          "max_score":10
        },
        {"original_name":"cascade_pltr",
          "debias_name":"cascade_pltr_debias",
          "max_score":10
        },
        {"original_name":"cascade_pwtr",
          "debias_name":"cascade_pwtr_debias",
          "max_score":10
        },
        {"original_name":"cascade_longview_score",
          "debias_name":"cascade_longview_score_debias",
          "max_score":200
        },
        {"original_name":"cascade_psvtr",
          "debias_name":"cascade_psvtr_debias",
          "max_score":10
        },
        {"original_name":"cascade_pwatch_time",
          "debias_name":"cascade_pwatch_time_debias",
          "max_score":10
        }
      ]
    )

    return self

  def _cascade_count_hetu_after(self):
    self \
    .if_("skip_fountain_cascade_hetu_count == 0") \
      .pack_item_attr(
          item_source = {
            "reco_results": True,
          },
          mappings = [{
            "from_item_attr": "hetu_level_one_v2_index_cascade",
            "to_common_attr": "hetu_level_one_v2_index_cascade_list",
            "dedup_to_common_attr":True,
          }],
      ) \
      .enrich_attr_by_lua(
        import_common_attr = [
          "hetu_level_one_v2_index_cascade_list",
        ],
        export_common_attr = [
          "hetu_level_one_v2_index_cascade_list_size_after",
        ],
        function_for_common = "cascade_hetu_list_size",
        lua_script_file = "fountain/cascade/lua/cascade_control.lua",
      ) \
      .perflog_attr_value(
        check_point="{{after_cascade_hetu_count}}",
        common_attrs=["hetu_level_one_v2_index_cascade_list_size_after"],
        aggregator="avg",
      ) \
    .end_if_()

    return self

  def _negative_feedback_discount(self):
    self \
    .fountain_negative_feedback_discount_v2(
      user_info_attr = "userInfoPb",
      save_score_to_attr = "cascade_discount_ratio",
      enable_fountain_user_profile = "{{fountain_cascade_nfd_v2_enable_fountain_profile}}",
      enable_hot_user_profile = "{{fountain_cascade_nfd_v2_enable_hot_profile}}",
      enable_not_click_list = "{{fountain_cascade_nfd_v2_enable_not_click_list}}",
      enable_play_stat_list = "{{fountain_cascade_nfd_v2_enable_play_stat_list}}",
      enable_hate_list = "{{fountain_cascade_nfd_v2_enable_hate_list}}",
      discount_score = "{{fountain_cascade_nfd_v2_discount_score}}",
      neg_feedback_threshold = "{{fountain_cascade_nfd_v2_neg_feedback_threshold}}",
      period_decay_factor = "{{fountain_cascade_nfd_v2_period_decay_factor}}",
      no_click_factor = "{{fountain_cascade_nfd_v2_not_click_discount_factor}}",
      video_play_stat_factor = "{{fountain_cascade_nfd_v2_play_stat_discount_factor}}",
      hate_list_factor = "{{fountain_cascade_nfd_v2_hate_list_discount_factor}}",
      play_time_thresold_0 = "{{fountain_cascade_nfd_v2_play_time_thresold_0}}",
      play_time_thresold_1 = "{{fountain_cascade_nfd_v2_play_time_thresold_1}}",
      time_limit_second = "{{fountain_cascade_nfd_v2_time_limit_second}}",
      attr_keys = ["hetu_level_one","hetu_level_two"],
    ) \
    .enrich_attr_by_lua(
      import_common_attr = [
      ],
      import_item_attr = [
        "cascade_discount_ratio",
        "cascade_ensemble_score",
      ],
      export_item_attr = [
        "cascade_ensemble_score",
      ],
      function_for_item = "cascade_ensemble_score_discount_calc",
      lua_script_file = "fountain/cascade/lua/cascade_control.lua",
    )

    return self

  def _multipy_gate_score(self):
    self \
    .if_("skip_fountain_fast_cascade_ensemble_score_multiply_gate == 0") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_cascade_psvtr_gate_alpha", "as": "svtr_alpha"},
          {"name": "fountain_cascade_psvtr_gate_beta", "as": "svtr_beta"},
          {"name": "fountain_cascade_pctr_gate_alpha", "as": "ctr_alpha"},
          {"name": "fountain_cascade_pctr_gate_beta", "as": "ctr_beta"},
        ],
        import_item_attr = [
          {"name": "cascade_ensemble_score", "as": "es_score"},
          {"name": "cascade_psvtr", "as": "svtr_score"},
          {"name": "cascade_pctr", "as": "ctr_score"},
        ],
        export_item_attr = [
          {"name": "es_score", "as": "cascade_ensemble_score"},
        ],
        function_name = "EsScoreMultiplyGate",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_if_() \

    return self

  def _debias_by_user_personal_weight(self):
    self \
    .if_("enable_cal_user_group_emp_xtr_in_cascade == 1") \
      ._cal_user_group_emp_xtr_in_cascade() \
    .end_if_() \
    .if_("fountain_enable_personal_weight_with_emp_xtr == 1") \
      .enrich_attr_by_light_function( # 粗排一轮 个性化调整互动权重
        import_common_attr = [
          "user_emp_ltr",
          "user_emp_wtr",
          "user_emp_ftr",
          "user_emp_cmtr",
          "user_emp_eptr",
          {"name": "fountain_cascade_ensemble_power_weight_cascade_like_emp", "as": "all_user_emp_ltr"},
          {"name": "fountain_cascade_ensemble_power_weight_cascade_follow_emp", "as": "all_user_emp_wtr"},
          {"name": "fountain_cascade_ensemble_power_weight_cascade_forward_emp", "as": "all_user_emp_ftr"},
          {"name": "fountain_cascade_ensemble_power_weight_cascade_comment_emp", "as": "all_user_emp_cmtr"},
          {"name": "fountain_cascade_ensemble_power_weight_cascade_eps_emp", "as": "all_user_emp_eptr"},
          {"name": "fountain_variant_cluster_sort_weight_cascade_like_score", "as": "user_ori_ltr_weight"},
          {"name": "fountain_variant_cluster_sort_weight_cascade_follow_score", "as": "user_ori_wtr_weight"},
          {"name": "fountain_variant_cluster_sort_weight_cascade_forward_score", "as": "user_ori_ftr_weight"},
          {"name": "fountain_variant_cluster_sort_weight_cascade_comment_score", "as": "user_ori_cmtr_weight"},
          {"name": "fountain_variant_cluster_sort_weight_cascade_epstr_score", "as": "user_ori_eptr_weight"},
          {"name": "fountain_weight_adjust_coeff_min", "as": "explore_weight_adjust_coeff_min"},
          {"name": "fountain_weight_adjust_coeff_max", "as": "explore_weight_adjust_coeff_max"},
        ],
        export_common_attr = [
          {"name": "user_ltr_weight", "as": "fountain_variant_cluster_sort_weight_cascade_like_score"},
          {"name": "user_wtr_weight", "as": "fountain_variant_cluster_sort_weight_cascade_follow_score"},
          {"name": "user_ftr_weight", "as": "fountain_variant_cluster_sort_weight_cascade_forward_score"},
          {"name": "user_cmtr_weight", "as": "fountain_variant_cluster_sort_weight_cascade_comment_score"},
          {"name": "user_eptr_weight", "as": "fountain_variant_cluster_sort_weight_cascade_epstr_score"},
        ],
        function_name = "UserSortWeightAdjust",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .if_("fountain_enable_personal_weight_by_single_queue == 1") \
      .calc_by_formula1(
        kconf_key = "formula.scenarioKey75.McFountainActiveScoreAdjust",
        import_common_attr = [
          "user_emp_ltr",
          "user_emp_wtr",
          "user_emp_ftr",
          "user_emp_cmtr",
          "user_emp_eptr",
          "fountain_cascade_ensemble_power_weight_cascade_like_emp",
          "fountain_cascade_ensemble_power_weight_cascade_follow_emp",
          "fountain_cascade_ensemble_power_weight_cascade_forward_emp",
          "fountain_cascade_ensemble_power_weight_cascade_comment_emp",
          "fountain_cascade_ensemble_power_weight_cascade_eps_emp",
          "uLikeActiveScore",
          "uFollowActiveScore",
          "uShareActiveScore",
          "uCommentActiveScore",
          "uCollectActiveScore",
          "fountain_variant_cluster_sort_weight_cascade_like_score",
          "fountain_variant_cluster_sort_weight_cascade_follow_score",
          "fountain_variant_cluster_sort_weight_cascade_forward_score",
          "fountain_variant_cluster_sort_weight_cascade_comment_score",
          "fountain_variant_cluster_sort_weight_cascade_epstr_score",
          "fountain_variant_cluster_sort_weight_cascade_pcltr",
          "fountain_variant_cluster_sort_weight_action_once_interact_score"
        ],
        export_formula_value = [
          {"name": "final_like_score", "as": "fountain_variant_cluster_sort_weight_cascade_like_score", "to_common": True},
          {"name": "final_follow_score", "as": "fountain_variant_cluster_sort_weight_cascade_follow_score", "to_common": True},
          {"name": "final_forward_score", "as": "fountain_variant_cluster_sort_weight_cascade_forward_score", "to_common": True},
          {"name": "final_comment_score", "as": "fountain_variant_cluster_sort_weight_cascade_comment_score", "to_common": True},
          {"name": "final_epstr_score", "as": "fountain_variant_cluster_sort_weight_cascade_epstr_score", "to_common": True},
          {"name": "final_collect_score", "as": "fountain_variant_cluster_sort_weight_cascade_pcltr", "to_common": True},
          {"name": "final_action_once_score", "as": "fountain_variant_cluster_sort_weight_action_once_interact_score", "to_common": True}
        ],
        abtest_biz_name = "KUAISHOU_APPS"
      ) \
    .end_() \
    .if_("enable_fountain_cascade_s2_personal_interact_weight == 1") \
      .enrich_attr_by_light_function( # 粗排二轮 个性化调整互动权重
        import_common_attr = [
          "user_emp_ltr",
          "user_emp_wtr",
          "user_emp_ftr",
          "user_emp_cmtr",
          "user_emp_eptr",
          {"name": "fountain_cascade_ensemble_power_weight_cascade_like_emp", "as": "all_user_emp_ltr"},
          {"name": "fountain_cascade_ensemble_power_weight_cascade_follow_emp", "as": "all_user_emp_wtr"},
          {"name": "fountain_cascade_ensemble_power_weight_cascade_forward_emp", "as": "all_user_emp_ftr"},
          {"name": "fountain_cascade_ensemble_power_weight_cascade_comment_emp", "as": "all_user_emp_cmtr"},
          {"name": "fountain_cascade_ensemble_power_weight_cascade_eps_emp", "as": "all_user_emp_eptr"},
          {"name": "fountain_fast_ensemble_power_weight_cascade_like_score", "as": "user_ori_ltr_weight"},
          {"name": "fountain_fast_ensemble_power_weight_cascade_follow_score", "as": "user_ori_wtr_weight"},
          {"name": "fountain_fast_ensemble_power_weight_cascade_forward_score", "as": "user_ori_ftr_weight"},
          {"name": "fountain_fast_ensemble_power_weight_cascade_comment_score", "as": "user_ori_cmtr_weight"},
          {"name": "fountain_fast_ensemble_power_weight_cascade_epstr_score", "as": "user_ori_eptr_weight"},
          {"name": "fountain_cascade_ensemble_power_weight_adjust_ratio_min", "as": "explore_weight_adjust_coeff_min"},
          {"name": "fountain_cascade_ensemble_power_weight_adjust_ratio_max", "as": "explore_weight_adjust_coeff_max"},
        ],
        export_common_attr = [
          {"name": "user_ltr_weight", "as": "fountain_fast_ensemble_power_weight_cascade_like_score"},
          {"name": "user_wtr_weight", "as": "fountain_fast_ensemble_power_weight_cascade_follow_score"},
          {"name": "user_ftr_weight", "as": "fountain_fast_ensemble_power_weight_cascade_forward_score"},
          {"name": "user_cmtr_weight", "as": "fountain_fast_ensemble_power_weight_cascade_comment_score"},
          {"name": "user_eptr_weight", "as": "fountain_fast_ensemble_power_weight_cascade_epstr_score"},
        ],
        function_name = "UserSortWeightAdjust",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .enrich_attr_by_lua(
      skip = "{{skip_cascade_user_adaptive_weight_cal}}",
      import_common_attr = [
        "fountain_fast_ensemble_weight_cascade_like_score",
        "fountain_fast_ensemble_weight_cascade_follow_score",
        "fountain_fast_ensemble_weight_cascade_comment_score",
        "fountain_fast_ensemble_weight_cascade_profile_score",
        "fountain_fast_ensemble_weight_cascade_forward_score",
        "fountain_fast_ensemble_weight_cascade_epstr_score",
        "fountain_cascade_ensemble_power_weight_adjust_ratio_min",
        "fountain_cascade_ensemble_power_weight_adjust_ratio_max",
        "fountain_cascade_ensemble_power_weight_cascade_like_emp",
        "fountain_cascade_ensemble_power_weight_cascade_follow_emp",
        "fountain_cascade_ensemble_power_weight_cascade_comment_emp",
        "fountain_cascade_ensemble_power_weight_cascade_profile_emp",
        "fountain_cascade_ensemble_power_weight_cascade_forward_emp",
        "fountain_cascade_ensemble_power_weight_cascade_eps_emp",
        "userExpLtr",
        "userExpWtr",
        "userExpCmtr",
        "userExpPtr",
        "userExpFtr",
        "userExpEptr"
      ],
      export_common_attr = [
        "fountain_fast_ensemble_weight_cascade_like_score",
        "fountain_fast_ensemble_weight_cascade_follow_score",
        "fountain_fast_ensemble_weight_cascade_comment_score",
        "fountain_fast_ensemble_weight_cascade_profile_score",
        "fountain_fast_ensemble_weight_cascade_forward_score",
        "fountain_fast_ensemble_weight_cascade_epstr_score"
      ],
      function_for_common = "cal_cascade_adaptive_weights",
      lua_script_file = "fountain/cascade/lua/cal_personality_weight.lua",
    ) \
    .enrich_attr_by_lua(
      skip = "{{skip_cascade_user_adaptive_watch_time_weight_cal}}",
      import_common_attr = [
        "fountain_variant_cluster_sort_weight_cascade_pwatch_time",
        "fountain_fast_ensemble_power_weight_cascade_pwatch_time",
        "fountain_variant_cluster_sort_weight_cascade_pwtd",
        "fountain_fast_ensemble_weight_cascade_pwtd",
        "fountain_cascade_fast_cluster_sort_use_personal_pwatch_time_weight",
        "fountain_cascade_fast_ensemble_sort_use_personal_pwatch_time_weight",
        "fountain_cascade_fast_cluster_sort_use_personal_pwtd_weight",
        "fountain_cascade_fast_ensemble_sort_use_personal_pwtd_weight",
        "cascade_watch_time_weight_adaptive_ratio_power",
        "cascade_watch_time_weight_adaptive_ratio_offset",
        "fountain_cascade_ensemble_power_weight_adjust_min_ratio_pwatch_time",
        "fountain_cascade_ensemble_power_weight_adjust_max_ratio_pwatch_time",
        "fountain_cascade_ensemble_power_weight_cascade_watch_time_emp",
        "userAvgEffectiveWatchTime",
        "fountain_cascade_watch_time_reweight_use_colossus_res",
        "user_emp_watch_time",
      ],
      export_common_attr = [
        "fountain_variant_cluster_sort_weight_cascade_pwatch_time",
        "fountain_fast_ensemble_power_weight_cascade_pwatch_time",
        "fountain_variant_cluster_sort_weight_cascade_pwtd",
        "fountain_fast_ensemble_weight_cascade_pwtd",
      ],
      function_for_common = "cal_cascade_adaptive_watch_time_weights",
      lua_script_file = "fountain/cascade/lua/cal_personality_weight.lua",
    ) \

    return self

  def _adjust_forward_social_params(self):
    self \
    .if_("fountain_cascade_enable_adjust_forward_social_params > 0 and bid_follow_num > 0") \
      .gen_common_attr_by_lua(
        attr_map={
          "fountain_variant_cluster_sort_weight_cascade_forward_score": "fountain_variant_cluster_sort_weight_cascade_forward_score * fountain_variant_cluster_sort_weight_cascade_forward_score_social_coeff",
          }
      ) \
    .end_()
    return self

  def _boost_low_follow_user_follow_weight(self):
    self \
    .if_("fountain_cascade_s1_weight_low_follow == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_cascade_s1_low_follow_pwtr_weight", "as": "rank_low_follow_pwtr_weight"},
          {"name": "fountain_cascade_s1_low_follow_thres_s", "as": "rank_low_follow_thres_s"},
          {"name": "fountain_cascade_s1_enable_no_follow_boost", "as": "enable_no_follow_boost"},
          "follow_timestamps",
          {"name": "fountain_variant_cluster_sort_weight_cascade_follow_score", "as": "input_pwtr_score"},
          {"name": "fountain_cascade_s1_enable_low_follow_boost", "as": "enable_low_follow_boost"},
          {"name": "fountain_cascade_s1_low_follow_boost_threshold", "as": "low_follow_boost_threshold"},
          {"name": "user_follow_type", "as": "user_follow_type"},
          {"name": "fountain_cascade_s1_no_follow_pwtr_weight", "as": "no_follow_pwtr_weight"},
          {"name": "fountain_cascade_s1_valid_follow_pwtr_weight", "as": "valid_follow_pwtr_weight"},
          {"name": "fountain_cascade_s1_valid_low_follow_pwtr_weight", "as": "valid_low_follow_pwtr_weight"},
          {"name": "fountain_cascade_s1_valid_media_follow_pwtr_weight", "as": "valid_media_follow_pwtr_weight"},
          {"name": "fountain_cascade_s1_valid_high_follow_pwtr_weight", "as": "valid_high_follow_pwtr_weight"}
        ],
        export_common_attr = [
          {"name": "output_pwtr_score", "as": "fountain_variant_cluster_sort_weight_cascade_follow_score"},
        ],
        function_name = "UserSortWeightLowFollow",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .if_("fountain_cascade_s2_weight_low_follow == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_cascade_s2_low_follow_pwtr_weight", "as": "rank_low_follow_pwtr_weight"},
          {"name": "fountain_cascade_s2_low_follow_thres_s", "as": "rank_low_follow_thres_s"},
          {"name": "fountain_cascade_s2_enable_no_follow_boost", "as": "enable_no_follow_boost"},
          "follow_timestamps",
          {"name": "fountain_fast_ensemble_power_weight_cascade_follow_score", "as": "input_pwtr_score"},
          {"name": "fountain_cascade_s2_enable_low_follow_boost", "as": "enable_low_follow_boost"},
          {"name": "fountain_cascade_s2_low_follow_boost_threshold", "as": "low_follow_boost_threshold"},
          {"name": "user_follow_type", "as": "user_follow_type"},
          {"name": "fountain_cascade_s2_no_follow_pwtr_weight", "as": "no_follow_pwtr_weight"},
          {"name": "fountain_cascade_s2_valid_follow_pwtr_weight", "as": "valid_follow_pwtr_weight"},
          {"name": "fountain_cascade_s2_valid_low_follow_pwtr_weight", "as": "valid_low_follow_pwtr_weight"},
          {"name": "fountain_cascade_s2_valid_media_follow_pwtr_weight", "as": "valid_media_follow_pwtr_weight"},
          {"name": "fountain_cascade_s2_valid_high_follow_pwtr_weight", "as": "valid_high_follow_pwtr_weight"}
        ],
        export_common_attr = [
          {"name": "output_pwtr_score", "as": "fountain_fast_ensemble_power_weight_cascade_follow_score"},
        ],
        function_name = "UserSortWeightLowFollow",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \

    return self


  def _get_personally_ensemble_sort(self):
    self \
    .get_common_attr_from_redis(
      cluster_name = "recoNewUserPhotos",
      timeout_ms = 10,
      cache_bits = 2,
      cache_expire_second = 600,
      redis_params = [
        {
          "redis_key": "{{fountain_cascade_personnally_maxtrix_redis_key}}",
          "output_attr_name": "fountain_cascade_ensemble_cem_maxtrix"
        },
        {
          "redis_key": "{{fountain_cascade_personnally_feature_redis_key}}",
          "output_attr_name": "fountain_cascade_ensemble_cem_feature"
        },
      ]
    ) \
    .pack_item_attr(
      item_source = {"reco_results": True},
      mappings = [
        {
          "from_item_attr": "cascade_pwtd",
          "to_common_attr": "cascade_pwtd_avg",
          "aggregator":"avg"
        },
        {
          "from_item_attr": "cascade_pwtd",
          "to_common_attr": "cascade_pwtd_dev",
          "aggregator":"dev"
        },
        {
          "from_item_attr": "cascade_pctr",
          "to_common_attr": "cascade_pctr_avg",
          "aggregator":"avg"
        },
        {
          "from_item_attr": "cascade_pctr",
          "to_common_attr": "cascade_pctr_dev",
          "aggregator":"dev"
        },
         {
          "from_item_attr": "cascade_pwatch_time",
          "to_common_attr": "cascade_pwatch_time_avg",
          "aggregator":"avg"
        },
        {
          "from_item_attr": "cascade_pwatch_time",
          "to_common_attr": "cascade_pwatch_time_dev",
          "aggregator":"dev"
        },
        {
          "from_item_attr": "cascade_action_once_interact_score",
          "to_common_attr": "cascade_action_once_interact_score_avg",
          "aggregator":"avg"
        },
        {
          "from_item_attr": "cascade_action_once_interact_score",
          "to_common_attr": "cascade_action_once_interact_score_dev",
          "aggregator":"dev"
        },
      ]
    ) \
    .explore_personally_ensemble_weight(
      matrix_weight_attr = "fountain_cascade_ensemble_cem_maxtrix",
      feature_vector_attr = "fountain_cascade_ensemble_cem_feature",
      weight_config =[
        {
          "weight_name":"fountain_fast_ensemble_weight_cascade_pwtd",
          "weight_config_key":"wtd"
        },
        {
          "weight_name":"fountain_fast_ensemble_power_weight_cascade_click_score",
          "weight_config_key":"click"
        },
        {
          "weight_name":"fountain_fast_ensemble_power_weight_cascade_pwatch_time",
          "weight_config_key":"watchtime"
        },
        {
          "weight_name":"fountain_fast_ensemble_weight_action_once_interact_score",
          "weight_config_key":"actiononce"
        },
      ],
      feature_config=[
        {
          "fature_name":"page",
          "treat_type":"maxmin",
          "value_type":"int",
          "min_value_attr":"fountain_cascade_page_feature_min",
          "max_value_attr":"fountain_cascade_page_feature_max",
        },
        {
          "fature_name":"userRequestHour",
          "treat_type":"maxmin",
          "value_type":"int",
          "min_value_attr":"fountain_cascade_hour_feature_min",
          "max_value_attr":"fountain_cascade_hour_feature_max",
        },
        {
          "fature_name":"user_emp_ltr",
          "treat_type":"maxmin",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_emp_ltr_feature_min",
          "max_value_attr":"fountain_cascade_emp_ltr_feature_max",
        },
        {
          "fature_name":"user_emp_wtr",
          "treat_type":"maxmin",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_emp_wtr_feature_min",
          "max_value_attr":"fountain_cascade_emp_wtr_feature_max",
        },
        {
          "fature_name":"user_emp_cmtr",
          "treat_type":"maxmin",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_emp_cmtr_feature_min",
          "max_value_attr":"fountain_cascade_emp_cmtr_feature_max",
        },
        {
          "fature_name":"user_emp_ftr",
          "treat_type":"maxmin",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_emp_ftr_feature_min",
          "max_value_attr":"fountain_cascade_emp_ftr_feature_max",
        },
        {
          "fature_name":"user_emp_watch_time",
          "treat_type":"maxmin",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_emp_watchtime_feature_min",
          "max_value_attr":"fountain_cascade_emp_watchtime_feature_max",
        },
        {
          "fature_name":"user_emp_evtr",
          "treat_type":"maxmin",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_emp_evtr_feature_min",
          "max_value_attr":"fountain_cascade_emp_evtr_feature_max",
        },
        {
          "fature_name":"featureUserClickCount",
          "treat_type":"maxmin",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_user_click_cnt_feature_min",
          "max_value_attr":"fountain_cascade_user_click_cnt_feature_max",
        },
        {
          "fature_name":"cascade_pwtd_avg",
          "treat_type":"maxmin",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_pwtd_avg_feature_min",
          "max_value_attr":"fountain_cascade_pwtd_avg_feature_max",
        },
        {
          "fature_name":"cascade_pwtd_dev",
          "treat_type":"original",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_pwtd_dev_feature_min",
          "max_value_attr":"fountain_cascade_pwtd_dev_feature_max",
        },
        {
          "fature_name":"cascade_pctr_avg",
          "treat_type":"maxmin",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_pctr_avg_feature_min",
          "max_value_attr":"fountain_cascade_pctr_avg_feature_max",
        },
        {
          "fature_name":"cascade_pctr_dev",
          "treat_type":"original",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_pctr_dev_feature_min",
          "max_value_attr":"fountain_cascade_pctr_dev_feature_max",
        },
        {
          "fature_name":"cascade_pwatch_time_avg",
          "treat_type":"maxmin",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_pwatch_time_avg_feature_min",
          "max_value_attr":"fountain_cascade_pwatch_time_avg_feature_max",
        },
        {
          "fature_name":"cascade_pwatch_time_dev",
          "treat_type":"original",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_pwatch_time_dev_feature_min",
          "max_value_attr":"fountain_cascade_pwatch_time_dev_feature_max",
        },
        {
          "fature_name":"cascade_action_once_interact_score_avg",
          "treat_type":"maxmin",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_action_once_interact_score_avg_feature_min",
          "max_value_attr":"fountain_cascade_action_once_interact_score_avg_feature_max",
        },
        {
          "fature_name":"cascade_action_once_interact_score_dev",
          "treat_type":"original",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_action_once_interact_score_dev_feature_min",
          "max_value_attr":"fountain_cascade_action_once_interact_score_dev_feature_max",
        },
      ],
    )

    return self



  def _cascade_ensemble_sort(self):
    """
    粗排第二阶段排序
    """
    self \
    .if_('fountain_fast_cascade_ensemble_enable_personally_weight == 1') \
      ._get_personally_ensemble_sort() \
    .end_if_() \
    .switch_("fountain_fast_mc_s2_calc_es_score_method") \
      .case_(2) \
        .copy_attr( # 摸底:一轮纯值排序分用于二轮,提高链路一致性,后续删除
          attrs=[{
            "from_item": "cascade_variant_sort_score",
            "to_item": "cascade_ensemble_score"
          }]
        ) \
      .default_() \
        .fountain_calc_ensemble_score(
          use_dist_calc = "{{fountain_cascade_ensemble_use_dist_calc}}",
          dis_factor = "{{fountain_cascade_ensemble_dis_factor}}",
          range_end = "{{fountain_fast_cascade_ensemble_range_end}}",
          user_new_proportion = "{{fountain_fast_cascade_ensemble_sort_enable_proportion}}",
          user_power_calc = "{{fountain_fast_cascade_ensemble_sort_enable_power_calc}}",
          user_power_calc_v2 = "{{fountain_fast_cascade_ensemble_sort_enable_power_calc_v2}}",
          enable_time_cost_opt = "{{fountain_cascade_enable_time_cost_opt}}",
          user_info_ptr_attr = "user_info_ptr",
          action_day = "{{mc_variant_weight_action_day_num}}",
          queues = cascade_ensemble_sort_queues,
          save_score_to_attr = "cascade_ensemble_score",
          rank_smooth = "{{fountain_fast_cascade_rank_smooth}}",
          use_queue_smooth_as_rank_smooth = "{{fountain_fast_cascade_ensemble_use_queue_smooth_as_rank_smooth}}",
          rank_score_calculate_method = "{{fountain_fast_cascade_ensemble_sort_rank_score_calculate_method}}",
          hyperbolic_scale = "{{fountain_fast_cascade_ensemble_sort_hyperbolic_scale}}",
          hyperbolic_alpha = "{{fountain_fast_cascade_ensemble_sort_hyperbolic_alpha}}",
          hyperbolic_beta = "{{fountain_fast_cascade_ensemble_sort_hyperbolic_beta}}",
          min_rank_weight = "{{fountain_mc_fullrank_min_rank_weight}}"
        ) \
    .end_() \
    ._dump_attr_to_kafka( # 混合排序之后, 将全部item的重要 item attr 落盘
      stage_name = "mc_s2_score",
      dump_item_attr_list = [
        # 分桶排序分
        "cascade_variant_sort_score",
        # 仅在二轮计算使用的队列
        "cascade_min_act_rank_reci",
        # 混合排序分
        "cascade_ensemble_score",
        # 和入口视频的相关分
        "fountain_related_score_v2_detail"
      ]
    ) \
    .if_("fountain_cascade_skip_rrr_discount == 0") \
      ._cascade_rrr_discount() \
    .end_if_() \
    .if_("skip_cascade_negative_feedback_discount_v2 == 0", to_be_delete = "date=2024-05-29;committer=gengxiao03") \
      ._negative_feedback_discount() \
    .end_if_() \
    ._multipy_gate_score() \
    ._cascade_score_boost_discount() \
    ._audit_adjust_score() \
    ._cascade_final_hetu_distribution_adjust() \
    .switch_("fountain_fast_mc_s2_sort_method") \
      .case_(0) \
        .sort(
          range_end = "{{fountain_cascade_ensemble_range_end}}",
          score_from_attr = "cascade_ensemble_score",
        ) \
      .default_() \
        .sort(
          range_end = "{{fountain_cascade_ensemble_range_end}}",
          score_from_attr = "cascade_ensemble_score",
        ) \
    .end_()

    return self

  def _final_truncate(self):
    """
    粗排第二阶段后截断
    """
    self \
    .if_("fountain_mc_enable_dedup_on_same_author == 1") \
      .deduplicate(
        on_item_attr = "author__id",
      ) \
    .end_() \
    .if_("fountain_enable_cascade_variant_sort_by_adjust_score == 1") \
      .sort(
        score_from_attr = "cascade_variant_sort_adjust_score",
      ) \
    .end_() \
    .if_('fountain_fast_cascade_final_control_hetu_count == 1') \
      .if_("fountain_enable_dynamic_hetu_control_diversity_by_age == 1") \
        .split_string(
          input_common_attr = "fountain_mc_dynamic_hetu_control_diversity_age_coeff_str",
          output_common_attr = "fountain_mc_dynamic_hetu_control_diversity_age_coeff_list",
          delimiters = ",",
          parse_to_double = True,
          trim_spaces = True,
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_mc_dynamic_hetu_control_diversity_age_coeff_list", "as": "age_group_coeff_list"},
            "basic_info_age_segment_v2",
          ],
          export_common_attr = [
            {"name": "coeff", "as": "fountain_mc_dynamic_hetu_control_diversity_coeff"}
          ],
          function_name = "GetCoeffByAgeGroup",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .explore_control_hetu_count_arranger(
        user_hetu_stat_attr = "user_mixed_interest_stat",
        user_actual_distribution_attr = "colossus_actual_reward_hetu_stat",
        hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
        hetu_level_two_attr = "hetu_tag_level_info__hetu_level_two",
        hetu_level_five_attr = "hetu_tag_level_info__hetu_level_five",
        duration_ms_attr = "duration_ms",
        author_attr = "author__id",
        enable_hetu_control_interest = "{{cascade_enable_hetu_control_interest}}",
        enable_hetu_control_diversity = "{{cascade_enable_hetu_control_diversity}}",
        enable_duration_control_diversity = "{{cascade_enable_duration_control_diversity}}",
        enable_author_control_diversity = "{{cascade_enable_author_control_diversity}}",
        keep_size = "{{fullrank_fast_before_variant_mc_limit_size}}",
        hetu1_max_size = "{{cascade_control_hetu1_max_size}}",
        hetu2_max_size = "{{cascade_control_hetu2_max_size}}",
        hetu5_max_size = "{{cascade_control_hetu5_max_size}}",
        duration_0_7s_max_size = "{{cascade_control_duration_0_7s_max_size}}",
        duration_7_9s_max_size = "{{cascade_control_duration_7_9s_max_size}}",
        duration_9_12s_max_size = "{{cascade_control_duration_9_12s_max_size}}",
        duration_12_17s_max_size = "{{cascade_control_duration_12_17s_max_size}}",
        duration_17_20s_max_size = "{{cascade_control_duration_17_20s_max_size}}",
        duration_300_400s_max_size = "{{cascade_control_duration_300_400s_max_size}}",
        duration_400s_inf_max_size = "{{cascade_control_duration_400s_inf_max_size}}",
        hetu_adjust_max_value = "{{fountain_mc_hetu_control_hetu_adjust_max_value}}",
        hetu_adjust_min_value = "{{fountain_mc_hetu_control_hetu_adjust_min_value}}",
        hetu_adjust_coef = "{{fountain_mc_hetu_control_hetu_adjust_coef}}",
        enable_actual_hetu_control = "{{enable_fountain_mc_hetu_actual_hetu_control}}",
        same_author_max_size = "{{cascade_control_same_author_max_size}}",
        enable_dynamic_hetu_control_diversity = "{{fountain_mc_enable_dynamic_hetu_control_diversity}}",
        enable_dynamic_hetu_control_diversity_v2 = "{{fountain_mc_enable_dynamic_hetu_control_diversity_v2}}",
        dynamic_hetu_control_diversity_coeff = "{{fountain_mc_dynamic_hetu_control_diversity_coeff}}",
        enable_dynamic_hetu_control_diversity_level_one = "{{fountain_mc_enable_dynamic_hetu_control_diversity_level_one}}",
        enable_dynamic_hetu_control_diversity_level_two = "{{fountain_mc_enable_dynamic_hetu_control_diversity_level_two}}",
        enable_adjust_quota_by_avg_reward_coeff = "{{fountain_mc_enable_adjust_quota_by_avg_reward_coeff}}",
        enable_dynamic_hetu_control_diversity_normal = "{{fountain_mc_enable_dynamic_hetu_control_diversity_normal}}",
        avg_reward_coeff_hetu_stat_attr = "colossus_avg_reward_coeff_hetu_stat"
      ) \
    .else_() \
      .truncate(
        size_limit = "{{fullrank_fast_before_variant_mc_limit_size}}",
      ) \
    .end_if_()
    return self

  def _perf_cascade_before(self):
    self \
    ._cascade_count_hetu_before() \
    .perflog_attr_value(
      check_point="before_cascade_",
      item_attrs=["hetu_level_one_v2_index_cascade","duration_perf_id"],
      aggregator="count",
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
        "cascade_phtr",
        "cascade_psvtr",
        "cascade_pwatch_time",
        "cascade_pwtd",
        "cascade_longview_score",
        "cascade_shortview_score2",
        "questionnaire_score",
        "cascade_ftr_kai",
        "cascade_slide_kai",
        "cascade_pepstr",
        "cascade_pcltr",
        "cascade_pcotr",
        "cascade_pcmtr",
        "cascade_pcestr",
        "cascade_wtd_kai"
      ],
      fractile_list = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0],
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
    )

    return self

  def _perf_cascade_mid(self):
    self \
    .count_reco_result(save_count_to = "cascade_variant") \
    .perflog_reason_count(
      check_point = "post_cascade_stage1",
    ) \
    .perflog_attr_value(
      check_point="after_cascade_first_",
      item_attrs=["hetu_level_one_v2_index_cascade","duration_perf_id"],
      aggregator="count",
    ) \
    .if_("skip_cascade_cluster_type_cal==0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .enrich_attr_by_lua(
          import_item_attr = [
            "cascade_cluster_id",
          ],
          export_item_attr = [
            "cascade_cluster_type",
          ],
          function_for_item = "calc_cascade_cluster_type",
          lua_script_file = "fountain/cascade/lua/cascade_control.lua",
        ) \
        .perflog_attr_value(
        check_point="before_variant_sort_",
        item_attrs=["cascade_cluster_type"],
        aggregator="count",
        ) \
      .perflog_attr_value(
        check_point="after_variant_sort_",
        item_attrs=["cascade_cluster_type"],
        aggregator="count",
      ) \
    .end_if_() \
    .perflog_attr_value(
      # 统计河图类目分布
      check_point = "{{fountain_mc_hetu_perf_post_stage1_ckpt}}",
      item_attrs=["hetu_tag_level_info__hetu_level_one", "hetu_tag_level_info__hetu_level_two"],
      aggregator="count"
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
        "cascade_phtr",
        "cascade_psvtr",
        "cascade_pwatch_time",
        "cascade_pwtd",
        "cascade_longview_score",
        "cascade_shortview_score2",
        "questionnaire_score",
        "cascade_ftr_kai",
        "cascade_slide_kai",
        "cascade_pepstr",
        "cascade_pcltr",
        "cascade_pcotr",
        "cascade_pcmtr",
        "cascade_pcestr",
        "cascade_wtd_kai"
      ],
      fractile_list = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0],
      check_point = "cascade_stage2"
    )

    return self

  def _perf_cascade_final(self):
    self \
    .perflog_attr_value(
      skip = "{{skip_cascade_cluster_type_cal}}",
      check_point="after_ensemble_",
      item_attrs=["cascade_cluster_type"],
      aggregator="count",
      ) \
    .perflog_attr_value(
      check_point="after_cascade_",
      item_attrs=["hetu_level_one_v2_index_cascade","duration_perf_id"],
      aggregator="count",
    ) \
    .perflog_reason_count(
      check_point = "cascade_finish",
    ) \
    .perflog_attr_value(check_point="cascade_item_num",
      common_attrs=[
        "cascade_enter",
        "cacade_first_truncate",
        "cascade_pre_filter",
        "cacade_prerank_truncate",
        "cascade_variant",
        "cascade_combine_variant",
        "cascade_second_truncate",
        "cascade_duration_variant",
        "cascade_final"
      ]
    ) \
    ._cascade_count_hetu_after() \
    .perflog_attr_value(
      # 统计河图类目分布
      check_point = "{{fountain_mc_hetu_perf_cascade_final_ckpt}}",
      item_attrs=["hetu_tag_level_info__hetu_level_one", "hetu_tag_level_info__hetu_level_two"],
      aggregator="count"
    )

    return self

  def _full_link_distill_sample(self):
    self \
    .get_kconf_params(
      kconf_configs=[
        {
          "kconf_key": "reco.offline.fountainFulllinkDistillRankParam",
          "value_type": "json",
          "json_path": "cascade_s2_full_link_distill_sample_begin",
          "export_common_attr": "fountain_cascade_s2_full_link_distill_sample_begin"
        },
        {
          "kconf_key": "reco.offline.fountainFulllinkDistillRankParam",
          "value_type": "json",
          "json_path": "cascade_s2_full_link_distill_sample_end",
          "export_common_attr": "fountain_cascade_s2_full_link_distill_sample_end"
        },
        {
          "kconf_key": "reco.offline.fountainFulllinkDistillRankParam",
          "value_type": "json",
          "json_path": "cascade_s2_full_link_distill_sample_num",
          "export_common_attr": "fountain_cascade_s2_full_link_distill_sample_num"
        },
        {
          "kconf_key": "reco.offline.fountainFulllinkDistillRankParam",
          "value_type": "json",
          "json_path": "cascade_s2_full_link_distill_sample_ratio",
          "export_common_attr": "fountain_cascade_s2_full_link_distill_sample_ratio"
        },
      ]
    ) \
    .explore_full_link_context_sample_reco_log_enricher(
      sample_config = [
        {
          "sample_begin": "fountain_cascade_s2_full_link_distill_sample_begin",
          "sample_end": "fountain_cascade_s2_full_link_distill_sample_end",
          "sample_num": "fountain_cascade_s2_full_link_distill_sample_num",
          "label_name": "cas_neg",
        },
      ],
      sample_ratio = "fountain_cascade_s2_full_link_distill_sample_ratio",
      user_info_attr = "userInfoPb",
      tab = 666,
      enable_set_user_info = True,
      load_attr = "fountain_full_link_reco_log_message",
      output_attr = "fountain_full_link_reco_log_message",
      cascade_pctr = "cascade_pctr",
      cascade_pltr = "cascade_pltr",
      cascade_pwtr = "cascade_pwtr",
      cascade_pftr = "cascade_pftr",
      cascade_pptr = "cascade_ptr",
      cascade_pcmtr = "cascade_pcmtr",
      cascade_plvtr = "cascade_plvtr",
      cascade_pvtr = "cascade_pwatch_time",
    )

    return self
  
  def _mc_merchant_photo_boost_by_buyer_type(self):
    """
    功能: 【内流-粗排-挂车短视频】根据买家分层调整对电商视频调权，新买家降权，老买家提权，整体控电商load
    Owner: zhanglinjiang
    Date: 2023-07-05
    :return:
    """
    self.enrich_attr_by_light_function( # 计算挂车粗排权重系数
      import_common_attr = [
        {"name": "merchant_buyer_type", "as": "buyer_type"},
        {"name": "fountain_mc_merchant_photo_boost_coef", "as": "buyer_boost_coef"},
      ],
      export_common_attr = [
        {"name": "merchant_boost_coef", "as": "mc_fountain_merchant_photo_coef"}
      ],
      function_name = "MerchantCalcBoostCoef",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "mc_fountain_merchant_photo_coef", "as": "boost_discount_coeff"},
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
        "is_merchant_cart" : 1
      }
    )

    return self
  
  def _mc_living_photo_boost_by_paying_type(self):
    """
    功能: 【内流-粗排-直播短视频】根据用户付费分层调整对直播短视频调权
    Owner: chenliangliang03
    Date: 2024-03-19
    :return:
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "is_live_big_g_user", "as": "is_live_big_g_user"},
        {"name": "uUserKuaishouLivePayTag", "as": "user_live_paying_type"},
        {"name": "fountain_mc_living_photo_boost_coef_str", "as": "paying_user_boost_coef_str"},
        {"name": "fountain_mc_living_photo_boost_coef_big_g", "as": "boost_coef_big_g"},
      ],
      export_common_attr = [
        {"name": "living_boost_coef", "as": "mc_fountain_living_photo_coef"}
      ],
      function_name = "LivingCalcBoostCoef",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "mc_fountain_living_photo_coef", "as": "boost_discount_coeff"},
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
        "live_photo_info__is_living" : 1, "is_merchant_living" : 0
      }
    )

    return self
  
  def _mc_merchant_live_boost_by_buyer_type(self):
    """
    功能: 【内流-粗排-live头像】根据买家分层调整对电商视频调权，新买家降权，老买家提权，整体控电商load
    Owner: zhanglinjiang
    Date: 2023-07-05
    :return:
    """
    self.enrich_attr_by_light_function( # 计算live头像粗排权重系数
      import_common_attr = [
        {"name": "merchant_buyer_type", "as": "buyer_type"},
        {"name": "fountain_mc_merchant_live_boost_coef", "as": "buyer_boost_coef"},
      ],
      export_common_attr = [
        {"name": "merchant_boost_coef", "as": "mc_fountain_merchant_live_coef"}
      ],
      function_name = "MerchantCalcBoostCoef",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "mc_fountain_merchant_live_coef", "as": "boost_discount_coeff"},
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
        "is_merchant_living" : 1
      }
    )

    return self

  def _mc_merchant_reduce_cart_show(self):
    """
    功能: 打压低效率电商视频曝光
    Owner: zhanglinjiang
    Date: 2023-11-09
    :return:
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_mc_merchant_reduce_cart_show_coef", "as": "boost_discount_coeff"},
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
        "is_merchant_cart" : 1
      }
    )

    return self
  
  def _mc_merchant_reduce_live_show(self):
    """
    功能: 打压低效率电商live头像曝光
    Owner: zhanglinjiang
    Date: 2023-11-09
    :return:
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_mc_merchant_reduce_live_show_coef", "as": "boost_discount_coeff"},
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
        "is_merchant_living" : 1
      }
    )

    return self

  def _mc_produce_uploads_boost(self):
    self \
      .if_("enable_fountain_cascade_produce_predict_all == 1 and fountain_produce_user_type > fountain_cascade_upload_boost_user_switch", to_be_delete = "date=2024-05-29;committer=liuyipeng03") \
        .enrich_attr_by_light_function( # enable_fountain_cascade_produce_predict_all 全局反转开关
          import_common_attr = [
            {"name": "fountain_cascade_produce_uploads_boost_coef", "as": "boost_discount_coeff"}
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
            "fountain_produce_mc_is_produce_uploads_item": 1
          }
        ) \
      .end_()
    return self

  def _mc_produce_item_boost(self):
    self \
      .if_("enable_fountain_cascade_produce_photo_predict == 2 and fountain_produce_user_type > fountain_cascade_produce_user_switch", to_be_delete = "date=2024-05-29;committer=liuyipeng03") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_produce_user_type", "as": "produce_user_type"},
            {"name": "fountain_cascade_new_user_produce_boost_coef_l1", "as": "new_user_produce_boost_coef_l1"},
            {"name": "fountain_cascade_month_user_produce_boost_coef_l1", "as": "month_user_produce_boost_coef_l1"},
            {"name": "fountain_cascade_weeks_user_produce_boost_coef_l1", "as": "weeks_user_produce_boost_coef_l1"},
            {"name": "fountain_cascade_week_user_produce_boost_coef_l1", "as": "week_user_produce_boost_coef_l1"},
          ],
          export_common_attr = [
            {"name": "produce_boost_coef", "as": "fountain_cascade_produce_boost_coef"},
          ],
          function_name = "CalProduceBoostScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_cascade_produce_boost_coef", "as": "boost_discount_coeff"}
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
            "fountain_produce_mc_is_produce_item_l1": 1
          }
        ) \
      .end_()
    return self

  def _cascade_calc_opportunity_score(self):
    self \
    .if_("fountain_mc_enable_calc_opportunity_score == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_mc_opportunity_cost_queue_cost_weight", "as": "mc_opportunity_cost_queue_cost_weight"},
          {"name": "fountain_mc_opportunity_cost_queue_reward_weight", "as": "mc_opportunity_cost_queue_reward_weight"},
          {"name": "fountain_mc_opportunity_cost_queue_pltr_weight", "as": "mc_opportunity_cost_queue_pltr_weight"},
          {"name": "fountain_mc_opportunity_cost_queue_pwtr_weight", "as": "mc_opportunity_cost_queue_pwtr_weight"},
          {"name": "fountain_mc_opportunity_cost_queue_pftr_weight", "as": "mc_opportunity_cost_queue_pftr_weight"},
          {"name": "fountain_mc_opportunity_cost_queue_pcmtr_weight", "as": "mc_opportunity_cost_queue_pcmtr_weight"},
          {"name": "fountain_mc_opportunity_cost_queue_pepstr_weight", "as": "mc_opportunity_cost_queue_pepstr_weight"},
          {"name": "fountain_mc_opportunity_cost_queue_pcltr_weight", "as": "mc_opportunity_cost_queue_pcltr_weight"},
          {"name": "fountain_mc_opportunity_cost_queue_ctr_power_weight", "as": "mc_opportunity_cost_queue_ctr_power_weight"},
        ],
        import_item_attr = [
          "cascade_pctr",
          "cascade_pltr",
          "cascade_pwtr",
          "cascade_pftr",
          "cascade_pcltr",
          "cascade_pepstr",
          "cascade_pcmtr",
          "cascade_pwatch_time",
        ],
        export_item_attr = [
          "mc_ensemble_opportunity_cost_score",
        ],
        function_name = "CalcOpportunityCostScore",
        class_name = "ExploreLightFunctionSetV2",
      )\
    .end_()

    return self

  def _full_link_cascade_s1_distill_sample(self):
    self \
    .get_kconf_params(
      kconf_configs=[
        {
          "kconf_key": "reco.offline.fountainFulllinkDistillRankParam",
          "value_type": "json",
          "json_path": "cascade_s1_full_link_distill_sample_begin",
          "export_common_attr": "fountain_cascade_s1_full_link_distill_sample_begin"
        },
        {
          "kconf_key": "reco.offline.fountainFulllinkDistillRankParam",
          "value_type": "json",
          "json_path": "cascade_s1_full_link_distill_sample_end",
          "export_common_attr": "fountain_cascade_s1_full_link_distill_sample_end"
        },
        {
          "kconf_key": "reco.offline.fountainFulllinkDistillRankParam",
          "value_type": "json",
          "json_path": "cascade_s1_full_link_distill_sample_num",
          "export_common_attr": "fountain_cascade_s1_full_link_distill_sample_num"
        },
        {
          "kconf_key": "reco.offline.fountainFulllinkDistillRankParam",
          "value_type": "json",
          "json_path": "cascade_s1_full_link_distill_sample_ratio",
          "export_common_attr": "fountain_cascade_s1_full_link_distill_sample_ratio"
        },
      ]
    ) \
    .explore_full_link_context_sample_reco_log_enricher(
      target_item = { "cascade_s1_filter_flag": 1},
      sample_config = [
        {
          "sample_begin": "fountain_cascade_s1_full_link_distill_sample_begin",
          "sample_end": "fountain_cascade_s1_full_link_distill_sample_end",
          "sample_num": "fountain_cascade_s1_full_link_distill_sample_num",
          "label_name": "cas_neg_stage1",
        },
      ],
      sample_ratio = "fountain_cascade_s1_full_link_distill_sample_ratio",
      user_info_attr = "userInfoPb",
      enable_set_user_info = True,
      output_attr = "fountain_full_link_reco_log_message",
      cascade_pctr = "cascade_pctr",
      cascade_pltr = "cascade_pltr",
      cascade_pwtr = "cascade_pwtr",
      cascade_pftr = "cascade_pftr",
      cascade_pptr = "cascade_ptr",
      cascade_pcmtr = "cascade_pcmtr",
      cascade_plvtr = "cascade_plvtr",
      cascade_pvtr = "cascade_pwatch_time",
    )

    return self

  def _prerank_hot_content_retr_boost(self):
    self \
      .if_("enable_hot_content_thompson_sampling_corr_calculate == 1", to_be_delete = "date=2024-05-29;committer=guohao") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_mc_prerank_hot_content_retr_boost_coef", "as": "value"},
            {"name": "hot_content_corr", "as": "weight"},
          ],
          export_common_attr = [
            {"name": "new_value", "as": "fountain_mc_prerank_hot_content_retr_boost_coef"},
          ],
          function_name = "CalExploreDoubleMultiDouble",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_mc_prerank_hot_content_retr_boost_coef", "as": "boost_discount_coeff"},
        ],
        import_item_attr = [
          {"name": "cascade_prerank_ensemble_score", "as": "score"},
        ],
        export_item_attr = [
          {"name": "score", "as": "cascade_prerank_ensemble_score"},
        ],
        function_name = "BoostOrDiscountV2",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          "reason" : [341, 416]  # 爆款作品
        }
      )
    return self

  def _prerank_female_porn_discount(self):
    self \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "basic_info_gender_v2",
        ],
        import_item_attr = [
          "audit_b_second_tag",
        ],
        export_item_attr = [
          "is_porn_for_female",
        ],
        function_name = "IsPornForFemale",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_prerank_female_porn_discount_coef", "as": "boost_discount_coeff"},
        ],
        import_item_attr = [
          {"name": "cascade_prerank_ensemble_score", "as": "score"},
        ],
        export_item_attr = [
          {"name": "score", "as": "cascade_prerank_ensemble_score"},
        ],
        function_name = "BoostOrDiscountV2",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          "is_porn_for_female" : 1
        }
      )
    return self

  def _cascade_ensemble_sort_v2(self):
    """
    粗排第二阶段排序
    """
    self \
      .calc_by_formula1(
        kconf_key = "formula.scenarioKey38.FountainMcS2EnsembleSort",
        import_item_attr = [
          {"name": "cascade_pwatch_time", "as": "pwatchtime"},
          {"name": "cascade_pwtd", "as": "pwtd"},
          {"name": "cascade_pctr", "as": "pevtr"},
          {"name": "cascade_slide_kai", "as": "pslide_kai"},
          {"name": "cascade_plvtr", "as": "plvtr"},
          {"name": "cascade_longview_score", "as": "longview_score"},
          {"name": "cascade_pltr", "as": "pltr"},
          {"name": "cascade_pwtr", "as": "pwtr"},
          {"name": "cascade_pftr", "as": "pftr"},
          {"name": "cascade_ptr", "as": "ptr"},
          {"name": "cascade_pepstr", "as": "pepstr"},
          {"name": "cascade_pcmtr", "as": "pcmtr"},
          {"name": "cascade_pcestr", "as": "pcestr"},
          {"name": "cascade_pcltr", "as": "pcltr"},
          {"name": "cascade_act_kai", "as": "act_once_kai"},
          {"name": "cascade_phtr", "as": "phtr"},
          {"name": "cascade_distill_fast_rank", "as": "distill_fast_rank"},
          {"name": "cascade_distill_show", "as": "distill_show"},
          {"name": "cascade_pcotr", "as": "pcotr"},
          {"name": "cascade_fl_realshow_reward", "as": "fl_realshow_reward"},
          {"name": "explore_stat__click_count", "as": "explore_click_count"},
          {"name": "explore_stat__real_show_count", "as": "explore_realshow_count"},
        ],
        import_common_attr = [
          "user_emp_ltr",
          "user_emp_wtr",
          "user_emp_ftr",
          "user_emp_cmtr",
          "user_emp_eptr",
          "fountain_cascade_ensemble_power_weight_cascade_like_emp",
          "fountain_cascade_ensemble_power_weight_cascade_follow_emp",
          "fountain_cascade_ensemble_power_weight_cascade_forward_emp",
          "fountain_cascade_ensemble_power_weight_cascade_comment_emp",
          "fountain_cascade_ensemble_power_weight_cascade_eps_emp"
        ],
        export_formula_value = [
          {"name": "final_score", "as": "cascade_ensemble_score"}
        ],
        abtest_biz_name = "KUAISHOU_APPS"
      ) \
      .if_("enable_fountain_mc_s2_sort_v2_adjust == 1") \
        .item_attr_operation(
          item_attr_a = "cascade_ensemble_score",
          item_attr_b = "mc_adjust_coeff_final",
          operator = "*",
          output_attr = "cascade_ensemble_score"
        ) \
      .end_() \
      ._dump_attr_to_kafka( # 混合排序之后, 将全部item的重要 item attr 落盘
        stage_name = "mc_s2_score",
        dump_item_attr_list = [
          # 分桶排序分
          "cascade_variant_sort_score",
          # 混合排序分
          "cascade_ensemble_score",
          # 和入口视频的相关分
          "fountain_related_score_v2_detail"
        ]
      ) \
      .sort(
        score_from_attr = "cascade_ensemble_score",
      )
    return self

  def _get_recent_emp_xtr(self):
    self \
    .explore_user_emp_xtr_enricher(
      colossus_resp_attr = "colossus_resp_v2",
      enable_colossus_item_limit = 1,
      max_colossus_item_num = 10000,
      user_colossus_min_sec_ago = "{{recent_colossus_min_sec_ago}}",
      user_colossus_max_sec_ago = "{{recent_colossus_max_sec_ago}}",
      save_user_stats_click_count = "user_recent_stats_click_count",
      save_user_emp_watch_time = "user_recent_emp_watch_time",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        "user_recent_stats_click_count", "user_recent_emp_watch_time",
        "user_colossus_click_count", "user_emp_watch_time",
        "fountain_longterm_significant_thresh", "fountain_shorterm_significant_thresh",
        "fountain_trend_significant_upper", "fountain_trend_significant_lower",
        "fountain_trend_weight_max", "fountain_trend_weight_min",
        "fountain_trend_weight_alpha", "fountain_trend_weight_beta",
      ],
      export_common_attr = ["fountain_watchtime_trend_weight"],
      function_name = "CalcFountainWatchtimeTrendWeight",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def _his_cur_duration_longview_adjust(self):
    self \
    .enrich_attr_by_light_function(
      import_common_attr = [
        "is_work_day",
        {"name": "fountain_enable_longview_adjust_is_work_day", "as": "enable_longview_adjust_is_work_day"},
        {"name": "fountain_longview_adjust_weekend_factor", "as": "longview_adjust_weekend_factor"},
        "colossus_duration_list",
        "colossus_play_time_list",
        "colossus_timestamp_list",
        {"name": "fountain_duration_longview_adjust_num_threshold", "as": "duration_longview_adjust_num_threshold"},
        {"name": "fountain_duration_longview_adjust_long_duration_threshold", "as": "duration_longview_adjust_long_duration_threshold"},
        {"name": "fountain_duration_longview_adjust_long_play_threshold", "as": "duration_longview_adjust_long_play_threshold"},
        {"name": "fountain_duration_longview_adjust_interval_threshold", "as": "duration_longview_adjust_interval_threshold"},
        {"name": "fountain_duration_longview_adjust_significant_upper", "as": "duration_longview_adjust_significant_upper"},
        {"name": "fountain_duration_longview_adjust_significant_lower", "as": "duration_longview_adjust_significant_lower"},
        {"name": "fountain_duration_longview_adjust_weight_alpha", "as": "duration_longview_adjust_weight_alpha"},
        {"name": "fountain_duration_longview_adjust_weight_beta", "as": "duration_longview_adjust_weight_beta"},
        {"name": "fountain_duration_longview_adjust_weight_max", "as": "duration_longview_adjust_weight_max"},
        {"name": "fountain_duration_longview_adjust_weight_min", "as": "duration_longview_adjust_weight_min"},
      ],
      export_common_attr = ["fountain_duration_longview_adjust_weight"],
      function_name = "HisCurDurationLongviewAdjust",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self
  
  def _cascade_boost_top_photo_topk(self):
    self \
    .if_("fountain_enable_gen_is_reason_top_photo_modify_day0 == 1") \
      .gen_is_reason_top_photo_modify_day0() \
    .else_() \
      .gen_is_reason_top_photo() \
    .end_() \
    .item_attr_operation(
      item_attr_a = "cascade_variant_sort_adjust_score",
      common_attr_b = "{{fountain_reason_top_photo_boost_coeff}}",
      operator = "*",
      target_item = {"is_top_reason_topk_boost_photo": 1},
      output_attr = "cascade_variant_sort_adjust_score"
    )
    return self