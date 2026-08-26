from cascading.common_module import CommonModule
from cascading.module.queue.cascade_queues import cluster_variant_sort_weight_param_dict

# coding: utf-8
"""
- Description:
- Author: linpengpeng@kuaishou.com
- Date: 2022-07-06
"""

class CascadingChannelSortPrepareModule(CommonModule):

  def __init__(self, module_name):
    super().__init__(module_name)
  
  def process(self) -> None:
    self.flow \
    .if_("enable_mc_tag_sim == 1") \
      .explore_colossus_cluster_enricher(
        colossus_v2_attr_name = "colossus_resp_v2",
        user_info_ptr_attr = "user_info_ptr",
        export_colossus_attr_one = "sim_one_tags",
        export_colossus_attr_two = "sim_two_tags",
        export_colossus_attr_three = "sim_three_tags",
        export_colossus_attr_explore = "sim_explore_tags",
        enable_mc_explore_cluster = "{{enable_mc_explore_cluster}}",
        mc_explore_cluster_limit = "{{mc_explore_cluster_limit}}",
        mc_explore_cluster_score_limit = "{{mc_explore_cluster_score_limit}}",
        enable_mc_interact_cluster = "{{enable_mc_interact_cluster}}",
        export_colossus_attr_interact = "sim_interact_tags",
        enable_mc_explore_cluster_v2 = "{{enable_mc_explore_cluster_v2}}",
        enable_mc_explore_cluster_target = "{{enable_mc_explore_cluster_target}}",
        mc_explore_cluster_target_count_limit = "{{mc_explore_cluster_target_count_limit}}",
        high_quality_tags_attr = "high_quality_tags",
        mc_explore_cluster_recent_click_count_limit = "{{mc_explore_cluster_recent_click_count_limit}}",
        mc_explore_cluster_recent_show_count_limit = "{{mc_explore_cluster_recent_show_count_limit}}",
        mc_explore_cluster_click_count_limit = "{{mc_explore_cluster_click_count_limit}}",
        mc_explore_cluster_click_time_limit = "{{mc_explore_cluster_click_time_limit}}",
        mc_explore_cluster_recent_click_top_ratio = "{{mc_explore_cluster_recent_click_top_ratio}}",
        mc_explore_cluster_recent_show_top_ratio = "{{mc_explore_cluster_recent_show_top_ratio}}",
        enable_longterm_interest_cluster_opt = "{{enable_longterm_interest_cluster_opt}}",
        enable_interest_vary_by_scenario = "{{enable_mc_long_term_interest_vary_by_scenario}}",
        fountain_interest_ratio = "{{mc_long_term_fountain_interest_ratio}}",
        gamora_interest_ratio = "{{mc_long_term_gamora_interest_ratio}}",
        colossus_day_upper = "{{mc_long_term_colossus_day_upper}}"
      ) \
    .end_() \
    .if_("explore_mc_sort_weight_adjust_s1 == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "enable_explore_weight_adjust_v2",
          "explore_colossus_user_emp_xtr_map_ptr",
          "explore_weight_adjust_coeff_a",
          "explore_weight_adjust_coeff_b",
          "explore_weight_adjust_coeff_c",
          "explore_weight_adjust_coeff_d",
          "user_emp_ltr",
          "user_emp_wtr",
          "user_emp_ftr",
          "user_emp_cmtr",
          "user_emp_eptr",
          {"name": "explore_weight_adjust_avg_emp_ltr", "as": "all_user_emp_ltr"},
          {"name": "explore_weight_adjust_avg_emp_wtr", "as": "all_user_emp_wtr"},
          {"name": "explore_weight_adjust_avg_emp_ftr", "as": "all_user_emp_ftr"},
          {"name": "explore_weight_adjust_avg_emp_cmtr", "as": "all_user_emp_cmtr"},
          {"name": "explore_weight_adjust_avg_emp_eptr", "as": "all_user_emp_eptr"},
          {"name": "explore_mc_ensemble_s1_pltr_weight", "as": "user_ori_ltr_weight"},
          {"name": "explore_mc_ensemble_s1_pwtr_weight", "as": "user_ori_wtr_weight"},
          {"name": "explore_mc_ensemble_s1_pftr_weight", "as": "user_ori_ftr_weight"},
          {"name": "explore_mc_ensemble_s1_pcmtr_weight", "as": "user_ori_cmtr_weight"},
          {"name": "explore_mc_ensemble_s1_pepstr_weight", "as": "user_ori_eptr_weight"},
          "explore_weight_adjust_coeff_min",
          "explore_weight_adjust_coeff_max"
        ],
        export_common_attr = [
          {"name": "user_ltr_weight", "as": "explore_mc_ensemble_s1_pltr_weight"},
          {"name": "user_wtr_weight", "as": "explore_mc_ensemble_s1_pwtr_weight"},
          {"name": "user_ftr_weight", "as": "explore_mc_ensemble_s1_pftr_weight"},
          {"name": "user_cmtr_weight", "as": "explore_mc_ensemble_s1_pcmtr_weight"},
          {"name": "user_eptr_weight", "as": "explore_mc_ensemble_s1_pepstr_weight"},
        ],
        function_name = "UserSortWeightAdjust",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .enrich_attr_by_lua(
      import_common_attr = [
        "photo_count_cascading_begin",
        "explore_mc_phtr_max_filter_rate"
      ],
      export_common_attr = [
        "phtr_filter_reserved_num"
      ],
      function_for_common = "htr_filter_threshold",
      lua_script_file = "explore/cascading/lua/module/cascading_channel_sort_prepare__attr_trans.lua"
    ) \
    .if_("explore_enable_mc_phtr_filter == 1", to_be_delete = "date=2024-05-29;committer=linpengpeng") \
      .filter_by_attr(
        attr_name = "cascade_phtr",
        remove_if = ">",
        compare_to = "{{explore_mc_phtr_filter_threshold}}",
        cancel_num = "{{phtr_filter_reserved_num}}"
      ) \
    .end_() \
    .if_("explore_enable_mc_phtr_filter_v2 == 1") \
      .enrich_with_protobuf(
        from_extra_var = "user_info_ptr",
        attrs = [
          dict(name="hate_ts_list", path="user_profile_v1.hate_list.time_ms"),
        ]
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "hate_ts_list",
          "base_phtr_filter_threshold",
          "min_phtr_filter_threshold",
          "recent_minute_for_high_freq_hate",
          "phtr_thrshold_temperature",
          "phtr_thrshold_smooth",
          "mc_htr_filter_ltr_threshold",
          "mc_htr_filter_wtr_threshold",
        ],
        import_item_attr = [
          "cascade_phtr",
          "mc_ensemble_pltr",
          "mc_ensemble_pwtr", 
        ],
        export_item_attr = [
          "mc_need_htr_filter",
        ],
        function_name = "NeedHtrFilter",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .filter_by_attr(
        name = "explore_mc_prepare_phtr_filter",
        traceback = True,
        attr_name = "mc_need_htr_filter",
        remove_if = "==",
        compare_to = 1,
        cancel_num = "{{phtr_filter_reserved_num}}"
      ) \
    .end_() \
    .if_("explore_enable_mc_pctr_filter == 1") \
      .count_reco_result(
        save_count_to = "explore_mc_prepare_input_count"
      ) \
      .count_reco_result(
        save_count_to = "explore_mc_prepare_input_count_video",
        target_item = {"is_picture" : 0}
      ) \
      .count_reco_result(
        save_count_to = "explore_mc_prepare_input_count_pic",
        target_item = {"is_picture" : 1}
      ) \
      .sort(
        score_from_attr = "cascade_corr_pctr"
      ) \
      .if_("enable_explore_mc_prepare_dynamic_filter_coff == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_mc_pctr_filter_distribue_coff", "as": "xtr_weight"},
            {"name": "active_days_avg_vv", "as": "user_vv"},
            {"name": "explore_mc_prepare_dynamic_filter_coff_exp_upper", "as": "exp_upper"},
            {"name": "explore_mc_prepare_dynamic_filter_coff_alpha", "as": "alpha"},
            {"name": "explore_mc_prepare_dynamic_filter_coff_beta", "as": "beta"},
            {"name": "explore_mc_prepare_dynamic_filter_coff_omega", "as": "omega"},
            {"name": "explore_mc_prepare_dynamic_filter_coff_max", "as": "coeff_max"},
            {"name": "explore_mc_prepare_dynamic_filter_coff_min", "as": "coeff_min"},
          ],
          export_common_attr = [
            {"name": "xtr_weight", "as": "explore_mc_pctr_filter_distribue_coff"},
          ],
          function_name = "AdjustWeightByUserVv",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_enable_mc_pic_pctr_filter_quota == 1") \
        .limit(
          size = "{{return math.floor(explore_mc_prepare_input_count_video * (1 - explore_mc_pctr_filter_distribue_coff))}}",
          target_item = {"is_picture" : 0}
        ) \
        .limit(
          size = "{{return math.floor(explore_mc_prepare_input_count_pic * (1 - explore_mc_pic_pctr_filter_distribue_coff))}}",
          target_item = {"is_picture" : 1}
        ) \
      .else_() \
        .limit(
          size = "{{return math.floor(explore_mc_prepare_input_count * (1 - explore_mc_pctr_filter_distribue_coff))}}",
          name = "explore_mc_prepare_pctr_filter",
          traceback = True,
        ) \
      .end_() \
    .end_() \
    .if_("explore_enable_mc_s1_ef_weight_adjust == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "mc_eftr_ensemble_sort_weight_s1": "mc_eftr_ensemble_sort_weight_s1 * explore_fountain_view_weight",
          "mc_efctr_ensemble_sort_weight_s1": "mc_efctr_ensemble_sort_weight_s1 * explore_fountain_view_weight",
        }
      ) \
    .end_() \
    .if_("enable_cascading_mc_ensemble_s1_pwtr_weight_new_follow_adjust == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "explore_mc_ensemble_s1_pwtr_weight": "explore_mc_ensemble_s1_pwtr_weight * new_follow_pwtr_cascading_s1_adjust_ratio",
        }
      ) \
    .end_() \
    .if_("enable_mc_s1_user_age_tgi_product_first_refresh_weight_adjust == 1 and is_first_refresh == 1") \
      .gen_common_attr_by_lua( # user_age_interest_tagnex_tgi_product_pxtr_score 首屏权重独立设置
        attr_map = {
          "explore_mc_ensemble_s1_user_age_interest_tagnex_tgi_product_pxtr_score_weight" : "explore_mc_s1_user_age_tgi_product_first_refresh_weight",
          "explore_mc_ensemble_s1_user_age_interest_tagnex_tgi_product_pxtr_score_power_weight" : "explore_mc_s1_user_age_tgi_product_first_refresh_power_weight",
          "explore_mc_ensemble_s1_user_age_interest_tagnex_tgi_product_pxtr_score_raw_weight" : "explore_mc_s1_user_age_tgi_product_first_refresh_raw_weight",
          "explore_mc_ensemble_s1_user_age_interest_tagnex_tgi_product_pxtr_score_raw_power_weight" : "explore_mc_s1_user_age_tgi_product_first_refresh_raw_power_weight",
        }
      ) \
    .end_() \
    .if_("enable_user_age_tgi_score_population_weight_adjust == 1 and basic_info_age_segment_v2 > user_age_tgi_score_population_age_segment_threshold and active_days_gt_5min_rate < user_age_tgi_score_population_active_days_threshold") \
      .copy_attr(
        attrs = [{
          "from_common": "explore_mc_ensemble_s1_user_age_interest_tagnex_tgi_product_pxtr_score_population_weight",
          "to_common": "explore_mc_ensemble_s1_user_age_interest_tagnex_tgi_product_pxtr_score_weight"
        },
        {
          "from_common": "explore_mc_ensemble_s1_user_age_interest_tagnex_tgi_product_pxtr_score_population_power_weight",
          "to_common": "explore_mc_ensemble_s1_user_age_interest_tagnex_tgi_product_pxtr_score_power_weight"
        },
        {
          "from_common": "explore_mc_ensemble_s1_user_age_interest_tagnex_tgi_product_pxtr_score_population_raw_weight",
          "to_common": "explore_mc_ensemble_s1_user_age_interest_tagnex_tgi_product_pxtr_score_raw_weight"
        },
        {
          "from_common": "explore_mc_ensemble_s1_user_age_interest_tagnex_tgi_product_pxtr_score_population_raw_power_weight",
          "to_common": "explore_mc_ensemble_s1_user_age_interest_tagnex_tgi_product_pxtr_score_raw_power_weight"
        }]
      ) \
    .end_() \
    .if_("explore_user_group_consume_weight_adjust_prepare == 1") \
      .cast_attr_type(
        attr_type_cast_configs = [{
          "to_type": "string",
          "from_common_attr": "uMultiDimensionGroupDetailKV",
          "to_common_attr": "uMultiDimensionGroupDetailKV_str"
        }]
      ) \
      .get_kconf_params(
        kconf_configs = [{
          "kconf_key": "reco.offline.userGroupConsumeXtrStat",
          "json_path": "all",
          "default_value": "",
          "export_common_attr": "explore_all_user_consume_str"
        }, {
          "kconf_key": "reco.offline.userGroupConsumeXtrStat",
          "json_path": "{{uMultiDimensionGroupDetailKV_str}}",
          "default_value": "",
          "export_common_attr": "explore_user_group_consume_str"
        }]
      ) \
    .end_() \
    .if_("explore_user_group_consume_weight_adjust_mc_s1 == 1") \
      .user_group_consume_weight_adjust(cluster_variant_sort_weight_param_dict, "mc_s1") \
    .end_()


  def post_process(self) -> None:
    pass
