from cascading.common_module import CommonModule

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
      .if_("enable_life_colossus_cluster_new == 1") \
        .explore_life_colossus_cluster_enricher(
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
          enable_longterm_interest_calc_new = "{{life_cluster_enable_longterm_interest_calc_new}}",
          skip_recent_interest_calc = "{{life_cluster_skip_recent_interest_calc}}",
          play_time_second_threshold = "{{life_cluster_play_time_second_threshold}}",
          time_decay_coeff_min = "{{life_cluster_time_decay_coeff_min}}",
          like_weight = "{{life_cluster_like_weight}}",
          follow_weight = "{{life_cluster_follow_weight}}",
          comment_weight = "{{life_cluster_comment_weight}}",
          forward_weight = "{{life_cluster_forward_weight}}",
          enter_profile_weight = "{{life_cluster_enter_profile_weight}}",
          finish_play_weight = "{{life_cluster_finish_play_weight}}",
          play_time_weight = "{{life_cluster_play_time_weight}}",
        ) \
      .else_() \
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
          enable_longterm_interest_cluster_opt = "{{enable_longterm_interest_cluster_opt}}"
        ) \
      .end_() \
      .if_("enable_explore_hetu_tags_replace == 1") \
        .copy_attr(
          attrs = [{
            "from_common": "explore_hetu_tags",
            "to_common": "sim_explore_tags",
          }]
        ) \
      .end_() \
      .if_("enable_longterm_interest_cluster_opt == 1") \
        .copy_attr(
          attrs = [{
            "from_common": "interest_explore_longterm_hetu_one",
            "to_common": "sim_one_tags",
          },
          {
            "from_common": "interest_explore_longterm_hetu_two",
            "to_common": "sim_two_tags",
          },
          {
            "from_common": "interest_explore_longterm_hetu_three",
            "to_common": "sim_three_tags",
          }]
        ) \
      .end_() \
    .end_() \
    .if_("enable_life_unbias_interest_cluster == 1 and life_unbias_interest_list == nil") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "life_unbias_interest_cluster_prefix", "as": "key_prefix"},
          "basic_info_age_segment_v2",
          "basic_info_gender_v2",
        ],
        export_common_attr = [
          "user_age_gender_key"
        ],
        function_name = "GetUserAgeGenderKey",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .get_kconf_params(
        kconf_configs = [{
          "kconf_key": "reco.eyeshot.life_unbias_interest_hetu2_map",
          "json_path": "{{user_age_gender_key}}",
          "default_value": "",
          "export_common_attr": "life_unbias_interest_str"
        }]
      ) \
      .split_string(
        input_common_attr = "life_unbias_interest_str",
        output_common_attr = "life_unbias_interest_list",
        delimiters = ",",
        skip_empty_tokens = True,
        trim_spaces = True,
        parse_to_int = True
      ) \
      .if_("enable_life_unbias_interest_adjust_low_active == 1 and uIsLifeHighActive ~= 1") \
        .gen_common_attr_by_lua(
          attr_map={
            "life_unbias_interest_start_hetu_count": "life_unbias_interest_start_hetu_count_low_active",
            "life_unbias_interest_final_hetu_count": "life_unbias_interest_final_hetu_count_low_active",
          }
        ) \
      .end_() \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "user_info_ptr",
          {"name": "life_unbias_interest_list", "as": "unbias_interest_list"},
          {"name": "life_unbias_interest_time_gap_min", "as": "time_gap_min"},
          {"name": "life_unbias_interest_short_play_thresh", "as": "short_play_thresh"},
          {"name": "life_unbias_interest_short_play_rate_thresh", "as": "short_play_rate_thresh"},
          {"name": "life_unbias_interest_start_hetu_count", "as": "start_hetu_count"},
          {"name": "life_unbias_interest_final_hetu_count", "as": "final_hetu_count"},
          {"name": "life_unbias_interest_filter_hate_hetu", "as": "filter_hate_hetu"},
        ],
        export_common_attr = [
          {"name": "final_unbias_interest_list", "as": "life_unbias_interest_list"},
        ],
        function_name = "ShuffleUnbiasInterestList",
        class_name = "ExploreLifeLightFunctionSet"
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
          "user_group_emp_ltr",
          "user_group_emp_wtr",
          "user_group_emp_ftr",
          "user_group_emp_cmtr",
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
    .if_("enable_mc_htr_weight_adjust_s1 == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_mc_ensemble_s1_phtr_power_weight", "as": "raw_htr_weight"},
          "find_user_active_degree",
          "page_index",
          "boost_htr_weight_by_mid_degree",
          "boost_htr_weight_by_high_degree",
          "boost_htr_weight_by_full_degree",
          "boost_htr_weight_by_p_idx",
          "boost_htr_weight_p_idx_down",
          "boost_htr_weight_p_idx_up" 
        ],
        export_common_attr = [
          {"name": "raw_htr_weight", "as": "explore_mc_ensemble_s1_phtr_power_weight"},
        ],
        function_name = "HtrWeightAdjust",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .if_("explore_cascade_s1_weight_low_follow == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "user_follow_type", "as": "user_follow_type"},
          {"name": "cascade_s1_low_follow_pwtr_weight", "as": "rank_low_follow_pwtr_weight"},
          {"name": "cascade_s1_no_follow_pwtr_weight", "as": "no_follow_pwtr_weight"},
          {"name": "cascade_s1_valid_follow_pwtr_weight", "as": "valid_follow_pwtr_weight"},
          {"name": "cascade_s1_valid_low_follow_pwtr_weight", "as": "valid_low_follow_pwtr_weight"},
          {"name": "cascade_s1_valid_media_follow_pwtr_weight", "as": "valid_media_follow_pwtr_weight"},
          {"name": "cascade_s1_valid_high_follow_pwtr_weight", "as": "valid_high_follow_pwtr_weight"},
          {"name": "cascade_s1_low_follow_thres_s", "as": "rank_low_follow_thres_s"},
          {"name": "cascade_s1_enable_no_follow_boost", "as": "enable_no_follow_boost"},
          {"name": "cascade_s1_enable_low_follow_boost", "as": "enable_low_follow_boost"},
          {"name": "cascade_s1_low_follow_boost_threshold", "as": "low_follow_boost_threshold"},
          "follow_timestamps",
          {"name": "explore_mc_ensemble_s1_pwtr_weight", "as": "input_pwtr_score"}
        ],
        export_common_attr = [
          {"name": "output_pwtr_score", "as": "explore_mc_ensemble_s1_pwtr_weight"},
        ],
        function_name = "UserSortWeightLowFollow",
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
      lua_script_file = "life/cascading/lua/module/cascading_channel_sort_prepare__attr_trans.lua"
    ) \
    .if_("life_enable_intere_score_str == 1") \
    .enrich_attr_by_lua(
      import_common_attr = [
          "intere_str_list",
        ],
        export_common_attr = [
          "intere_score_str","cluster_id_str","cluster_vv_str"
        ],
        function_for_common = "get_intere_lists",
        lua_script_file = "life/cascading/lua/module/cascading_channel_sort_prepare__attr_trans.lua"
    )\
    .split_string(
      input_common_attr = "intere_score_str",
      output_common_attr = "intere_score_list", 
      delimiters=",",
      parse_to_double=True,
    )\
      .split_string(
      input_common_attr = "cluster_id_str",
      output_common_attr = "cluster_id_list", 
      delimiters=",",
      parse_to_int=True,
    )\
      .split_string(
      input_common_attr = "cluster_vv_str",
      output_common_attr = "cluster_vv_list", 
      delimiters=",",
      parse_to_int=True,
    )\
    .enrich_attr_by_lua(
      import_common_attr = [
          "intere_score_list",
          "cluster_id_list"
        ],
        export_common_attr = [
          "cluster_intere_score_map"
        ],
        function_for_common = "get_intere_scores",
        lua_script_file = "life/cascading/lua/module/cascading_channel_sort_prepare__attr_trans.lua"
    ) \
    .enrich_attr_by_lua(
      import_common_attr = [
          "cluster_intere_score_map",
          "remap_cluster_id_632_list"
        ],
        import_item_attr = [
          "hetu_tag_level_info__hetu_cluster_id"
        ],
        export_item_attr = [
          "intere_score", "is_eff_intere", "is_not_intere","nonneg_intere_score"
        ],
        function_for_item = "get_item_intere_score",
        lua_script_file = "life/cascading/lua/module/cascading_channel_sort_prepare__attr_trans.lua"
    ) \
    .log_debug_info(
        item_attrs = [
          "hetu_sim_cluster_id","hetu_sim_cluster_id862","nonneg_intere_score",
          "hetu_tag_level_info__hetu_cluster_id","intere_score", "is_eff_intere", "is_not_intere"
        ],
        for_debug_request_only = True,
        respect_sample_loggging = True,
      ) \
      .log_debug_info(
        common_attrs = [
          "cluster_intere_score_map",
          "cluster_vv_list", 
          "intere_score_list", 
          "cluster_id_list", 
           "intere_score_str",
           "cluster_id_str",
           "intere_str_list",
           "combo_intere_str",
          "remap_cluster_id_632_list"
        ],
        for_debug_request_only = True,
        respect_sample_loggging = True,
      ) \
    .end_() \
    .if_("explore_enable_mc_phtr_filter == 1") \
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
        attr_name = "mc_need_htr_filter",
        remove_if = "==",
        compare_to = 1,
        cancel_num = "{{phtr_filter_reserved_num}}"
      ) \
    .end_() \
    .if_("mc_ctr_filter_ensemble == 1") \
      .count_reco_result(
        save_count_to="explore_reco_leaf_mc_model_input_count"
      ) \
      .if_("mc_absolute_ctr_filter == 1") \
        .filter_by_attr(
          attr_name = "mc_ensemble_pctr",
          remove_if = "<",
          compare_to = "{{explore_mc_pctr_filter_threshold}}",
          cancel_num = "{{return math.floor(explore_reco_leaf_mc_model_input_count * (1 - mc_ctr_filter_distribue_coff))}}"
        ) \
      .else_() \
        .explore_calc_ensemble_score(
          save_score_to_attr = "mc_ctr_filter_ensemble_score",
          use_superscript_rank = True,
          user_power_calc_v2 = 1,
          user_info_ptr_attr = "user_info_ptr",
          rank_smooth = 10,
          queues = [
            {
              "name": "mc_ensemble_pctr",
              "weight": 1.0,
              "power_weight_attr": "explore_mc_filter_ctr_weight"
            },
            {
              "name": "mc_ensemble_peftr",
              "weight": 0.0,
              "power_weight_attr": "explore_mc_filter_eftr_weight"
            },
            {
              "name": "mc_ensemble_pefctr",
              "weight": 0.0,
              "power_weight_attr": "explore_mc_filter_efctr_weight"
            },
            {
              "name": "mc_ensemble_plvtr2",
              "weight": 0.0,
              "power_weight_attr": "explore_mc_filter_lvtr2_weight"
            },
            {
              "name": "mc_ensemble_psvtr",
              "weight": 0.0,
              "power_weight_attr": "explore_mc_filter_svtr_weight"
            },
            {
              "name": "mc_ensemble_pwtd",
              "weight": 0.0,
              "power_weight_attr": "explore_mc_filter_wtd_weight"
            },
          ]
        ) \
        .sort(
          score_from_attr = "mc_ctr_filter_ensemble_score"
        ) \
        .if_("mc_ctr_filter_ensemble_pic_vd == 1") \
          .count_reco_result(
            save_count_to = "mc_pic_count",
            target_item = {"is_picture" : 1}
          ) \
          .count_reco_result(
            save_count_to = "mc_vd_count",
            target_item = {"is_picture" : 0}
          ) \
          .limit(
            size = "{{return math.floor(mc_pic_count * (1 - mc_pic_ctr_filter_distribue_coff))}}",
            target_item = {"is_picture" : 1}
          ) \
          .limit(
            size = "{{return math.floor(mc_vd_count * (1 - mc_ctr_filter_distribue_coff))}}",
            target_item = {"is_picture" : 0}
          ) \
        .else_() \
          .limit(size = "{{return math.floor(explore_reco_leaf_mc_model_input_count * (1 - mc_ctr_filter_distribue_coff))}}") \
        .end_() \
      .end_() \
    .end_()\
    .if_("enable_calc_same_location_flag == 1") \
      .transform_item_attr(
        mappings=[{
          "check_attr_name": "location__province_id",
          "check_attr_type": "int",
          "output_attr_name": "is_same_location",
          "output_attr_type": "int",
          "output_default_value": 0,
          "rules": [{
            "check_values": ["{{uProvinceId}}"],
            "output_value": 1
          }]
        }]
      ) \
    .end_() \

  def post_process(self) -> None:
    pass
