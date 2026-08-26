#!/usr/bin/env python3
# coding=utf-8

from dragonfly.ext.common_leaf_base_mixin import CommonLeafBaseMixin

class ExploreCascadeStrategyMixin(CommonLeafBaseMixin):
  """
  双列发现页外流粗排策略函数 Mixin 实现
  """

  def boost_young_photo_by_vv(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        "basic_info_age_segment_v2",
        {"name": "mc_young_vv_photo_threshold", "as": "young_vv_photo_threshold"},
        {"name": "mc_young_vv_pic_threshold", "as": "young_vv_pic_threshold"},
        {"name": "mc_city_vv_boost_threshold", "as": "city_vv_boost_threshold"},
        {"name": "mc_young_photo_18_23_prob_boost_threshold", "as": "young_photo_18_23_prob_boost_threshold"},
        {"name": "mc_young_photo_24_30_prob_boost_threshold", "as": "young_photo_24_30_prob_boost_threshold"},
        {"name": "mc_young_vv_boost_coeff", "as": "young_vv_boost_coeff"},
        {"name": "mc_city_vv_boost_coeff", "as": "city_vv_boost_coeff"},
        {"name": "mc_young_photo_18_23_prob_boost_coeff", "as": "young_photo_18_23_prob_boost_coeff"},
        {"name": "mc_young_photo_24_30_prob_boost_coeff", "as": "young_photo_24_30_prob_boost_coeff"},
        {"name": "mc_young_age_score_cliff_ratio", "as": "young_age_score_cliff_ratio"},
        {"name": "mc_age_0_12_score_cliff_ratio", "as": "age_0_12_score_cliff_ratio"},
        {"name": "mc_age_12_17_score_cliff_ratio", "as": "age_12_17_score_cliff_ratio"},
        {"name": "mc_age_18_23_score_cliff_ratio", "as": "age_18_23_score_cliff_ratio"},
        {"name": "mc_age_24_30_score_cliff_ratio", "as": "age_24_30_score_cliff_ratio"},
        {"name": "mc_age_31_40_score_cliff_ratio", "as": "age_31_40_score_cliff_ratio"},
        {"name": "mc_age_41_49_score_cliff_ratio", "as": "age_41_49_score_cliff_ratio"},
        {"name": "mc_age_greater_50_score_cliff_ratio", "as": "age_greater_50_score_cliff_ratio"},
        {"name": "mc_enable_personal_cliff_ratio", "as": "enable_personal_cliff_ratio"},
        {"name": "mc_enable_young_photo_boost_rate_threshold", "as": "enable_young_photo_boost_rate_threshold"},
        {"name": "mc_young_photo_boost_rate_threshold", "as": "young_photo_boost_rate_threshold"}
      ],
      import_item_attr = [
        "is_picture",
        "da_young_18_30_vv_rate",
        "da_1_2_city_vv_rate",
        "young_photo_18_23_prob",
        "young_photo_24_30_prob",
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        "young_age_boost_rate",
        {"name": "score", "as": score_attr},
      ],
      target_item = { flag_attr: 1 },
      function_name = "YoungAgeBoost",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def multiply_gate_score(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "hot_cascade_psvtr_gate_alpha", "as": "svtr_alpha"},
        {"name": "hot_cascade_psvtr_gate_beta", "as": "svtr_beta"},
        {"name": "hot_cascade_pctr_gate_alpha", "as": "ctr_alpha"},
        {"name": "hot_cascade_pctr_gate_beta", "as": "ctr_beta"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "es_score"},
        {"name": "mc_ensemble_psvtr", "as": "svtr_score"},
        {"name": "mc_ensemble_pctr", "as": "ctr_score"},
      ],
      export_item_attr = [
        {"name": "es_score", "as": score_attr},
      ],
      target_item = { flag_attr: 1 },
      function_name = "EsScoreMultiplyGate",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def boost_click_count(self, score_attr):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": score_attr, "as": "ensemble_score"},
        "explore_stat__click_count"
      ],
      import_common_attr = [
        "click_thred",
        "boost_click_count_alpha",
        "boost_click_count_beta",
        "boost_click_count_omega",
        "boost_click_val_max",
        "boost_click_val_min"
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr}
      ],
      function_name = "BoostClickCount",
      class_name = "ExploreLightFunctionSetV2"
    )
    return self

  def prerank_select_photo_by_interest(self, score_attr_name, flag_attr_name):
    self.sort(
       score_from_attr = score_attr_name,
       target_item = {
         flag_attr_name : 1
       }
    ) \
    .if_("explore_enable_user_need_break_cocoon_prerank == 1 and user_need_break_cocoon_flag == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "prerank_control_hetu5_max_size", "as": "value"},
          {"name": "user_need_break_cocoon_prerank_control_hetu5_coef", "as": "weight"}
        ],
        export_common_attr = [
          {"name": "new_value", "as": "prerank_control_hetu5_max_size"}
        ],
        function_name = "CalExploreIntMultiDouble",
        class_name = "ExploreLightFunctionSetV2"
      ) \
    .end_() \
    .explore_control_hetu_count_enricher(
      save_flag_to_attr = "prerank_diversity_select_flag",
      keep_size = "{{prerank_final_candidate_num}}", 
      enable_hetu_control_diversity = "{{prerank_enable_hetu_control_diversity}}",
      hetu_level_five_attr = "hetu_tag_level_info__hetu_level_five",
      hetu5_max_size = "{{prerank_control_hetu5_max_size}}",
      enable_minority_control_diversity = "{{prerank_enable_minority_control_diversity}}",
      is_minority_photo_attr = "is_minority_photo",
      minority_max_size = "{{prerank_control_minority_max_size}}",
      enable_reach_content_keep_candidates = "{{prerank_enable_reach_content_keep_candidates}}",
      protect_content_attr = "reach_content",
      protect_content_min_size = "{{prerank_control_reach_content_min_size}}",
      protect_content_coef = "{{explore_prerank_reach_content_boost_coef}}",
      save_is_degraded_common_attr = "prerank_hetu_quota_control_is_degraded",
      target_item = {
        flag_attr_name : 1
      }
    ) \
    .if_("prerank_enable_reach_content_keep_candidates == 1") \
      .item_attr_operation(
        item_attr_a = score_attr_name,
        common_attr_b = "{{explore_prerank_reach_content_boost_coef}}",
        operator = "*",
        output_attr = score_attr_name,
        select_item = {
          "join": "and",
          "filters": [{
              "attr_name": flag_attr_name,
              "select_if": "==",
              "compare_to": 1,
          }, {
              "attr_name": "reach_content",
              "select_if": "==",
              "compare_to": 1,
          }, {
              "attr_name": "prerank_diversity_select_flag",
              "select_if": "==",
              "compare_to": 1,
          }],
          "limit": "{{prerank_control_reach_content_min_size}}",
        }
      ) \
    .end_() \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "prerank_diversity_select_flag", "as": "flag"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr_name},
      ],
      function_name = "SetMinimumScoreByFlag",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr_name : 1
      }
    )
    return self

  def partial_time_based_tagnex_stat(self):
    self.enrich_attr_by_light_function(
      item_list_from_attr = "partial_time_based_selected_pids",
      import_common_attr = [
        {"name": "explore_tagnex_id_min", "as": "attr_min"},
        {"name": "explore_tagnex_id_max", "as": "attr_max"},
      ],
      import_item_attr = [
        {"name": "hetu_tag_level_info__hetu_tag", "as": "item_list_attr_name"},
      ],
      export_common_attr = [
        {"name": "key_list", "as": "partial_time_based_tagnex_keys"},
        {"name": "value_list", "as": "partial_time_based_tagnex_ratios"},
      ],
      function_name = "CalItemListAttrFrequency",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def partial_time_based_tagnex_boost(self, score_attr, flag_attr, stage="prerank"):
    adjust_param = "explore_partial_time_based_tagnex_boost_adjust_coef_" + stage
    boost_coef = "explore_partial_time_based_tagnex_boost_coef_" + stage
    current_keys_name = "partial_time_based_tagnex_keys_" + stage
    current_values_name = "partial_time_based_tagnex_ratios_" + stage
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_tagnex_id_min", "as": "attr_min"},
        {"name": "explore_tagnex_id_max", "as": "attr_max"},
      ],
      import_item_attr = [
        {"name": "hetu_tag_level_info__hetu_tag", "as": "item_list_attr_name"},
      ],
      export_common_attr = [
        {"name": "key_list", "as": current_keys_name},
        {"name": "value_list", "as": current_values_name},
      ],
      function_name = "CalItemListAttrFrequency",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": adjust_param, "as": "adjust_param"},
        {"name": "partial_time_based_tagnex_keys", "as": "history_key_list"},
        {"name": "partial_time_based_tagnex_ratios", "as": "history_value_list"},
        {"name": current_keys_name, "as": "candidate_key_list"},
        {"name": current_values_name, "as": "candidate_value_list"},
        {"name": "explore_tagnex_id_min", "as": "attr_min"},
        {"name": "explore_tagnex_id_max", "as": "attr_max"},
      ],
      import_item_attr = [
        {"name": "hetu_tag_level_info__hetu_tag", "as": "item_list_attr_name"},
      ],
      export_item_attr = [
        {"name": "final_coef", "as": boost_coef},
      ],
      function_name = "CalTagListRatioDiff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1
      }
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": boost_coef, "as": "boost_discount_coeff"},
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr},
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1
      }
    )
    return self

  def partial_time_based_interest_stat(self):
    self.enrich_attr_by_light_function(
      item_list_from_attr = "partial_time_based_selected_pids",
      import_item_attr = [
        {"name": "cluster_id_632", "as": "item_attr_name"},
      ],
      export_common_attr = [
        {"name": "key_list", "as": "partial_time_based_interest_keys"},
        {"name": "value_list", "as": "partial_time_based_interest_ratios"},
      ],
      function_name = "CalItemAttrFrequency",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def partial_time_based_interest_boost(self, score_attr, flag_attr, stage="prerank"):
    adjust_param = "explore_partial_time_based_interest_boost_adjust_coef_" + stage
    boost_coef = "explore_partial_time_based_interest_boost_coef_" + stage
    current_keys_name = "partial_time_based_interest_keys_" + stage
    current_values_name = "partial_time_based_interest_ratios_" + stage
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "cluster_id_632", "as": "item_attr_name"},
      ],
      export_common_attr = [
        {"name": "key_list", "as": current_keys_name},
        {"name": "value_list", "as": current_values_name},
      ],
      function_name = "CalItemAttrFrequency",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": adjust_param, "as": "adjust_param"},
        {"name": "partial_time_based_interest_keys", "as": "history_key_list"},
        {"name": "partial_time_based_interest_ratios", "as": "history_value_list"},
        {"name": current_keys_name, "as": "candidate_key_list"},
        {"name": current_values_name, "as": "candidate_value_list"},
      ],
      import_item_attr = [
        {"name": "cluster_id_632", "as": "item_attr_name"},
      ],
      export_item_attr = [
        {"name": "final_coef", "as": boost_coef},
      ],
      function_name = "CalTagRatioDiff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1
      }
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": boost_coef, "as": "boost_discount_coeff"},
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr},
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1
      }
    )
    return self

  def user_vv_type_int_value_adjust(self, int_value_name):
    adjust_coef = int_value_name + "_vv_type_adjust_coef"
    self \
      .if_("user_vv_flag == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": adjust_coef, "as": "weight"},
            {"name": int_value_name, "as": "value"}
          ],
          export_common_attr = [
            {"name": "new_value", "as": int_value_name}
          ],
          function_name = "CalExploreIntMultiDouble",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()
    return self

  def user_cocoon_int_value_adjust(self, int_value_name):
    adjust_coef = int_value_name + "_cocoon_adjust_coef"
    self \
      .if_("user_cocoon_flag == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": adjust_coef, "as": "weight"},
            {"name": int_value_name, "as": "value"}
          ],
          export_common_attr = [
            {"name": "new_value", "as": int_value_name}
          ],
          function_name = "CalExploreIntMultiDouble",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()
    return self

  def user_vv_type_weight_adjust(self, weight_name):
    adjust_coef = weight_name + "_vv_type_adjust_coef"
    self \
      .if_("user_vv_flag == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": adjust_coef, "as": "weight"},
            {"name": weight_name, "as": "value"}
          ],
          export_common_attr = [
            {"name": "new_value", "as": weight_name}
          ],
          function_name = "CalExploreDoubleMultiDouble",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()
    return self

  def user_cocoon_weight_adjust(self, weight_name):
    adjust_coef = weight_name + "_cocoon_adjust_coef"
    self \
      .if_("user_cocoon_flag == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": adjust_coef, "as": "weight"},
            {"name": weight_name, "as": "value"}
          ],
          export_common_attr = [
            {"name": "new_value", "as": weight_name}
          ],
          function_name = "CalExploreDoubleMultiDouble",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()
    return self

  def explore_cover_sense_view_score_trans(self):
    self \
      .set_attr_default_value(
        item_attrs=[{
          "name": "sense_view_predict_trans_score",
          "type": "double",
          "value": 0.0
        }, {
          "name": "cover_view_predict_trans_score",
          "type": "double",
          "value": 0.0       
        }],
      ) \
      .switch_("explore_cover_sense_view_score_version") \
        .case_(2) \
          .explore_cover_sense_view_score_unreview_trans(sense_view_predict_score = "sense_view_predict_score_v2",
                                                       cover_view_predict_score = "cover_view_predict_score_v2") \
        .default_() \
          .explore_cover_sense_view_score_unreview_trans(sense_view_predict_score = "sense_view_predict_score",
                                                       cover_view_predict_score = "cover_view_predict_score") \
        .end_()
    return self

  def explore_cover_sense_view_score_unreview_trans(self, sense_view_predict_score, cover_view_predict_score):
    self \
      .copy_attr(
        attrs=[{
          "from_item": sense_view_predict_score,
          "to_item": "sense_view_predict_trans_score"
        }],
        select_item = {
          "attr_name": "audit_b_second_tag",
          "compare_to": 0,
          "select_if": "<=",
          "select_if_attr_missing": True,
        }
      ) \
      .copy_attr(
        attrs=[{
          "from_item": cover_view_predict_score,
          "to_item": "cover_view_predict_trans_score"
        }],
        select_item = {
          "attr_name": "audit_hot_cover_level",
          "compare_to": 0,
          "select_if": "<=",
          "select_if_attr_missing": True,
        }
      )
    return self

  def explore_frist_screen_customization_use_interest_cluster_id_632(self):
    self \
      .copy_attr(
        attrs=[
          {
            "from_common": "enable_explore_use_interest_cluster_id_632_frist_screen",
            "to_common": "enable_explore_use_interest_cluster_id_632"
          },
          {
            "from_common": "fr_s2_valid_interest_user_boost_alpha_coeff_frist_screen",
            "to_common": "fr_s2_valid_interest_user_boost_alpha_coeff"
          }
        ]
      )
    return self

  def explore_frist_screen_customization_explore_cascading_comment_predict_model(self):
    self \
      .copy_attr(
        attrs=[
          {
            "from_common": "enable_explore_cascading_comment_predict_model_frist_screen",
            "to_common": "enable_explore_cascading_comment_predict_model"
          },
          {
            "from_common": "explore_cascading_comment_predict_kess_service_frist_screen",
            "to_common": "explore_cascading_comment_predict_kess_service"
          },
          {
            "from_common": "explore_mc_ensemble_s2_cascading_valid_play_score_power_weight_frist_screen",
            "to_common": "explore_mc_ensemble_s2_cascading_valid_play_score_power_weight"
          },
          {
            "from_common": "explore_fr_cascading_valid_play_score_ranking_weight_frist_screen",
            "to_common": "explore_fr_cascading_valid_play_score_ranking_weight"
          },
          {
            "from_common": "explore_rerank_gen_seed_ensemble_cascading_valid_play_score_weight_frist_screen",
            "to_common": "explore_rerank_gen_seed_ensemble_cascading_valid_play_score_weight"
          }
        ]
      )
    return self

  def explore_frist_screen_customization_interest_migration_photo_boost(self):
    self \
      .copy_attr(
        attrs=[
          {
            "from_common": "enable_interest_migration_photo_coef_calculator_frist_screen",
            "to_common": "enable_interest_migration_photo_coef_calculator"
          },
          {
            "from_common": "interest_migration_score_threshold_frist_screen",
            "to_common": "interest_migration_score_threshold"
          },
          {
            "from_common": "interest_migration_migration_threshold_frist_screen",
            "to_common": "interest_migration_migration_threshold"
          },
          {
            "from_common": "interest_migration_migration_coef_frist_screen",
            "to_common": "interest_migration_migration_coef"
          },
          {
            "from_common": "interest_migration_ignore_cluster_lv1_classes_str_frist_screen",
            "to_common": "interest_migration_ignore_cluster_lv1_classes_str"
          },
          {
            "from_common": "enable_mc_s2_interest_migration_photo_boost_frist_screen",
            "to_common": "enable_mc_s2_interest_migration_photo_boost"
          },
          {
            "from_common": "enable_fr_s2_interest_migration_photo_boost_frist_screen",
            "to_common": "enable_fr_s2_interest_migration_photo_boost"
          }
        ]
      )
    return self

  def interest_migration_coef_calculator(self, flag_attr):
    self.split_string(
      input_common_attr = "interest_migration_ignore_cluster_lv1_classes_str",
      output_common_attr = "interest_migration_ignore_cluster_lv1_classes",
      delimiters = ",", 
      skip_empty_tokens = True,
      trim_spaces = True,
      parse_to_int = True 
    ) \
    .explore_interest_migration_coef_calculator_enricher(
      explore_realshow_pids_attr = "explore_realshow_pids",
      gamora_play_pids_attr = "interest_migration_pids",
      gamora_play_scores_attr = "interest_migration_scores",
      cluster_id_attr = "hetu_sim_cluster_id",
      output_coef_attr = "interest_migration_photo_coef",
      gamora_score_threshold = "{{interest_migration_score_threshold}}",
      migration_threshold = "{{interest_migration_migration_threshold}}",
      migration_coef = "{{interest_migration_migration_coef}}",
      cluster_id_lv1_attr = "hetu_sim_cluster_id862_lv1",
      filter_by_cluster_lv1_classes_attr = "interest_migration_ignore_cluster_lv1_classes",
      targer_item = {
        flag_attr : 1
      }
    )
    return self

  def mc_s2_select_photo_by_interest(self, score_attr_name, flag_attr_name):
    self.if_("enable_mc_s2_cluster_id_control_dynamic == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "uOldMmuClusterId300ListList",
          {"name": "hot_cascade_control_cluster_id_max_size", "as": "old_window_size"},
          {"name": "explore_mc_s2_cid_dynamic_mode", "as": "cluster_id_window_size_dynamic_mode"},
          {"name": "explore_mc_s2_cid_upper_bound", "as": "cluster_id_window_size_upper_bound"},
          {"name": "explore_mc_s2_cid_discount_coef", "as": "cluster_id_window_size_discount_coef"},
          {"name": "explore_mc_s2_cid_lower_bound", "as": "cluster_id_window_size_lower_bound"},
        ],
        export_common_attr = [
          {"name": "new_window_size", "as": "hot_cascade_control_cluster_id_max_size"},
        ],
        function_name = "DynamicClusterIdWindowSize",
        class_name = "ExploreLightFunctionSetV2"
      ) \
    .end_() \
    .if_("explore_enable_user_need_break_cocoon_mc_s2 == 1 and user_need_break_cocoon_flag == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "hot_cascade_control_hetu2_max_size", "as": "value"},
          {"name": "user_need_break_cocoon_mc_s2_control_hetu2_coef", "as": "weight"}
        ],
        export_common_attr = [
          {"name": "new_value", "as": "hot_cascade_control_hetu2_max_size"}
        ],
        function_name = "CalExploreIntMultiDouble",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "hot_cascade_control_hetu5_max_size", "as": "value"},
          {"name": "user_need_break_cocoon_mc_s2_control_hetu5_coef", "as": "weight"}
        ],
        export_common_attr = [
          {"name": "new_value", "as": "hot_cascade_control_hetu5_max_size"}
        ],
        function_name = "CalExploreIntMultiDouble",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "hot_cascade_control_cluster_id_max_size", "as": "value"},
          {"name": "user_need_break_cocoon_mc_s2_control_cid_coef", "as": "weight"}
        ],
        export_common_attr = [
          {"name": "new_value", "as": "hot_cascade_control_cluster_id_max_size"}
        ],
        function_name = "CalExploreIntMultiDouble",
        class_name = "ExploreLightFunctionSetV2"
      ) \
    .end_() \
    .explore_control_hetu_count_enricher(
      user_hetu_stat_attr = "colossus_hetu_distribution_hetu_stat",
      hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
      hetu_level_two_attr = "hetu_tag_level_info__hetu_level_two",
      hetu_level_five_attr = "hetu_tag_level_info__hetu_level_five",
      cluster_id_attr = "mounted_interest_cluster_id",
      old_cluster_id_interest_list_attr = "uOldMmuClusterId300ListList",
      duration_ms_attr = "duration_ms",
      save_flag_to_attr = "mc_s2_diversity_select_flag",
      enable_hetu_control_interest = "{{hot_cascade_enable_hetu_control_interest}}",
      enable_hetu_control_diversity = "{{hot_cascade_enable_hetu_control_diversity}}",
      enable_cluster_id_control_diversity = "{{hot_cascade_enable_cluster_id_control_diversity}}",
      enable_duration_control_diversity = "{{hot_cascade_enable_duration_control_diversity}}",
      hetu_control_interest_start = "{{hot_cascade_hetu_control_interest_start}}",
      hetu_control_diversity_start = "{{hot_cascade_hetu_control_diversity_start}}",
      cluster_id_control_diversity_start = "{{hot_cascade_cluster_id_control_diversity_start}}",
      duration_control_diversity_start = "{{hot_cascade_duration_control_diversity_start}}",
      keep_size = "{{mc_final_candidate_num}}",
      hetu1_max_size = "{{hot_cascade_control_hetu1_max_size}}",
      hetu2_max_size = "{{hot_cascade_control_hetu2_max_size}}",
      hetu5_max_size = "{{hot_cascade_control_hetu5_max_size}}",
      cluster_id_max_size = "{{hot_cascade_control_cluster_id_max_size}}",
      duration_0s_max_size = "{{hot_cascade_control_duration_0s_max_size}}",
      duration_0_7s_max_size = "{{hot_cascade_control_duration_0_7s_max_size}}",
      duration_7_9s_max_size = "{{hot_cascade_control_duration_7_9s_max_size}}",
      duration_9_12s_max_size = "{{hot_cascade_control_duration_9_12s_max_size}}",
      duration_12_17s_max_size = "{{hot_cascade_control_duration_12_17s_max_size}}",
      duration_17_20s_max_size = "{{hot_cascade_control_duration_17_20s_max_size}}",
      old_cluster_id_interest_coef = "{{hot_cascade_control_cluster_id_interest_boost_coef}}",
      enable_minority_control_diversity = "{{hot_cascade_enable_minority_control_diversity}}",
      is_minority_photo_attr = "is_minority_photo",
      minority_max_size = "{{hot_cascade_control_minority_max_size}}",
      enable_protect_content_keep_candidates = "{{hot_cascade_enable_reach_content_keep_candidates}}",
      protect_content_attr = "reach_content",
      protect_content_min_size = "{{hot_cascade_control_reach_content_min_size}}",
      save_protect_content_flag_to_attr = "mc_s2_diversity_protect_content_flag",
      save_is_degraded_common_attr = "mc_s2_hetu_quota_control_is_degraded",
      target_item = {
        flag_attr_name : 1
      }
    ) \
    .if_("hot_cascade_enable_reach_content_keep_candidates == 1") \
      .item_attr_operation(
        item_attr_a = score_attr_name,
        common_attr_b = "{{explore_mc_s2_reach_content_boost_coef}}",
        operator = "*",
        output_attr = score_attr_name,
        select_item = {
          "join": "and",
          "filters": [{
              "attr_name": flag_attr_name,
              "select_if": "==",
              "compare_to": 1,
          }, {
              "attr_name": "reach_content",
              "select_if": "==",
              "compare_to": 1,
          }, {
              "attr_name": "mc_s2_diversity_select_flag",
              "select_if": "==",
              "compare_to": 1,
          }],
          "limit": "{{hot_cascade_control_reach_content_min_size}}",
        }
      ) \
    .end_() \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "mc_s2_diversity_select_flag", "as": "flag"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr_name},
      ],
      function_name = "SetMinimumScoreByFlag",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr_name : 1
      }
    )
    return self

  def mc_s2_select_photo_by_interest_directly_reach_fullrank(self, score_attr_name, flag_attr_name, left_count_attr_name):
    self.if_("hot_cascade_directly_reach_fullrank_enable_pic_filter == 1") \
      .set_attr_value(
        item_attrs=[
          {
            "name": score_attr_name,
            "type": "double",
            "value": 0.0
          }
        ],
        target_item = {
          flag_attr_name : 1,
          "is_picture" : 1
        }
      ) \
    .end_() \
    .sort(
       score_from_attr = score_attr_name,
       target_item = {
         flag_attr_name : 1
       }
    ) \
    .explore_control_hetu_count_enricher(
      hetu_level_five_attr = "hetu_tag_level_info__hetu_level_five",
      cluster_id_attr = "mounted_interest_cluster_id",
      old_cluster_id_interest_list_attr = "uOldMmuClusterId300ListList",
      save_flag_to_attr = "mc_s2_diversity_select_flag_directly_reach_fullrank",
      enable_hetu_control_diversity = "{{hot_cascade_directly_reach_fullrank_enable_hetu_control_diversity}}",
      enable_cluster_id_control_diversity = "{{hot_cascade_directly_reach_fullrank_enable_cluster_id_control_diversity}}",
      keep_size = "{{" + left_count_attr_name + "}}",
      hetu5_max_size = "{{hot_cascade_directly_reach_fullrank_control_hetu5_max_size}}",
      cluster_id_max_size = "{{hot_cascade_directly_reach_fullrank_control_cluster_id_max_size}}",
      old_cluster_id_interest_coef = "{{hot_cascade_directly_reach_fullrank_control_cluster_id_interest_boost_coef}}",
      save_is_degraded_common_attr = "mc_s2_directly_reach_fullrank_hetu_quota_control_is_degraded",
      target_item = {
        flag_attr_name : 1
      }
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "mc_s2_diversity_select_flag_directly_reach_fullrank", "as": "flag"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr_name},
      ],
      function_name = "SetMinimumScoreByFlag",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr_name : 1
      }
    )
    return self

  def boost_hot_content_retr(self, score_attr_name, flag_attr_name):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "mc_s2_hot_content_retr_boost_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": score_attr_name, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr_name},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr_name : 1,
        "reason" : [10030, 10031, 10032]
      }
    )
    return self

  def refinement_boost_personified_author(self, score_attr, flag_attr):
    """
    Module: photo_queue
    功能: 细分用户和视频维度，精细化对人格化账号提权
    Owner: xubaoquan
    Date: 2023-07-12
    :return:
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "basic_info_age_segment_v2", "as": "basic_info_age_segment_v2"},
        {"name": "basic_info_gender_v2", "as": "basic_info_gender_v2"},
        {"name": "explore_personifed_author_boost_ptr", "as": "boost_map_ptr"},
        {"name": "refinement_boost_personified_author_redis_prefix", "as": "redis_prefix"},
        {"name": "cascade_refinement_boost_personified_author_power_weight", "as": "power_weight"},
      ],
      import_item_attr = [
        {"name": "author__gender", "as": "author__gender"},
        {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_level_one_list"},
        {"name": score_attr, "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr},
      ],
      target_item = { 
        flag_attr: 1,
        "eyeshot_source" : 1
      },
      function_name = "UniverseRefinementBoost",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def hack_act_pic_es_decay(self, score_attr, flag_attr):
    """
    Module: photo_queue
    功能: 诱导互动图文降权
    Owner: zhuwenyong
    Date: 2024-02-27
    :return:
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_pic_hack_act_mc_decay_weight", "as": "decay_weight"},
        {"name": "explore_pic_hack_act_mc_decay_only_single_pic", "as": "only_single_pic"},
      ],
      import_item_attr = [
        "picture_type",
        "high_value_pic_flag",
        "audit_b_second_tag",
        "author__fans_count",
        {"name": score_attr, "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr},
      ],
      target_item = {
        flag_attr: 1
      },
      function_name = "PicHackActEsDecay",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def calc_pic_set_variety_score(self, flag_attr):
    """
    图文粗排多样性队列
    通过计算 item 与候选集中心 embedding 的距离，及该 item 的离散程度，作为多样性分
    Owner: caozhong
    Date: 2023-07-31
    :param flag_attr: 粗排分组 attr
    :return:
    """
    self.enrich_attr_by_light_function(
        import_common_attr=[
          {"name": "pic_cascade_variety_embedding_size", "as": "emb_size"},
          {"name": "pic_cascade_variety_score_alpha", "as": "alpha"},
          {"name": "pic_cascade_variety_score_beta", "as": "beta"},
          {"name": "pic_cascade_variety_score_ctr_alpha", "as": "ctr_alpha"},
          {"name": "pic_cascade_variety_score_ltr_alpha", "as": "ltr_alpha"},
          {"name": "pic_cascade_variety_score_wtr_alpha", "as": "wtr_alpha"},
          {"name": "pic_cascade_variety_score_cmtr_alpha", "as": "cmtr_alpha"},
          {"name": "pic_cascade_variety_score_reward_alpha", "as": "reward_alpha"},
          {"name": "pic_cascade_variety_score_ctr_beta", "as": "ctr_beta"},
          {"name": "pic_cascade_variety_score_reward_beta", "as": "reward_beta"},
        ],
        import_item_attr=[
          {"name": "pic_mmu_embedding", "as": "item_embedding"},
          {"name": "mc_ensemble_pctr", "as": "mc_pctr"},
          {"name": "mc_ensemble_pltr", "as": "mc_pltr"},
          {"name": "mc_ensemble_pwtr", "as": "mc_pwtr"},
          {"name": "mc_ensemble_pcmtr", "as": "mc_pcmtr"},
        ],
        export_item_attr=[
          "pic_variety_score",
        ],
        function_name="CalcPicVarietyScore",
        class_name="ExploreLightFunctionSetV2",
        target_item={
          flag_attr: 1,
        },
      )
    return self

  def boost_pic_cascade_s1_es_by_follow_author(self, score_attr, flag_attr):
    """
    关注作者 pic_s1_es boost
    Owner: chenqiaojun, caozhong
    Date: 2023-07-31
    :param score_attr:
    :param flag_attr:
    :return:
    """
    self.enrich_attr_by_light_function(
          import_common_attr=[
            {"name": "cascading_follow_author_pic_boost_coef", "as": "boost_discount_coeff"},
          ],
          import_item_attr=[
            {"name": "is_picture_follow_author", "as": "need_item_attr"},
            {"name": score_attr, "as": "ensemble_score"},
          ],
          export_item_attr=[
            {"name": "ensemble_score", "as": score_attr}
          ],
          function_name="BoostOrDiscount",
          class_name="ExploreLightFunctionSetV2",
          target_item={
            flag_attr: 1
          },
        )
    return self

  def boost_pic_cascade_s1_es_by_caption(self, score_attr, flag_attr):
    """
    长文本图文 pic_s1_es boost
    Owner: chenqiaojun, caozhong
    Date: 2023-07-31
    :param score_attr:
    :param flag_attr:
    :return:
    """
    self.enrich_attr_by_light_function(
          import_common_attr=[
            {"name": "cascade_channel_caption_boost_coef", "as": "caption_boost_coef"},
            {"name": "cascade_channel_caption_boost_len_thresh", "as": "caption_boost_len_thresh"},
            {"name": "cascade_channel_caption_boost_len_max", "as": "caption_boost_len_max"},
            {"name": "cascade_channel_boost_only_xhs_photo", "as": "boost_only_xhs_photo"},
            {"name": "cascade_channel_boost_only_picture", "as": "boost_only_picture"},
          ],
          import_item_attr=[
            {"name": score_attr, "as": "score"},
            "caption_length",
            "is_xhs_type_photo",
            "is_picture",
          ],
          export_item_attr=[
            {"name": "score", "as": score_attr},
          ],
          export_common_attr=[
            {"name": "boost_count", "as": "cascade_channel_caption_photo_boost_count"},
          ],
          function_name="BoostWithCaption",
          class_name="ExploreLightFunctionSetV2",
          target_item={
            flag_attr: 1
          }
        )
    return self

  def calc_pic_cascade_s1_real_pctr(self):
    self.enrich_attr_by_light_function(  # 计算粗排真实 ctr
      import_item_attr = [
        {"name": "cascade_pctr", "as": "pctr"},
        {"name": "cascade_psvtr", "as": "psvr"},
      ],
      export_item_attr = [
        {"name": "real_pctr", "as": "cascade_real_pctr"},
      ],
      function_name = "CalcRealPctr",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        "is_picture" : 1
      }
    ) \
    .if_("enable_explore_pic_mc_real_pctr_personalized_weight > 0") \
      .enrich_attr_by_light_function(  # 计算个性化权重
        import_common_attr = [
          {"name": "explore_pic_mc_real_pctr_weight_config_str", "as": "pxtr_attr_config_str"},
          {"name": "explore_pic_mc_real_pctr_weight_avg_top_num", "as": "avg_top_num"},
          {"name": "explore_pic_mc_real_pctr_weight_trans_params_str", "as": "trans_params_str"},
          {"name": "explore_pic_mc_real_pctr_weight_enable_trans", "as": "enable_trans"},
          {"name": "explore_pic_mc_real_pctr_weight_trans_alpha", "as": "trans_alpha"},
          {"name": "explore_pic_mc_real_pctr_weight_trans_bias", "as": "trans_bias"},
          {"name": "explore_pic_mc_real_pctr_weight_trans_pow", "as": "trans_pow"},
          {"name": "explore_pic_mc_real_pctr_weight_trans_min", "as": "trans_min"},
          {"name": "explore_pic_mc_real_pctr_weight_trans_max", "as": "trans_max"},
        ],
        export_common_attr = [
          {"name": "pxtr_topn_avg_score", "as": "explore_pic_mc_real_pctr_pow_weight"},
        ],
        import_item_attr = [
          "mc_ensemble_pctr",
          "cascade_pltr",
          "cascade_pwtr",
          "cascade_pcltr",
          "cascade_pcmtr",
          "cascade_pftr",
        ],
        function_name = "CalcPxtrStatScore",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          "is_picture": 1
        }
      ) \
    .end_()
    return self

  def mc_pic_boost_pctr_on_not_click_user(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        "pic_recent_realshow_not_click_cnt",
        {"name": "explore_pic_mc_boost_pctr_on_not_click_user_lower_bound", "as": "not_click_count_lower_bound"},
        {"name": "explore_pic_mc_boost_pctr_on_not_click_user_upper_bound", "as": "not_click_count_upper_bound"},
        {"name": "explore_pic_mc_boost_pctr_on_not_click_user_alpha", "as": "boost_pctr_alpha"},
        {"name": "explore_pic_mc_boost_pctr_on_not_click_user_beta", "as": "boost_pctr_beta"},
        {"name": "explore_pic_mc_boost_pctr_on_not_click_user_power_weight", "as": "boost_pctr_power_weight"},
      ],
      export_common_attr = [
        {"name": "pctr_power_weight_boost_coeff", "as": "mc_pic_not_click_pctr_power_weight_boost_coeff"},
      ],
      function_name = "CalPicNotClickBoostCoeff",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .gen_common_attr_by_lua(
      attr_map = {
        "explore_mc_ensemble_pic_pctr_power_weight": "explore_mc_ensemble_pic_pctr_power_weight * mc_pic_not_click_pctr_power_weight_boost_coeff"
      }
    )
    return self

  def boost_pic_cascade_s1_es_by_target_hetu(self, score_attr, flag_attr):
    """
    特定河图类目 pic_s1_es boost
    Owner: chenqiaojun, caozhong
    Date: 2023-07-31
    :param score_attr:
    :param flag_attr:
    :return:
    """
    self.enrich_attr_by_light_function(
          import_common_attr=[
            {"name": "cascade_target_hetu_pic_mc_s1_boost_coeff", "as": "boost_discount_coeff"},
          ],
          import_item_attr=[
            {"name": "is_boost_hetu_pic", "as": "need_item_attr"},
            {"name": score_attr, "as": "ensemble_score"},
          ],
          export_item_attr=[
            {"name": "ensemble_score", "as": score_attr},
          ],
          function_name="BoostOrDiscount",
          class_name="ExploreLightFunctionSetV2",
          target_item={
            flag_attr: 1,
          }
        )
    return self

  def boost_pic_cascade_s1_es_by_hetu_ratio(self, score_attr, flag_attr):
    """
    根据河图占比调整 pic_s1_es
    Owner: chenqiaojun, caozhong
    Date: 2023-07-31
    :param score_attr:
    :param flag_attr:
    :return:
    """
    self.sort(
          score_from_attr=score_attr,
          target_item={
            flag_attr: 1
          }
        ) \
        .enrich_attr_by_light_function(
          import_common_attr=[
            {"name": "cascade_s1_hetu_decay_coeff", "as": "decay_coeff"},
            {"name": "cascade_s1_hetu_decay_keep_size_coeff", "as": "decay_keep_size_coeff"},
          ],
          import_item_attr=[
            {"name": score_attr, "as": "score"},
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_level_one_list"},
            {"name": "is_key_target_hetu_pic", "as": "is_target_hetu"},
          ],
          export_item_attr=[
            {"name": "score", "as": score_attr},
          ],
          function_name="HetuRatioDecay",
          class_name="ExploreLightFunctionSetV2",
          target_item={
            flag_attr: 1
          }
        )
    return self

  def boost_pic_cascade_s1_es_by_revisited(self, score_attr, flag_attr):
    """
    复访作品调整 pic_s1_es
    Owner: chenqiaojun, caozhong
    Date: 2023-07-31
    :param score_attr:
    :param flag_attr:
    :return:
    """
    self.enrich_attr_by_light_function(
          import_common_attr=[
            {"name": "cascade_revisited_item_boost_coef", "as": "boost_weight"}
          ],
          import_item_attr=[
            {"name": score_attr, "as": "ensemble_score"},
          ],
          export_item_attr=[
            {"name": "ensemble_score", "as": score_attr}
          ],
          function_name="EnsembleScoreBoost",
          class_name="ExploreLightFunctionSetV2",
          target_item={
            flag_attr: 1,
            "reason": 13071,
          },
        )
    return self

  def boost_pic_cascade_s1_es_by_hetu_distribution(self, score_attr, flag_attr):
    """
    根据候选集hetu分布 和 用户历史河图分布 调整 pic_s1_es
    Owner: gaodong, gengxiao
    Date: 2023-07-31
    :param score_attr:
    :param flag_attr:
    :return:
    """
    self.sort(
          score_from_attr=score_attr,
        ) \
        .explore_photo_distribution_adjust_enricher(
          colossus_total_count_attr="colossus_hetu_distribution_total_count",
          user_hetu_stat_attr="colossus_hetu_distribution_hetu_stat",
          colossus_total_count_threshold="{{cascading_s1_pic_hetu_distribution_colossus_total_count_threshold}}",
          max_count="{{cascading_s1_pic_hetu_distribution_max_count}}",
          global_fuse_corr="{{cascading_s1_pic_hetu_distribution_global_fuse_corr}}",
          hetu_level_one_attr="hetu_tag_level_info__hetu_level_one",
          candidate_hetu_adjust_coeff_map_attr="candidate_hetu_adjust_coeff_map"
        ) \
        .enrich_attr_by_light_function(
          import_common_attr=[
            {"name": "cascading_s1_pic_hetu_distribution_hetu_coef_alpha", "as": "hetu_coef_alpha"},
            {"name": "cascading_s1_pic_hetu_distribution_hetu_coef_beta", "as": "hetu_coef_beta"},
            {"name": "cascading_s1_pic_hetu_distribution_hetu_discount_threshold", "as": "hetu_discount_threshold"},
            {"name": "cascading_s1_pic_hetu_distribution_hetu_encourage_threshold", "as": "hetu_encourage_threshold"},
            "candidate_hetu_adjust_coeff_map",
          ],
          import_item_attr=[
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_level_one_list"},
            {"name": score_attr, "as": "es_score"},
          ],
          export_item_attr=[
            {"name": "es_score", "as": score_attr},
          ],
          function_name="AdjustScoreByHetuDistribution",
          class_name="ExploreLightFunctionSetV2",
        )
    return self

  def not_cover_audit_photo_discount(self, score_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_mc_not_cover_audit_enable_follow_author_exemption", "as": "enable_follow_author_exemption"},
        "follow_aids",
        {"name": "page_index", "as": "page"},
      ],
      import_item_attr = [
        "author__id",
        "audit_hot_cover_level",
      ],
      export_item_attr = [
        {"name": "is_not_cover_audit_for_first_page", "as": "is_not_cover_audit_for_first_page_mc"}
      ],
      function_name = "IsNotCoverAuditForFirstPage",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_mc_not_cover_audit_photo_discount_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        "is_not_cover_audit_for_first_page_mc" : 1
      }
    )
    return self

  def user_age_based_weight_adjust_all(self):
    self \
    .user_attr_based_weight_adjust(
      "user_age_segment",
      "explore_mc_s2_age_based_pctr_weight_adjust_list",
      "explore_mc_ensemble_s2_pctr_power_weight"
    ) \
    .user_attr_based_weight_adjust(
      "user_age_segment",
      "explore_mc_s2_age_based_pltr_weight_adjust_list",
      "explore_mc_ensemble_s2_pltr_power_weight"
    ) \
    .user_attr_based_weight_adjust(
      "user_age_segment",
      "explore_mc_s2_age_based_pwtr_weight_adjust_list",
      "explore_mc_ensemble_s2_pwtr_power_weight"
    ) \
    .user_attr_based_weight_adjust(
      "user_age_segment",
      "explore_mc_s2_age_based_pftr_weight_adjust_list",
      "explore_mc_ensemble_s2_pftr_power_weight"
    ) \
    .user_attr_based_weight_adjust(
      "user_age_segment",
      "explore_mc_s2_age_based_pcmtr_weight_adjust_list",
      "explore_mc_ensemble_s2_pcmtr_power_weight"
    ) \
    .user_attr_based_weight_adjust(
      "user_age_segment",
      "explore_mc_s2_age_based_awesome_wtd_weight_adjust_list",
      "explore_mc_ensemble_s2_pwtd_inverse_power_weight"
    )
    return self

  def user_attr_based_weight_adjust(self, user_attr, weight_list_str, weight_attr):
    weight_list = weight_list_str + "_to_list"
    weight_adjust_coef = weight_attr + "_adjust_coef_by_" + user_attr

    self.split_string(
      input_common_attr = weight_list_str,
      output_common_attr = weight_list,
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_double = True,
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": user_attr, "as": "index"},
        {"name": weight_list, "as": "weight_list"},
      ],
      export_common_attr = [
        {"name": "weight", "as": weight_adjust_coef},
      ],
      function_name = "GetDoubleValueInList",
      class_name = "ExploreLightFunctionSetV2",\
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": weight_attr, "as": "value"},
        {"name": weight_adjust_coef, "as": "weight"},
      ],
      export_common_attr = [
        {"name": "new_value", "as": weight_attr},
      ],
      function_name = "CalExploreDoubleMultiDouble",
      class_name = "ExploreLightFunctionSetV2",\
    )
    return self

  def get_user_group_emp_xtr(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        "basic_info_age_segment_v2",
        "basic_info_gender_v2",
      ],
      export_common_attr = [
        "emp_xtr_user_group_prefix_ltr",
        "emp_xtr_user_group_prefix_wtr",
        "emp_xtr_user_group_prefix_ftr",
        "emp_xtr_user_group_prefix_cmtr",
        "emp_xtr_user_group_prefix_svtr",
      ],
      function_name = "CalUserGroupBucket",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .get_kconf_params(
      kconf_configs = [
        {
          "kconf_key": "reco.author.exploreUserGroupAgeGenderEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_ltr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_ltr"
        },
        {
          "kconf_key": "reco.author.exploreUserGroupAgeGenderEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_wtr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_wtr"
        },
        {
          "kconf_key": "reco.author.exploreUserGroupAgeGenderEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_ftr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_ftr"
        },
        {
          "kconf_key": "reco.author.exploreUserGroupAgeGenderEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_cmtr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_cmtr"
        },
        {
          "kconf_key": "reco.author.exploreUserGroupAgeGenderEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_svtr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_svtr"
        },
        {
          "kconf_key": "reco.author.exploreUserGroupAgeGenderRankEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_svtr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_rank_svtr"
        }
      ]
    ) \

    return self

  def get_user_new_group_emp_xtr(self):
    self \
    .set_attr_value(
      no_overwrite = True,
      common_attrs = [
        {
          "name": "explore_gender_data_type",
          "type": "int",
          "value": 0,
        }
      ]
    ) \
    .enrich_attr_by_light_function(         
      import_common_attr = [
        "basic_info_age_segment_v2",
        "user_gender",
        {"name": "explore_gender_data_type", "as": "is_gender_data_type_list"},
      ],
      export_common_attr = [
        "emp_xtr_user_group_prefix_ltr",
        "emp_xtr_user_group_prefix_wtr",
        "emp_xtr_user_group_prefix_ftr",
        "emp_xtr_user_group_prefix_cmtr",
        "emp_xtr_user_group_prefix_svtr",
      ],
      function_name = "CalNewUserGroupBucket",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .get_kconf_params(
      kconf_configs = [
        {
          "kconf_key": "reco.author.userExploreGroupAgeGenderEmpXtr_new",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_ltr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_ltr"
        },
        {
          "kconf_key": "reco.author.userExploreGroupAgeGenderEmpXtr_new",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_wtr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_wtr"
        },
        {
          "kconf_key": "reco.author.userExploreGroupAgeGenderEmpXtr_new",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_ftr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_ftr"
        },
        {
          "kconf_key": "reco.author.userExploreGroupAgeGenderEmpXtr_new",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_cmtr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_cmtr"
        },
        {
          "kconf_key": "reco.author.userExploreGroupAgeGenderEmpXtr_new",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_svtr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_svtr"
        },
        {
          "kconf_key": "reco.author.userExploreGroupAgeGenderEmpXtr_new",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_svtr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_rank_svtr"
        }
      ]
    )
    return self

  def get_user_gemini_refresh_scene(self):
    self \
    .enrich_attr_by_json(
      json_attr = "recoReportContext",
      json_configs = [
        {
          "json_path": "geminiRefreshScene",
          "export_common_attr": "gemini_refresh_scene",
          "default_value": 0
        }
      ]
    ) \

    return self

  def get_user_mau_emp_xtr(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        "find_visit_days_30d",
      ],
      export_common_attr = [
        "emp_xtr_user_mau_prefix_ltr",
        "emp_xtr_user_mau_prefix_wtr",
        "emp_xtr_user_mau_prefix_ftr",
        "emp_xtr_user_mau_prefix_cmtr",
        "emp_xtr_user_mau_prefix_evtr",
        "emp_xtr_user_mau_prefix_play",
      ],
      function_name = "CalUserMauBucket",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .get_kconf_params(
      kconf_configs = [
        {
          "kconf_key": "reco.author.FountainUserMauRerankEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_mau_prefix_ltr}}",
          "default_value": 1.0,
          "export_common_attr": "user_mau_emp_ltr"
        },
        {
          "kconf_key": "reco.author.FountainUserMauRerankEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_mau_prefix_wtr}}",
          "default_value": 1.0,
          "export_common_attr": "user_mau_emp_wtr"
        },
        {
          "kconf_key": "reco.author.FountainUserMauRerankEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_mau_prefix_ftr}}",
          "default_value": 1.0,
          "export_common_attr": "user_mau_emp_ftr"
        },
        {
          "kconf_key": "reco.author.FountainUserMauRerankEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_mau_prefix_cmtr}}",
          "default_value": 1.0,
          "export_common_attr": "user_mau_emp_cmtr"
        },
        {
          "kconf_key": "reco.author.FountainUserMauRerankEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_mau_prefix_evtr}}",
          "default_value": 1.0,
          "export_common_attr": "user_mau_emp_evtr"
        },
        {
          "kconf_key": "reco.author.FountainUserMauRerankEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_mau_prefix_play}}",
          "default_value": 1.0,
          "export_common_attr": "user_mau_emp_rank_play"
        }
      ]
    ) \

    return self

  def _diversify_interactive_power_weight_in_mc_s2(self):
    self \
    .enrich_attr_by_light_function( # 粗排二轮 个性化调整互动权重
      import_common_attr = [
        "user_emp_ltr",
        "user_emp_wtr",
        "user_emp_ftr",
        "user_emp_cmtr",
        "user_emp_eptr",
        "user_group_emp_ltr",
        "user_group_emp_wtr",
        "user_group_emp_ftr",
        "user_group_emp_cmtr",
        {"name": "explore_mc_s2_power_weight_cascade_like_emp", "as": "all_user_emp_ltr"},
        {"name": "explore_mc_s2_power_weight_cascade_follow_emp", "as": "all_user_emp_wtr"},
        {"name": "explore_mc_s2_power_weight_cascade_forward_emp", "as": "all_user_emp_ftr"},
        {"name": "explore_mc_s2_power_weight_cascade_comment_emp", "as": "all_user_emp_cmtr"},
        {"name": "explore_mc_ensemble_s2_pltr_power_weight", "as": "user_ori_ltr_weight"},
        {"name": "explore_mc_ensemble_s2_pwtr_power_weight", "as": "user_ori_wtr_weight"},
        {"name": "explore_mc_ensemble_s2_pftr_power_weight", "as": "user_ori_ftr_weight"},
        {"name": "explore_mc_ensemble_s2_pcmtr_power_weight", "as": "user_ori_cmtr_weight"},
        {"name": "explore_mc_s2_power_weight_adjust_ratio_min", "as": "explore_weight_adjust_coeff_min"},
        {"name": "explore_mc_s2_power_weight_adjust_ratio_max", "as": "explore_weight_adjust_coeff_max"},
      ],
      export_common_attr = [
        {"name": "user_ltr_weight", "as": "explore_mc_ensemble_s2_pltr_power_weight"},
        {"name": "user_wtr_weight", "as": "explore_mc_ensemble_s2_pwtr_power_weight"},
        {"name": "user_ftr_weight", "as": "explore_mc_ensemble_s2_pftr_power_weight"},
        {"name": "user_cmtr_weight", "as": "explore_mc_ensemble_s2_pcmtr_power_weight"},
      ],
      function_name = "UserSortWeightAdjust",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def user_vv_weight_adjust_mc_s2(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_mc_s2_global_emp_phtr_in_order_power_weight", "as": "xtr_weight"},
        {"name": "active_days_avg_vv", "as": "user_vv"},
        {"name": "explore_mc_ensemble_s2_hate_like_weight_adjust_exp_upper", "as": "exp_upper"},
        {"name": "explore_mc_ensemble_s2_hate_like_weight_adjust_alpha", "as": "alpha"},
        {"name": "explore_mc_ensemble_s2_hate_like_weight_adjust_beta", "as": "beta"},
        {"name": "explore_mc_ensemble_s2_hate_like_weight_adjust_omega", "as": "omega"},
        {"name": "explore_mc_ensemble_s2_hate_like_weight_adjust_max", "as": "coeff_max"},
        {"name": "explore_mc_ensemble_s2_hate_like_weight_adjust_min", "as": "coeff_min"},
      ],
      export_common_attr = [
        {"name": "xtr_weight", "as": "explore_mc_s2_global_emp_phtr_in_order_power_weight"},
      ],
      function_name = "AdjustWeightByUserVv",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def boost_impression_audit(self, score_attr, flag_attr):
    self.split_string(
      input_common_attr = "explore_audit_b_second_tag_blacklist",
      output_common_attr = "audit_b_second_tag_blacklist_list",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": score_attr, "as": "ensemble_score"},
        {"name": "content_safety_level_with_namespace__level_hot_online", "as": "level_hot_online_attr"},
        "audit_b_second_tag"
      ],
      import_common_attr = [
        "enable_open_each_page",
        "enable_open_first_page",
        "good_impression_weight",
        "normal_impression_weight",
        "normal_bad_impression_weight",
        "page_index",
        "audit_b_second_tag_blacklist_list"
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr}
      ],
      target_item = { flag_attr: 1 },
      function_name = "BoostImpressionAudit",
      class_name = "ExploreLightFunctionSetV2"
    )
    return self

  def mc_high_global_emphtr_discount(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_mc_high_global_emphtr_discount_coeff", "as": "boost_discount_coeff"},
        {"name": "explore_mc_high_global_emphtr_discount_thres", "as": "boost_discount_thres"},
      ],
      import_item_attr = [
        {"name": "global_emphtr_score", "as": "need_item_attr"},
        {"name": score_attr, "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr}
      ],
      target_item = { flag_attr: 1 },
      function_name = "BoostOrDiscountWithThres",
      class_name = "ExploreLightFunctionSetV2",
    ) \

    return self

  def prerank_search_score_boost(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_prerank_search_score_boost_coeff", "as": "boost_discount_coeff"},
        {"name": "explore_prerank_search_score_boost_thres", "as": "boost_discount_thres"},
      ],
      import_item_attr = [
        {"name": "search_score", "as": "need_item_attr"},
        {"name": score_attr, "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr}
      ],
      target_item = { flag_attr: 1 },
      function_name = "BoostOrDiscountWithThres",
      class_name = "ExploreLightFunctionSetV2",
    ) \

    return self
  
  def unbias_interest_photo_boost(self, score_attr, stage_name="prerank"):
    coeff_param_name = "explore_" + stage_name + "_unbias_interest_photo_boost_coeff"
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": coeff_param_name, "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "is_in_selected_cids", "as": "need_item_attr"},
        {"name": score_attr, "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr}
      ],
      function_name = "BoostOrDiscount",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    
    return self

  def interest_score_history_coef_calculator(self):
    self.split_string(
      input_common_attr = "explore_interest_score_select_channel_str",
      output_common_attr = "explore_interest_score_select_channel",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True,
    ) \
    .split_string(
      input_common_attr = "explore_interest_score_select_channel_weight_str",
      output_common_attr = "explore_interest_score_select_channel_weight",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_double = True,
    ) \
    .explore_history_interest_score(
      pid_list_attr = "interest_score_based_pids",
      real_show_pid_attr = "standard_explore_realshow_pid_list",
      channel_list_attr = "interest_score_based_channels",
      score_list_attr = "interest_score_based_scores",
      select_channel_list_attr = "explore_interest_score_select_channel",
      select_channel_weight_attr = "explore_interest_score_select_channel_weight",
      default_channel_weight = "{{explore_interest_score_channel_default_weight}}",
      tagnex_v3_alpha = "{{explore_interest_score_tagnex_v3_alpha}}",
      real_show_hetu_tag_n_limit = "{{explore_interest_score_real_show_hetu_tag_n_limit}}",
      cluster_632_alpha = "{{explore_interest_score_cluster_632_alpha}}",
      real_show_cluster_632_n_limit = "{{explore_interest_score_real_show_cluster_632_n_limit}}",
      hetu_tag_attr = "hetu_tag_level_info__hetu_tag",
      cluster_id_632_attr = "cluster_id_632",
      output_score_attr = "photo_history_interest_score",
    )
    return self

  def interest_score_history_boost(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_history_interest_score_boost_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      select_item = {
        "join": "and",
        "filters": [{
          "attr_name": flag_attr,
          "select_if": "==",
          "compare_to": 1,
        }, {
          "attr_name": "photo_history_interest_score",
          "select_if": ">",
          "compare_to": 1.0,
        }]
      }
    )
    return self

  def hot_list_coef_calculator(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name" : "explore_stat__real_show_count", "as" : "denominator"},
        {"name" : "explore_stat__click_count", "as" : "numerator"}
      ],
      export_item_attr = [
        {"name" : "fraction", "as" : "empirical_ctr_for_hot_list"}
      ],
      function_name = "CalExploreIntDevideIntItemAttr",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        "is_hot_list_flag": 1,
      }
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        "upload_time"
      ],
      export_item_attr = [
        "photo_age_hour"
      ],
      function_name = "CalcAgeHour",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        "is_hot_list_flag": 1,
      }
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name" : "active_days_avg_vv", "as" : "user_avg_vv"},
        {"name" : "hot_list_coef_user_avg_vv_threshold", "as" : "user_avg_vv_threshold"},
        {"name" : "hot_list_coef_user_avg_vv_slope", "as" : "user_avg_vv_slope"},
        {"name" : "hot_list_coef_user_avg_vv_scale", "as" : "user_avg_vv_scale"},
        {"name" : "hot_list_coef_average_ctr", "as" : "average_ctr"},
        {"name" : "hot_list_coef_ctr_adjust_slope", "as" : "ctr_adjust_slope"},
        {"name" : "hot_list_coef_ctr_adjust_scale", "as" : "ctr_adjust_scale"},
        {"name" : "hot_list_coef_photo_age_hour_max", "as" : "photo_age_hour_max"},
        {"name" : "hot_list_coef_photo_age_hour_decay_rate", "as" : "photo_age_hour_decay_rate"}
      ],
      import_item_attr = [
        "empirical_ctr_for_hot_list",
        "photo_age_hour",
      ],
      export_item_attr = [
        "hot_list_adjust_coeff"
      ],
      function_name = "CalHotListAdjustCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        "is_hot_list_flag": 1,
      }
    )
    return self
  
  
  def hot_list_photo_boost(self, score_attr, flag_attr, stage_name="prerank"):
    coeff_param_name = "explore_" + stage_name + "_hot_list_photo_boost_coeff"
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": coeff_param_name, "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1,
        "is_hot_list_flag" : 1
      }
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "hot_list_adjust_coeff", "as": "boost_discount_coeff"},
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1,
        "is_hot_list_flag" : 1
      }
    )
    
    return self
  
  def mc_cal_interest_cid_coeff(self, interest_and_score_list_name="user_develop_interest_cid_and_score_list"):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        "uOldMmuClusterId300ListList",
        {"name": interest_and_score_list_name, "as": "interest_and_score_list"},
        {"name": "mc_s2_interest_cluster_id_num_threshold", "as": "interest_cluster_id_num_threshold"},
        {"name": "mc_s2_identified_interest_cluster_id_num_threshold", "as": "identified_interest_cluster_id_num_threshold"},
        {"name": "mc_s2_unidentified_interest_cluster_id_num_threshold", "as": "unidentified_interest_cluster_id_num_threshold"},
        {"name": "mc_s2_interest_score_cids_ori_boost_coeff", "as": "interest_score_cids_ori_boost_coeff"},
        {"name": "mc_s2_identified_interest_boost_alpha_coeff", "as": "identified_interest_boost_alpha_coeff"},
        {"name": "mc_s2_identified_interest_boost_beta_coeff", "as": "identified_interest_boost_beta_coeff"},
        {"name": "mc_s2_identified_interest_boost_omega_coeff", "as": "identified_interest_boost_omega_coeff"},
        {"name": "mc_s2_unidentified_interest_boost_alpha_coeff", "as": "unidentified_interest_boost_alpha_coeff"},
        {"name": "mc_s2_unidentified_interest_boost_beta_coeff", "as": "unidentified_interest_boost_beta_coeff"},
        {"name": "mc_s2_unidentified_interest_boost_omega_coeff", "as": "unidentified_interest_boost_omega_coeff"},
        {"name": "mc_s2_develop_interest_score_lower_bound", "as": "develop_interest_score_lower_bound"},
        {"name": "mc_s2_develop_interest_score_upper_bound", "as": "develop_interest_score_upper_bound"},
        {"name": "mc_s2_identified_interest_score_lower_bound", "as": "identified_interest_score_lower_bound"},
        {"name": "mc_s2_unidentified_interest_score_lower_bound", "as": "unidentified_interest_score_lower_bound"},
        {"name": "enable_mc_s2_interest_score_boost", "as": "enable_interest_score_cids_boost"},
        {"name": "enable_mc_s2_identified_interest_score_boost", "as": "enable_identified_interest_score_cids_boost"},
        {"name": "enable_mc_s2_unidentified_interest_score_boost", "as": "enable_unidentified_interest_score_cids_boost"},
      ],
      import_item_attr = [
        {"name": "interest_cluster_id", "as": "hetu_sim_cluster_id862"},
      ],
      export_item_attr = [
        {"name": "interest_cids_coeff", "as": "cascade_interest_cids_coeff"},
      ],
      function_name = "CalInterestCidsCoeff",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    
    return self

  def mc_interest_score_cids_boost(self, score_attr):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "cascade_interest_cids_coeff", "as": "boost_discount_coeff"},
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
    ) \

    return self
    
  def mc_cal_valid_interest_cid_coeff(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        "uOldMmuClusterId300ListList",
        "user_valid_interest_cid_and_score_list",
        {"name": "mc_s2_valid_interest_cluster_id_num_threshold", "as": "valid_interest_cluster_id_num_threshold"},
        {"name": "mc_s2_valid_interest_user_boost_alpha_coeff", "as": "valid_interest_user_boost_alpha_coeff"},
        {"name": "mc_s2_valid_interest_user_boost_beta_coeff", "as": "valid_interest_user_boost_beta_coeff"},
        {"name": "mc_s2_valid_interest_user_boost_omega_coeff", "as": "valid_interest_user_boost_omega_coeff"},
        {"name": "mc_s2_develop_valid_interest_score_lower_bound", "as": "develop_valid_interest_score_lower_bound"},
        {"name": "enable_mc_s2_valid_interest_score_boost", "as": "enable_valid_interest_score_cids_boost"},
      ],
      import_item_attr = [
        {"name": "interest_cluster_id", "as": "hetu_sim_cluster_id862"},
      ],
      export_item_attr = [
        {"name": "valid_interest_cids_coeff", "as": "cascade_valid_interest_cids_coeff"},
      ],
      function_name = "CalValidInterestCidsCoeff",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    
    return self

  def mc_valid_interest_score_cids_boost(self, score_attr):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "cascade_valid_interest_cids_coeff", "as": "boost_discount_coeff"},
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
    ) \

    return self

  def mc_cal_short_valid_interest_first_refresh_coeff(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        "uOldMmuClusterId300ListList",
        {"name": "uExploreShortValidInterestAndScoreList", "as": "user_valid_interest_cid_and_score_list"},
        {"name": "mc_s2_short_valid_interest_first_refresh_cluster_id_num_threshold", "as": "valid_interest_cluster_id_num_threshold"},
        {"name": "mc_s2_short_valid_interest_first_refresh_user_boost_alpha_coeff", "as": "valid_interest_user_boost_alpha_coeff"},
        {"name": "mc_s2_short_valid_interest_first_refresh_user_boost_beta_coeff", "as": "valid_interest_user_boost_beta_coeff"},
        {"name": "mc_s2_short_valid_interest_first_refresh_user_boost_omega_coeff", "as": "valid_interest_user_boost_omega_coeff"},
        {"name": "mc_s2_short_valid_interest_first_refresh_develop_score_lower_bound", "as": "develop_valid_interest_score_lower_bound"},
        {"name": "enable_mc_s2_short_valid_interest_first_refresh_score_boost", "as": "enable_valid_interest_score_cids_boost"},
      ],
      import_item_attr = [
        {"name": "interest_cluster_id", "as": "hetu_sim_cluster_id862"},
      ],
      export_item_attr = [
        {"name": "valid_interest_cids_coeff", "as": "cascade_short_valid_interest_first_refresh_coeff"},
      ],
      function_name = "CalValidInterestCidsCoeff",
      class_name = "ExploreLightFunctionSetV2",
    ) \

    return self

  def mc_short_valid_interest_first_refresh_boost(self, score_attr):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "cascade_short_valid_interest_first_refresh_coeff", "as": "boost_discount_coeff"},
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
    ) \

    return self

  def interest_generalization_boost(self, score_attr, flag_attr, stage_name):
    boost_coef_name = "explore_" + stage_name + "_interest_generalization_boost_coef"
    self.enrich_attr_by_light_function(
      import_common_attr = [ 
        {"name": "explore_interest_interest_generalization_prefix", "as": "key_prefix"},
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
        "kconf_key": "reco.offline.tagnexLv2GeneralizaionMap",
        "json_path": "{{user_age_gender_key}}",
        "default_value": "", 
        "export_common_attr": "explore_interest_generalization_map_str"
      },
      {
        "kconf_key": "reco.offline.tagnexMapLv3ToLv2MapStr",
        "json_path": "lv3ToLv2",
        "default_value": "", 
        "export_common_attr": "explore_tagnex_lv3_to_lv2_map_str"
      }]  
    ) \
    .pack_item_attr(
      item_source = {
        "reco_results": False,
        "common_attr": ["explore_recent_play_list"],
      },
      mappings = [{
        "from_item_attr": "hetu_tag_level_info__hetu_tag",
        "to_common_attr": "explore_recent_view_hetu_tag",
        "dedup_to_common_attr": True,
      }]
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        "explore_recent_view_hetu_tag",
        "explore_interest_generalization_map_str",
        "explore_tagnex_lv3_to_lv2_map_str",
      ],
      import_item_attr = [
        {"name": "hetu_tag_level_info__hetu_tag", "as": "item_tag_list"}
      ],
      export_item_attr = [
        "is_generalization_photo",
      ],
      function_name = "CalInterestGeneralizationFlag",
      class_name = "ExploreLightFunctionSetV2",
      target_item={ flag_attr: 1 }
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": boost_coef_name, "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "is_generalization_photo", "as": "need_item_attr"},
        {"name": score_attr, "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr}
      ],
      function_name = "BoostOrDiscount",
      class_name = "ExploreLightFunctionSetV2",
      target_item={ flag_attr: 1 }
    )
    return self

  def unbias_interest_cluster_is_in_set(self):
    # cascade_cluster_id在130000～140000内为无偏兴趣分桶
    self.set_attr_value(
      item_attrs = [
        {
          "name": "is_in_cluster_unbias_cids",
          "type": "int",
          "value": 1
        }
      ],
      select_item = { 
        "join": "and",
        "filters": [{
            "attr_name": "cascade_cluster_id",
            "select_if": ">=",
            "compare_to": 130000,
        }, {
            "attr_name": "cascade_cluster_id",
            "select_if": "<",
            "compare_to": 140000,
        }]
      }
    ) \

    return self

  def unbias_interest_cluster_boost(self, score_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "unbias_interest_in_mc_s2_cids_coeff", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "is_in_cluster_unbias_cids", "as": "need_item_attr"},
        {"name": score_attr, "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr}
      ],
      function_name = "BoostOrDiscount",
      class_name = "ExploreLightFunctionSetV2",
    ) \

    return self

  def short_uninterest_photo_discount(self, score_attr, stage_name="prerank"):
    coeff_param_name = "explore_" + stage_name + "_short_uninterest_photo_discount_coeff"
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": coeff_param_name, "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "is_short_uninterested_photo", "as": "need_item_attr"},
        {"name": score_attr, "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr}
      ],
      function_name = "BoostOrDiscount",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    
    return self

  def interest_migration_photo_boost(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "interest_migration_photo_coef", "as": "boost_discount_coeff"},
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1,
      }
    )
    return self

  def mc_calc_search_score(self, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_mc_search_score_threshold", "as": "ann_dist_threshold"}, 
      ],
      import_item_attr = [
        {"name": "q2i_ann_score", "as": "ann_dist_list"},  
      ],
      export_item_attr = [
         {"name": "ann_dist", "as": "search_score"},  
      ],
      target_item = { flag_attr: 1 },
      function_name = "AnnCalThresholdValueForDistList",
      class_name = "ExploreLightFunctionSetV2",
    ) \

    return self

  def mc_search_score_boost(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_mc_search_score_boost_coeff", "as": "boost_discount_coeff"},
        {"name": "explore_mc_search_score_boost_thres", "as": "boost_discount_thres"},
      ],
      import_item_attr = [
        {"name": "search_score", "as": "need_item_attr"},
        {"name": score_attr, "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr}
      ],
      target_item = { flag_attr: 1 },
      function_name = "BoostOrDiscountWithThres",
      class_name = "ExploreLightFunctionSetV2",
    ) \

    return self

  def boost_ua_long_view(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "mc_boost_ua_long_view_weight", "as": "boost_weight"},
        {"name": "mc_weaken_ua_long_view_weight", "as": "weaken_weight"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr},
      ],
      target_item = {"is_long_view_author": 1},
      function_name = "EnsembleScoreBoost",
      class_name = "ExploreLightFunctionSetV2",
    ) \

    return self

  def deboost_merchant_car_photo(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": score_attr, "as": "ensemble_score"},
        "is_merchant_cart",
      ],
      import_common_attr = [
        "weaken_merchant_car_weight",
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr}
      ],
      target_item = { flag_attr: 1 },
      function_name = "DeboostMerchantCarPhoto",
      class_name = "ExploreLightFunctionSetV2"
    )
    return self

  def explore_replace_cascade_ctr_corr(self):
    """
    Module: RankingScoreModule
    功能: 真实ctr替换ctr队列
    Owner: xuwei09
    Date: 2024-05-09
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_replace_cascade_ctr_corr_alpha", "as": "eff_ctr_corr_alpha"},
        {"name": "explore_replace_cascade_ctr_corr_power", "as": "eff_ctr_corr_power"},
      ],
      import_item_attr = [
        {"name": "cascade_corr_pctr", "as": "pctr"},
        "hetu_tag_level_info_v2__hetu_level_one",
        {"name": "cascade_psvtr", "as": "psvr"},
        "is_picture"
      ],
      export_item_attr = [
        {"name": "pctr", "as": "cascade_corr_pctr_psvr"},
      ],
      function_name = "RealCtrReplaceCtr",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def explore_cascade_pxtr_calibration(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_cascade_pctr_calibration_upload_time", "as": "pctr_calibration_upload_time"},
      ],
      import_item_attr = [
        "upload_time",
        {"name": "cascade_pctr", "as": "pctr"},
      ],
      export_item_attr = [
        {"name": "pctr", "as": "cascade_corr_pctr"},
      ],
      function_name = "PxtrCalibration",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .if_("enable_explore_cascade_new_information_pxtr_calibration == 1") \
      .split_string(
        input_common_attr = "explore_cascade_new_information_pxtr_calibrate_hetu",
        output_common_attr = "explore_cascade_new_information_pxtr_calibrate_hetu_list",
        delimiters = ",",
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .split_string(
        input_common_attr = "explore_cascade_pctr_calibration_new_information",
        output_common_attr = "explore_cascade_pctr_calibration_new_information_list",
        delimiters = ",",
        skip_empty_tokens = True,
        parse_to_double = True
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_cascade_new_information_pxtr_calibrate_hetu_list", "as": "calibrate_hetu_list"},
          {"name": "explore_cascade_pctr_calibration_new_information_list", "as": "pxtr_calibration_upload_time"},
        ],
        import_item_attr = [
          "upload_time",
          "hetu_tag_level_info_v2__hetu_level_one",
          {"name": "cascade_corr_pctr", "as": "pxtr"},
        ],
        export_item_attr = [
          {"name": "pxtr", "as": "cascade_corr_pctr"},
        ],
        function_name = "PxtrCalibrationV2",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_()
    return self

  def explore_cascade_cal_debias_xtr_by_pcoc_score(self, prefix, raw_xtr, debias_xtr_by_pcoc):
    self.str_format(
      format_string = f"{prefix}_%d",
      input_attrs = ["basic_info_age_segment_v2"],
      output_attr = "cascade_debias_xtr_pcoc_key",
    ) \
    .get_kconf_params(
      kconf_configs = [{
        "kconf_key": "reco.offline.userHetuInteractPcocStat",
        "json_path": "{{cascade_debias_xtr_pcoc_key}}",
        "value_type": "list_double",
        "default_value": [],
        "export_common_attr": "cascade_debias_xtr_by_pcoc_list"
      }]
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "cascade_debias_xtr_by_pcoc_list", "as": "value_list"},
      ],
      import_item_attr = [
        {"name": "hetu_level_one_top1", "as" : "item_key_attr"}
      ],
      export_item_attr = [
        {"name": "target_item_attr", "as": "cascade_debias_xtr_by_pcoc"}
      ],
      function_name = "AddItemAttrByCommonList",
      class_name = "ExploreLightFunctionSetV2"
    ) \
    .set_attr_default_value(
      item_attrs=[{
        "name": "cascade_debias_xtr_by_pcoc",
        "type": "double", 
        "value": 1.0
      }]
    ) \
    .item_attr_operation(
      item_attr_a = raw_xtr,
      item_attr_b = "cascade_debias_xtr_by_pcoc",
      operator = "/",
      output_attr = debias_xtr_by_pcoc
    )
    return self

  def explore_cascade_eff_ctr_corr(self):
    """
    Module: CascadingScoreModule
    功能: 真实ctr替换ctr
    Owner: wangyalong03
    Date: 2024-06-21
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "cascade_eff_ctr_corr_pic_coeff", "as": "eff_ctr_corr_pic_coeff"},
        {"name": "cascade_eff_ctr_corr_power", "as": "eff_ctr_corr_power"},
        {"name": "cascade_eff_ctr_corr_alpha", "as": "eff_ctr_corr_alpha"},
      ],
      import_item_attr = [
        {"name": "cascade_corr_pctr", "as": "pctr"},
        {"name": "cascade_psvtr", "as": "psvr"},
        "hetu_tag_level_info_v2__hetu_level_one",
        "is_picture"
      ],
      export_item_attr = [
        {"name": "pctr", "as": "cascade_corr_pctr"},
      ],
      function_name = "RealCtrReplaceCtr",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def gen_is_outer_field_interest_photo(self):
    self.enrich_attr_by_light_function(
      item_list_from_attr = "explore_realshow_pids",
      import_common_attr = [
        {"name": "explore_outer_field_rate_threshold", "as": "rate_threshold"},
        {"name": "explore_outer_field_cnt_threshold", "as": "cnt_threshold"},
        {"name": "cluster_id_632_default_value", "as": "default_value"},
      ],
      import_item_attr = [
        {"name" : "cluster_id_632", "as" : "attr"}
      ],
      export_common_attr = [
        {"name": "final_list", "as": "hot_show_interest_cids_632"}
      ],
      function_name = "PackItemAttrWithFilter",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "hot_show_interest_cids_632", "as": "attr_list"},
        {"name": "cluster_id_632_default_value", "as": "default_value"},
      ],
      import_item_attr = [
        {"name" : "cluster_id_632", "as" : "attr"}
      ],
      export_item_attr = [
        {"name": "is_in_set", "as": "is_hot_show_interest_632"}
      ],
      function_name = "AttrIsInSet",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .set_attr_value(
      no_overwrite=True,
      item_attrs=[
        {
          "name": "is_outer_field_interest",
          "type": "int",
          "value": 1
        }
      ],
      select_item = { 
        "join": "and",
        "filters": [{
          "attr_name": "is_hot_show_interest_632",
          "select_if": "!=",
          "compare_to": 1,
          "select_if_attr_missing": True
        }, {
          "attr_name": "is_all_page_valid_interest",
          "select_if": "==",
          "compare_to": 1,
        }]
      }
    )
    return self

  def cal_explore_fountain_view_weight(self):
    self.enrich_attr_by_light_function(
      import_common_attr=[
        "colossus_channel_list",
        "explore_min_explore_view_cnt",
        "explore_min_fountain_view_cnt",
        "explore_ef_weight_alpha",
        "explore_ef_weight_beta",
        "explore_ef_weight_min",
        "explore_ef_weight_max",
      ],
      export_common_attr=[
        "explore_fountain_view_weight",
      ],
      function_name="CalcExploreFountainViewWeight",
      class_name="ExploreLightFunctionSetV2",
    )
    return self

  def gen_is_new_interest_explore(self):
    self.if_("explore_calc_new_interest_use_632 == 1 or is_traceback_request == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name" : "user_long_term_interest_cid_list", "as" : "attr_list"},
          {"name" : "explore_new_interest_max_num", "as" : "max_num_threshold"}
        ],
        import_item_attr = [
          {"name" : "cluster_id_632", "as" : "attr"}
        ],
        export_item_attr = [
          {"name" : "is_not_in_set", "as" : "is_new_interest_explore"}
        ],
        function_name = "AttrIsNotInSet",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .enrich_attr_by_light_function(
        item_list_from_attr = "explore_recent_play_list",
        import_common_attr = [
          {"name" : "user_long_term_interest_cid_list", "as" : "attr_list"},
          {"name" : "explore_new_interest_max_num", "as" : "max_num_threshold"}
        ],
        import_item_attr = [
          {"name" : "cluster_id_632", "as" : "attr"}
        ],
        export_item_attr = [
          {"name" : "is_not_in_set", "as" : "is_new_interest_explore"}
        ],
        function_name = "AttrIsNotInSet",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .if_("explore_ranking_diversity_enable_standard_explore_realshow_pid_list == 1") \
        .enrich_attr_by_light_function(
          item_list_from_attr = "standard_explore_realshow_pid_list",
          import_common_attr = [
            {"name" : "user_long_term_interest_cid_list", "as" : "attr_list"},
            {"name" : "explore_new_interest_max_num", "as" : "max_num_threshold"}
          ],
          import_item_attr = [
            {"name" : "cluster_id_632", "as" : "attr"}
          ],
          export_item_attr = [
            {"name" : "is_not_in_set", "as" : "is_new_interest_explore"}
          ],
          function_name = "AttrIsNotInSet",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_() \
    .else_() \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name" : "uOldMmuClusterId300ListList", "as" : "attr_list"},
          {"name" : "explore_new_interest_max_num", "as" : "max_num_threshold"}
        ],
        import_item_attr = [
          {"name" : "mounted_interest_cluster_id", "as" : "attr"}
        ],
        export_item_attr = [
          {"name" : "is_not_in_set", "as" : "is_new_interest_explore"}
        ],
        function_name = "AttrIsNotInSet",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .enrich_attr_by_light_function(
        item_list_from_attr = "explore_recent_play_list",
        import_common_attr = [
          {"name" : "uOldMmuClusterId300ListList", "as" : "attr_list"},
          {"name" : "explore_new_interest_max_num", "as" : "max_num_threshold"}
        ],
        import_item_attr = [
          {"name" : "mounted_interest_cluster_id", "as" : "attr"}
        ],
        export_item_attr = [
          {"name" : "is_not_in_set", "as" : "is_new_interest_explore"}
        ],
        function_name = "AttrIsNotInSet",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .if_("explore_ranking_diversity_enable_standard_explore_realshow_pid_list == 1") \
        .enrich_attr_by_light_function(
          item_list_from_attr = "standard_explore_realshow_pid_list",
          import_common_attr = [
            {"name" : "uOldMmuClusterId300ListList", "as" : "attr_list"},
            {"name" : "explore_new_interest_max_num", "as" : "max_num_threshold"}
          ],
          import_item_attr = [
            {"name" : "mounted_interest_cluster_id", "as" : "attr"}
          ],
          export_item_attr = [
            {"name" : "is_not_in_set", "as" : "is_new_interest_explore"}
          ],
          function_name = "AttrIsNotInSet",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_() \
    .end_()
    return self


  def gen_is_marketing_compensation_photo(self):
    self.split_string(
      input_common_attr = "explore_marketing_compensation_photo_tags_list_str",
      output_common_attr = "explore_marketing_compensation_photo_tags_list",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_marketing_compensation_photo_tags_list", "as": "tags_list"},
        {"name": "explore_marketing_compensation_high_value_author_ignore", "as": "high_value_author_ignore"},
        {"name": "explore_marketing_compensation_open_reason_thres", "as": "open_reason_thres"},
        "high_value_black_author_map_ptr"
      ],
      import_item_attr = [
        "sirius_distribution_info__mark_cod",
        "author__id"
      ],
      export_item_attr = [
        "is_marketing_compensation_photo"
      ],
      function_name = "GenIsMarketingCompensationPhoto",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def user_group_consume_weight_adjust(self, weight_param_dict, stage_name):
    input_common_attrs = [
      {"name": "explore_all_user_consume_str", "as": "all_user_consume_stat_str"},
      {"name": "explore_user_group_consume_str", "as": "user_consume_stat_str"}
    ]
    output_common_attrs = []
    ratio_prefix = "explore_user_group_consume_weight_adjust_ratio_" + stage_name + "_"
    for xtr in weight_param_dict.keys():
      input_common_attrs.append({"name": weight_param_dict[xtr], "as": xtr + "_weight"})
      input_common_attrs.append({"name": ratio_prefix + xtr, "as": xtr + "_adjust_ratio"})
      output_common_attrs.append({"name": xtr + "_weight", "as": weight_param_dict[xtr]})
    
    self.enrich_attr_by_light_function(
      import_common_attr = input_common_attrs,
      export_common_attr = output_common_attrs,
      function_name = "UserGroupWeightAdjustCoef",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def get_user_group_interest_tgi(self):
    self.switch_("explore_user_group_interest_tgi_version") \
      .case_(2) \
        .enrich_attr_by_light_function(
          export_common_attr = [
            "is_work_day"
          ],
          function_name = "IsWorkDay",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .str_format(
          format_string = "%s_%d_%d",
          input_attrs = ["explore_user_group_interest_tgi_prefix", "uMultiDimensionGroupDetailKV", "is_work_day"],
          output_attr = "user_group_interest_tgi_key",
        ) \
      .case_(1) \
        .str_format(
          format_string = "%s_%d",
          input_attrs = ["explore_user_group_interest_tgi_prefix", "uMultiDimensionGroupDetailKV"],
          output_attr = "user_group_interest_tgi_key",
        ) \
      .default_() \
        .cast_attr_type(
          attr_type_cast_configs = [{
            "to_type": "string",
            "from_common_attr": "uMultiDimensionGroupKV",
            "to_common_attr": "user_group_interest_tgi_key"
          }]
        ) \
    .end_() \
    .get_kconf_params(
      kconf_configs = [{
        "kconf_key": "reco.offline.userGroupInterestTgiStat",
        "json_path": "{{user_group_interest_tgi_key}}",
        "value_type": "list_double",
        "default_value": [],
        "export_common_attr": "explore_user_group_interest_tgi_list"
      }]
    ) \
    .set_attr_default_value(
      item_attrs=[{
        "name": "user_group_interest_tgi_score",
        "type": "double",
        "value": 1.0
      }]
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_user_group_interest_tgi_list", "as": "value_list"},
      ],
      import_item_attr = [
        {"name" : "hetu_sim_cluster_id", "as" : "item_key_attr"}
      ],
      export_item_attr = [
        {"name": "target_item_attr", "as": "user_group_interest_tgi_score"}
      ],
      function_name = "AddItemAttrByCommonList",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_user_group_interest_tgi_shrink_center", "as": "center"},
        {"name": "explore_user_group_interest_tgi_shrink_ratio", "as": "ratio"},
      ],
      import_item_attr = [
        {"name" : "user_group_interest_tgi_score", "as" : "target_item_attr"}
      ],
      export_item_attr = [
        {"name": "target_item_attr", "as": "user_group_interest_tgi_score"}
      ],
      function_name = "ShrinkItemAttr",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        "is_new_interest_explore": 1,
      }
    ) 

    return self

  def prerank_marketing_compensation_discount(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "prerank_marketing_compensation_discount_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
          flag_attr : 1,
          "is_marketing_compensation_photo" : 1
        }
    )
    return self

  def mc_s2_marketing_compensation_discount(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "mc_s2_marketing_compensation_discount_ctr_weight", "as": "ctr_weight"},
        {"name": "mc_s2_marketing_compensation_discount_watchtime_weight", "as": "watchtime_weight"},
        {"name": "mc_s2_marketing_compensation_discount_score_base", "as": "score_base"},
        {"name": "mc_s2_marketing_compensation_discount_score_base_ratio", "as": "score_base_ratio"},
        {"name": "mc_s2_marketing_compensation_discount_coef", "as": "old_coeff"},
      ],
      import_item_attr = [
        {"name": "mc_ensemble_pctr", "as": "ctr"},
        {"name": "mc_ensemble_pwtd_inverse", "as": "watchtime"},
      ],
      export_common_attr = [
       {"name": "coeff", "as": "mc_s2_marketing_compensation_discount_reward_coeff"}
      ],
      function_name = "CalcRewardCoeff",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "mc_s2_marketing_compensation_discount_scale_factor", "as": "scale_factor"},
        {"name": "mc_s2_marketing_compensation_discount_reward_coeff", "as": "reward_coeff"},
        {"name": "mc_s2_marketing_compensation_discount_coef", "as": "old_coeff"},
      ],
      export_common_attr = [
       {"name": "new_coeff", "as": "mc_s2_marketing_compensation_discount_coef"}
      ],
      function_name = "MarketingCompensationPhotoAdjust",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "mc_s2_marketing_compensation_discount_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1,
        "is_marketing_compensation_photo" : 1
      }
    )
    return self

  def mc_s2_marketing_compensation_personal_discount(self, score_attr, flag_attr):
    self \
    .calc_by_formula1(
      kconf_key = "formula.scenarioKey77.McExploreMarketingPhotoDeboost",
      import_item_attr = [
        "explore_marketing_compensation_positive_trigger_similarity_score",
      ],
      import_common_attr = [
        "explore_marketing_compensation_positive_trigger_size",
      ],
      export_formula_value = [
        {"name": "final_score", "as": "mc_s2_marketing_compensation_personal_discount_coef"}
      ],
      abtest_biz_name = "KUAISHOU_APPS",
      target_item = {
        "is_marketing_compensation_photo": 1,
        flag_attr: 1
      }
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "mc_s2_marketing_compensation_personal_discount_coef", "as": "boost_discount_coeff"},
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr},
      ],
      function_name = "BoostOrDiscountWithItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1,
        "is_marketing_compensation_photo": 1
      }
    )
    return self
  
  def prerank_olympic_latest_boost(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_prerank_olympic_latest_boost_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1,
        "is_olympic_latest" : 1
      }
    ) 
    return self
  
  def mc_s2_olympic_latest_boost(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_mc_s2_olympic_latest_boost_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1,
        "is_olympic_latest" : 1
      }
    ) 
    return self

  def gen_is_protogenetic_advertise_photo(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
         "protogenetic_advertise_type_list_str"
      ],
      import_item_attr = [
        "data_set_tags_bit"
      ],
      export_item_attr = [
        "is_protogenetic_advertise_photo"
      ],
      function_name = "IsProtogeneticAdvertisePhoto",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .if_("enable_explore_prev_items_gen_is_protogenetic_advertise_photo == 1") \
      .enrich_attr_by_light_function(
        item_list_from_attr = "explore_recent_play_list",
        import_common_attr = [
          "protogenetic_advertise_type_list_str"
        ],
        import_item_attr = [
          "data_set_tags_bit"
        ],
        export_item_attr = [
          "is_protogenetic_advertise_photo"
        ],
        function_name = "IsProtogeneticAdvertisePhoto",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .if_("explore_ranking_diversity_enable_standard_explore_realshow_pid_list == 1") \
        .enrich_attr_by_light_function(
          item_list_from_attr = "standard_explore_realshow_pid_list",
          import_common_attr = [
            "protogenetic_advertise_type_list_str"
          ],
          import_item_attr = [
            "data_set_tags_bit"
          ],
          export_item_attr = [
            "is_protogenetic_advertise_photo"
          ],
          function_name = "IsProtogeneticAdvertisePhoto",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
    .end_()
    return self

  def gen_is_olympic_photo(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_olympic_latest_photo_hour_limit", "as": "hour_limit"}
      ],
      import_item_attr = [
        "upload_time",
        "hetu_tag_level_info__hetu_tag"
      ],
      export_item_attr = [
        "is_olympic",
        "is_olympic_latest"
      ],
      function_name = "GenIsOlympicPhoto",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def gen_upload_time_day(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        "upload_time"
      ],
      export_item_attr = [
        "upload_time_day"
      ],
      function_name = "GenUploadTimeDay",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self
  
  def gen_upload_time_second(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "upload_time", "as": "upload_time"}
      ],
      export_item_attr = [
        {"name": "upload_time_second", "as": "item_upload_second"},
      ],
      function_name = "GenUploadTimeSecond",
      class_name = "ExploreLightFunctionSetV2"
    )
    return self
  
  def gen_lowvv_tag(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "explore_stat__real_show_count", "as": "real_show_count"}
      ],
      import_common_attr = [
        {"name": "explore_cs_photo_quality_boost_lower_bound", "as": "lower_bound"},
        {"name": "explore_cs_photo_quality_boost_upper_bound", "as": "upper_bound"},
      ],
      export_item_attr = [
        {"name": "is_lowvv", "as": "is_lowvv"},
      ],
      function_name = "IsLowvvPhoto",
      class_name = "ExploreLightFunctionSetV2"
    )
    return self
  
  def gen_same_author_tail_tag(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "video_cold_start_info__explore_author_exp_tail", "as": "item_attr"}
      ],
      import_common_attr = [
        {"name": "explore_author_exp_tail", "as": "common_attr"}
      ],
      export_item_attr = [
        {"name": "judge", "as": "is_same_author_tail"},
      ],
      function_name = "JudgeItemAttrAndCommonAttrEqual",
      class_name = "ExploreLightFunctionSetV2"
    )
    return self

  def gen_photo_quality_score(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "cold_item_quality_score_map_ptr", "as": "map_ptr"}
      ],
      import_item_attr = [
        {"name": "photo_id", "as": "key_attr"}
      ],
      export_item_attr = [
        {"name": "target_item_attr", "as": "cold_item_quality_score"},
      ],
      function_name = "GetItemAttrByIntToDoubleMapPtr",
      class_name = "ExploreLightFunctionSetV2"
    )
    return self

  def gen_is_diversity_degraded(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_diversity_degraded_min_per_hetu1", "as": "min_per_hetu1"},
        {"name": "explore_diversity_degraded_hetu1_min", "as": "hetu1_min"},
      ],
      import_item_attr = [
        "hetu_tag_level_info_v2__hetu_level_one"
      ],
      export_common_attr = [
        "is_diversity_hetu1_degraded"
      ],
      function_name = "GenDiversityDegraded",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def gen_is_user_recent_consume_photo(self, flag_attr):
    self.enrich_attr_by_light_function(
      item_list_from_attr = "standard_explore_realshow_pid_list",
      import_common_attr = [
        {"name": "uStandardExploreRealshowTimestampList", "as": "real_show_timestamp_list"},
        {"name": "uStandardExploreRealshowLabelList", "as": "real_show_label_list"},
        {"name": "explore_user_recent_consume_photo_minute_threshold", "as": "minute_threshold"},
      ],
      import_item_attr = [
        "cluster_id_632",
        "hetu_tag_level_info__hetu_tag"
      ],
      export_common_attr = [
        {"name": "unclick_cid_list", "as": "user_recent_unclick_cid_list"},
        {"name": "click_cid_list", "as": "user_recent_click_cid_list"},
        {"name": "unclick_tagnex_list", "as": "user_recent_unclick_tagnex_list"},
        {"name": "click_tagnex_list", "as": "user_recent_click_tagnex_list"},
      ],
      function_name = "GenUserRecentConsumeTags",
      class_name = "ExploreLightFunctionSetV2"
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "user_recent_unclick_cid_list", "as": "unclick_cid_list"},
        {"name": "user_recent_click_cid_list", "as": "click_cid_list"},
        {"name": "user_recent_unclick_tagnex_list", "as": "unclick_tagnex_list"},
        {"name": "user_recent_click_tagnex_list", "as": "click_tagnex_list"},
        {"name": "explore_user_recent_consume_photo_enable_unclick", "as": "enable_unclick"},
        {"name": "explore_user_recent_consume_photo_enable_cid", "as": "enable_cid"},
        {"name": "explore_user_recent_consume_photo_enable_tagnex", "as": "enable_tagnex"},
      ],
      import_item_attr = [
        "cluster_id_632",
        "hetu_tag_level_info__hetu_tag"
      ],
      export_item_attr = [
        "is_user_recent_consume_photo"
      ],
      function_name = "GenIsUserRecentConsumePhoto",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1,
      }
    )
    return self

  def boost_user_recent_consume_photo(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_mc_s2_recent_consume_photo_boost_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        "is_user_recent_consume_photo" : 1,
        flag_attr : 1,
      }
    )
    return self

  def gen_is_audit_good_photo(self, flag_attr):
    self.split_string(
      input_common_attr = "explore_user_skip_audit_impression_ignore_tag_str",
      output_common_attr = "explore_user_skip_audit_impression_ignore_tag",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True,
    ) \
    .split_string(
      input_common_attr = "explore_user_skip_audit_hot_high_ignore_tag_str",
      output_common_attr = "explore_user_skip_audit_hot_high_ignore_tag",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True,
    ) \
    .split_string(
      input_common_attr = "explore_user_skip_audit_topk_ignore_tag_str",
      output_common_attr = "explore_user_skip_audit_topk_ignore_tag",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True,
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_user_skip_audit_impression_ignore_tag", "as": "audit_impression_ignore_tag"},
        {"name": "explore_user_skip_audit_hot_high_ignore_tag", "as": "audit_hot_high_ignore_tag"},
        {"name": "explore_user_skip_audit_topk_ignore_tag", "as": "audit_topk_ignore_tag"},
      ],
      import_item_attr = [
        "audit_cold_review_level",
        "content_safety_level_with_namespace__level_hot_online",
        "audit_b_second_tag",
        "audit_hot_high_tag_level",
        "explore_operation_c_review_level",
        "topk_audit_level",
        "topk_audit_tag",
      ],
      export_item_attr = [
        "is_audit_good_photo"
      ],
      function_name = "GenIsAuditGoodPhoto",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1,
      }
    )
    return self

  def boost_audit_good_photo(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_mc_s2_audit_good_photo_boost_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        "is_audit_good_photo" : 1,
        flag_attr : 1,
      }
    )
    return self

  def gen_is_meinv_photo(self):
    self.split_string(
      input_common_attr = "explore_meinv_hetu5_list_str",
      output_common_attr = "explore_meinv_hetu5_list",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True,
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_meinv_hetu5_list", "as": "attr_list"},
      ],
      import_item_attr = [
        {"name" : "hetu_tag_level_info__hetu_level_five", "as" : "attrs"}
      ],
      export_item_attr = [
        {"name": "is_in_set", "as": "is_in_meinv_hetu5_set"}
      ],
      function_name = "AttrListIsInSet",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .split_string(
      input_common_attr = "explore_meinv_cid_list_str",
      output_common_attr = "explore_meinv_cid_list",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True,
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_meinv_cid_list", "as": "attr_list"},
      ],
      import_item_attr = [
        {"name" : "hetu_sim_cluster_id", "as" : "attr"}
      ],
      export_item_attr = [
        {"name": "is_in_set", "as": "is_in_meinv_cid_set"}
      ],
      function_name = "AttrIsInSet",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .set_attr_value(
      item_attrs = [
        {
          "name": "is_meinv_photo",
          "type": "int",
          "value": 1
        }
      ],
      select_item = { 
        "join": "or",
        "filters": [{
            "attr_name": "is_in_meinv_hetu5_set",
            "select_if": "==",
            "compare_to": 1,
        }, {
            "attr_name": "is_in_meinv_cid_set",
            "select_if": "==",
            "compare_to": 1,
        }]
      }
    )
    return self

  def gen_photo_show_ration(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        "show_ration_realshow_threshold",
      ],
      import_item_attr = [
        "explore_stat__real_show_count",
        "thanos_stats__real_show_count"
      ],
      export_item_attr = [
        "show_ration_level",
      ],
      function_name = "GenPhotoShowRation",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def gen_is_low_cost_photo(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "negative_aid_set_ptr", "as": "aid_set_ptr"}
      ],
      import_item_attr = [
        "author__id"
      ],
      export_item_attr = [
        {"name": "is_target_photo", "as": "is_low_cost_photo"}
      ],
      function_name = "AidInSet",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def gen_is_minority_photo(self):    
    self.split_string(
      input_common_attr = "explore_minority_photo_tags_bits_list_str",
      output_common_attr = "explore_minority_photo_tags_bits_list",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True,
    ) \
    .split_string(
      input_common_attr = "explore_minority_photo_manjiao_markcode_tags_str",
      output_common_attr = "explore_minority_photo_manjiao_markcode_tags",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True,
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_minority_photo_tags_bits_list", "as": "minority_photo_bits_list"},
        {"name": "explore_minority_photo_manjiao_markcode_tags", "as": "manjiao_markcode_tags"}
      ],
      import_item_attr = [
        "data_set_tags_bit",
        "manjiao_markcode"
      ],
      export_item_attr = [
        "is_minority_photo",
      ],
      function_name = "IsMinorityPhotoV2",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .if_("enable_explore_prev_items_gen_minority_photo == 1") \
      .enrich_attr_by_light_function(
        item_list_from_attr = "explore_recent_play_list",
        import_common_attr = [
          {"name": "explore_minority_photo_tags_bits_list", "as": "minority_photo_bits_list"},
          {"name": "explore_minority_photo_manjiao_markcode_tags", "as": "manjiao_markcode_tags"}
        ],
        import_item_attr = [
          "data_set_tags_bit",
          "manjiao_markcode"
        ],
        export_item_attr = [
          "is_minority_photo",
        ],
        function_name = "IsMinorityPhotoV2",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .if_("explore_ranking_diversity_enable_standard_explore_realshow_pid_list == 1") \
        .enrich_attr_by_light_function(
          item_list_from_attr = "standard_explore_realshow_pid_list",
          import_common_attr = [
            {"name": "explore_minority_photo_tags_bits_list", "as": "minority_photo_bits_list"},
            {"name": "explore_minority_photo_manjiao_markcode_tags", "as": "manjiao_markcode_tags"}
          ],
          import_item_attr = [
            "data_set_tags_bit",
            "manjiao_markcode"
          ],
          export_item_attr = [
            "is_minority_photo",
          ],
          function_name = "IsMinorityPhotoV2",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
    .end_()
    return self

  def gen_user_develop_interest_score(self):
    self \
      .cal_is_in_set(input_set_name = "user_short_develop_interest_cid_list", default_value = "cluster_id_632_default_value",
                       item_flag = "cluster_id_632", output_name = "is_user_short_develop_interest") \
      .set_attr_default_value(
        item_attrs=[
          {
            "name": "is_user_short_develop_interest",
            "type": "int",
            "value": 0
          }
        ]
      )
    return self

  def boost_user_short_develop_interest(self, score_attr, flag_attr, stage_name="mc_s2", strategy_name="user_short_develop_interest"):
    all_item_count = "explore_" + stage_name + "_" + strategy_name + "_all_item_count"
    target_item_count = "explore_" + stage_name + "_" + strategy_name + "_target_item_count"
    boost_coef = "explore_" + stage_name + "_" + strategy_name + "_boost_coef"
    alpha = "explore_" + stage_name + "_" + strategy_name + "_alpha"
    max_ratio = "explore_" + stage_name + "_" + strategy_name + "_max_ratio"
    empirical_ctr_threshold = "explore_" + strategy_name + "_emp_ctr_threshold"
    self.count_reco_result(
      save_count_to = all_item_count,
      target_item = {flag_attr : 1}, 
    ) \
    .count_reco_result(
      save_count_to = target_item_count,
      target_item = {
        "is_user_short_develop_interest" : 1,
        flag_attr : 1,
      }, 
    ) \
    .gen_common_attr_by_lua(
      attr_map={
        boost_coef: "{coef} * (1 + {alpha} * ({max_ratio} - {target_cnt} / {total_cnt} > 0.0 and {max_ratio} - {target_cnt} / {total_cnt} or 0.0))".format(coef=boost_coef, alpha=alpha, max_ratio=max_ratio, target_cnt=target_item_count, total_cnt=all_item_count)
      }   
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": boost_coef, "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      select_item = {
        "join": "and",
        "filters": [{
            "attr_name": "is_user_short_develop_interest",
            "select_if": "==",
            "compare_to": 1,
        }, {
            "attr_name": flag_attr,
            "select_if": "==",
            "compare_to": 1,
        }, {
            "attr_name": "empirical_ctr",
            "select_if": ">",
            "compare_to": "{{" + empirical_ctr_threshold + "}}",
        }]
      }
    )
    return self

  def prerank_low_cost_photo_discount(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_prerank_low_cost_photo_discount_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1,
        "is_low_cost_photo" : 1
      }
    )
    return self

  def mc_s2_boost_similar_author(self, score_attr, flag_attr):#开实验 9.28删代码
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "mc_boost_similar_author_reason_topk", "as": "fr_quality_signal_exptag_reason_topk"},
        {"name": "mc_boost_similar_author_reason_weight", "as": "quality_signal_exptag_reason_weight"}
      ],
      import_item_attr = [
        {"name": score_attr, "as": "explore_fr_ensemble_score"},
      ],
      export_item_attr = [
        {"name": "explore_fr_ensemble_score", "as": score_attr},
        {"name": "is_quality_singal_topk", "as": "is_pid_for_similar_author"}
      ],
      function_name = "BoostTopkQualitySignalPhoto",
      class_name = "ExploreLightFunctionSetV2",
      target_item={
        flag_attr: 1,
        "reason": 3200,
      },
    )
    return self 

  def mc_s2_boost_unbias_interest_photo(self, score_attr, flag_attr):#开实验 9.28删代码
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "mc_boost_unbias_interest_reason_topk", "as": "fr_quality_signal_exptag_reason_topk"},
        {"name": "mc_boost_unbias_interest_reason_weight", "as": "quality_signal_exptag_reason_weight"}
      ],
      import_item_attr = [
        {"name": score_attr, "as": "explore_fr_ensemble_score"},
      ],
      export_item_attr = [
        {"name": "explore_fr_ensemble_score", "as": score_attr},
        {"name": "is_quality_singal_topk", "as": "is_unbias_interest_pid_for_crows"}
      ],
      function_name = "BoostTopkQualitySignalPhoto",
      class_name = "ExploreLightFunctionSetV2",
      target_item={
        flag_attr: 1,
        "reason": 10043,
      },
    )
    return self

  def merchant_hetu_tag_discount(self, score_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_mc_s2_merchant_tag_discount_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        "is_merchant_hetu_tag_id" : 1
      }
    )
    return self

  def cal_is_in_set(self, input_set_name, default_value, item_flag, output_name):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": input_set_name, "as": "attr_list"},
        {"name": default_value, "as": "default_value"},
      ],
      import_item_attr = [
        {"name": item_flag, "as": "attr"}
      ],
      export_item_attr = [
        {"name": "is_in_set", "as": output_name}
      ],
      function_name = "AttrIsInSet",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def gen_is_top_audit_photo(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_top_audit_ctr_limit", "as": "top_audit_ctr_limit"},
        {"name": "explore_top_audit_day_new", "as":"top_audit_day_new"},
        {"name": "explore_top_audit_day_old", "as":"top_audit_day_old"},
        {"name": "explore_top_guangan_audit_limit", "as":"top_guangan_audit_limit"},
        {"name": "explore_top_topk_audit_limit", "as":"top_topk_audit_limit"},
        {"name": "explore_top_hot_audit_limit", "as":"top_hot_audit_limit"},
      ],
      import_item_attr = [
        "upload_time",
        {"name": "content_safety_level_with_namespace__level_hot_online", "as": "level_hot_online_attr"},
        "topk_audit_level",
        "audit_hot_high_tag_level",
        "explore_stat__real_show_count",
        "explore_stat__click_count"
      ],
      export_item_attr = [
        "is_top_audit",
      ],
      function_name = "GenIsTopAuditPhoto",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def prerank_top_author_new_boost(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_prerank_top_author_new_boost_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1,
        "is_top_audit" : 1
      }
    ) 
    return self

  def mc_s2_top_author_new_boost(self, score_attr, flag_attr): 
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_mc_s2_top_author_new_boost_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1,
        "is_top_audit" : 1,
      }
    )
    return self

  def gen_is_new_hot_photo(self):
    self.split_string(
      input_common_attr = "explore_new_hot_photo_bits_list_str",
      output_common_attr = "explore_new_hot_photo_bits_list",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True,
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_new_hot_day_limit", "as": "new_hot_day_limit"},
        {"name": "explore_new_hot_photo_bits_list", "as": "new_hot_photo_bits_list"}
      ],
      import_item_attr = [
        "upload_time",
        "data_set_tags_bit",
      ],
      export_item_attr = [
        "is_new_hot_photo",
      ],
      function_name = "IsNewHotPhoto",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def prerank_new_hot_boost(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_prerank_new_hot_boost_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1,
        "is_new_hot_photo" : 1
      }
    ) 
    return self

  def mc_s2_new_hot_boost(self, score_attr, flag_attr): 
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_mc_s2_new_hot_boost_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1,
        "is_new_hot_photo" : 1,
      }
    )

    return self

  def prerank_pic_search_boost(self, score_attr, flag_attr, target_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_prerank_pic_search_boost_coef", "as": "boost_discount_coeff"},
        {"name": "explore_prerank_pic_search_boost_topk", "as": "topk"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1,
        target_attr: 1,
      }
    )
    return self

  def prerank_pic_recent_search_boost(self, score_attr, flag_attr, target_attr):
    self \
    .if_("enable_explore_prerank_pic_recent_search_boost_sort_topk == 1") \
      .sort(
        score_from_attr = score_attr,
        target_item = {
          flag_attr: 1,
          target_attr: 1
        },
        partial_sort = True,
        partial_num = "{{explore_prerank_pic_recent_search_boost_topk}}"
      ) \
    .end_() \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_prerank_pic_recent_search_boost_coef", "as": "boost_discount_coeff"},
        {"name": "explore_prerank_pic_recent_search_boost_topk", "as": "topk"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1,
        target_attr: 1,
      }
    )
    return self

  def mc_pic_search_boost(self, score_attr, flag_attr, target_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_mc_pic_search_boost_coef", "as": "boost_discount_coeff"},
        {"name": "explore_mc_pic_search_boost_topk", "as": "topk"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1,
        target_attr: 1,
      }
    )
    return self

  def mc_pic_recent_search_boost(self, score_attr, flag_attr, target_attr):
    self \
    .if_("enable_explore_mc_pic_recent_search_boost_sort_topk == 1") \
      .sort(
        score_from_attr = score_attr,
        target_item = {
          flag_attr: 1,
          target_attr: 1
        },
        partial_sort = True,
        partial_num = "{{explore_mc_pic_recent_search_boost_topk}}"
      ) \
    .end_() \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_mc_pic_recent_search_boost_coef", "as": "boost_discount_coeff"},
        {"name": "explore_mc_pic_recent_search_boost_topk", "as": "topk"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1,
        target_attr: 1,
      }
    )
    return self
  
  def prerank_pic_valid_interest_cluster_boost(self, score_attr, flag_attr):
    self.sort(
      score_from_attr = score_attr,
      target_item = {
        flag_attr: 1,
        "is_pic_valid_interest_cluster": 1
      },
      partial_sort = True,
      partial_num = "{{explore_prerank_pic_valid_interest_cluster_boost_topk}}"
    ) \
    .if_("enable_explore_prerank_pic_valid_interest_cluster_boost_low_interest_user_enhance == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_prerank_pic_valid_interest_cluster_boost_coef", "as": "base_coeff"},
          {"name": "explore_prerank_pic_valid_interest_cluster_boost_low_interest_user_enhance_base_coef", "as": "enhance_base_coeff"},
          {"name": "explore_prerank_pic_valid_interest_cluster_boost_low_interest_user_enhance_weight", "as": "enhance_weight"},
          {"name": "explore_prerank_pic_valid_interest_cluster_boost_low_interest_thres", "as": "low_interest_thres"},
          {"name": "pic_double_outside_valid_interest_num", "as": "user_interest_num"},
        ],
        export_common_attr = [
          {"name": "enhance_coeff", "as": "explore_prerank_pic_valid_interest_cluster_boost_low_interest_user_enhance_coef"}
        ],
        function_name = "CalcPicLowInterestUserBoost",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_prerank_pic_valid_interest_cluster_boost_low_interest_user_enhance_coef", "as": "boost_discount_coeff"},
          {"name": "explore_prerank_pic_valid_interest_cluster_boost_thres", "as": "boost_discount_thres"},
          {"name": "explore_prerank_pic_valid_interest_cluster_boost_topk", "as": "topk"},
        ],
        import_item_attr = [
          {"name": "cascade_prerank_pctr", "as": "need_item_attr"},
          {"name": score_attr, "as": "ensemble_score"},
        ],
        export_item_attr = [
          {"name": "ensemble_score", "as": score_attr}
        ],
        function_name = "BoostOrDiscountWithThres",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          flag_attr: 1,
          "is_pic_valid_interest_cluster": 1
        }
      ) \
    .else_() \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_prerank_pic_valid_interest_cluster_boost_coef", "as": "boost_discount_coeff"},
          {"name": "explore_prerank_pic_valid_interest_cluster_boost_thres", "as": "boost_discount_thres"},
          {"name": "explore_prerank_pic_valid_interest_cluster_boost_topk", "as": "topk"},
        ],
        import_item_attr = [
          {"name": "cascade_prerank_pctr", "as": "need_item_attr"},
          {"name": score_attr, "as": "ensemble_score"},
        ],
        export_item_attr = [
          {"name": "ensemble_score", "as": score_attr}
        ],
        function_name = "BoostOrDiscountWithThres",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          flag_attr: 1,
          "is_pic_valid_interest_cluster": 1
        }
      ) \
    .end_()
    return self

  def mc_pic_valid_interest_cluster_boost(self, score_attr, flag_attr):
    self.sort(
      score_from_attr = score_attr,
      target_item = {
        flag_attr: 1,
        "is_pic_valid_interest_cluster": 1
      },
      partial_sort = True,
      partial_num = "{{explore_mc_pic_valid_interest_cluster_boost_topk}}"
    ) \
    .if_("enable_explore_mc_pic_valid_interest_cluster_boost_low_interest_user_enhance == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_mc_pic_valid_interest_cluster_boost_coef", "as": "base_coeff"},
          {"name": "explore_mc_pic_valid_interest_cluster_boost_low_interest_user_enhance_base_coef", "as": "enhance_base_coeff"},
          {"name": "explore_mc_pic_valid_interest_cluster_boost_low_interest_user_enhance_weight", "as": "enhance_weight"},
          {"name": "explore_mc_pic_valid_interest_cluster_boost_low_interest_thres", "as": "low_interest_thres"},
          {"name": "pic_double_outside_valid_interest_num", "as": "user_interest_num"},
        ],
        export_common_attr = [
          {"name": "enhance_coeff", "as": "explore_mc_pic_valid_interest_cluster_boost_low_interest_user_enhance_coef"}
        ],
        function_name = "CalcPicLowInterestUserBoost",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_mc_pic_valid_interest_cluster_boost_low_interest_user_enhance_coef", "as": "boost_discount_coeff"},
          {"name": "explore_mc_pic_valid_interest_cluster_boost_thres", "as": "boost_discount_thres"},
          {"name": "explore_mc_pic_valid_interest_cluster_boost_topk", "as": "topk"},
        ],
        import_item_attr = [
          {"name": "mc_ensemble_pctr", "as": "need_item_attr"},
          {"name": score_attr, "as": "ensemble_score"},
        ],
        export_item_attr = [
          {"name": "ensemble_score", "as": score_attr}
        ],
        function_name = "BoostOrDiscountWithThres",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          flag_attr: 1,
          "is_pic_valid_interest_cluster": 1
        }
      ) \
    .else_() \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_mc_pic_valid_interest_cluster_boost_coef", "as": "boost_discount_coeff"},
          {"name": "explore_mc_pic_valid_interest_cluster_boost_thres", "as": "boost_discount_thres"},
          {"name": "explore_mc_pic_valid_interest_cluster_boost_topk", "as": "topk"},
        ],
        import_item_attr = [
          {"name": "mc_ensemble_pctr", "as": "need_item_attr"},
          {"name": score_attr, "as": "ensemble_score"},
        ],
        export_item_attr = [
          {"name": "ensemble_score", "as": score_attr}
        ],
        function_name = "BoostOrDiscountWithThres",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          flag_attr: 1,
          "is_pic_valid_interest_cluster": 1
        }
      ) \
    .end_()
    return self

  def prerank_pic_long_interest_cluster_boost(self, score_attr, flag_attr):
    self.sort(
      score_from_attr = score_attr,
      target_item = {
        flag_attr: 1,
        "is_pic_long_interest_cluster": 1
      },
      partial_sort = True,
      partial_num = "{{explore_prerank_pic_long_interest_cluster_boost_topk}}"
    ) \
    .if_("enable_explore_prerank_pic_long_interest_cluster_boost_low_interest_user_enhance == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_prerank_pic_long_interest_cluster_boost_coef", "as": "base_coeff"},
          {"name": "explore_prerank_pic_long_interest_cluster_boost_low_interest_user_enhance_base_coef", "as": "enhance_base_coeff"},
          {"name": "explore_prerank_pic_long_interest_cluster_boost_low_interest_user_enhance_weight", "as": "enhance_weight"},
          {"name": "explore_prerank_pic_long_interest_cluster_boost_low_interest_thres", "as": "low_interest_thres"},
          {"name": "pic_double_outside_valid_interest_num", "as": "user_interest_num"},
        ],
        export_common_attr = [
          {"name": "enhance_coeff", "as": "explore_prerank_pic_long_interest_cluster_boost_low_interest_user_enhance_coef"}
        ],
        function_name = "CalcPicLowInterestUserBoost",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_prerank_pic_long_interest_cluster_boost_low_interest_user_enhance_coef", "as": "explore_prerank_pic_long_interest_cluster_boost_discount_coeff"},
          {"name": "explore_prerank_pic_long_interest_cluster_boost_thres", "as": "boost_discount_thres"},
          {"name": "explore_prerank_pic_long_interest_cluster_boost_topk", "as": "topk"},
        ],
        import_item_attr = [
          {"name": "cascade_prerank_pctr", "as": "need_item_attr"},
          {"name": score_attr, "as": "ensemble_score"},
        ],
        export_item_attr = [
          {"name": "ensemble_score", "as": score_attr}
        ],
        function_name = "BoostOrDiscountWithThres",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          flag_attr: 1,
          "is_pic_long_interest_cluster": 1
        }
      ) \
    .else_() \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_prerank_pic_long_interest_cluster_boost_coef", "as": "boost_discount_coeff"},
          {"name": "explore_prerank_pic_long_interest_cluster_boost_thres", "as": "boost_discount_thres"},
          {"name": "explore_prerank_pic_long_interest_cluster_boost_topk", "as": "topk"},
        ],
        import_item_attr = [
          {"name": "cascade_prerank_pctr", "as": "need_item_attr"},
          {"name": score_attr, "as": "ensemble_score"},
        ],
        export_item_attr = [
          {"name": "ensemble_score", "as": score_attr}
        ],
        function_name = "BoostOrDiscountWithThres",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          flag_attr: 1,
          "is_pic_long_interest_cluster": 1
        }
      ) \
    .end_()
    return self
  
  def mc_pic_boost_coef_with_flag(self, coef_attr, score_attr, flag_attrs, boost_num_max_attr, boost_num_ratio_attr):
    target_map = {flag_attr: 1 for flag_attr in flag_attrs}
    
    self.count_reco_result( # 统计 item 数量
      save_count_to = "mc_pic_flag_target_item_count", 
      target_item = target_map
    ) \
    .gen_common_attr_by_lua( # 计算要 boost 多少个, 同时控制最大比例和个数
      attr_map={
        "mc_pic_flag_target_item_boost_num": f"math.min(math.ceil(mc_pic_flag_target_item_count * {boost_num_ratio_attr}), {boost_num_max_attr})",
      }
    ) \
    .enrich_attr_by_light_function( # 执行 boost
      import_common_attr=[
        {"name": coef_attr, "as": "boost_discount_coeff"},
        {"name": "mc_pic_flag_target_item_boost_num", "as": "topk"}
      ],
      import_item_attr=[{"name": score_attr, "as": "score"},],
      export_item_attr=[{"name": "score", "as": score_attr},],
      function_name="BoostOrDiscountV2",
      class_name="ExploreLightFunctionSetV2",
      target_item=target_map
    )
    return self

  def mc_pic_long_interest_cluster_boost(self, score_attr, flag_attr):
    self.sort(
      score_from_attr = score_attr,
      target_item = {
        flag_attr: 1,
        "is_pic_long_interest_cluster": 1
      },
      partial_sort = True,
      partial_num = "{{explore_mc_pic_long_interest_cluster_boost_topk}}"
    ) \
    .if_("enable_explore_mc_pic_long_interest_cluster_boost_low_interest_user_enhance == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_mc_pic_long_interest_cluster_boost_coef", "as": "base_coeff"},
          {"name": "explore_mc_pic_long_interest_cluster_boost_low_interest_user_enhance_base_coef", "as": "enhance_base_coeff"},
          {"name": "explore_mc_pic_long_interest_cluster_boost_low_interest_user_enhance_weight", "as": "enhance_weight"},
          {"name": "explore_mc_pic_long_interest_cluster_boost_low_interest_thres", "as": "low_interest_thres"},
          {"name": "pic_double_outside_valid_interest_num", "as": "user_interest_num"},
        ],
        export_common_attr = [
          {"name": "enhance_coeff", "as": "explore_mc_pic_long_interest_cluster_boost_low_interest_user_enhance_coef"}
        ],
        function_name = "CalcPicLowInterestUserBoost",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_mc_pic_long_interest_cluster_boost_low_interest_user_enhance_coef", "as": "boost_discount_coeff"},
          {"name": "explore_mc_pic_long_interest_cluster_boost_thres", "as": "boost_discount_thres"},
          {"name": "explore_mc_pic_long_interest_cluster_boost_topk", "as": "topk"},
        ],
        import_item_attr = [
          {"name": "mc_ensemble_pctr", "as": "need_item_attr"},
          {"name": score_attr, "as": "ensemble_score"},
        ],
        export_item_attr = [
          {"name": "ensemble_score", "as": score_attr}
        ],
        function_name = "BoostOrDiscountWithThres",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          flag_attr: 1,
          "is_pic_long_interest_cluster": 1
        }
      ) \
    .else_() \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_mc_pic_long_interest_cluster_boost_coef", "as": "boost_discount_coeff"},
          {"name": "explore_mc_pic_long_interest_cluster_boost_thres", "as": "boost_discount_thres"},
          {"name": "explore_mc_pic_long_interest_cluster_boost_topk", "as": "topk"},
        ],
        import_item_attr = [
          {"name": "mc_ensemble_pctr", "as": "need_item_attr"},
          {"name": score_attr, "as": "ensemble_score"},
        ],
        export_item_attr = [
          {"name": "ensemble_score", "as": score_attr}
        ],
        function_name = "BoostOrDiscountWithThres",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          flag_attr: 1,
          "is_pic_long_interest_cluster": 1
        }
      ) \
    .end_()
    return self

  def prerank_pic_double_valid_interest_cluster_boost(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_prerank_pic_double_valid_interest_cluster_boost_coef", "as": "boost_discount_coeff"},
        {"name": "explore_prerank_pic_double_valid_interest_cluster_boost_thres", "as": "boost_discount_thres"},
        {"name": "explore_prerank_pic_double_valid_interest_cluster_boost_topk", "as": "topk"},
      ],
      import_item_attr = [
        {"name": "cascade_prerank_pctr", "as": "need_item_attr"},
        {"name": score_attr, "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountWithThres",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1,
        "is_pic_double_valid_interest_cluster": 1
      }
    )
    return self
  
  def mc_pic_double_valid_interest_cluster_boost(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_mc_pic_double_valid_interest_cluster_boost_coef", "as": "boost_discount_coeff"},
        {"name": "explore_mc_pic_double_valid_interest_cluster_boost_thres", "as": "boost_discount_thres"},
        {"name": "explore_mc_pic_double_valid_interest_cluster_boost_topk", "as": "topk"},
      ],
      import_item_attr = [
        {"name": "mc_ensemble_pctr", "as": "need_item_attr"},
        {"name": score_attr, "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountWithThres",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1,
        "is_pic_double_valid_interest_cluster": 1
      }
    )
    return self
  
  def prerank_pic_single_valid_interest_cluster_boost(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_prerank_pic_single_valid_interest_cluster_boost_coef", "as": "boost_discount_coeff"},
        {"name": "explore_prerank_pic_single_valid_interest_cluster_boost_thres", "as": "boost_discount_thres"},
        {"name": "explore_prerank_pic_single_valid_interest_cluster_boost_topk", "as": "topk"},
      ],
      import_item_attr = [
        {"name": "cascade_prerank_pctr", "as": "need_item_attr"},
        {"name": score_attr, "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountWithThres",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1,
        "is_pic_single_valid_interest_cluster": 1
      }
    )
    return self
  
  def mc_pic_single_valid_interest_cluster_boost(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_mc_pic_single_valid_interest_cluster_boost_coef", "as": "boost_discount_coeff"},
        {"name": "explore_mc_pic_single_valid_interest_cluster_boost_thres", "as": "boost_discount_thres"},
        {"name": "explore_mc_pic_single_valid_interest_cluster_boost_topk", "as": "topk"},
      ],
      import_item_attr = [
        {"name": "mc_ensemble_pctr", "as": "need_item_attr"},
        {"name": score_attr, "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountWithThres",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1,
        "is_pic_single_valid_interest_cluster": 1
      }
    )
    return self

  def prerank_pic_recent_interest_cluster_boost(self, score_attr, flag_attr):
    self \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_pic_recent_interest_cluster_id_list", "as": "key_list"},
        {"name": "explore_pic_recent_interest_cluster_score_list", "as": "value_list"},
        {"name": "explore_prerank_pic_recent_interest_cluster_score_alpha", "as": "score_alpha"},
        {"name": "explore_prerank_pic_recent_interest_cluster_score_beta", "as": "score_beta"},
        {"name": "explore_prerank_pic_recent_interest_cluster_score_sigma", "as": "score_sigma"},
        {"name": "enable_explore_prerank_pic_recent_interest_cluster_score_range_limit", "as": "enable_score_range_limit"},
        {"name": "explore_prerank_pic_recent_interest_cluster_score_lower_bound", "as": "score_lower_bound"},
        {"name": "explore_prerank_pic_recent_interest_cluster_score_upper_bound", "as": "score_upper_bound"},
        {"name": "enable_explore_prerank_pic_recent_interest_cluster_boost_only_low_interest_user", "as": "enable_only_low_interest_user"},
        {"name": "enable_explore_prerank_pic_recent_interest_cluster_boost_low_interest_user_enhance", "as": "enable_low_interest_user_enhance"},
        {"name": "explore_prerank_pic_recent_interest_cluster_boost_low_interest_user_enhance_base_coef", "as": "enhance_base_coeff"},
        {"name": "explore_prerank_pic_recent_interest_cluster_boost_low_interest_user_enhance_weight", "as": "enhance_weight"},
        {"name": "explore_prerank_pic_recent_interest_cluster_boost_low_interest_thres", "as": "low_interest_thres"},
        {"name": "uDoubleOutsideValidPicCluster7dList", "as": "user_interest"},
        {"name": "enable_explore_prerank_pic_recent_interest_cluster_boost_only_not_interest_cluster", "as": "enable_only_not_interest_cluster"},
      ],
      import_item_attr = [
        "cluster_id_632",
      ],
      export_item_attr = [
        {"name": "recent_interest_score", "as": "explore_prerank_pic_recent_interest_score"}
      ],
      function_name = "CalcPicRecentInterestScore",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1
      }
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "explore_prerank_pic_recent_interest_score", "as": "boost_discount_coeff"},
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1
      }
    )
    return self

  def mc_pic_recent_interest_cluster_boost(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_pic_recent_interest_cluster_id_list", "as": "key_list"},
        {"name": "explore_pic_recent_interest_cluster_score_list", "as": "value_list"},
        {"name": "explore_mc_pic_recent_interest_cluster_score_alpha", "as": "score_alpha"},
        {"name": "explore_mc_pic_recent_interest_cluster_score_beta", "as": "score_beta"},
        {"name": "explore_mc_pic_recent_interest_cluster_score_sigma", "as": "score_sigma"},
        {"name": "enable_explore_mc_pic_recent_interest_cluster_score_range_limit", "as": "enable_score_range_limit"},
        {"name": "explore_mc_pic_recent_interest_cluster_score_lower_bound", "as": "score_lower_bound"},
        {"name": "explore_mc_pic_recent_interest_cluster_score_upper_bound", "as": "score_upper_bound"},
        {"name": "enable_explore_mc_pic_recent_interest_cluster_boost_only_low_interest_user", "as": "enable_only_low_interest_user"},
        {"name": "enable_explore_mc_pic_recent_interest_cluster_boost_low_interest_user_enhance", "as": "enable_low_interest_user_enhance}"},
        {"name": "explore_mc_pic_recent_interest_cluster_boost_low_interest_user_enhance_base_coef", "as": "enhance_base_coeff"},
        {"name": "explore_mc_pic_recent_interest_cluster_boost_low_interest_user_enhance_weight", "as": "enhance_weight"},
        {"name": "explore_mc_pic_recent_interest_cluster_boost_low_interest_thres", "as": "low_interest_thres"},
        {"name": "uDoubleOutsideValidPicCluster7dList", "as": "user_interest"},
        {"name": "enable_explore_mc_pic_recent_interest_cluster_boost_only_not_interest_cluster", "as": "enable_only_not_interest_cluster"},
      ],
      import_item_attr = [
        "cluster_id_632",
      ],
      export_item_attr = [
        {"name": "recent_interest_score", "as": "explore_mc_pic_recent_interest_score"}
      ],
      function_name = "CalcPicRecentInterestScore",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1
      }
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "explore_mc_pic_recent_interest_score", "as": "boost_discount_coeff"},
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1
      }
    )
    return self
  
  def mc_pic_hot_content_topk_boost(self, score_attr, flag_attr, target_attr):
    """
    Module: picture_queue
    功能: 高热图文topk提权
    Owner: zhongchao03
    Date: 2025-08-25
    :return:
    """
    self.sort(
      score_from_attr = score_attr,
      target_item = {
        flag_attr: 1,
        target_attr: 1
      },
      partial_sort = True,
      partial_num = "{{explore_mc_pic_hot_content_boost_topk}}"
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_mc_pic_hot_content_boost_coef", "as": "boost_discount_coeff"},
        {"name": "explore_mc_pic_hot_content_boost_thres", "as": "boost_discount_thres"},
        {"name": "explore_mc_pic_hot_content_boost_topk", "as": "topk"},
      ],
      import_item_attr = [
        {"name": "mc_ensemble_pctr", "as": "need_item_attr"},
        {"name": score_attr, "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountWithThres",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1,
        target_attr: 1,
      }
    )
    return self

  def prerank_update_bar_boost(self, score_attr, flag_attr):
    self.split_string(
      input_common_attr = "explore_prerank_update_bar_proportion_str",
      output_common_attr = "explore_prerank_update_bar_proportion_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "explore_prerank_update_bar_score_weight_str",
      output_common_attr = "explore_prerank_update_bar_score_weight_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_prerank_update_bar_proportion_list", "as": "update_bar_proportion_list"},
        {"name": "explore_prerank_update_bar_score_weight_list", "as": "update_bar_score_weight_list"},
        {"name": "explore_prerank_update_bar_audit_limit", "as": "update_bar_audit_limit"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
        {"name": "content_safety_level_with_namespace__level_hot_online", "as": "level_hot_online"},
        "upload_time",
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostUpdateTimeBar",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1,
      }
    )
    return self

  def prerank_user_intrest_adjust(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "output_intrest_key_list", "as": "intrest_key_list"},
        {"name": "output_intrest_value_list", "as": "intrest_value_list"},
        {"name": "explore_prerank_user_intrest_adjust_boost_coef", "as": "boost_coef"},
        {"name": "explore_prerank_user_intrest_adjust_discount_coef", "as": "discount_coef"},
        {"name": "explore_enable_hetu1_user_intrest_adjust", "as": "enable_hetu1"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "input_score"},
        "hetu_tag_level_info__hetu_level_two",
        "hetu_tag_level_info__hetu_level_one",
      ],
      export_item_attr = [
        {"name": "output_score", "as": score_attr},
        "intrest_adjust_score"
      ],
      function_name = "IntrestAdjustScore",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1
      }
    )
    return self
    
  def explore_cascade_cs_boost(self, score_attr, flag_attr, stage="prerank"):
    if stage=="prerank":
      kconf_key = "formula.scenarioKey10.explore_cold_combined_score_prerank"
    else:
      kconf_key = "formula.scenarioKey75.explore_cold_combined_score_mc_s2"
    
    self.calc_by_formula1(
      kconf_key = kconf_key,
      import_item_attr = [
        {"name": "explore_stat__real_show_count", "as": "current_impr", "default_val": 0},
        {"name": "explore_stat__click_count", "as": "current_click", "default_val": 0},
        {"name": "cold_item_quality_score", "as": "item_quality_score", "default_val": 0.0},
        {"name": "item_upload_second", "as": "created_second", "default_val": 0},
        {"name": "is_cold_recall", "as": "is_cold_recall", "default_val": 0},
      ],
      export_formula_value = [
        {"name": "cold_combined_score", "as": "explore_cold_photo_score_%s"%stage}
      ],
      abtest_biz_name = "KUAISHOU_APPS",
      perf_tag = "explore_cold_combined_score_%s"%stage,
      target_item = {
        flag_attr: 1,
        "is_same_author_tail": 1
      }
    )
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "explore_cold_photo_score_%s"%stage, "as": "boost_discount_coeff"},
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr},
      ],
      function_name = "BoostOrDiscountWithItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1,
        "is_same_author_tail": 1
      }
    )
    return self

  def explore_first_refresh_good_boost(self, score_attr, flag_attr, stage="prerank"):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_first_refresh_good_boost_weight_%s"%stage, "as": "boost_discount_coeff"}
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        "is_first_refresh_good_photo": 1,
        flag_attr : 1
      }
    ) 
    return self

  def explore_cover_video_not_correlation_deboost(self, score_attr, flag_attr, stage="prerank"):
    self.enrich_attr_by_light_function(
          import_item_attr = [
            "hetu_tag_level_info__hetu_tag"
          ],
          export_item_attr = [
            {"name": "mmu_not_correlation_tag", "as": "mmu_not_correlation_tag"}
          ],
          function_name = "ExploreCoverVideoNotCorrelationTag",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {flag_attr : 1}
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_not_correlation_deboost_weight_%s"%stage, "as": "boost_discount_coeff"}
          ],
          import_item_attr = [
            {"name": score_attr, "as": "score"},
          ],
          export_item_attr = [
            {"name": "score", "as": score_attr},
          ],
          function_name = "BoostOrDiscountV2",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "mmu_not_correlation_tag": 1,
            flag_attr : 1
          }
        ) 
    return self

  def mc_s2_update_bar_boost(self, score_attr, flag_attr):
    self.split_string(
      input_common_attr = "explore_mc_s2_update_bar_proportion_str",
      output_common_attr = "explore_mc_s2_update_bar_proportion_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "explore_mc_s2_update_bar_score_weight_str",
      output_common_attr = "explore_mc_s2_update_bar_score_weight_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_mc_s2_update_bar_proportion_list", "as": "update_bar_proportion_list"},
        {"name": "explore_mc_s2_update_bar_score_weight_list", "as": "update_bar_score_weight_list"},
        {"name": "explore_mc_s2_update_bar_audit_limit", "as": "update_bar_audit_limit"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
        {"name": "content_safety_level_with_namespace__level_hot_online", "as": "level_hot_online"},
        "upload_time",
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostUpdateTimeBar",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1,
      }
    )
    return self
  
  def high_photo_count_author_adjust(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        "high_upload_photo_author_map_ptr",
        {"name": "explore_mc_s2_high_photo_count_author_photo_coeff", "as": "boost_discount_coeff"},
        {"name": "explore_mc_s2_high_photo_count_author_pos_neg_ratio_coeff", "as": "pos_neg_ratio_coeff"},
      ],
      import_item_attr = [
        "author__id",
        {"name": score_attr, "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr}
      ],
      function_name = "HighPhotoCountAuthorPhotoAdjustV2",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def llm_negative_photo_adjust(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_mc_s2_llm_negative_photo_adjust_tag_coeff_map_str", "as": "tag_coeff_map_str"},
      ],
      import_item_attr = [
        "hetu_tag_level_info_v2__hetu_tag",
        "explore_stat__click_count",
        "explore_stat__report_count",
        "fountain_stats__real_show_count",
        "fountain_stats__report_count",
        {"name": score_attr, "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr}
      ],
      function_name = "LlmNegativePhotoAdjust",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def gen_short_window_ctr_coeff(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_all_short_window_ctr_cali_upload_time_limit", "as": "upload_time_limit"},
        {"name": "explore_all_short_window_ctr_cali_coeff_lower_bound", "as": "coeff_lower_bound"},
        {"name": "explore_all_short_window_ctr_cali_coeff_upper_bound", "as": "coeff_upper_bound"},
        {"name": "explore_window_ctr_power_coeff", "as": "window_ctr_power_coeff"}
      ],
      import_item_attr = [
        {"name": "explore_stat__real_show_count", "as": "rc_old"},
        {"name": "explore_stat__click_count", "as": "pc_old"},
        {"name": "rc12h", "as": "rc_new"},
        {"name": "pc12h", "as": "pc_new"},
        "upload_time",
      ],
      export_item_attr = [
        {"name": "ctr_cali_coeff", "as": "short_window_ctr_cali_coeff"}
      ],
      function_name = "CalcShortWindowCtrCaliCoeff",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def cascade_short_window_ctr_cali(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "cascade_corr_pctr", "as": "score"},
        {"name": "short_window_ctr_cali_coeff", "as": "boost_discount_coeff"},
      ],
      export_item_attr = [
        {"name": "score", "as": "cascade_corr_pctr"}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def explore_cal_update_xtr_score_mc_s1(self):
    self.split_string(
      input_common_attr = "explore_update_fix_xtr_weight_mc_s1_str",
      output_common_attr = "explore_update_fix_xtr_weight_mc_s1_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "explore_update_fix_xtr_power_mc_s1_str",
      output_common_attr = "explore_update_fix_xtr_power_mc_s1_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "explore_update_fix_xtr_buttom_mc_s1_str",
      output_common_attr = "explore_update_fix_xtr_buttom_mc_s1_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "explore_update_fix_xtr_upper_mc_s1_str",
      output_common_attr = "explore_update_fix_xtr_upper_mc_s1_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .set_attr_value( 
      no_overwrite=True,
      common_attrs=[
        {
          "name": "explore_mc_s1_update_xtr_name_list",
          "type": "string_list",
          "value": self.update_fix_xtr_name()
        }
      ]
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_update_fix_xtr_weight_mc_s1_list", "as": "update_fix_xtr_weight_list"},
        {"name": "explore_update_fix_xtr_power_mc_s1_list", "as": "update_fix_xtr_power_list"},
        {"name": "explore_update_fix_xtr_buttom_mc_s1_list", "as": "update_fix_xtr_buttom_list"},
        {"name": "explore_update_fix_xtr_upper_mc_s1_list", "as": "update_fix_xtr_upper_list"},
        {"name": "explore_update_window_width_mc_s1", "as": "window_width"},
        {"name": "explore_mc_ensemble_s1_window_duration_ratio", "as": "window_duration_ratio"},
        {"name": "explore_mc_s1_update_xtr_name_list", "as": "fix_xtr_list"},
      ],
      import_item_attr = [
        "upload_time",
        "duration_ms",
        "mc_ensemble_pctr",
        "mc_ensemble_pltr",
        "mc_ensemble_pwtr",
        "mc_ensemble_pcmtr",
        "mc_ensemble_pcltr",
        "mc_ensemble_pwtd_inverse",
        "mc_ensemble_pwatch_time",
        "mc_ensemble_peftr",
        "mc_ensemble_plvtr2",
        "cascade_pctr",
        "mc_ensemble_pftr"
      ],
      export_item_attr = [
        {"name": "update_bar_score", "as": "cascade_update_xtr_fix_mc_s1_score"}
      ],
      function_name = "FixWindowXtr",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def explore_cascade_s2_low_time_active_weight_adjust(self):
    self.calc_by_formula1(
      kconf_key = "formula.scenarioKey58.cascade_s2_low_time_active_weight_adjust_f1",
      import_common_attr = [
        "explore_mc_ensemble_s2_pltr_power_weight",
        "explore_mc_ensemble_s2_pwtr_power_weight",
        "explore_mc_ensemble_s2_pftr_power_weight",
        "explore_mc_ensemble_s2_pcmtr_power_weight",
        "explore_mc_ensemble_s2_pctr_power_weight",
        "explore_mc_ensemble_s2_pwatch_time_power_weight",
        "explore_mc_ensemble_ppwatch_time_raw_weight",
        "active_days_high_time_rate"
      ],
      export_formula_value = [
        {"name": "explore_mc_ensemble_s2_pltr_power_weight", "to_common": True},
        {"name": "explore_mc_ensemble_s2_pwtr_power_weight", "to_common": True},
        {"name": "explore_mc_ensemble_s2_pftr_power_weight", "to_common": True},
        {"name": "explore_mc_ensemble_s2_pcmtr_power_weight", "to_common": True},
        {"name": "explore_mc_ensemble_s2_pctr_power_weight", "to_common": True},
        {"name": "explore_mc_ensemble_s2_pwatch_time_power_weight", "to_common": True},
        {"name": "explore_mc_ensemble_ppwatch_time_raw_weight", "to_common": True}
      ],
      abtest_biz_name = "KUAISHOU_APPS"
    )

    return self

  def explore_cal_mc_ensemble_pftr_dur(self):
    self \
    .if_("explore_mc_ensemble_pltr_dur_social_condition == 1 and (bid_follow_num == 0 or (u_inside_share_active_degree_detail_code > 3))") \
      .gen_common_attr_by_lua(
        attr_map={
          "explore_mc_ensemble_s1_pftr_dur_power_weight_social": "0.0",
          "explore_mc_ensemble_s1_pftr_dur_raw_power_weight_social" : "0.0",
        }
      ) \
    .end_() \
    .if_("explore_mc_ensemble_pltr_dur_social_condition == 2 and (bid_follow_num == 0 or (u_share_num_30d == 0 and u_message_active_degree ~= 5 and u_message_active_degree ~= 6))") \
      .gen_common_attr_by_lua(
        attr_map={
          "explore_mc_ensemble_s1_pftr_dur_power_weight_social": "0.0",
          "explore_mc_ensemble_s1_pftr_dur_raw_power_weight_social" : "0.0",
        }
      ) \
    .end_() \
    .if_("explore_mc_ensemble_pltr_dur_social_condition == 3 and (bid_follow_num == 0 or (user_msg_cnt_ssm_today + user_msg_cnt_gsm_today > 0))") \
      .gen_common_attr_by_lua(
        attr_map={
          "explore_mc_ensemble_s1_pftr_dur_power_weight_social": "0.0",
          "explore_mc_ensemble_s1_pftr_dur_raw_power_weight_social" : "0.0",
        }
      ) \
    .end_() \
    .split_string(
      input_common_attr = "explore_mc_pftr_dur_percentile_str",
      output_common_attr = "explore_mc_pftr_dur_percentile_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_mc_pftr_dur_percentile_list", "as": "percentile_list"},
        {"name": "explore_mc_pftr_dur_gama", "as": "gama"},
        {"name": "explore_mc_pftr_dur_threshold", "as": "threshold"}
      ],
      import_item_attr = [
        {"name": "duration_ms", "as": "duration"},
        {"name": "mc_ensemble_pftr", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "mc_ensemble_pftr_dur_social"},
      ],
      function_name = "CalculateCascadePftrDurScore",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def update_fix_xtr_name(self):
    update_fix_xtrs = [
      "mc_ensemble_pctr",
      "mc_ensemble_pltr",
      "mc_ensemble_pwtr",
      "mc_ensemble_pcmtr",
      "mc_ensemble_pcltr",
      "mc_ensemble_pwtd_inverse",
      "mc_ensemble_pwatch_time",
      "mc_ensemble_peftr",
      "mc_ensemble_plvtr2",
      "cascade_pctr",
      "mc_ensemble_pftr"
    ]
    return update_fix_xtrs
  
  def mc_s2_all_page_interest_boost(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_mc_s2_all_page_valid_interest_boost_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1,
        "is_all_page_valid_interest" : 1,
      }
    )
    return self

  def gen_is_sexy_induce_photo(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "sexy_induce_photo_set_ptr", "as": "aid_set_ptr"}
      ],
      import_item_attr = [
        "author__id"
      ],
      export_item_attr = [
        {"name": "is_target_photo", "as": "is_sexy_induce_photo"}
      ],
      function_name = "AidInSet",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def extract_hetu_info_tag_for_llm(self,flag_attr):
    self.split_string(
      input_common_attr = "explore_tag_llm_negative_set_str",
      output_common_attr = "explore_tag_llm_negative_set_str_list",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        "hetu_tag_level_info_v2__hetu_tag"
      ],
      import_common_attr = [
        {"name": "explore_tag_llm_negative_set_str_list", "as": "tag_llm_negative_set_list"},
      ],
      export_item_attr = [
        {"name": "hetu_target_info_tag", "as": "hetu_info_for_llm_negative"}
      ],
      function_name = "ExtractHetuInfoTag",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1,
      }
    )
    return self

  def mc_llm_negative_photo_personal_adjust(self,score_attr, flag_attr):
    self.calc_by_formula1(
      kconf_key = "formula.scenarioKey21.McExploreLlmNeagtivePhotoDeboost",
      import_item_attr = [
        "hetu_info_for_llm_negative",
        "explore_stat__click_count",
        "explore_stat__report_count",
        "fountain_stats__real_show_count",
        "fountain_stats__report_count",
      ],
      import_common_attr = [
        "uToleranceScoreKV"
      ],
      export_formula_value = [
        {"name": "final_score", "as": "final_llm_explore_personal_score"}
      ],
      abtest_biz_name = "KUAISHOU_APPS"
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": score_attr, "as": "score"},
        {"name": "final_llm_explore_personal_score", "as": "boost_discount_coeff"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1,
      }
    )
    return self

  def mc_s2_sexy_induce_deboost(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_mc_s2_sexy_induce_deboost_coeff", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1,
        "is_sexy_induce_photo" : 1,
      }
    )
    return self

  def mc_s2_sexy_induce_personal_deboost(self, score_attr, flag_attr):
    self.calc_by_formula1(
      kconf_key = "formula.scenarioKey61.ExploreSexyInduceDeboost",
      import_item_attr = [
      ],
      import_common_attr = [
        "uSexyInterestScore",
      ],
      export_formula_value = [
        {"name": "final_score", "as": "final_sexy_deboost_score", "to_common": True}
      ],
      abtest_biz_name = "KUAISHOU_APPS"
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "final_sexy_deboost_score", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1,
        "is_sexy_induce_photo" : 1,
      }
    )
    return self

  def explore_cal_upload_xtr_score_mc_s2(self, flag_attr):
    self.split_string(
      input_common_attr = "explore_upload_fix_xtr_weight_mc_s2_str",
      output_common_attr = "explore_upload_fix_xtr_weight_mc_s2_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "explore_upload_fix_xtr_power_mc_s2_str",
      output_common_attr = "explore_upload_fix_xtr_power_mc_s2_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "explore_upload_fix_xtr_buttom_mc_s2_str",
      output_common_attr = "explore_upload_fix_xtr_buttom_mc_s2_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "explore_upload_fix_xtr_upper_mc_s2_str",
      output_common_attr = "explore_upload_fix_xtr_upper_mc_s2_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .set_attr_value( 
      no_overwrite=True,
      common_attrs=[
        {
          "name": "explore_mc_s2_update_xtr_name_list",
          "type": "string_list",
          "value": self.update_fix_xtr_name()
        }
      ]
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_upload_fix_xtr_weight_mc_s2_list", "as": "update_fix_xtr_weight_list"},
        {"name": "explore_upload_fix_xtr_power_mc_s2_list", "as": "update_fix_xtr_power_list"},
        {"name": "explore_upload_fix_xtr_buttom_mc_s2_list", "as": "update_fix_xtr_buttom_list"},
        {"name": "explore_upload_fix_xtr_upper_mc_s2_list", "as": "update_fix_xtr_upper_list"},
        {"name": "explore_upload_window_width_mc_s2", "as": "window_width"},
        {"name": "explore_mc_ensemble_s2_window_upload_ratio", "as": "window_duration_ratio"},
        {"name": "explore_mc_s2_update_xtr_name_list", "as": "fix_xtr_list"},
      ],
      import_item_attr = [
        "upload_time",
        "duration_ms",
        "mc_ensemble_pctr",
        "mc_ensemble_pltr",
        "mc_ensemble_pwtr",
        "mc_ensemble_pcmtr",
        "mc_ensemble_pcltr",
        "mc_ensemble_pwtd_inverse",
        "mc_ensemble_pwatch_time",
        "mc_ensemble_peftr",
        "mc_ensemble_plvtr2",
        "cascade_pctr",
        "mc_ensemble_pftr"
      ],
      export_item_attr = [
        {"name": "update_bar_score", "as": "cascade_upload_xtr_fix_mc_s2_score"}
      ],
      function_name = "FixWindowXtr",
      class_name = "ExploreLightFunctionSetV2",
      targer_item = {
        flag_attr : 1
      }
    )
    return self
  
  def explore_cal_hetu_one_debias_score_mc_s2(self, flag_attr):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "hetu_tag_level_info__hetu_level_one", "as": "extract_hetu_tag_list"},
      ],
      export_item_attr = [
        {"name": "first_hetu_tag", "as": "hetu_level_one_top1"},
      ],
      function_name = "ExtractFirstHetuTag",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .split_string(
      input_common_attr = "explore_hetu_one_debias_xtr_weight_mc_s2_str",
      output_common_attr = "explore_hetu_one_debias_xtr_weight_mc_s2_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "explore_hetu_one_debias_xtr_power_mc_s2_str",
      output_common_attr = "explore_hetu_one_debias_xtr_power_mc_s2_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "explore_hetu_one_debias_xtr_buttom_mc_s2_str",
      output_common_attr = "explore_hetu_one_debias_xtr_buttom_mc_s2_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "explore_hetu_one_debias_xtr_upper_mc_s2_str",
      output_common_attr = "explore_hetu_one_debias_xtr_upper_mc_s2_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .set_attr_value( 
      no_overwrite=True,
      common_attrs=[
        {
          "name": "explore_mc_s2_hetu_one_debias_xtr_name_list",
          "type": "string_list",
          "value": self.update_fix_xtr_name()
        }
      ]
    ) \
    .if_("enable_mc_s2_hetu_debias_target_limit == 1")  \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_hetu_one_debias_xtr_weight_mc_s2_list", "as": "id_debias_xtr_weight_list"},
          {"name": "explore_hetu_one_debias_xtr_power_mc_s2_list", "as": "id_debias_xtr_power_list"},
          {"name": "explore_hetu_one_debias_xtr_buttom_mc_s2_list", "as": "id_debias_xtr_buttom_list"},
          {"name": "explore_hetu_one_debias_xtr_upper_mc_s2_list", "as": "id_debias_xtr_upper_list"},
          {"name": "explore_mc_s2_hetu_one_debias_xtr_name_list", "as": "fix_xtr_list"},
        ],
        import_item_attr = [
          {"name": "hetu_level_one_top1", "as": "debias_id_feature"},
          "mc_ensemble_pctr",
          "mc_ensemble_pltr",
          "mc_ensemble_pwtr",
          "mc_ensemble_pcmtr",
          "mc_ensemble_pcltr",
          "mc_ensemble_pwtd_inverse",
          "mc_ensemble_pwatch_time",
          "mc_ensemble_peftr",
          "mc_ensemble_plvtr2",
          "cascade_pctr",
          "mc_ensemble_pftr"
        ],
        export_item_attr = [
          {"name": "debias_score", "as": "cascade_hetu_one_xtr_debias_mc_s2_score"}
        ],
        function_name = "GenXtrScoreByIdFeature",
        class_name = "ExploreLightFunctionSetV2",
        targer_item = {
          flag_attr : 1
        }
      ) \
    .else_() \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_hetu_one_debias_xtr_weight_mc_s2_list", "as": "id_debias_xtr_weight_list"},
          {"name": "explore_hetu_one_debias_xtr_power_mc_s2_list", "as": "id_debias_xtr_power_list"},
          {"name": "explore_hetu_one_debias_xtr_buttom_mc_s2_list", "as": "id_debias_xtr_buttom_list"},
          {"name": "explore_hetu_one_debias_xtr_upper_mc_s2_list", "as": "id_debias_xtr_upper_list"},
          {"name": "explore_mc_s2_hetu_one_debias_xtr_name_list", "as": "fix_xtr_list"},
        ],
        import_item_attr = [
          {"name": "hetu_level_one_top1", "as": "debias_id_feature"},
          "mc_ensemble_pctr",
          "mc_ensemble_pltr",
          "mc_ensemble_pwtr",
          "mc_ensemble_pcmtr",
          "mc_ensemble_pcltr",
          "mc_ensemble_pwtd_inverse",
          "mc_ensemble_pwatch_time",
          "mc_ensemble_peftr",
          "mc_ensemble_plvtr2",
          "cascade_pctr",
          "mc_ensemble_pftr"
        ],
        export_item_attr = [
          {"name": "debias_score", "as": "cascade_hetu_one_xtr_debias_mc_s2_score"}
        ],
        function_name = "GenXtrScoreByIdFeature",
        class_name = "ExploreLightFunctionSetV2"
      ) \
    .end_()
    return self

  def sort_and_cal_slide_pxtr(self, strategies):
    for strategy_prefix, sort_base_name, numerator_name, denominator_name, pxtr_name in strategies:
      mul_switch_attr_name = "explore_enable_" + strategy_prefix + "_mul_universal_score"
      type_attr_name = "explore_" + sort_base_name + "_type"
      win_size_attr_name = "explore_" + strategy_prefix + "_window_size"
      numerator_base_name = numerator_name + "_base"
      denominator_base_name = denominator_name + "_base"
      slide_score_name = strategy_prefix + "_score"
      self \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": mul_switch_attr_name, "as": "enable_mul_universal_score"},
            {"name": type_attr_name, "as": "input_attr_type"},
            {"name": win_size_attr_name, "as": "window_size"},
            {"name": numerator_base_name, "as": "pc_base"},
            {"name": denominator_base_name, "as": "rc_base"},
          ],
          import_item_attr = [
            {"name": denominator_name, "as": "rc"},
            {"name": numerator_name, "as": "pc"},
            {"name": pxtr_name, "as": "origin_pxtr"},
          ],
          export_item_attr = [
            {"name": "slide_pxtr_score", "as": slide_score_name}
          ],
          function_name = "CalcSlidePxtrScore",
          class_name = "ExploreLightFunctionSetV2",
        )
    return self
  
  def gen_u2a_author_circle_cluster_id(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "u2a_author_id_circle_id_detail_kuaishou_ptr", "as": "map_ptr"}
      ],
      import_item_attr = [
        {"name": "author__id", "as": "key_attr"}
      ],
      export_item_attr = [
        {"name": "target_item_attr", "as": "author_circle_cluster_id"},
      ],
      function_name = "GetItemAttrByIntToIntMapPtr",
      class_name = "ExploreLightFunctionSetV2"
    )
    return self

  def explore_cal_hetu_one_debias_score_mc_s1(self):
    self.split_string(
      input_common_attr = "explore_hetu_one_debias_xtr_weight_mc_s1_str",
      output_common_attr = "explore_hetu_one_debias_xtr_weight_mc_s1_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "explore_hetu_one_debias_xtr_power_mc_s1_str",
      output_common_attr = "explore_hetu_one_debias_xtr_power_mc_s1_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "explore_hetu_one_debias_xtr_buttom_mc_s1_str",
      output_common_attr = "explore_hetu_one_debias_xtr_buttom_mc_s1_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "explore_hetu_one_debias_xtr_upper_mc_s1_str",
      output_common_attr = "explore_hetu_one_debias_xtr_upper_mc_s1_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .set_attr_value( 
      no_overwrite=True,
      common_attrs=[
        {
          "name": "explore_mc_s1_hetu_one_debias_xtr_name_list",
          "type": "string_list",
          "value": self.update_fix_xtr_name()
        }
      ]
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_hetu_one_debias_xtr_weight_mc_s1_list", "as": "id_debias_xtr_weight_list"},
        {"name": "explore_hetu_one_debias_xtr_power_mc_s1_list", "as": "id_debias_xtr_power_list"},
        {"name": "explore_hetu_one_debias_xtr_buttom_mc_s1_list", "as": "id_debias_xtr_buttom_list"},
        {"name": "explore_hetu_one_debias_xtr_upper_mc_s1_list", "as": "id_debias_xtr_upper_list"},
        {"name": "explore_mc_s1_hetu_one_debias_xtr_name_list", "as": "fix_xtr_list"},
      ],
      import_item_attr = [
        {"name": "hetu_level_one_first_tag", "as": "debias_id_feature"},
        "mc_ensemble_pctr",
        "mc_ensemble_pltr",
        "mc_ensemble_pwtr",
        "mc_ensemble_pcmtr",
        "mc_ensemble_pcltr",
        "mc_ensemble_pwtd_inverse",
        "mc_ensemble_pwatch_time",
        "mc_ensemble_peftr",
        "mc_ensemble_plvtr2",
        "cascade_pctr",
        "mc_ensemble_pftr"
      ],
      export_item_attr = [
        {"name": "debias_score", "as": "cascade_hetu_one_xtr_debias_mc_s1_score"}
      ],
      function_name = "GenXtrScoreByIdFeature",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def gen_hetu_first_tag(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "hetu_tag_level_info__hetu_level_one", "as": "extract_hetu_tag_list"},
      ],
      export_item_attr = [
        {"name": "first_hetu_tag", "as": "hetu_level_one_first_tag"},
      ],
      function_name = "ExtractFirstHetuTag",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self
  
  def gen_is_good_author_pool_photo(self):
    self.set_attr_value(
      no_overwrite=True,
      item_attrs=[
        {
          "name": "is_good_author_pool_photo",
          "type": "int",
          "value": 1
        }
      ],
      select_item = {
        "join": "or",
        "filters": [{
          "attr_name": "original_submission_author_tag",
          "select_if": ">",
          "compare_to": "{{explore_good_author_submission_author_tag_limit}}",
        },
        {
          "attr_name": "personalization_author_tag",
          "select_if": ">",
          "compare_to": "{{explore_good_author_personalization_author_tag_limit}}",
        },
        {
          "attr_name": "userfulness_author_tag",
          "select_if": ">",
          "compare_to": "{{explore_good_author_userfulness_author_tag_limit}}",
        },
        {
          "attr_name": "author_grade_key",
          "select_if": ">=",
          "compare_to": "{{explore_good_author_author_grade_key_limit}}",
        },
        {
          "attr_name": "topk_audit_level",
          "select_if": ">",
          "compare_to": "{{explore_good_author_topk_audit_level_limit}}",
        },
        {
          "attr_name": "audit_hot_high_tag_level",
          "select_if": ">",
          "compare_to": "{{explore_good_author_audit_hot_high_tag_level_limit}}",
        },
        {
          "attr_name": "content_safety_level_with_namespace__level_hot_online",
          "select_if": ">",
          "compare_to": "{{explore_good_author_content_safety_level_with_namespace__level_hot_online_limit}}",
        }]
      }  
    )
    return self

  def gen_is_first_refresh_good_photo(self):
    self.set_attr_value(
      item_attrs=[
        {
          "name": "is_first_refresh_good_photo",
          "type": "int",
          "value": 1
        }
      ],
      target_item = {"is_follow_author": 1}
    )
    return self

  def gen_is_picture_follow_author(self):
    self.copy_attr(
      attrs = [{
        "from_item": "is_follow_author",
        "to_item": "is_picture_follow_author",
      }],
      target_item = {
        "is_picture": 1,
        "is_follow_author": 1
      }
    )
    return self

  def cal_mc_s2_es_score_f1(self, score_attr, flag_attr):
    self.calc_by_formula1(
      kconf_key = "formula.scenarioKey63.ExploreMcS2EnsembleSort",
      import_item_attr = [
        "mc_ensemble_pwatch_time",
        "mc_ensemble_pctr",
        "mc_interact_fusion_score",
        "mc_ensemble_pltr",
        "mc_ensemble_pwtr",
        "mc_ensemble_pcltr",
        "mc_ensemble_pcmtr"
      ],
      import_common_attr = [
        "explore_today_vv",
        "active_days_avg_vv",
        "uExploreActiveDays",
        "user_explore_last_like_gap_hour",
        "user_explore_last_follow_gap_hour",
        "user_explore_last_comment_gap_hour",
        "user_explore_last_collect_gap_hour"
      ],
      export_formula_value = [
        {"name": "final_score", "as": "mc_s2_es_score_f1"}
      ],
      abtest_biz_name = "KUAISHOU_APPS",
      target_item = {
        flag_attr : 1
      }
    ) \
    .item_attr_operation(
      item_attr_a = score_attr,
      item_attr_b = "mc_s2_es_score_f1",
      operator = "*",
      output_attr = score_attr,
      target_item = {
        flag_attr : 1
      }
    )
    return self

  def cal_mc_s2_unaudit_deboost_score_f1(self, score_attr, flag_attr):
    self.calc_by_formula1(
      kconf_key = "formula.scenarioKey62.ExploreMcS2AuditScore",
      import_item_attr = [
        "explore_mc_bad_sense_similarity_score",
        "explore_mc_bad_hot_audit_similarity_score",
        "explore_mc_bad_cover_similarity_score"
      ],
      import_common_attr = [
        "active_days_gt_5min_rate",
        "uExploreActiveDays"
      ],
      export_formula_value = [
        {"name": "final_score", "as": "mc_s2_unaudit_deboost_score_f1"}
      ],
      abtest_biz_name = "KUAISHOU_APPS",
      target_item = {
        flag_attr : 1
      }
    ) \
    .item_attr_operation(
      item_attr_a = score_attr,
      item_attr_b = "mc_s2_unaudit_deboost_score_f1",
      operator = "*",
      output_attr = score_attr,
      target_item = {
        flag_attr : 1
      }
    )
    return self
  def mc_user_vv_ensemble_power_weight_adjust(self, weight, weight_adjust_exp_upper, weight_adjust_alpha, weight_adjust_beta, weight_adjust_omega, weight_adjust_max, weight_adjust_min):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": weight, "as": "xtr_weight"},
        {"name": "explore_recent_valid_click_count", "as": "user_vv"},
        {"name": weight_adjust_exp_upper, "as": "exp_upper"},
        {"name": weight_adjust_alpha, "as": "alpha"},
        {"name": weight_adjust_beta, "as": "beta"},
        {"name": weight_adjust_omega, "as": "omega"},
        {"name": weight_adjust_max, "as": "coeff_max"},
        {"name": weight_adjust_min, "as": "coeff_min"},
      ],
      export_common_attr = [
        {"name": "xtr_weight", "as": weight},
      ],
      function_name = "AdjustWeightByUserVv",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self
  
  def pic_phtr_mc_filter(self, flag_attr):
    self.calc_by_formula1(
      kconf_key = "formula.scenarioKey96.explore_pic_mc_pxtr_filter_f1",
      import_common_attr = [
        {"name": "explore_pic_phtr_mc_filter_rate", "as": "filter_rate"},
        {"name": "explore_pic_phtr_mc_filter_age_thresh", "as": "age_thresh"},
      ],
      import_item_attr = [
        "cascade_pctr",
        "cascade_pltr",
        "cascade_pwtr",
        "cascade_pftr",
        "cascade_pcltr",
        "cascade_pcmtr",
        "cascade_pdtr",
        "cascade_phtr",
        "cascade_psvtr",
        "mc_ensemble_pctr",
        "mc_ensemble_pltr",
        "mc_ensemble_pftr",
        "mc_ensemble_pwtr",
        "mc_ensemble_pcltr",
        "mc_ensemble_pcmtr",
        "mc_ensemble_psvtr"
      ],
      export_formula_value = [
        {"name": "need_filter", "as": "is_phtr_mc_filter_pic", "to_int": True},
      ],
      abtest_biz_name = "KUAISHOU_APPS",
      perf_tag = "{{explore_pic_mc_pxtr_filter_f1_perf_tag}}",
      target_item = {
        flag_attr : 1
      }
    )
    return self

  def gen_is_ugc_photo(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        "data_set_tags_bit"
      ],
      export_item_attr = [
        "is_ugc_photo"
      ],
      function_name = "IsUgcPhoto",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def short_term_photo_tagnex_score_adjust(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "short_term_item_tagnex_score", "as": "boost_discount_coeff"},
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1
      }
    )
    return self

  def short_term_photo_cluster_id_score_adjust(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "short_term_item_cluster_id_score", "as": "boost_discount_coeff"},
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1
      }
    )
    return self

  def short_term_photo_hetu_level2_score_adjust(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "short_term_item_hetu_level2_score", "as": "boost_discount_coeff"},
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1
      }
    )
    return self

  def short_term_photo_hashtag_score_adjust(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "short_term_item_hashtag_score", "as": "boost_discount_coeff"},
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1
      }
    )
    return self

  def short_term_photo_hetu_tag_score_adjust(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "short_term_item_hetu_tag_score", "as": "boost_discount_coeff"},
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1
      }
    )
    return self

  def short_term_photo_interest_community_tag_score_adjust(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "short_term_item_interest_community_tag_score", "as": "boost_discount_coeff"},
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1
      }
    )
    return self

  def short_term_photo_sid_score_adjust(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "short_term_item_sid_score", "as": "boost_discount_coeff"},
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1
      }
    )
    return self

  def mc_s2_user_recent_hate_count_ensemble_koc_htr_power_weight_adjust(self, weight, weight_adjust_exp_upper, weight_adjust_alpha, weight_adjust_beta, weight_adjust_omega, weight_adjust_max, weight_adjust_min):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": weight, "as": "xtr_weight"},
        {"name": "recent_hate_count", "as": "user_vv"},
        {"name": weight_adjust_exp_upper, "as": "exp_upper"},
        {"name": weight_adjust_alpha, "as": "alpha"},
        {"name": weight_adjust_beta, "as": "beta"},
        {"name": weight_adjust_omega, "as": "omega"},
        {"name": weight_adjust_max, "as": "coeff_max"},
        {"name": weight_adjust_min, "as": "coeff_min"},
      ],
      export_common_attr = [
        {"name": "xtr_weight", "as": weight},
      ],
      function_name = "AdjustWeightByUserVv",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self
  
  def explore_mc_s2_cal_svtr_rid_ctr_score(self, flag_attr):
    self.item_attr_operation(
      item_attr_a = "cascade_psvtr",
      common_attr_b = "{{explore_mc_s2_svtr_shift_coef}}",
      operator = "+",
      output_attr = "cascade_shift_svtr",
      target_item = {
        flag_attr : 1
      }
    ) \
    .item_attr_operation(
      item_attr_a = "cascade_corr_pctr",
      common_attr_b = "{{explore_mc_s2_ctr_shift_coef}}",
      operator = "+",
      output_attr = "cascade_shift_ctr",
      target_item = {
        flag_attr : 1
      }
    ) \
    .item_attr_operation(
      item_attr_a = "cascade_shift_svtr",
      item_attr_b = "cascade_shift_ctr",
      operator = "/",
      output_attr = "cascade_svtr_rid_ctr_mc_s2_score",
      target_item = {
        flag_attr : 1
      }
    )
    return self

  def gen_is_reason_top_photo(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_reason_top_photo_white_list", "as": "reason_white_list"},
        {"name": "explore_reason_top_photo_top_k", "as": "top_k"},
      ],
      export_item_attr = [
        {"name": "is_reason_top_photo", "as": "is_directly_reach_fullrank"},
      ],
      function_name = "CalIsReasonTopPhoto",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def cal_user_stage_interest_tagnex_tgi(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_user_stage_interest_tagnex_tgi_list", "as": "match_list"},
        {"name": "explore_user_stage_interest_tagnex_tgi_coeff", "as": "coeff"},
        {"name": "explore_user_stage_interest_tagnex_tgi_bias", "as": "bias"},
        {"name": "explore_user_stage_interest_tagnex_circle_attr_min", "as": "attr_min"},
        {"name": "explore_user_stage_interest_tagnex_circle_attr_max", "as": "attr_max"},
        {"name": "explore_user_stage_interest_tagnex_circle_use_single_match_item", "as": "use_single_match_item"},
      ],
      import_item_attr = [
        {"name" : "hetu_tag_level_info__hetu_tag", "as" : "hetu_tag"},
      ],
      export_item_attr = [
        {"name": "match_score", "as": "user_stage_interest_tagnex_tgi_score"}
      ],
      function_name = "CalMatchScore",
      class_name = "ExploreLightFunctionSetV2",
    ) 
    return self

  def cal_user_career_interest_tagnex_tgi(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_user_career_interest_tagnex_tgi_list", "as": "match_list"},
        {"name": "explore_user_career_interest_tagnex_tgi_coeff", "as": "coeff"},
        {"name": "explore_user_career_interest_tagnex_tgi_bias", "as": "bias"},
        {"name": "explore_user_career_interest_tagnex_circle_attr_min", "as": "attr_min"},
        {"name": "explore_user_career_interest_tagnex_circle_attr_max", "as": "attr_max"},
        {"name": "explore_user_career_interest_tagnex_circle_use_single_match_item", "as": "use_single_match_item"},
      ],
      import_item_attr = [
        {"name" : "hetu_tag_level_info__hetu_tag", "as" : "hetu_tag"},
      ],
      export_item_attr = [
        {"name": "match_score", "as": "user_career_interest_tagnex_tgi_score"}
      ],
      function_name = "CalMatchScore",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def cal_user_age_interest_tagnex_tgi(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_user_age_interest_tagnex_tgi_list", "as": "match_list"},
        {"name": "explore_user_age_interest_tagnex_tgi_coeff", "as": "coeff"},
        {"name": "explore_user_age_interest_tagnex_tgi_bias", "as": "bias"},
        {"name": "explore_user_age_interest_tagnex_circle_attr_min", "as": "attr_min"},
        {"name": "explore_user_age_interest_tagnex_circle_attr_max", "as": "attr_max"},
        {"name": "explore_user_age_interest_tagnex_circle_use_single_match_item", "as": "use_single_match_item"},
      ],
      import_item_attr = [
        {"name" : "hetu_tag_level_info__hetu_tag", "as" : "hetu_tag"}
      ],
      export_item_attr = [
        {"name": "match_score", "as": "user_age_interest_tagnex_tgi_score"}
      ],
      function_name = "CalMatchScore",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def prerank_operation_pic_boost(self, score_attr, flag_attr, target_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_prerank_operation_pic_boost_coeff", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1,
        target_attr: 1
      }
    )
    return self

  def mc_operation_pic_boost(self, score_attr, flag_attr, target_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_mc_operation_pic_boost_coef", "as": "boost_discount_coeff"},
        {"name": "explore_mc_operation_pic_boost_thres", "as": "boost_discount_thres"},
        {"name": "explore_mc_operation_pic_boost_topk", "as": "topk"},
      ],
      import_item_attr = [
        {"name": "mc_ensemble_pctr", "as": "need_item_attr"},
        {"name": score_attr, "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountWithThres",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1,
        target_attr: 1,
      }
    )
    return self

  def prerank_pic_type_boost(self, score_attr, flag_attr):
    self.if_("enable_explore_prerank_pic_type_age_boost == 1") \
      .calc_by_formula1(
        kconf_key = "formula.scenarioKey42.PrerankExploreAgePictureTypeBoost",
        import_common_attr = ["basic_info_age_segment_v2"],
        import_item_attr = [
          "picture_type",
          score_attr
        ],
        export_formula_value = [
          {"name": "final_score", "as": score_attr}
        ],
        abtest_biz_name = "KUAISHOU_APPS",
        target_item = { flag_attr: 1 }
      ) \
    .else_() \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_prerank_single_pic_boost_coeff", "as": "single_boost_weight"},
          {"name": "explore_prerank_long_pic_boost_coeff", "as": "long_boost_weight"},
          {"name": "explore_prerank_pic_set_boost_coeff", "as": "set_boost_weight"},
        ],
        import_item_attr = [
          "picture_type",
          {"name": score_attr, "as": "ensemble_score"},
        ],
        export_item_attr = [
          {"name": "ensemble_score", "as": score_attr},
        ],
        function_name = "PictureTypeEsBoost",
        class_name = "ExploreLightFunctionSetV2",
        target_item = { flag_attr: 1 }
      ) \
    .end_()
    return self

  def prerank_pic_recent_repeated_realshow_deboost(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        "user_info_ptr",
        {"name": "uStandardRealShowPicAllIdList", "as": "realshow_pic_ids"},
        {"name": "explore_prerank_pic_recent_repeated_realshow_deboost_mode",         "as": "mode"},
        {"name": "explore_prerank_pic_recent_repeated_realshow_deboost_power_weight", "as": "power_weight"},
        {"name": "explore_prerank_pic_recent_repeated_realshow_history_num",          "as": "history_num"},
        {"name": "explore_prerank_pic_recent_repeated_realshow_time_gap_min",         "as": "time_gap_min"},
        {"name": "explore_prerank_pic_recent_repeated_realshow_only_unclick",         "as": "only_unclick"},
        {"name": "explore_prerank_pic_recent_repeated_realshow_tagnex_min",           "as": "tagnex_min"},
        {"name": "explore_prerank_pic_recent_repeated_realshow_tagnex_max",           "as": "tagnex_max"},
        {"name": "explore_prerank_pic_recent_repeated_realshow_repeat_threshold",     "as": "repeat_threshold"},
      ],
      import_item_attr = [
        "hetu_tag_level_info__hetu_level_one",
        "hetu_tag_level_info__hetu_level_two",
        "hetu_tag_level_info__hetu_tag",
        {"name": score_attr, "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr},
      ],
      function_name = "PicRecentRepeatedRealshowEsDeboost",
      class_name = "ExploreLightFunctionSetV2",
      target_item = { flag_attr: 1 }
    )
    return self

  def mc_pic_type_boost(self, score_attr, flag_attr):
    self.if_("enable_explore_mc_pic_type_age_boost == 1") \
      .calc_by_formula1(
        kconf_key = "formula.scenarioKey38.McExploreAgePictureTypeBoost",
        import_common_attr = ["basic_info_age_segment_v2"],
        import_item_attr = [
          "picture_type",
          score_attr
        ],
        export_formula_value = [
          {"name": "final_score", "as": score_attr}
        ],
        abtest_biz_name = "KUAISHOU_APPS",
        target_item = { flag_attr: 1 }
      ) \
    .else_() \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_mc_single_pic_boost_coeff", "as": "single_boost_weight"},
          {"name": "explore_mc_long_pic_boost_coeff", "as": "long_boost_weight"},
          {"name": "explore_mc_pic_set_boost_coeff", "as": "set_boost_weight"},
        ],
        import_item_attr = [
          "picture_type",
          {"name": score_attr, "as": "ensemble_score"},
        ],
        export_item_attr = [
          {"name": "ensemble_score", "as": score_attr},
        ],
        function_name = "PictureTypeEsBoost",
        class_name = "ExploreLightFunctionSetV2",
        target_item = { flag_attr: 1 }
      ) \
    .end_()
    return self
  
  def cal_rise_follow_boost_light_score(self):
    self.split_string(
      input_common_attr = "explore_boost_follow_xtr_weight_mc_s2_str",
      output_common_attr = "explore_boost_follow_xtr_weight_mc_s2_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "explore_boost_follow_xtr_alpha_mc_s2_str",
      output_common_attr = "explore_boost_follow_xtr_alpha_mc_s2_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "explore_boost_follow_xtr_beta_mc_s2_str",
      output_common_attr = "explore_boost_follow_xtr_beta_mc_s2_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .set_attr_value(
      no_overwrite=True,
      common_attrs=[
        {
          "name": "explore_mc_s2_boost_follow_xtr_name_list",
          "type": "string_list",
          "value": self.boost_follow_xtr_name()
        }
      ]
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_cal_rise_pwtr_reshape_alpha", "as": "reshape_alpha"},
        {"name": "explore_cal_rise_pwtr_reshape_max_value", "as": "reshape_max_value"},
      ],
      import_item_attr = [
        "cascade_pwtr",
      ],
      export_item_attr = [
        "cascade_reshape_pwtr",
      ],
      function_name = "CalSigmoidReshapeScore",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_boost_follow_xtr_weight_mc_s2_list", "as": "boost_follow_xtr_weight_list"},
        {"name": "explore_boost_follow_xtr_alpha_mc_s2_list", "as": "boost_follow_xtr_alpha_list"},
        {"name": "explore_boost_follow_xtr_beta_mc_s2_list", "as": "boost_follow_xtr_beta_list"},
        {"name": "explore_mc_s2_boost_follow_xtr_name_list", "as": "boost_follow_xtr_list"},
      ],
      import_item_attr = [
        "cascade_reshape_pwtr",
        "cascade_pwtr",
        "mc_ensemble_pwtd_inverse",
        "mc_ensemble_pwatch_time",
        "cascade_pctr"
      ],
      export_item_attr = [
        {"name": "cascade_follow_score", "as": "cascade_rise_follow_boost_score"}
      ],
      function_name = "CalNewFollowBoostScore",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def boost_follow_xtr_name(self):
    update_fix_xtrs = [
      "cascade_reshape_pwtr",
      "cascade_pwtr",
      "mc_ensemble_pwtd_inverse",
      "mc_ensemble_pwatch_time",
      "cascade_pctr"
    ]
    return update_fix_xtrs

  def explore_cascade_interest_card_photo_score_adjust(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "interest_card_adjust_score", "as": "boost_discount_coeff"},
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1
      }
    )
    return self

  def cal_user_no_bias_interest_tagnex_tgi(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_no_bias_interest_tagnex_tgi_prefix", "as": "key_prefix"},
        "basic_info_age_segment_v2",
        "basic_info_gender_v2",
      ],
      export_common_attr = [
        {"name": "user_age_gender_key", "as": "no_bias_interest_tagnex_tgi_user_key"}
      ],
      function_name = "GetUserAgeGenderKey",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .get_kconf_params(
      kconf_configs = [{
        "kconf_key": "reco.offline.noBiasInterestTagnexTgiStat",
        "json_path": "{{no_bias_interest_tagnex_tgi_user_key}}",
        "value_type": "list_int64",
        "default_value": [],
        "export_common_attr": "explore_no_bias_interest_tagnex_tgi_list"
      }]
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_no_bias_interest_tagnex_tgi_list", "as": "match_list"},
        {"name": "explore_no_bias_interest_tagnex_tgi_coeff", "as": "coeff"},
        {"name": "explore_no_bias_interest_tagnex_tgi_bias", "as": "bias"},
        {"name": "explore_no_bias_interest_tagnex_circle_attr_min", "as": "attr_min"},
        {"name": "explore_no_bias_interest_tagnex_circle_attr_max", "as": "attr_max"},
        "explore_no_bias_tagnex_lv3_to_lv2_map_ptr",
      ],
      import_item_attr = [
        {"name": "hetu_tag_level_info__hetu_tag", "as": "hetu_tag"},
      ],
      export_item_attr = [
        {"name": "match_score", "as": "user_no_bias_interest_tagnex_tgi_score"}
      ],
      function_name = "CalTagnexLv2MatchScore",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self
