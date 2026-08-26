from ranking import CommonModule

class RankingStageOneTruncateModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def ensemble_filter_queues(self):
    queues = [
      {
        "name": "pctr",
        "weight_attr": "explore_ensemble_filter_pctr_weight",
        "tail_ratio_attr": "explore_ensemble_filter_pctr_tail_ratio",
        "global_quantile_value_attr": "explore_ensemble_filter_pctr_global_quantile_value"
      },
      {
        "name": "pltr",
        "weight_attr": "explore_ensemble_filter_pltr_weight",
        "tail_ratio_attr": "explore_ensemble_filter_pltr_tail_ratio",
        "global_quantile_value_attr": "explore_ensemble_filter_pltr_global_quantile_value"
      },
      {
        "name": "pwtr",
        "weight_attr": "explore_ensemble_filter_pwtr_weight",
        "tail_ratio_attr": "explore_ensemble_filter_pwtr_tail_ratio",
        "global_quantile_value_attr": "explore_ensemble_filter_pwtr_global_quantile_value"
      },
      {
        "name": "pftr",
        "weight_attr": "explore_ensemble_filter_pftr_weight",
        "tail_ratio_attr": "explore_ensemble_filter_pftr_tail_ratio",
        "global_quantile_value_attr": "explore_ensemble_filter_pftr_global_quantile_value"
      },
      {
        "name": "pcmtr",
        "weight_attr": "explore_ensemble_filter_pcmtr_weight",
        "tail_ratio_attr": "explore_ensemble_filter_pcmtr_tail_ratio",
        "global_quantile_value_attr": "explore_ensemble_filter_pcmtr_global_quantile_value"
      },
      {
        "name": "pptr",
        "weight_attr": "explore_ensemble_filter_pptr_weight",
        "tail_ratio_attr": "explore_ensemble_filter_pptr_tail_ratio",
        "global_quantile_value_attr": "explore_ensemble_filter_pptr_global_quantile_value"
      },
      {
        "name": "pcltr",
        "weight_attr": "explore_ensemble_filter_pcltr_weight",
        "tail_ratio_attr": "explore_ensemble_filter_pcltr_tail_ratio",
        "global_quantile_value_attr": "explore_ensemble_filter_pcltr_global_quantile_value"
      },
      {
        "name": "fr_score1",
        "weight_attr": "explore_ensemble_filter_fr_score1_weight",
        "tail_ratio_attr": "explore_ensemble_filter_fr_score1_tail_ratio",
        "global_quantile_value_attr": "explore_ensemble_filter_fr_score1_global_quantile_value"
      },
      {
        "name": "fr_score2",
        "weight_attr": "explore_ensemble_filter_fr_score2_weight",
        "tail_ratio_attr": "explore_ensemble_filter_fr_score2_tail_ratio",
        "global_quantile_value_attr": "explore_ensemble_filter_fr_score2_global_quantile_value"
      },
      {
        "name": "pepstr",
        "weight_attr": "explore_ensemble_filter_pepstr_weight",
        "tail_ratio_attr": "explore_ensemble_filter_pepstr_tail_ratio",
        "global_quantile_value_attr": "explore_ensemble_filter_pepstr_global_quantile_value"
      },
      {
        "name": "phtr",
        "weight_attr": "explore_ensemble_filter_phtr_weight",
        "tail_ratio_attr": "explore_ensemble_filter_phtr_tail_ratio",
        "reverse_order": True,
        "global_quantile_value_attr": "explore_ensemble_filter_phtr_global_quantile_value"
      },
      {
        "name": "psvr",
        "weight_attr": "explore_ensemble_filter_psvr_weight",
        "tail_ratio_attr": "explore_ensemble_filter_psvr_tail_ratio",
        "reverse_order": True,
        "global_quantile_value_attr": "explore_ensemble_filter_psvr_global_quantile_value"
      },
      {
        "name": "awesome_wtd",
        "weight_attr": "explore_ensemble_filter_awesome_wtd_weight",
        "tail_ratio_attr": "explore_ensemble_filter_awesome_wtd_tail_ratio",
        "global_quantile_value_attr": "explore_ensemble_filter_awesome_wtd_global_quantile_value"
      },
      {
        "name": "fetr",
        "weight_attr": "explore_ensemble_filter_fetr_weight",
        "tail_ratio_attr": "explore_ensemble_filter_fetr_tail_ratio",
        "global_quantile_value_attr": "explore_ensemble_filter_fetr_global_quantile_value"
      },
      {
        "name": "fountain_eff",
        "weight_attr": "explore_ensemble_filter_fountain_eff_weight",
        "tail_ratio_attr": "explore_ensemble_filter_fountain_eff_tail_ratio",
        "global_quantile_value_attr": "explore_ensemble_filter_fountain_eff_global_quantile_value"
      },
      {
        "name": "corr_fountain_eff",
        "weight_attr": "explore_ensemble_filter_corr_fountain_eff_weight",
        "tail_ratio_attr": "explore_ensemble_filter_corr_fountain_eff_tail_ratio",
        "global_quantile_value_attr": "explore_ensemble_filter_corr_fountain_eff_global_quantile_value"
      }
    ]
    return queues

  def ensemble_sort_queues(self):
    queues = [
      # interactive_queues
      {
        "name": "score_pctr",
        "weight": 0.0,
        "power_weight_attr": "explore_ensemble_power_weight_rank_s1_pctr_score",
        "raw_weight_attr": "explore_ensemble_power_weight_rank_s1_pctr_score_alpha",
        "raw_power_weight_attr": "explore_ensemble_power_weight_rank_s1_pctr_score_beta",
        "use_min_rank": True
      },
      {
        "name": "score_pltr",
        "weight": 0.0,
        "power_weight_attr": "explore_ensemble_power_weight_rank_s1_pltr_score",
        "raw_weight_attr": "explore_ensemble_power_weight_rank_s1_pltr_score_alpha",
        "raw_power_weight_attr": "explore_ensemble_power_weight_rank_s1_pltr_score_beta",
        "use_min_rank": True
      },
      {
        "name": "score_pwtr",
        "weight": 0.0,
        "power_weight_attr": "explore_ensemble_power_weight_rank_s1_pwtr_score",
        "raw_weight_attr": "explore_ensemble_power_weight_rank_s1_pwtr_score_alpha",
        "raw_power_weight_attr": "explore_ensemble_power_weight_rank_s1_pwtr_score_beta",
        "use_min_rank": True
      },
      {
        "name": "score_pftr",
        "weight": 0.0,
        "power_weight_attr": "explore_ensemble_power_weight_rank_s1_pftr_score",
        "raw_weight_attr": "explore_ensemble_power_weight_rank_s1_pftr_score_alpha",
        "raw_power_weight_attr": "explore_ensemble_power_weight_rank_s1_pftr_score_beta",
        "use_min_rank": True
      },
      {
        "name": "score_pcmtr",
        "weight": 0.0,
        "power_weight_attr": "explore_ensemble_power_weight_rank_s1_pcmtr_score",
        "raw_weight_attr": "explore_ensemble_power_weight_rank_s1_pcmtr_score_alpha",
        "raw_power_weight_attr": "explore_ensemble_power_weight_rank_s1_pcmtr_score_beta",
        "use_min_rank": True
      },
      {
        "name": "score_pptr",
        "weight": 0.0,
        "power_weight_attr": "explore_ensemble_power_weight_rank_s1_pptr_score",
        "raw_weight_attr": "explore_ensemble_power_weight_rank_s1_pptr_score_alpha",
        "raw_power_weight_attr": "explore_ensemble_power_weight_rank_s1_pptr_score_beta",
      },
      {
        "name": "score_pcmef",
        "weight": 0.0,
        "power_weight_attr": "explore_ensemble_power_weight_rank_s1_pcmef_score",
        "raw_weight_attr": "explore_ensemble_power_weight_rank_s1_pcmef_score_alpha",
        "raw_power_weight_attr": "explore_ensemble_power_weight_rank_s1_pcmef_score_beta",
      },
      {
        "name": "score_pcltr",
        "weight": 0.0,
        "power_weight_attr": "explore_ensemble_power_weight_rank_s1_pcltr_score",
        "raw_weight_attr": "explore_ensemble_power_weight_rank_s1_pcltr_score_alpha",
        "raw_power_weight_attr": "explore_ensemble_power_weight_rank_s1_pcltr_score_beta",
        "use_min_rank": True
      },
      {
        "name": "score_phtr",
        "weight": 0.0,
        "reverse_order": True,
        "power_weight_attr": "explore_ensemble_power_weight_rank_s1_phtr_score",
      },
      {
        "name": "svr_act_score",
        "weight": 0.0,
        "power_weight_attr": "explore_ensemble_power_weight_rank_s1_svr_act_score",
      },
      {
        "name": "pcmef_debias_score",
        "weight": 0.0,
        "power_weight_attr": "explore_ensemble_power_weight_rank_s1_pcmef_debias_score",
      },
      {
        "name": "min_act_rank_score",
        "weight": 0.0,
        "power_weight_attr": "explore_ensemble_power_weight_rank_s1_min_act_rank_score",
      },
      # time_queues
      {
        "name": "score_psvr",
        "weight": 0.0,
        "power_weight_attr": "explore_ensemble_power_weight_rank_s1_psvr_score",
      },
      {
        "name": "fr_score1",
        "weight": 0.0,
        "power_weight_attr": "explore_ensemble_power_weight_rank_s1_fr_score1",
        "raw_weight_attr": "explore_ensemble_power_weight_rank_s1_fr_score1_alpha",
        "raw_power_weight_attr": "explore_ensemble_power_weight_rank_s1_fr_score1_beta",
      },
      {
        "name": "fr_score2",
        "weight": 0.0,
        "power_weight_attr": "explore_ensemble_power_weight_rank_s1_fr_score2",
        "raw_weight_attr": "explore_ensemble_power_weight_rank_s1_fr_score2_alpha",
        "raw_power_weight_attr": "explore_ensemble_power_weight_rank_s1_fr_score2_beta",
      },
      {
        "name": "score_pepstr",
        "weight": 0.0,
        "power_weight_attr": "explore_ensemble_power_weight_rank_s1_pepstr_score",
        "raw_weight_attr": "explore_ensemble_power_weight_rank_s1_pepstr_score_alpha",
        "raw_power_weight_attr": "explore_ensemble_power_weight_rank_s1_pepstr_score_beta",
      },
      {
        "name": "corr_fetr",
        "weight": 0.0,
        "power_weight_attr": "explore_ensemble_power_weight_rank_s1_corr_fetr",
      },
      {
        "name": "corr_fountain_eff",
        "weight": 0.0,
        "power_weight_attr": "explore_ensemble_power_weight_rank_s1_corr_fountain_eff",
      },
      {
        "name": "awesome_wtd",
        "weight": 0.0,
        "power_weight_attr": "explore_ensemble_power_weight_rank_s1_awesome_wtd",
        "raw_weight_attr": "explore_ensemble_power_weight_rank_s1_awesome_wtd_alpha",
        "raw_power_weight_attr": "explore_ensemble_power_weight_rank_s1_awesome_wtd_beta",
        "use_min_rank": True
      },
      {
        "name": "corr_cpr",
        "weight": 0.0,
        "power_weight_attr": "explore_ensemble_power_weight_rank_s1_corr_cpr",
        "raw_weight_attr": "explore_ensemble_power_weight_rank_s1_corr_cpr_alpha",
        "raw_power_weight_attr": "explore_ensemble_power_weight_rank_s1_corr_cpr_beta",
      },
    ]
    return queues

  def process(self) -> None:
    self.flow \
    .count_reco_result(
      save_count_to="explore_reco_leaf_rank_model_input_count"
    ) \
    .count_reco_result(
      save_count_to = "explore_reco_leaf_rank_model_pic_input_count",
      target_item = {"is_picture" : 1}
    ) \
    .count_reco_result(
      save_count_to = "explore_reco_leaf_rank_model_vd_input_count",
      target_item = {"is_picture" : 0}
    ) \
    .if_("explore_enable_rank_write_rank_stage1_result_to_redis == 1") \
      .pack_item_attr(
        item_source = {
          "reco_results": True,
        },
        mappings = [{
          "from_item_attr": "photo_id",
          "to_common_attr": "rank_photo_id_before_stage1",
        }]
      ) \
    .end_() \
    ._dump_attr_to_kafka(
      stage_name = "fr_s1_score", 
      dump_item_attr_list = [
        # 模型预估分和 ensemble filter 使用队列
        "fr_score1",
        "fr_score2",
        "corr_pctr",
        "pctr",
        "pltr",
        "pwtr",
        "pftr",
        "psvr",
        "pcmtr",
        "pptr", 
        "pcmef",
        "phtr",
        "pevtr", 
        "plvtr", 
        "pepstr",
        "pdtr", 
        "pcltr", 
        "cpr",
        "corr_fountain_eff"
      ]
    ) \
    .if_("explore_gen_single_pic_type == 1") \
      .enrich_attr_by_light_function(    
        import_item_attr = [
          "duration_ms",
          "upload_type",
          "picture_type",
          "photo_picture_count",
        ],
        export_item_attr = [
          "is_single_picture",
        ],
        function_name = "GenSinglePicType",
        class_name = "ExploreLightFunctionSetV2",
       ) \
    .end_() \
    .if_("explore_rank_stage1_adjust_ratio_by_user_feature == 1") \
      .trucate_by_user_features()\
    .end_() \
    .if_("enable_distribute_ctr_filter == 1") \
      .if_("enable_explore_rank_stage1_personal_cem == 1") \
        .rank_stage1_personal_cem() \
      .end_() \
      .if_("explore_ctr_filter_ensemble == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "ctr_filter_debias_string",
          ],
          import_item_attr = [
            "corr_pctr",
            "corr_fetr",
            "corr_fountain_eff",
            "upload_type"
          ],
          export_item_attr = [
            "debias_pctr",
            "debias_fetr",
            "debias_fountain_eff",
          ],
          function_name = "FrPxtrDebiasFilter",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .if_("life_enable_fr_s1_debias_hetu_cluster_id == 1") \
          .explore_life_uninterest_tag_exit_enricher(
            prev_item_from_attr = "life_realshow_common_list",
            prev_item_from_attr_timestamp = "life_realshow_timestamp_common_list",
            prev_click_item_from_attr = "life_click_common_list",
            cluster_id_attr = "hetu_sim_cluster_id",
            input_pctr_attr = "debias_pctr",
            output_pctr_attr = "debias_pctr",
            realshow_num_threshold = "{{life_fr_s1_realshow_num_threshold}}",
            time_window = "{{life_fr_s1_timestamp_threshold}}",
            calculate_mode = "{{life_fr_s1_calculate_mode}}",
            discount_coef = "{{life_fr_s1_discount_coef}}",
            realshow_unclick_num = "{{life_fr_s1_realshow_unclick_num}}",
          ) \
        .end_() \
        .if_("enable_life_fr_s1_hetu_debias_pctr == 1") \
          .explore_life_uninterest_hetu_exit_enricher(
            user_info_ptr_attr = "user_info_ptr",
            realshow_num_threshold = "{{life_fr_s1_hetu_debias_pctr_realshow_num_threshold}}",
            time_gap_s = "{{life_fr_s1_hetu_debias_pctr_time_gap_s}}",
            hetu_tag_attr = "hetu_tag_level_info__hetu_level_two",
            input_pctr_attr = "debias_pctr",
            output_pctr_attr = "debias_pctr",
            calculate_mode = "{{life_fr_s1_hetu_debias_pctr_calculate_mode}}",
            discount_coef = "{{life_fr_s1_hetu_debias_pctr_discount_coef}}",
            realshow_unclick_num_thr = "{{life_fr_s1_hetu_debias_pctr_realshow_unclick_num_thr}}",
          ) \
        .end_() \
        .if_("enable_unbind_parameter_in_stage1 == 1") \
          .explore_calc_ensemble_score(
            save_score_to_attr = "ctr_filter_ensemble_score",
            user_power_calc = "{{explore_fr_fullrank_variant_s1_enable_power_calc}}",
            rank_smooth = "{{explore_fr_fullrank_ctr_filter_essemble_smooth}}",
            rank_power_weight = "{{explore_fr_fullrank_rank_power_weight}}",
            use_reciprocal = "{{explore_fr_use_reciprocal}}",
            duration_min = "{{explore_fr_duration_min}}",
            duration_max = "{{explore_fr_duration_max}}",
            enable_perf_pxtr_pic = "{{explore_enable_perf_pxtr_pic_limit}}",
            perf_pxtr_pic_num = "{{explore_perf_pxtr_pic_num}}",
            duration_add = 10,
            action_day = "{{rk_collect_queue_boost_active_day_num}}",
            enable_dynamic_weight_by_user_degree = "{{fr_enable_pcltr_adjust_by_gender_age}}",
            fr_rank_max_num = "{{explore_fr_rank_max_num}}",
            fr_rank_specified_num = "{{explore_fr_rank_specified_num}}",
            fr_rank_has_sec_str = "{{explore_fr_rank_has_sec_str}}",
            queues = [
              {
                "name": "debias_pctr",
                "weight": 1.0,
                "power_weight_attr": "explore_fullrank_filter_ctr_weight"
              },
              {
                "name": "debias_fetr",
                "weight": 0.0,
                "power_weight_attr": "explore_fullrank_filter_fetr_weight"
              },
              {
                "name": "debias_fountain_eff",
                "weight": 0.0,
                "power_weight_attr": "explore_fullrank_filter_eff_weight"
              },
              {
                "name": "fr_score2",
                "weight": 0.0,
                "power_weight_attr": "explore_fullrank_filter_fr_score2_weight"
              } 
            ],
          ) \
        .else_() \
          .if_("enable_explore_rank_stage1_personal_cem_es_weight == 1") \
            .rank_stage1_personal_cem_es_weight_adjust() \
          .end_() \
          .explore_calc_ensemble_score(
            save_score_to_attr = "ctr_filter_ensemble_score",
            user_power_calc = "{{explore_fr_fullrank_variant_enable_power_calc}}",
            rank_smooth = "{{explore_fr_fullrank_ctr_filter_essemble_smooth}}",
            rank_power_weight = "{{explore_fr_fullrank_rank_power_weight}}",
            use_reciprocal = "{{explore_fr_use_reciprocal}}",
            duration_min = "{{explore_fr_duration_min}}",
            duration_max = "{{explore_fr_duration_max}}",
            enable_perf_pxtr_pic = "{{explore_enable_perf_pxtr_pic_limit}}",
            perf_pxtr_pic_num = "{{explore_perf_pxtr_pic_num}}",
            duration_add = 10,
            action_day = "{{rk_collect_queue_boost_active_day_num}}",
            enable_dynamic_weight_by_user_degree = "{{fr_enable_pcltr_adjust_by_gender_age}}",
            fr_rank_max_num = "{{explore_fr_rank_max_num}}",
            fr_rank_specified_num = "{{explore_fr_rank_specified_num}}",
            fr_rank_has_sec_str = "{{explore_fr_rank_has_sec_str}}",
            queues = [
              {
                "name": "debias_pctr",
                "weight": 1.0,
                "power_weight_attr": "explore_fullrank_filter_ctr_weight"
              },
              {
                "name": "debias_fetr",
                "weight": 0.0,
                "power_weight_attr": "explore_fullrank_filter_fetr_weight"
              },
              {
                "name": "debias_fountain_eff",
                "weight": 0.0,
                "power_weight_attr": "explore_fullrank_filter_eff_weight"
              },
              {
                "name": "fr_score2",
                "weight": 0.0,
                "power_weight_attr": "explore_fullrank_filter_fr_score2_weight"
              }
            ]
          ) \
        .end_() \
        .if_("enable_life_direct_tab_boost == 1") \
          .set_attr_value(
            item_attrs=[
              {
                "name": "ctr_filter_ensemble_score",
                "type": "double",
                "value": 1000000000.0
              }
            ],
            target_item = {
              "reason": 2416
            }
          ) \
        .end_() \
        .if_("is_fresh_request == 1 and enable_life_active_interest_boost == 1 and (life_active_interest_boost_limit_low_active ~= 1 or uIsLifeHighActive ~= 1)") \
          .set_attr_value(
            item_attrs=[
              {
                "name": "ctr_filter_ensemble_score",
                "type": "double",
                "value": 1000000000.0
              }
            ],
            target_item = {
              "reason": [2422]
            }
          ) \
        .end_() \
        .switch_("explore_fullrank_stage1_sort_switch") \
          .case_(1) \
            .sort(
              score_from_attr = "ctr_filter_ensemble_score",
              target_item = {"is_picture" : 0}  # 视频排序
            ) \
            .sort(
              score_from_attr = "ctr_filter_ensemble_score",
              target_item = {"is_single_picture" : 1}  # 单图排序
            ) \
            .sort(
              score_from_attr = "corr_pctr", 
              target_item = { "is_picture": 1, "is_single_picture" : 0} # 多图排序
            ) \
          .case_(2) \
            .explore_ensemble_filter_score_enricher(
              queues = self.ensemble_filter_queues(),
              filter_function = "{{ensemble_filter_function}}",
              score_with_rank = "{{ensemble_filter_score_with_rank}}",
              save_score_to_attr = "explore_fr_ensemble_filter_score",
            ) \
            .sort(
              score_from_attr = "explore_fr_ensemble_filter_score",
              stable_sort = True,
              desc = False
            ) \
          .case_(3) \
            .explore_calc_ensemble_score(
              save_score_to_attr = "explore_fr_ensemble_score_s1",
              user_power_calc_v2 = "{{explore_rank_s1_ensemble_user_power_calc_v2}}",
              value_seq_fusion_status = "{{explore_rank_s1_ensemble_value_seq_fusion_status}}",
              queues = self.ensemble_sort_queues(),
            ) \
            .sort(
              score_from_attr = "explore_fr_ensemble_score_s1",
              stable_sort = True,
            ) \
          .default_() \
            .if_("explore_pic_ctr_filter_sort == 1") \
              .sort(
                score_from_attr = "ctr_filter_ensemble_score",
                target_item = {"is_picture" : 0}
              ) \
              .sort(
                score_from_attr = "corr_pctr",
                target_item = {"is_picture" : 1}
              ) \
            .else_() \
              .sort(
                score_from_attr = "ctr_filter_ensemble_score"
              ) \
            .end_() \
        .end_() \
      .else_() \
        .sort(
          score_from_attr = "corr_pctr"
        ) \
      .end_() \
      .switch_("explore_fullrank_stage1_truncate_switch") \
        .case_(2) \
          .enrich_attr_by_light_function(
            import_common_attr = [
              "ctr_filter_distribue_coff",
              {"name": "explore_reco_leaf_rank_model_input_count", "as": "input_count"},
              "user_emp_ctr",
              "ctr_filter_user_all_photo_emp_ctr",
              "ctr_filter_emp_ctr_ratio_coffe",
              "ctr_filter_retain_max_num",
              "ctr_filter_retain_min_num",
              "user_emp_ctr_conf_string",
              "enable_emp_ctr_section"
            ],
            export_common_attr = [
              "ctr_filter_retain_num",
              "user_emp_ctr_bucket",
              "emp_ratio"
            ],
            function_name = "UserEmpCtrFilterRetain",
            class_name = "ExploreLightFunctionSetV2",
          ) \
          .limit(size = "{{ctr_filter_retain_num}}") \
        .case_(3) \
          .transform_item_attr(
            mappings = [
            {
              "check_attr_name": "duration_ms",
              "check_attr_type": "int",
              "output_attr_name": "duration_bucket",
              "output_attr_type": "int",
              "output_default_value": -1,
              "rules": [
                {
                  "check_range": {
                    "upper_bound": 20000, # 不包含，可缺省
                  },
                  "output_value": 1
                },
                {
                  "check_range": {
                    "lower_bound": 20000, # 包含，可缺省
                    "upper_bound": 40000, # 不包含，可缺省
                  },
                  "output_value": 2
                },
                {
                  "check_range": {
                    "lower_bound": 40000, # 包含，可缺省
                    "upper_bound": 60000, # 不包含，可缺省
                  },
                  "output_value": 3
                },
                {
                  "check_range": {
                    "lower_bound": 60000, # 包含，可缺省
                    "upper_bound": 90000, # 不包含，可缺省
                  },
                  "output_value": 4
                },
                {
                  "check_range": {
                    "lower_bound": 90000, # 包含，可缺省
                    "upper_bound": 120000, # 不包含，可缺省
                  },
                  "output_value": 5
                },
                {
                  "check_range": {
                    "lower_bound": 120000, # 包含，可缺省
                    "upper_bound": 180000, # 不包含，可缺省
                  },
                  "output_value": 6
                },
                {
                  "check_range": {
                    "lower_bound": 180000, # 包含
                  },
                  "output_value": 7
                }
              ]
            }
          ]) \
          .transform_item_attr(
            mappings = [{
              "check_attr_name": "is_picture",
              "check_attr_type": "int",
              "output_attr_name": "duration_bucket",
              "output_attr_type": "int",
              "rules": [
                {
                  "check_values": [1],
                  "output_value": 0
                }
              ]
            }]
          )\
          .transform_item_attr(
            mappings = [{
              "check_attr_name": "is_single_picture",
              "check_attr_type": "int",
              "output_attr_name": "duration_bucket",
              "output_attr_type": "int",
              "rules": [
                {
                  "check_values": [1],
                  "output_value": 8
                }
              ]
            }]
          ) \
          .explore_cluster_truncate_arranger(
            cluster_name = "duration_bucket",
            skip_reserved_clusters_str = "{{explore_fullrank_duration_cluster_skip_reserved_clusters_str}}",
            cluster_truncate_ratio_map_str = "{{explore_fullrank_duration_cluster_ratio_map_str}}",
            truncate_ratio = "{{explore_fullrank_duration_cluster_truncate_ratio}}",
            min_reserved_percentile = "{{explore_fullrank_duration_cluster_reserved_min_percentile}}",
            truncate_min_size = "{{explore_fullrank_duration_cluster_truncate_min_size}}",
          ) \
        .case_(4) \
          .count_reco_result(
            save_count_to = "explore_reco_leaf_rank_model_vd_input_count",
            target_item = {"is_picture" : 0}
          ) \
          .count_reco_result(
            save_count_to = "explore_reco_leaf_rank_model_spic_input_count",
            target_item = {"is_single_picture" : 1}
          ) \
          .count_reco_result(
            save_count_to = "explore_reco_leaf_rank_model_pic_input_count",
            target_item = {"is_picture" : 1, "is_single_picture" : 0}
          ) \
          .limit(
            size = "{{return math.floor(explore_reco_leaf_rank_model_vd_input_count * (1 - ctr_filter_distribue_coff))}}",
            target_item = {"is_picture" : 0}
          ) \
          .limit(
            size = "{{return math.floor(explore_reco_leaf_rank_model_spic_input_count * (1 - spic_ctr_filter_distribue_coff))}}",
            target_item = {"is_single_picture" : 1}
          ) \
          .limit(
            size = "{{return math.floor(explore_reco_leaf_rank_model_pic_input_count * (1 - pic_ctr_filter_distribue_coff))}}",
            target_item = {"is_picture" : 1, "is_single_picture" : 0}
          ) \
        .case_(5) \
          .if_("explore_ensemble_filter_random_cut_off == 1") \
            .enrich_attr_by_light_function(
              import_common_attr = [
                {"name": "_USER_ID_", "as": "user_id"},
                {"name": "explore_ensemble_filter_random_cut_off_lower_bound", "as": "lower_bound"},
                {"name": "explore_ensemble_filter_random_cut_off_upper_bound", "as": "upper_bound"},
              ],
              export_common_attr = [
                {"name": "user_random_cut_off_coeff", "as": "ensemble_filter_coeff"}
              ],
              function_name = "GenRandomCutOffCoeff",
              class_name = "ExploreLightFunctionSetV2",
            ) \
          .end_() \
          .if_("enable_explore_rank_stage1_personal_cem_cut_off_ratio == 1") \
            .rank_stage1_personal_cem_cut_off_ratio_adjust() \
          .end_() \
          .limit(
            size = "{{return math.floor(explore_reco_leaf_rank_model_vd_input_count * (1 - ensemble_filter_coeff))}}",
            target_item = {"is_picture" : 0}
          ) \
          .limit(
            size = "{{return math.floor(explore_reco_leaf_rank_model_pic_input_count * (1 - pic_ctr_filter_distribue_coff))}}",
            target_item = {"is_picture" : 1}
          ) \
        .case_(6) \
            .enrich_attr_by_lua(
              import_common_attr = [
                  "explore_reco_leaf_rank_model_vd_input_count",
                  "explore_reco_leaf_rank_model_pic_input_count",
                  "life_fr_s2_pic_cutoff_rate_max_thred",
                  "life_fr_s2_out_num",
                ],
                export_common_attr = [
                  "photo_remain_num",
                  "pic_remain_num",
                ],
                function_for_common = "get_cutoff_rate",
                lua_script_file = "life/ranking/lua/module/ranking_ensemble_sort__score_coeff.lua"
            ) \
            .limit(
              size = "{{photo_remain_num}}",
              target_item = {"is_picture" : 0}
            ) \
            .limit(
              size = "{{pic_remain_num}}",
              target_item = {"is_picture" : 1}
            ) \
        .default_() \
          .if_("explore_pic_ctr_filter_quota == 1") \
            .limit(
              size = "{{return math.floor(explore_reco_leaf_rank_model_vd_input_count * (1 - ctr_filter_distribue_coff))}}",
              target_item = {"is_picture" : 0}
            ) \
            .limit(
              size = "{{return math.floor(explore_reco_leaf_rank_model_pic_input_count * (1 - pic_ctr_filter_distribue_coff))}}",
              target_item = {"is_picture" : 1}
            ) \
          .else_() \
            .limit(size = "{{return math.floor(explore_reco_leaf_rank_model_input_count * (1 - ctr_filter_distribue_coff))}}") \
          .end_() \
        .end_() \
    .else_if_("enable_skip_life_fr_s1_es == 1") \
      .do_nothing() \
    .else_() \
      .filter_by_attr(
        attr_name = "is_satisfy_ctr_filter",
        remove_if = "==",
        compare_to = 1,
      ) \
    .end_() \
    .if_("explore_enable_rank_write_rank_stage1_result_to_redis == 1") \
      .pack_item_attr(
        item_source = {
          "reco_results": True,
        },
        mappings = [{
          "from_item_attr": "photo_id",
          "to_common_attr": "rank_photo_id_after_stage1",
        }]
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "rank_photo_id_before_stage1", "as": "universal_set_list"},
          {"name": "rank_photo_id_after_stage1", "as": "sub_set_list"}
        ],
        export_common_attr = [
          {"name": "difference_list", "as": "photo_id_trunc_stage1"}
        ],
        function_name = "GetDifferenceSet",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .do_nothing(
      name = "explore_fr_stage1",
      traceback = True,
    )

  def post_process(self) -> None:
    self.flow \
      .if_("enable_distribute_ctr_filter == 1") \
        .count_reco_result( # TODO(guorongda) 基于打点透出比例和ab调试实验，预计22年12月初实验结束后删除,二期删除部分perf，新增部分，用于ab调试
          save_count_to = "filter_final_count"
        ) \
        .if_("enable_compute_user_emp_ctr_filter_coffe == 1") \
          .perflog_attr_value(
            check_point = "{{ctr_filter_emp_ctr_explore_check_point}}",
            item_attrs=[
              "corr_pctr"
            ],
            common_attrs=[
              "filter_final_count",
              "explore_reco_leaf_rank_model_input_count"
            ]
          ) \
        .end_() \
        .if_("enable_emp_ctr_section == 1") \
          .perflog_attr_value(
            check_point = "{{ctr_filter_emp_ctr_explore_check_point}}",
            item_attrs=[
              "corr_pctr"
            ],
            common_attrs=[
              "filter_final_count",
              "explore_reco_leaf_rank_model_input_count",
              "user_emp_ctr_bucket",
              "emp_ratio"
            ]
          ) \
        .end_() \
      .end_()
