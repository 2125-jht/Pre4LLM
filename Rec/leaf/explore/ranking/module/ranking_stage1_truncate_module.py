from ranking import CommonModule

class RankingStageOneTruncateModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def ctr_filter_queues(self):
    queues = [
      {
        "name": "corr_pctr",
        "weight": 1.0,
        "weight_attr": "explore_fullrank_filter_ctr_weight",
        "power_weight_attr": "explore_fullrank_filter_ctr_weight",
        "score_threshold": "explore_fullrank_filter_ctr_rank_cliff_threshold",
        "rank_height_attr": "explore_fullrank_filter_ctr_rank_height"
      },
      {
        "name": "corr_fetr",
        "weight": 0.0,
        "power_weight_attr": "explore_fullrank_filter_fetr_weight",
        "score_threshold": "explore_fullrank_filter_fetr_rank_cliff_threshold",
        "rank_height_attr": "explore_fullrank_filter_fetr_rank_height"
      },
      {
        "name": "corr_fountain_eff",
        "weight": 0.0,
        "power_weight_attr": "explore_fullrank_filter_eff_weight"
      },
      {
        "name": "awesome_wtd",
        "weight": 0.0,
        "weight_attr": "explore_fullrank_filter_awesome_wtd_weight",
        "power_weight_attr": "explore_fullrank_filter_awesome_wtd_weight"
      },
      {
        "name": "photo_history_interest_score_with_fr_ctr",
        "weight": 0.0,
        "weight_attr": "explore_fullrank_filter_photo_history_interest_score_with_fr_ctr_weight",
        "power_weight_attr": "explore_fullrank_filter_photo_history_interest_score_with_fr_ctr_weight"
      },
      {
        "name": "fr_score2",
        "weight": 0.0,
        "power_weight_attr": "explore_fullrank_filter_fr_score2_weight"
      },
      {
        "name": "explore_diversity_ctr_score",
        "weight": 0.0,
        "power_weight_attr": "explore_fullrank_filter_diversity_ctr_weight"
      },                 
      {
        "name": "fountain_eff",
        "weight": 0.0,
        "power_weight_attr": "explore_fullrank_fountain_eff_weight"
      },
      {
        "name": "cpr",
        "weight": 0.0,
        "power_weight_attr": "explore_fullrank_filter_corr_cpr_weight"
      },
      {
        "name": "plvtr",
        "weight": 0.0,
        "power_weight_attr": "explore_fullrank_filter_plvtr_weight"
      },
      {
        "name": "score_pltr",
        "weight": 0.0,
        "power_weight_attr": "explore_fullrank_filter_score_pltr_weight"
      },
      {
        "name": "score_psvr",
        "weight": 0.0,
        "power_weight_attr": "explore_fullrank_filter_score_psvr_weight"
      },
      {
        "name": "score_phtr",
        "weight": 0.0,
        "power_weight_attr": "explore_fullrank_filter_score_phtr_weight"
      },
      {
        "name": "bad_cover_similary_score",
        "weight": 0.0,
        "weight_attr": "explore_fullrank_filter_bad_cover_similary_weight",
        "power_weight_attr": "explore_fullrank_filter_bad_cover_similary_weight"
      },
      {
        "name": "bad_sense_similary_score",
        "weight": 0.0,
        "weight_attr": "explore_fullrank_filter_bad_sense_similary_weight",
        "power_weight_attr": "explore_fullrank_filter_bad_sense_similary_weight"
      },
      {
        "name": "sense_view_predict_trans_score",
        "weight": 0.0,
        "weight_attr": "explore_fr_ensemble_s1_sense_view_predict_trans_score_weight",
        "power_weight_attr": "explore_fr_ensemble_s1_sense_view_predict_trans_score_power_weight",
        "score_threshold": "explore_fr_ensemble_s1_sense_view_predict_trans_score_rank_cliff_threshold",
        "rank_height_attr": "explore_fr_ensemble_s1_sense_view_predict_trans_score_rank_weight",
        "raw_weight_attr": "explore_fr_ensemble_s1_sense_view_predict_trans_score_raw_weight",
        "raw_power_weight_attr": "explore_fr_ensemble_s1_sense_view_predict_trans_score_raw_power_weight",
        "rank_cliff_attr": "explore_fr_ensemble_s1_sense_view_predict_trans_score_rank_cliff",
        "rank_cliff_ratio_attr": "explore_fr_ensemble_s1_sense_view_predict_trans_score_rank_cliff_ratio",
        "rank_cliff_min_attr": "explore_fr_ensemble_s1_sense_view_predict_trans_score_rank_cliff_min",
      },
      {
        "name": "explore_uninterest_ctr_adjust_score",
        "weight": 0.0,
        "weight_attr": "explore_fr_ensemble_s1_uninterest_ctr_adjust_weight",
        "power_weight_attr": "explore_fr_ensemble_s1_uninterest_ctr_adjust_power_weight",
        "score_threshold": "explore_fr_ensemble_s1_uninterest_ctr_adjust_rank_cliff_threshold",
        "rank_height_attr": "explore_fr_ensemble_s1_uninterest_ctr_adjust_rank_weight",
        "raw_weight_attr": "explore_fr_ensemble_s1_uninterest_ctr_adjust_raw_weight",
        "raw_power_weight_attr": "explore_fr_ensemble_s1_uninterest_ctr_adjust_raw_power_weight",
        "rank_cliff_attr": "explore_fr_ensemble_s1_uninterest_ctr_adjust_rank_cliff",
        "rank_cliff_ratio_attr": "explore_fr_ensemble_s1_uninterest_ctr_adjust_rank_cliff_ratio",
        "rank_cliff_min_attr": "explore_fr_ensemble_s1_uninterest_ctr_adjust_rank_cliff_min",
      },
      {
        "name": "exclusive_ctr",
        "weight": 0.0,
        "weight_attr": "explore_fullrank_filter_exclusive_ctr_weight",
        "power_weight_attr": "explore_fullrank_filter_exclusive_ctr_weight",
        "score_threshold": "explore_fullrank_filter_exclusive_ctr_rank_cliff_threshold",
        "rank_height_attr": "explore_fullrank_filter_exclusive_ctr_rank_height"
      },
    ]
    return queues

  def process(self) -> None:
    self.flow \
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
        "awesome_wtd",
        "photo_history_interest_score_with_fr_ctr",
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
        "is_long_view_author", 
        "pcltr", 
        "fountain_eff",
        "cpr",
        "pdtr",
        "fetr"
      ]
    ) \
    .if_("enable_distribute_ctr_filter == 1") \
      .if_("explore_ctr_filter_ensemble == 1") \
        .if_("enable_first_screen_fr_s1_discount_by_cid == 1 and page_index == 1 and refreshTimes ~= 0 and gemini_refresh_scene > 0 and gemini_refresh_scene < 4", to_be_delete = "date=2024-05-29;committer=guohao") \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "first_screen_fr_s1_discount_coef_by_cid", "as": "boost_discount_coeff"},
            ],
            import_item_attr = [
              {"name": "is_first_screen_discount_by_cid", "as": "need_item_attr"},
              {"name": "corr_pctr", "as": "ensemble_score"},
            ],
            export_item_attr = [
              {"name": "ensemble_score", "as": "corr_pctr"}
            ],
            function_name = "BoostOrDiscount",
            class_name = "ExploreLightFunctionSetV2",
          ) \
        .end_() \
        .if_("enable_first_screen_fr_s1_discount == 1 and page_index >= 1 and page_index <= explore_mc_s2_first_screen_discount_threshold") \
          .if_("(explore_fr_s1_enable_gemini_refresh_scene_pxtr_adjust == 0) or (gemini_refresh_scene > 0 and gemini_refresh_scene < 4)") \
            .enrich_attr_by_light_function(
              import_common_attr = [
                {"name": "first_screen_fr_s1_discount_coef", "as": "boost_discount_coeff"},
              ],
              import_item_attr = [
                {"name": "is_first_screen_discount", "as": "need_item_attr"},
                {"name": "corr_pctr", "as": "ensemble_score"},
              ],
              export_item_attr = [
                {"name": "ensemble_score", "as": "corr_pctr"}
              ],
              function_name = "BoostOrDiscount",
              class_name = "ExploreLightFunctionSetV2",
            ) \
          .end_() \
        .end_() \
        .if_("enable_explore_fr_s1_uninterest_ctr_score_adjust == 1") \
          .explore_uninterest_ctr_score_adjust() \
        .end_() \
        .if_("explore_enable_rank_stage1_ef_weight_adjust == 1") \
          .gen_common_attr_by_lua(
            attr_map = {
              "explore_fullrank_filter_eff_weight": "explore_fullrank_filter_eff_weight * explore_fountain_view_weight",
            }
          ) \
        .end_() \
        .if_("explore_enable_rank_stage1_only_video == 1") \
          .explore_calc_ensemble_score(
            save_score_to_attr = "ctr_filter_ensemble_score",
            user_power_calc = "{{explore_fr_ctr_filter_variant_enable_power_calc}}",
            rank_smooth = "{{explore_fr_ctr_filter_essemble_smooth}}",
            rank_power_weight = "{{explore_fr_ctr_filter_rank_power_weight}}",
            use_reciprocal = "{{explore_fr_ctr_filter_use_reciprocal}}",
            duration_min = "{{explore_fr_ctr_filter_duration_min}}",
            duration_max = "{{explore_fr_ctr_filter_duration_max}}",
            duration_add = 10,
            action_day = "{{fr_s1_rk_collect_queue_boost_active_day_num}}",
            fr_rank_max_num = "{{explore_fr_ctr_filter_rank_max_num}}",
            fr_rank_specified_num = "{{explore_fr_ctr_filter_rank_specified_num}}",
            fr_rank_has_sec_str = "{{explore_fr_ctr_filter_rank_has_sec_str}}",
            queues = self.ctr_filter_queues(),
            target_item = {"is_picture": 0}
          ) \
        .else_() \
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
            queues = self.ctr_filter_queues(),
          ) \
        .end_() \
        .if_("explore_enable_rank_stage1_boost == 1") \
          .if_("explore_enable_fr_s1_reach_content_keep_item == 1") \
            .fr_reach_content_keep_item("ctr_filter_ensemble_score", "fr_s1") \
          .end_() \
          .if_("enable_fr_prior_author_id_boost == 1") \
            .enrich_attr_by_light_function(
              import_common_attr = [
                "prior_author_5w_set_ptr",
                "boost_prior_author_id_alpha"
              ],
              import_item_attr = [
                "author__id",
                {"name": "ctr_filter_ensemble_score", "as": "ensemble_score"},
              ],
              export_item_attr = [
                {"name": "ensemble_score", "as": "ctr_filter_ensemble_score"}
              ],
              function_name = "BoostPriorAid",
              class_name = "ExploreLightFunctionSetV2", 
            ) \
          .end_() \
          .if_("enable_fr_s1_hot_list_photo_boost == 1") \
            .hot_list_photo_boost("ctr_filter_ensemble_score", "fr_s1") \
          .end_() \
          .if_("enable_explore_partial_time_based_interest_boost_fr_s1 == 1") \
            .partial_time_based_interest_boost("ctr_filter_ensemble_score", "fr_s1") \
          .end_() \
          .if_("enable_explore_partial_time_based_tagnex_boost_fr_s1 == 1") \
            .partial_time_based_tagnex_boost("ctr_filter_ensemble_score", "fr_s1") \
          .end_() \
          .if_("enable_explore_fr_s1_boost_user_short_develop_interest == 1 and uExploreFountainPreferenceTypeKV == 1") \
            .boost_user_short_develop_interest("ctr_filter_ensemble_score", stage_name="fr_s1") \
          .end_() \
          .if_("enable_fr_s1_short_uninterest_decay_discount == 1") \
            .short_uninterest_decay_discount("ctr_filter_ensemble_score", "fr_s1") \
          .end_() \
          .if_("enable_fr_s1_fr_interest_generalization_boost == 1") \
            .fr_interest_generalization_boost("ctr_filter_ensemble_score", "fr_s1") \
          .end_() \
        .end_() \
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
          score_from_attr = "corr_pctr"
        ) \
      .end_() \
      .if_("enable_explore_gen_diversity_vol_candidate == 1") \
        .explore_custom_embedding_score_enricher(
          user_info_ptr_attr = "user_info_ptr",
          embedding_list_attr = "mc_embeddings_fr",
          source_pids_list_attr = "unexpected_source_pids",
          calc_type = "diversity_vol",
          diversity_vol_type = 1,
          explore_diversity_vol_max_num = "{{explore_diversity_vol_max_num_candidate}}",
          export_common_diversity_vol_attr = "diversity_vol_score_candidate",
          dim_size = 128,
          check_point_ = "fr"
        ) \
        .gen_common_attr_by_lua(
          attr_map = {
            "explore_is_low_diversity_status":
            "(diversity_vol_score_candidate ~= nil and diversity_vol_score_candidate < low_diversity_vol_score_threshold_candidate) \
             or (diversity_vol_score_realshow ~= nil and diversity_vol_score_realshow < low_diversity_vol_score_threshold_realshow)"
          }
        ) \
      .end_() \
      .if_("enable_explore_fullrank_stage1_dynamic_filter_coff == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "ctr_filter_distribue_coff", "as": "xtr_weight"},
            {"name": "explore_recent_valid_click_count", "as": "user_vv"},
            {"name": "explore_fullrank_stage1_dynamic_filter_coff_exp_upper", "as": "exp_upper"},
            {"name": "explore_fullrank_stage1_dynamic_filter_coff_alpha", "as": "alpha"},
            {"name": "explore_fullrank_stage1_dynamic_filter_coff_beta", "as": "beta"},
            {"name": "explore_fullrank_stage1_dynamic_filter_coff_omega", "as": "omega"},
            {"name": "explore_fullrank_stage1_dynamic_filter_coff_max", "as": "coeff_max"},
            {"name": "explore_fullrank_stage1_dynamic_filter_coff_min", "as": "coeff_min"},
          ],
          export_common_attr = [
            {"name": "xtr_weight", "as": "ctr_filter_distribue_coff"},
          ],
          function_name = "AdjustWeightByUserVv",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_explore_gen_diversity_entropy == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_gen_diversity_entropy_item_num", "as": "item_num"}
          ],
          import_item_attr = [
            "hetu_tag_level_info_v2__hetu_level_one"
          ],
          export_common_attr = [
            {"name": "diversity_entropy", "as": "explore_rank_s1_diversity_entropy"}
          ],
          function_name = "CalcDiversiyEntropy",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .if_("explore_rank_s1_diversity_entropy > explore_rank_s1_diversity_entropy_threshold") \
          .copy_attr(
            attrs=[{
              "from_common": "ctr_filter_distribue_coff_high_entropy",
              "to_common": "ctr_filter_distribue_coff"
            }]
          ) \
        .end_() \
      .end_() \
      .if_("enable_low_active_customization_ctr_filter_distribute_coff == 1 and is_explore_new_la_user == 1") \
        .copy_attr(
          attrs=[{
            "from_common": "ctr_filter_distribue_coff_low_active",
            "to_common": "ctr_filter_distribue_coff"
          }]
        ) \
      .end_() \
      .switch_("explore_fullrank_stage1_truncate_switch") \
        .case_(4) \
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
        .case_(6) \
          .count_reco_result(
            save_count_to = "explore_reco_leaf_rank_model_pic_input_count",
            target_item = {"is_picture" : 1}
          ) \
          .count_reco_result(
            save_count_to = "explore_reco_leaf_rank_model_vd_input_count",
            target_item = {"is_picture" : 0}
          ) \
          .explore_control_hetu_count_arranger(
            hetu_level_five_attr = "hetu_tag_level_info__hetu_level_five",
            enable_hetu_control_diversity = "{{explore_enable_fr_s1_hetu_control_diversity}}",
            keep_size = "{{return math.floor(explore_reco_leaf_rank_model_vd_input_count * (1 - ctr_filter_distribue_coff))}}",
            hetu5_max_size = "{{explore_fr_s1_control_hetu5_max_size}}",
            cluster_id_attr = "hetu_sim_cluster_id",
            enable_cluster_id_control_diversity = "{{explore_rank_enable_cluster_id_control_diversity}}",
            cluster_id_control_diversity_start = "{{explore_rank_cluster_id_control_diversity_start}}",
            cluster_id_max_size = "{{explore_rank_control_cluster_id_max_size}}", 
            target_item = {"is_picture" : 0}
          ) \
          .limit(
            size = "{{return math.floor(explore_reco_leaf_rank_model_pic_input_count * (1 - pic_ctr_filter_distribue_coff))}}",
            target_item = {"is_picture" : 1}
          ) \
        .default_() \
          .count_reco_result(
            save_count_to = "explore_reco_leaf_rank_model_pic_input_count",
            target_item = {"is_picture" : 1}
          ) \
          .count_reco_result(
            save_count_to = "explore_reco_leaf_rank_model_vd_input_count",
            target_item = {"is_picture" : 0}
          ) \
          .limit(
            size = "{{return math.floor(explore_reco_leaf_rank_model_vd_input_count * (1 - ctr_filter_distribue_coff))}}",
            target_item = {"is_picture" : 0}
          ) \
          .limit(
            size = "{{return math.floor(explore_reco_leaf_rank_model_pic_input_count * (1 - pic_ctr_filter_distribue_coff))}}",
            target_item = {"is_picture" : 1}
          ) \
      .end_() \
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

  def calc_result_count_to_ab_metric(self):
    return self.flow \
      .count_reco_result(
        save_count_to = "rank_s1_follow_author_count",
        target_item = {"is_follow_author": 1},
      ) \
      .count_reco_result(
        save_count_to = "rank_s1_all_page_valid_interest_count",
        target_item = {"is_all_page_valid_interest": 1},
      ) \
      .count_reco_result(
        save_count_to = "rank_s1_new_interest_count",
        target_item = {"is_new_interest_explore": 1},
      ) \
      .count_reco_result(
        save_count_to = "rank_s1_outer_field_interest_count",
        target_item = {"is_outer_field_interest": 1},
      ) \
      .count_reco_result(
        save_count_to = "rank_s1_show_ration_level6_count",
        target_item = {"show_ration_level": 6},
      ) \
      .count_reco_result(
        save_count_to = "rank_s1_upload_time_day0_count",
        target_item = {"upload_time_day": 0}
      ) \
      .count_reco_result(
        save_count_to = "rank_s1_upload_time_day1_count",
        target_item = {"upload_time_day": 1}
      ) \
      .count_reco_result(
        save_count_to = "rank_s1_upload_time_day2_count",
        target_item = {"upload_time_day": 2}
      ) \
      .count_reco_result(
        save_count_to = "rank_s1_upload_time_day3_7_count",
        target_item = {"upload_time_day": [3, 4, 5, 6, 7]}
      ) \
      .count_reco_result(
        save_count_to = "rank_s1_upload_time_day30_180_count",
        select_item = {
          "attr_name": "upload_time_day",
          "compare_to": 30,
          "select_if": ">=",
        } \
      ) \
      .count_reco_result(
        save_count_to = "rank_s1_result_count",
      ) \
      .count_reco_result(
        save_count_to = "rank_s1_explore_show_gt_show_ration_result_count",
        select_item = {
            "attr_name": "explore_stat__real_show_count",
            "compare_to": "{{show_ration_realshow_threshold}}",
            "select_if": ">"
        } \
      ) \
      .count_reco_result(
        save_count_to = "rank_s1_explore_noncoverview_result_count",
        select_item = {
          "attr_name": "audit_hot_cover_level",
          "compare_to": 0,
          "select_if": "<=",
          "select_if_attr_missing": True
        } \
      ) \
      .count_reco_result(
        save_count_to = "rank_s1_explore_nonsenseview_result_count",
        select_item = {
          "attr_name": "audit_b_second_tag",
          "compare_to": 0,
          "select_if": "<=",
          "select_if_attr_missing": True
        } \
      ) \
      .count_reco_result(
        save_count_to = "rank_s1_bias_interest_count",
        target_item = {"is_bias_interest_tagnex": 1},
      ) \
      .send_abtest_metrics(
        metrics = [
          "rank_s1_bias_interest_count",
          "rank_s1_follow_author_count",
          "rank_s1_all_page_valid_interest_count",
          "rank_s1_new_interest_count",
          "rank_s1_outer_field_interest_count",
          "rank_s1_show_ration_level6_count",
          "rank_s1_result_count",
          "rank_s1_upload_time_day0_count",
          "rank_s1_upload_time_day1_count",
          "rank_s1_upload_time_day2_count",
          "rank_s1_upload_time_day3_7_count",
          "rank_s1_upload_time_day30_180_count",
          "rank_s1_explore_show_gt_show_ration_result_count",
          "rank_s1_explore_noncoverview_result_count",
          "rank_s1_explore_nonsenseview_result_count"
        ],
        metric_name_prefix = "explore_reco_leaf_",
      )

  def post_process(self) -> None:
    self.flow.if_("_IS_ABTEST_METRICS_SAMPLING_REQUEST_ == 1 and _IS_ONLINE_SERVICE_ == 1 and _IS_NOT_BACKUP_ == 1")
    self.calc_result_count_to_ab_metric()
    self.flow.end_()
    self.flow \
      .log_debug_info(
        common_attrs = [
          "rank_s1_result_count"
        ],
        for_debug_request_only = True
      )
