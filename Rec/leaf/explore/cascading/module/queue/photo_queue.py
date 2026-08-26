from cascading.module.queue.cascading_channel_sort_queue import ChannelSortQueuePartitioner
from cascading.module.queue.cascading_channel_sort_queue import ChannelSortQueueScorer
from cascading.module.queue.cascade_queues import cluster_variant_sort_queue
from cascading.module.queue.cascade_prerank_queues import prerank_ensemble_sort_queues
from cascading.module.queue.cascade_final_queues import final_channel_sort_queues,cascades2_value_and_rank_score_queues

class PhotoQueueParitioner(ChannelSortQueuePartitioner):
  def __init__(self, name, flow, config):
    super().__init__(name, flow, config)

  def _calculate_flag(self):
    """
    默认队列，什么也不干，框架最后赋值
    """
    self.flow.do_nothing()


class PhotoQueuePrerankScorer(ChannelSortQueueScorer):
  def __init__(self, name, flow, config):
    super().__init__(name, flow, config)

  def _caculate_score(self, flag_attr, weight_attr, left_count_attr):
    self._calc_prerank_score(flag_attr, weight_attr)

  def _calc_prerank_score(self, flag_attr, weight_attr):
    self.flow.if_("explore_prerank_use_ensembe_score == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "cascade_prerank_pctr_weight",
          "cascade_prerank_pltr_weight",
        ],
        import_item_attr = [
          "prerank_mc_pctr",
          "prerank_mc_pltr",
          "prerank_mc_pwtr",
          "prerank_mc_pftr",
          "prerank_mc_plvtr",
          "prerank_mc_plvtr2",
          "prerank_mc_psvtr",
          "prerank_mc_ptr",
          "prerank_mc_pwatch_time",
          "prerank_mc_pepstr",
          "prerank_mc_pcmtr",
          "prerank_mc_pcltr",
          "prerank_mc_peftr",
          "prerank_mc_pefctr",
          "prerank_mc_pwtd",
          "prerank_mc_pcptr",
          "cascade_prerank_pctr",
          "cascade_prerank_pltr",
        ],
        export_item_attr = [
          "prerank_mc_ensemble_pctr",
          "prerank_mc_ensemble_pltr",
          "prerank_mc_ensemble_pwtr",
          "prerank_mc_ensemble_pftr",
          "prerank_mc_ensemble_plvtr",
          "prerank_mc_ensemble_plvtr2",
          "prerank_mc_ensemble_psvtr",
          "prerank_mc_ensemble_ptr",
          "prerank_mc_ensemble_pwatch_time",
          "prerank_mc_ensemble_pepstr",
          "prerank_mc_ensemble_pcmtr",
          "prerank_mc_ensemble_pcltr",
          "prerank_mc_ensemble_peftr",
          "prerank_mc_ensemble_pefctr",
          "prerank_mc_ensemble_pwtd",
          "prerank_mc_ensemble_pcptr",
          "prerank_mc_ensemble_cascade_score"
        ],
        function_name = "CalMcPrerankEnsembleQueueScore",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .explore_calc_ensemble_score(
        use_superscript_rank = True,
        user_power_calc_v2 = "{{explore_prerank_ensemble_user_power_calc_v2}}",
        user_info_ptr_attr = "user_info_ptr",
        rank_smooth = "{{explore_prerank_smooth}}",
        rank_power_weight = "{{explore_prerank_power_weight}}",
        queues = prerank_ensemble_sort_queues,
        save_score_to_attr = self._score_attr,
        target_item = {
          flag_attr : 1
        }
      ) \
    .else_() \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "cascade_prerank_pctr_weight",
          "cascade_prerank_pltr_weight",
          "cascade_prerank_prstr_weight",
          "cascade_prerank_calc_type",
          "prerank_ltr_weight",
          "prerank_ctr_weight",
          "prerank_wtd_weight",
          "prerank_life_ctr_weight",
          "prerank_duration_weight",
          "prerank_fountain_efficiency_vv_weight",
          "prerank_action_once_weight",
          "prerank_ctr_comirec_weight",
          "prerank_ctr_long_seq_weight",
          "prerank_cover_view_predict_score_weight",
          "prerank_sense_view_predict_score_weight",
          "explore_cover_sense_view_score_version",
          "prerank_user_age_interest_tagnex_tgi_score_weight",
        ],
        import_item_attr = [
          "cascade_prerank_pctr",
          "cascade_prerank_pltr",
          "cascade_prerank_prstr",
          "prerank_ltr",
          "prerank_ctr",
          "prerank_wtd",
          "prerank_life_ctr",
          "prerank_duration_score",
          "fountain_efficiency_vv",
          "prerank_mc_pctr", #prerank添加mc pxtr
          "prerank_mc_pltr",
          "prerank_mc_pwtr",
          "prerank_mc_pftr",
          "prerank_mc_plvtr",
          "prerank_mc_plvtr2",
          "prerank_mc_psvtr",
          "prerank_mc_ptr",
          "prerank_mc_pwatch_time",
          "prerank_mc_pepstr",
          # "prerank_mc_pcestr",
          "prerank_mc_pcmtr",
          "prerank_mc_plivingtr",
          "prerank_mc_pcltr",
          # "prerank_mc_pdtr",
          "prerank_mc_phtr",
          "prerank_mc_peftr",
          "prerank_mc_pefctr",
          "prerank_mc_pcptr",
          "prerank_mc_pwtd",
          "prerank_action_once_score",
          "prerank_ctr_comirec",
          "prerank_ctr_long_seq",
          "cover_view_predict_score",
          "sense_view_predict_score",
          "cover_view_predict_score_v2",
          "sense_view_predict_score_v2",
          "user_age_interest_tagnex_tgi_score",
        ],
        export_item_attr = [
          {"name": "cascade_prerank_score", "as": self._score_attr}
        ],
        function_name = "CalPreRankScore",
        class_name = "ExploreLightFunctionSetV2",
        target_item={ flag_attr: 1 }
      ) \
    .end_() \
    .if_("enable_explore_partial_time_based_interest_stat == 1") \
      .partial_time_based_interest_stat() \
    .end_() \
    .if_("enable_explore_partial_time_based_tagnex_stat == 1") \
      .partial_time_based_tagnex_stat() \
    .end_() \
    .if_("enable_explore_prerank_boost == 1") \
      .if_("enable_partial_time_based_interest_boost_prerank == 1") \
        .partial_time_based_interest_boost(self._score_attr, flag_attr, stage="prerank") \
      .end_() \
      .if_("enable_partial_time_based_tagnex_boost_prerank == 1") \
        .partial_time_based_tagnex_boost(self._score_attr, flag_attr, stage="prerank") \
      .end_() \
      .if_("enable_prerank_search_retri_boost == 1", to_be_delete = "date=2024-05-29;committer=liucong03") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "prerank_search_retri_boost_coef"
          ],
          import_item_attr = [
            {"name": self._score_attr, "as": "cascade_prerank_score"},
          ],
          export_item_attr = [
            {"name": "cascade_prerank_score", "as": self._score_attr},
          ],
          export_common_attr = [
            {"name": "boost_count", "as": "prerank_search_retri_photo_boost_count"},
            {"name": "total_count", "as": "prerank_search_retri_photo_total_count"},
          ],
          function_name = "BoostSearchRetri",
          class_name = "ExploreLightFunctionSetV2",
          target_item = { flag_attr: 1 }
        ) \
        .perflog_attr_value(
          check_point = "prerank_photo_search_retri_boost",
          common_attrs = [
            "prerank_search_retri_photo_boost_count",
            "prerank_search_retri_photo_total_count",
          ],
        ) \
      .end_() \
      .if_("explore_mc_enable_not_cover_audit_discount_for_first_page == 1") \
        .not_cover_audit_photo_discount(self._score_attr) \
      .end_() \
      .if_("enable_prerank_not_audit_discount == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "prerank_not_audit_discount_coef",
          ],
          import_item_attr = [
            {"name": self._score_attr, "as": "cascade_prerank_score"},
            "audit_b_second_tag",
          ],
          export_item_attr = [
            {"name": "cascade_prerank_score", "as": self._score_attr},
          ],
          function_name = "DiscountNotAuditPhotos",
          class_name = "ExploreLightFunctionSetV2",
          target_item = { flag_attr: 1 }
        ) \
      .end_() \
      .if_("enable_cascading_prerank_personified_author_boost == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "cascading_prerank_personified_author_boost_coef", "as": "personified_author_coeff"},
            {"name": "cascading_prerank_blacklist_author_boost_coef", "as": "blacklist_author_coeff"},
            {"name": "explore_personified_author_fans_thre_max", "as": "author_fans_thre_max"},
            {"name": "explore_personified_author_fans_thre_min", "as": "author_fans_thre_min"},
            {"name": "cascading_prerank_merchant_cart_boost_coef", "as": "merchant_cart_coeff"},
            {"name": "cascading_prerank_young_women_boost_coef", "as": "young_women_coeff"},
            {"name": "cascading_prerank_age_segment_18_23_coeff", "as": "age_segment_18_23_coeff"},
            "basic_info_gender_v2",
            "basic_info_age_segment_v2",
          ],
          import_item_attr = [
            {"name": "author__fans_count", "as": "author_fans_count"},
            {"name": "eyeshot_source", "as": "eyeshot_source"},
            {"name": "merchant_photo_cart_relation", "as": "cart_relation"},
            {"name": "merchant_item_info__item_id_list", "as": "cart_itemlist"},
            {"name": "live_photo_info__is_living", "as": "is_living"},
            {"name": self._score_attr, "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": self._score_attr},
          ],
          function_name = "PersonifiedAuthorBoost",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            flag_attr : 1
          }
        ) \
      .end_() \
      .if_("enable_cascading_prerank_female_porn_discount == 1", to_be_delete = "date=2024-05-29;committer=guohao") \
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
          target_item = {
            flag_attr : 1
          }
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "cascading_prerank_female_porn_discount_coef", "as": "boost_discount_coeff"},
          ],
          import_item_attr = [
            {"name": self._score_attr, "as": "score"},
          ],
          export_item_attr = [
            {"name": "score", "as": self._score_attr},
          ],
          function_name = "BoostOrDiscountV2",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_porn_for_female" : 1,
            flag_attr : 1
          }
        ) \
      .end_() \
      .if_("enable_cascading_prerank_sexy_sensitive_porn_discount == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "uSexyInterestScore",
            {"name": "cascading_prerank_sexy_sensitive_upper_bound", "as": "upper_bound"},
            {"name": "cascading_prerank_sexy_sensitive_lower_bound", "as": "lower_bound"},
          ],
          import_item_attr = [
            "audit_b_second_tag",
          ],
          export_item_attr = [
            "is_porn_for_sexy_sensitive",
          ],
          function_name = "IsPornForSexySensitive",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            flag_attr : 1
          }
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "cascading_prerank_sexy_sensitive_porn_discount_coef", "as": "boost_discount_coeff"},
          ],
          import_item_attr = [
            {"name": self._score_attr, "as": "score"},
          ],
          export_item_attr = [
            {"name": "score", "as": self._score_attr},
          ],
          function_name = "BoostOrDiscountV2",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_porn_for_sexy_sensitive" : 1,
            flag_attr : 1
          }
        ) \
      .end_() \
      .if_("enable_mc_calc_search_score == 1") \
        .mc_calc_search_score(flag_attr) \
      .end_() \
      .if_("enable_prerank_search_score_boost == 1") \
        .prerank_search_score_boost(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_prerank_unbias_interest_photo_boost == 1") \
        .if_("enable_prerank_unbias_interest_photo_boost_vv_adjust == 1") \
          .user_vv_type_weight_adjust("explore_prerank_unbias_interest_photo_boost_coeff") \
        .end_() \
        .if_("enable_prerank_unbias_interest_photo_boost_cocoon_adjust == 1") \
          .user_cocoon_weight_adjust("explore_prerank_unbias_interest_photo_boost_coeff") \
        .end_() \
        .unbias_interest_photo_boost(self._score_attr, "prerank") \
      .end_() \
      .if_("enable_hot_list_coef_calculator == 1") \
        .hot_list_coef_calculator() \
      .end_() \
      .if_("enable_prerank_hot_list_photo_boost == 1") \
        .hot_list_photo_boost(self._score_attr, flag_attr, "prerank") \
      .end_() \
      .if_("enable_prerank_short_uninterest_photo_discount == 1") \
        .short_uninterest_photo_discount(self._score_attr, "prerank") \
      .end_() \
      .if_("enable_prerank_marketing_compensation_discount == 1") \
        .prerank_marketing_compensation_discount(self._score_attr, flag_attr) \
      .end_() \
      .if_("explore_enable_prerank_olympic_latest_boost == 1") \
        .prerank_olympic_latest_boost(self._score_attr, flag_attr) \
      .end_() \
      .if_("explore_prerank_low_cost_photo_discount == 1") \
        .prerank_low_cost_photo_discount(self._score_attr, flag_attr) \
      .end_() \
      .if_("explore_enable_prerank_top_author_new_boost == 1") \
        .prerank_top_author_new_boost(self._score_attr, flag_attr) \
      .end_() \
      .if_("explore_enable_prerank_new_hot_boost == 1") \
        .prerank_new_hot_boost(self._score_attr, flag_attr) \
      .end_() \
      .if_("explore_enable_prerank_update_bar_boost == 1") \
        .prerank_update_bar_boost(self._score_attr, flag_attr) \
      .end_() \
      .if_("explore_enable_prerank_user_intrest_adjust == 1") \
        .prerank_user_intrest_adjust(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_explore_cs_photo_boost_prerank == 1") \
        .explore_cascade_cs_boost(self._score_attr, flag_attr, "prerank") \
      .end_() \
      .if_("enable_explore_not_correlation_deboost_prerank == 1") \
        .explore_cover_video_not_correlation_deboost(self._score_attr, flag_attr, "prerank") \
      .end_() \
      .if_("enable_explore_first_refresh_good_boost_prerank == 1 and is_first_refresh == 1") \
        .explore_first_refresh_good_boost(self._score_attr, flag_attr, "prerank") \
      .end_() \
      .if_("enable_explore_interest_score_history_boost_prerank == 1 and interest_score_based_valid_user == 1") \
        .interest_score_history_boost(self._score_attr, flag_attr) \
      .end_() \
    .end_() \
    .copy_attr(
      attrs=[{
        "from_item": self._score_attr,
        "to_item": "cascade_prerank_score"
      }]
    ) \
    .if_("enable_prerank_select_photo_by_interest == 1") \
      .prerank_select_photo_by_interest(self._score_attr, flag_attr) \
      .copy_item_meta_info(
        save_item_seq_to_attr = "prerank_final_index_photo",
        target_item = {
          flag_attr : 1
        }
      ) \
    .end_() \
    .if_("enable_cal_prerank_adjust_diversity_distribution == 1") \
      .enrich_attr_by_light_function(
        import_item_attr = [
          {"name": "hetu_tag_level_info__hetu_level_one", "as": "extract_hetu_tag_list"},
        ],
        export_item_attr = [
          {"name": "first_hetu_tag", "as": "hetu_level_one_top1"},
        ],
        function_name = "ExtractFirstHetuTag",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .enrich_attr_by_light_function(
        import_item_attr = [
          "hetu_level_one_top1",
        ],
        export_item_attr = [
          {"name": "hetu_level_one_ratio", "as": "explore_prerank_hetu_level_one_ratio"},
        ],
        function_name = "CalHetuOneRatio",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \

class PhotoQueueCascadingScorer(ChannelSortQueueScorer):
  def __init__(self, name, flow, config):
    super().__init__(name, flow, config)

  def _caculate_score(self, flag_attr, weight_attr, left_count_attr):
    self._calc_cascading_score(flag_attr, weight_attr, left_count_attr)

  def _calc_cascading_score(self, flag_attr, weight_attr, left_count_attr):
    self.flow \
    .if_("explore_mc_enable_mc_cluster_862_uninterest_cluster_by_u2c == 1") \
      .sort(
        score_from_attr = "cascade_explore_u2c_score",
        update_score = False
      ) \
      .pack_item_attr(
        item_source = {
          "reco_results": True,
        },
        mappings = [{
          "from_item_attr": "hetu_sim_cluster_id",
          "to_common_attr": "user_cluster862_sorted_list",
          "aggregator": "concat",
          "dedup_to_common_attr": True
        }]
      ) \
    .end_() \
    .if_("enable_mc_empctr_cluster == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "empctr_truncate_weight"
        ],
        import_item_attr = [
          "explore_stat__real_show_count",
          "explore_stat__click_count"
        ],
        export_item_attr = [
          "empctr_cluster"
        ],
        function_name = "CalEmpCtrFlagCluster",
        class_name = "ExploreLightFunctionSetV2"
      ) \
    .end_() \
    .switch_("explore_mc_cluster_method") \
      .case_("photo_duration_quantile", to_be_delete = "date=2023-11-16;committer=tangzhucheng03") \
        .sort(
          score_from_attr = "duration_ms",
          desc = False,
          target_item={ flag_attr: 1 },
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "explore_mc_time_cluster_num",
          ],
          export_item_attr = [
            "cascade_cluster_id",
          ],
          function_name = "EqualSizeCluster",
          class_name = "ExploreLightFunctionSetV2",
          target_item={ flag_attr: 1 },
        ) \
      .default_() \
        .if_("explore_mc_enable_mc_filter_u2c_score_by_cluster_862_lv1 == 1") \
          .split_string(
            input_common_attr = "explore_mc_filter_u2c_score_by_cluster_862_lv1_classes_str",
            output_common_attr = "explore_mc_filter_u2c_score_by_cluster_862_lv1_classes",
            delimiters = ",",
            skip_empty_tokens = True,
            trim_spaces = True,
            parse_to_int = True
          ) \
        .end_() \
        .if_("enable_unbias_interest_cluster_in_mc_s1 == 1") \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "explore_unbias_interest_cluster_in_mc_s1_prefix", "as": "key_prefix"},
              "basic_info_age_segment_v2",
              "basic_info_gender_v2",
            ],
            export_common_attr = [
              {"name": "user_age_gender_key", "as": "unbias_interest_user_age_gender_key"}
            ],
            function_name = "GetUserAgeGenderKey",
            class_name = "ExploreLightFunctionSetV2",
          ) \
          .get_kconf_params(
            kconf_configs = [{
              "kconf_key": "reco.offline.unbias_interest_cid_list_map",
              "json_path": "{{unbias_interest_user_age_gender_key}}",
              "value_type": "list_int64",
              "export_common_attr": "unbias_interest_in_mc_s1_cids"
            }]
          ) \
        .end_() \
        .if_("enable_unbias_interest_hetu_cluster_in_mc_s1 == 1") \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "explore_unbias_interest_hetu_cluster_in_mc_s1_prefix", "as": "key_prefix"},
              "basic_info_age_segment_v2",
              "basic_info_gender_v2",
            ],
            export_common_attr = [
              {"name": "user_age_gender_key", "as": "unbias_interest_hetu_user_age_gender_key"}
            ],
            function_name = "GetUserAgeGenderKey",
            class_name = "ExploreLightFunctionSetV2",
          ) \
          .get_kconf_params(
            kconf_configs = [{
              "kconf_key": "reco.offline.unbias_interest_hetu_list_map",
              "json_path": "{{unbias_interest_hetu_user_age_gender_key}}",
              "value_type": "list_int64",
              "export_common_attr": "unbias_interest_in_mc_s1_hetu"
            }]
          ) \
        .end_() \
        .if_("enable_mc_s1_c2c_cluster_num_vv_adjust == 1") \
          .user_vv_type_int_value_adjust("explore_mc_user_uninterest_cluster_862_count") \
        .end_() \
        .if_("enable_mc_s1_c2c_cluster_num_cocoon_adjust == 1") \
          .user_cocoon_int_value_adjust("explore_mc_user_uninterest_cluster_862_count") \
        .end_() \
        .if_("all_page_interest_user_migration_flag == 0") \
          .set_attr_value(
            common_attrs=[
              {
                "name": "explore_mc_enable_mc_all_page_valid_interest_cluster",
                "type": "int",
                "value": 0
              }
            ]
          ) \
        .end_() \
        .explore_cluster_by_custom_rule(
          skip = 0,
          use_extra_page = "{{explore_use_more_page_photos}}",
          user_info_ptr_attr = "user_info_ptr",
          save_cluster_id_to_attr = "cascade_cluster_id",
          enable_user_profile_top_hetu_level_one_cluster = "{{explore_enable_use_hetu_level1_id}}",
          enable_user_profile_top_hetu_level_two_cluster = "{{explore_enable_use_hetu_level2_id}}",
          enable_user_profile_top_hetu_level_three_cluster = "{{explore_enable_use_hetu_level3_id}}",
          enable_use_photo_age_cluster = "{{explore_enable_use_photo_age_cluster}}",
          enable_expired_time_on_action_list = "{{explore_enable_expired_time_on_action_list}}",
          expired_gap_second = "{{explore_expired_gap_second}}",
          user_profile_tag_score_limit = "{{mc_cluster_tag_score_limit}}", # 4
          user_profile_limit_num = "{{mc_cluster_limit_hetulevel1_num}}", # 3
          enable_use_real_show_list = "{{mc_cluster_use_real_show_list}}",
          enable_use_click_list = "{{mc_cluster_use_click_list}}",
          enable_use_like_list = "{{explore_mc_cluster_use_like_list}}",
          enable_use_follow_list = "{{explore_mc_cluster_use_follow_list}}",
          enable_use_forward_list = "{{explore_mc_cluster_use_forward_list}}",
          enable_use_collect_list = "{{explore_mc_cluster_use_collect_list}}",
          enable_use_comment_list = "{{explore_mc_cluster_use_comment_list}}",
          enable_user_video_play_stats = "{{explore_mc_cluster_use_video_play_list}}",
          enable_use_fountain_real_show_list = "{{enable_use_fountain_real_show_list}}",
          enable_use_fountain_click_list = "{{enable_use_fountain_click_list}}",
          enable_use_fountain_like_list = "{{enable_use_fountain_like_list}}",
          enable_use_fountain_follow_list = "{{enable_use_fountain_follow_list}}",
          enable_use_fountain_forward_list = "{{enable_use_fountain_forward_list}}",
          real_show_weight = "{{explore_mc_cluster_real_show_weight}}", # 1.0
          click_weight = "{{explore_mc_cluster_click_weight}}", #2.0,
          like_weight = "{{explore_mc_cluster_like_weight}}", #3.0,
          follow_weight = "{{explore_mc_cluster_follow_weight}}", # 3.0
          forward_weight = "{{explore_mc_cluster_forward_weight}}", # 3.0
          comment_weight = "{{explore_mc_cluster_comment_weight}}",
          collect_weight = "{{explore_mc_cluster_collect_weight}}",
          video_play_weight = "{{explore_mc_cluster_video_play_weight}}",
          min_effective_play_length = "{{explore_mc_cluster_effective_play_min_length}}",
          fountain_real_show_weight = "{{fountain_real_show_weight}}", # 1.0
          fountain_click_weight = "{{fountain_click_weight}}", #2.0,
          fountain_like_weight = "{{fountain_like_weight}}", #3.0,
          fountain_follow_weight = "{{fountain_follow_weight}}", # 3.0
          fountain_forward_weight = "{{fountain_forward_weight}}",
          enable_mc_use_realshow_no_click = "{{enable_mc_use_realshow_no_click}}",
          enable_uninterested_hetu_level_one_cluster = "{{mc_cluster_uninterested_use_hetu_level1_id}}",
          enable_uninterested_hetu_level_two_cluster = "{{mc_cluster_uninterested_use_hetu_level2_id}}",
          enable_uninterested_hetu_level_three_cluster = "{{mc_cluster_uninterested_use_hetu_level3_id}}",
          uninterested_tag_score_limit = "{{mc_cluster_uninterested_tag_score_limit}}", # 4
          uninterested_limit_num = "{{mc_cluster_uninterested_limit_num}}",
          real_show_no_click_weight = "{{mc_cluster_real_show_no_click_weight}}",
          uninterested_expired_gap_second = "{{mc_cluster_uninterested_expired_gap_second}}",
          enable_mc_use_xhs_tag = "{{enable_mc_use_xhs_tag}}",
          enable_colossus_cluster = "{{enable_use_colossus_cluster}}",
          enable_mc_explore_cluster = "{{enable_mc_explore_cluster}}",
          input_colossus_attr_one = "sim_one_tags",
          input_colossus_attr_two = "sim_two_tags",
          input_colossus_attr_three = "sim_three_tags",
          input_colossus_attr_explore = "sim_explore_tags",
          enable_mc_interact_cluster = "{{enable_mc_interact_cluster}}",
          input_colossus_attr_interact = "sim_interact_tags",
          hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
          hetu_level_two_attr = "hetu_tag_level_info__hetu_level_two",
          hetu_level_three_attr = "hetu_tag_level_info__hetu_level_three",
          hetu_level_four_attr = "hetu_tag_level_info__hetu_level_four",
          age_cluster_attr = "upload_time",
          mc_age_gap_str = "{{mc_mc_age_gap_str}}",
          input_xhs_hetu_tags_attr = "xhs_hetu_tags",
          enable_mc_follow_author_cluster = "{{enable_mc_follow_author_cluster}}",
          enable_mc_follow_author_cluster_first = "{{enable_mc_follow_author_cluster_first}}",
          white_author_attr = "is_white_author",
          is_follow_author_attr = "is_follow_author",
          enable_white_author_bucket = "{{explore_interactive_enable_white_author_bucket}}",
          enable_rough_default_cluster = "{{enable_rough_default_cluster}}",
          enable_mc_explore_cluster_mix = "{{enable_mc_explore_cluster_mix}}",
          enable_get_default_hetu_level_one_cluster = "{{enable_get_default_hetu_level_one_cluster}}",
          enable_get_default_hetu_level_two_cluster = "{{enable_get_default_hetu_level_two_cluster}}",
          enable_get_default_hetu_level_three_cluster = "{{enable_get_default_hetu_level_three_cluster}}",
          enable_set_bucket_limit_num_by_ratio = "{{enable_set_bucket_limit_num_by_ratio}}",
          enable_ignore_profile_candidate_limit_cut = "{{enable_ignore_profile_candidate_limit_cut}}",
          enable_mc_empctr_cluster = "{{enable_mc_empctr_cluster}}",
          empctr_cluster_flag_attr = "empctr_cluster",
          mc_realtime_bucket_limit_num_ratio = "{{mc_realtime_bucket_limit_num_ratio}}",
          target_item={ flag_attr: 1 },
          perf_checkpoint = "cascade",
          enable_shortterm_interest_cluster_opt = "{{enable_shortterm_interest_cluster_opt}}",
          shortterm_hetu_attr = "interest_explore_shortterm_hetu",
          enable_mc_cluster_862_cluster = "{{explore_mc_enable_mc_cluster_862_cluster}}",
          enable_mc_cluster_862_one_cluster = "{{explore_mc_enable_mc_cluster_862_one_cluster}}",
          enable_mc_cluster_862_cluster_ignore_recent_realshow = "{{explore_mc_enable_mc_cluster_862_cluster_ignore_recent_realshow}}",
          mc_user_interest_cluster_862_count = "{{explore_mc_user_interest_cluster_862_count}}",
          input_user_interest_cluster_862_attr = "uOldMmuClusterId300ListList",
          cluster_862_attr = "mounted_interest_cluster_id",
          enable_mc_cluster_862_uninterest_cluster = "{{explore_mc_enable_mc_cluster_862_uninterest_cluster}}",
          mc_user_uninterest_cluster_862_count = "{{explore_mc_user_uninterest_cluster_862_count}}",
          input_user_recent_realshow_cluster_862_attr = "user_recent_realshow_cids",
          enable_mc_cluster_862_uninterest_cluster_by_u2c = "{{explore_mc_enable_mc_cluster_862_uninterest_cluster_by_u2c}}",
          enable_mc_cluster_862_uninterest_cluster_impression_filter = "{{explore_mc_enable_cluster_862_uninterest_cluster_impression_filter}}",
          enable_mc_cluster_862_uninterest_cluster_cover_filter = "{{explore_mc_enable_cluster_862_uninterest_cluster_cover_filter}}",
          input_user_cluster862_sorted_list_attr = "user_cluster862_sorted_list",
          enable_mc_short_develop_interest_cluster = "{{explore_mc_enable_mc_short_develop_interest_cluster}}",
          mc_short_develop_interest_cluster_count = "{{explore_mc_short_develop_interest_cluster_count}}",
          input_user_short_develop_interest_tags_attr = "user_short_develop_interest_cid_list",
          enable_mc_all_page_valid_interest_cluster = "{{explore_mc_enable_mc_all_page_valid_interest_cluster}}",
          mc_all_page_valid_interest_cluster_count = "{{explore_mc_all_page_valid_interest_cluster_count}}",
          input_user_all_page_valid_interest_tags_attr = "uPicValidInterestClusterIdList",
          cluster_632_attr = "cluster_id_632",
          enable_mc_cluster_hot_list_cluster = "{{enable_mc_cluster_hot_list_cluster}}",
          is_hot_list_flag_attr = "is_hot_list_flag",
          audit_b_second_tag_attr = "audit_b_second_tag",
          audit_hot_cover_level_attr = "audit_hot_cover_level",
          enable_mc_filter_u2c_score_by_cluster_862_lv1 = "{{explore_mc_enable_mc_filter_u2c_score_by_cluster_862_lv1}}",
          filter_u2c_score_by_cluster_862_lv1_classes_attr = "explore_mc_filter_u2c_score_by_cluster_862_lv1_classes",
          enable_mc_unbias_interest_cluster = "{{enable_unbias_interest_cluster_in_mc_s1}}",
          mc_user_unbias_interest_cluster_count = "{{mc_user_unbias_interest_cluster_count}}",
          input_user_unbias_interest_cluster_attr = "unbias_interest_in_mc_s1_cids",
          enable_mc_unbias_interest_hetu_cluster = "{{enable_unbias_interest_hetu_cluster_in_mc_s1}}",
          mc_user_unbias_interest_hetu_cluster_count = "{{mc_user_unbias_interest_hetu_cluster_count}}",
          input_user_unbias_interest_hetu_cluster_attr = "unbias_interest_in_mc_s1_hetu",
          enable_interest_vary_by_scenario = "{{enable_mc_short_interest_vary_by_scenario}}",
          gamora_interest_ratio = "{{mc_short_gamora_interest_ratio}}",
          enable_mc_reach_content_cluster_first = "{{explore_mc_enable_mc_reach_content_cluster_first}}",
          reach_content_attr = "reach_content",
        ) \
    .end_() \
    .enrich_attr_by_light_function(
      import_common_attr = [
        "high_quality_tags",
      ],
      import_item_attr = [
        "cascade_cluster_id",
      ],
      export_item_attr = [
        "is_explore_photo",
        "is_high_quality_explore_photo",
      ],
      function_name = "IsExplorePhoto",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1
      }
    ) \
    .perflog_attr_value(
      check_point = "explore_interest_explore",
      item_attrs = [
        "is_explore_photo",
        "is_high_quality_explore_photo",
      ],
      aggregator = "sum",
    ) \
    .explore_cluster_variant_sort_v2_enrich(
      check_point = "cascade",
      use_superscript_rank = True,
      cluster_attr_name = "cascade_cluster_id",
      hetu_level_one_name = "hetu_tag_level_info__hetu_level_one",
      global_cut_ratio = "{{" + weight_attr + "}}",  #
      min_survival = "{{mc_bucket_shrink_min_num}}",
      use_reciprocal = "{{explore_use_multiply_fusion_s1}}",
      use_reciprocal_new_value_seq_fusion = "{{mc_use_new_value_seq_fusion}}",
      size_limit = "{{mc_final_candidate_num}}",
      user_info_ptr_attr = "user_info_ptr",
      action_day = "{{mc_variant_weight_action_day_num}}",
      enable_dynamic_weight_by_user_degree = "{{mc_enable_user_differently_by_degree_s1}}",
      enable_variant_cut_ratio = "{{explore_cascade_enable_variant_cut_ratio}}",
      variant_cut_ratio = "{{explore_cascade_variant_cut_ratio}}",
      save_score_to_attr = self._score_attr,
      rank_smooth = "{{explore_mc1_rank_smooth}}",
      use_rank_as_score = "{{cascade_channel_sort_use_rank_as_score}}",
      interest_explore_cluster_ratio_adjust = "{{enable_mc_interest_explore_cluster_ratio_adjust}}",
      interest_explore_cluster_ratio = "{{mc_interest_explore_cluster_ratio}}",
      realshow_no_click_cluster_ratio_adjust = "{{enable_mc_realshow_no_click_cluster_ratio_adjust}}",
      realshow_no_click_cluster_ratio = "{{mc_realshow_no_click_cluster_ratio}}",
      hot_list_cluster_ratio_adjust = "{{enable_mc_hot_list_cluster_ratio_adjust}}",
      hot_list_cluster_ratio = "{{mc_hot_list_cluster_ratio}}",
      colossus_interest_cluster_ratio_adjust = "{{colossus_interest_cluster_ratio_adjust}}",
      colossus_interest_cluster_ratio = "{{colossus_interest_cluster_ratio}}",
      colossus_interest_cluster_min_survival = "{{colossus_interest_cluster_min_survival}}",
      reach_content_cluster_ratio_adjust = "{{explore_mc_s1_reach_content_cluster_ratio_adjust}}",
      reach_content_cluster_ratio = "{{explore_mc_s1_reach_content_cluster_ratio}}",
      reach_content_cluster_min_survival = "{{explore_mc_s1_reach_content_cluster_min_survival}}",
      short_develop_interest_cluster_ratio_adjust = "{{explore_enable_short_develop_interest_cluster_ratio_adjust}}",
      short_develop_interest_cluster_ratio = "{{explore_mc_short_develop_interest_cluster_ratio}}",
      all_page_valid_interest_cluster_ratio_adjust = "{{explore_enable_all_page_valid_interest_cluster_ratio_adjust}}",
      all_page_valid_interest_cluster_ratio = "{{explore_mc_all_page_valid_interest_cluster_ratio}}",
      two_times_sort = "{{explore_enable_mc_two_times_sort}}",
      first_time_cut_ratio = "{{explore_mc_first_time_sort_cut_ratio}}",
      use_fractile_in_ensemble_sort = "{{explore_mc_s1_use_fractile_in_ensemble_sort}}",
      fractile_in_ensemble_sort_type = "{{explore_mc_s1_fractile_in_ensemble_sort_type}}",
      queues = cluster_variant_sort_queue,
      save_cluster_id_common_attr = "mc_s1_cluster_id",
      save_cluster_cnt_common_attr = "mc_s1_cluster_cnt",
      save_cluster_cnt_after_truncaton_common_attr = "mc_s1_cluster_after_truncation_cnt",
      target_item = { # 这个 processor 会对视频做逻辑截断，这里只对打上标签的 item 进行处理
        flag_attr : 1
      }
    ) \

    self.flow.log_debug_info(
      common_attrs=[],
      item_attrs=['cascade_score', 'mc_ensemble_pctr', self._score_attr],
      item_num_limit=10,
      target_item = {
        flag_attr : 1,
      })

class PhotoQueueFinalScorer(ChannelSortQueueScorer):
  def __init__(self, name, flow, config):
    super().__init__(name, flow, config)

  def _merchant_boost_score_by_buyer_type(self, flag_attr):
    self.flow \
    .if_("explore_mc_enable_merchant_photo_boost == 1") \
      .enrich_attr_by_light_function( # 计算挂车粗排权重系数
        import_common_attr = [
          {"name": "merchant_buyer_type", "as": "buyer_type"},
          {"name": "explore_mc_merchant_photo_boost_coef", "as": "buyer_boost_coef"},
        ],
        export_common_attr = [
          {"name": "merchant_boost_coef", "as": "mc_merchant_photo_boost_coef"}
        ],
        function_name = "MerchantCalcBoostCoef",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "mc_merchant_photo_boost_coef", "as": "boost_discount_coeff"},
        ],
        import_item_attr = [
          {"name": self._score_attr, "as": "score"},
        ],
        export_item_attr = [
          {"name": "score", "as": self._score_attr},
        ],
        function_name = "BoostOrDiscountV2",
        class_name="ExploreLightFunctionSetV2",
        target_item = {
          flag_attr : 1,
          "is_merchant_cart" : 1
        }
      ) \
    .end_() \
    .if_("explore_mc_enable_merchant_price_inferior_reduce_weight == 1", to_be_delete = "date=2024-05-29;committer=zhanglinjiang") \
      .enrich_attr_by_light_function( #【产品需求】【挂车短视频】产品侧要求对挂价格力劣质商品的短视频打压
        import_common_attr = [
          {"name": "explore_mc_merchant_price_inferior_reduce_weight", "as": "boost_discount_coeff"},
        ],
        import_item_attr = [
          {"name": self._score_attr, "as": "score"},
        ],
        export_item_attr = [
          {"name": "score", "as": self._score_attr},
        ],
        function_name = "BoostOrDiscountV2",
        class_name="ExploreLightFunctionSetV2",
        target_item = {
          flag_attr : 1,
          "is_merchant_cart" : 1,
          "price_info": 102
        }
      ) \
    .end_() \
    .if_("explore_mc_enable_merchant_live_boost == 1") \
      .enrich_attr_by_light_function( # 计算live头像粗排权重系数 
        import_common_attr = [
          {"name": "merchant_buyer_type", "as": "buyer_type"},
          {"name": "explore_mc_merchant_live_boost_coef", "as": "buyer_boost_coef"},
        ],
        export_common_attr = [
          {"name": "merchant_boost_coef", "as": "mc_merchant_live_boost_coef"}
        ],
        function_name = "MerchantCalcBoostCoef",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "mc_merchant_live_boost_coef", "as": "boost_discount_coeff"},
        ],
        import_item_attr = [
          {"name": self._score_attr, "as": "score"},
        ],
        export_item_attr = [
          {"name": "score", "as": self._score_attr},
        ],
        function_name = "BoostOrDiscountV2",
        class_name="ExploreLightFunctionSetV2",
        target_item = {
          flag_attr : 1,
          "is_merchant_living" : 1
        }
      ) \
    .end_()

  def _impression_audit_adjust(self):
    self.flow \
      .if_("enable_impression_audit_adjust == 1") \
        .transform_item_attr( # 观感审二级字段大于0才是已审核
          mappings = [{
            "check_attr_name": "audit_b_second_tag",
            "check_attr_type": "int",
            "output_attr_name": "is_impression_audit",
            "output_attr_type": "int",
            "output_default_value": 0,
            "rules": [{
              "check_range": {
                "lower_bound": 1
              },
              "output_value": 1
            }]
          }]
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "impression_audit_adjust_coeff_map_str", "as": "adjust_coeff_map_str_attr"},
            {"name": "impression_audit_emp_xtr_adjust_flag", "as": "emp_xtr_adjust_flag"},
            {"name": "impression_audit_emp_ctr_avg", "as": "emp_ctr_avg"},
            {"name": "impression_audit_emp_watchtime_avg", "as": "emp_watchtime_avg"},
            {"name": "impression_audit_emp_xtr_coeff_a", "as": "emp_xtr_coeff_a"},
            {"name": "impression_audit_emp_xtr_coeff_b", "as": "emp_xtr_coeff_b"}
          ],
          import_item_attr = [
            {"name": "content_safety_level_with_namespace__level_hot_online", "as": "audit_level_attr"},
            {"name": self._score_attr, "as": "ensemble_score_attr"},
            "upload_time",
            {"name": "explore_stat__real_show_count", "as": "realshow_count"},
            {"name": "explore_stat__click_count", "as": "click_count"},
            {"name": "explore_stat__view_length_sum", "as": "watchtime_sum"}
          ],
          export_item_attr = [
            {"name": "ensemble_score_attr", "as": self._score_attr},
          ],
          function_name = "AuditAdjustScore",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_impression_audit": 1,
          },
        ) \
      .end_()
    return self

  # 用户已知兴趣提权
  def _user_interest_cluster_862_adjust(self, flag_attr):
    self.flow \
      .if_("enable_mc_s2_user_uninterest_cluster_862_adjust == 1") \
        .enrich_attr_by_light_function( # 用户新兴趣调权
          import_common_attr = [
            {"name": "explore_mc_s2_user_uninterest_cluster_862_adjust_coeff_a", "as": "coeff_a"},
            {"name": "explore_mc_s2_user_uninterest_cluster_862_adjust_coeff_b", "as": "coeff_b"},
            {"name": "explore_mc_s2_user_uninterest_cluster_862_adjust_coeff_c", "as": "coeff_c"},
            {"name": "explore_mc_s2_user_uninterest_cluster_862_adjust_cluster_type", "as": "cluster_type"},
            {"name": "explore_mc_s2_user_uninterest_cluster_862_adjust_version", "as": "adjust_version"},
            {"name": "explore_mc_s2_user_uninterest_cluster_862_adjust_count", "as": "adjust_count"},
          ],
          import_item_attr = [
            "hetu_sim_cluster_id",
            "cascade_cluster_id",
            {"name": self._score_attr, "as": "ensemble_score_attr"},
          ],
          export_item_attr = [
            {"name": "ensemble_score_attr", "as": self._score_attr},
          ],
          function_name = "UserCluster862Adjust",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            flag_attr : 1
          }
        ) \
      .end_()
    return self

  def _beauty_caption_adjust(self, flag_attr):
    self.flow \
      .if_("enable_mc_s2_beauty_caption_adjust == 1") \
        .sort(
          score_from_attr = self._score_attr,
          target_item = {
            flag_attr : 1
          }
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "mc_final_candidate_num",
            {"name": "explore_mc_s2_beauty_caption_boost_coeff", "as": "boost_coeff"},
            {"name": "explore_mc_s2_beauty_caption_special_hetu_proportion", "as": "special_hetu_proportion"},
            {"name": "explore_mc_s2_beauty_caption_special_hetu_level_five_id", "as": "special_hetu_level_five_id"},
          ],
          import_item_attr = [
            {"name": "hetu_tag_level_info__hetu_level_five", "as": "hetu_level_five"},
            {"name": self._score_attr, "as": "ensemble_score_attr"},
          ],
          export_item_attr = [
            {"name": "ensemble_score_attr", "as": self._score_attr },
          ],
          function_name = "SpecialHetuCategoryDiversity",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            flag_attr : 1
          }
        ) \
      .end_()
    return self

  def _cascading_s2_boost_logic(self, flag_attr):
    self.flow \
    .if_("enable_explore_cascading_s2_recent_consume_photo_flag == 1 and user_need_saving_flag == 1") \
      .gen_is_user_recent_consume_photo(flag_attr) \
    .end_() \
    .if_("enable_explore_cascading_s2_audit_good_photo_flag == 1 and user_need_saving_flag == 1") \
      .gen_is_audit_good_photo(flag_attr) \
    .end_() \
    .if_("enable_explore_cascading_s2_boost == 1") \
      .if_("enable_cascading_caption_boost == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "cascading_caption_boost_coef", "as": "caption_boost_coef"},
            {"name": "cascading_caption_boost_len_thresh", "as": "caption_boost_len_thresh"},
            {"name": "cascading_caption_boost_len_max", "as": "caption_boost_len_max"},
            {"name": "cascading_boost_only_xhs_photo", "as": "boost_only_xhs_photo"},
            {"name": "cascading_boost_only_picture", "as": "boost_only_picture"},
          ],
          import_item_attr = [
            {"name": self._score_attr, "as": "score"},
            "caption_length",
            "is_xhs_type_photo",
            "is_picture",
          ],
          export_item_attr = [
            {"name": "score", "as": self._score_attr},
          ],
          export_common_attr = [
            {"name": "boost_count", "as": "cascading_caption_boost_count"},
            {"name": "total_count", "as": "cascading_caption_total_count"},
          ],
          function_name = "BoostWithCaption",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            flag_attr : 1
          }
        ) \
        .perflog_attr_value(
          check_point = "cascading_caption_boost",
          common_attrs = [
            "cascading_caption_boost_count",
            "cascading_caption_total_count",
          ],
          target_item = {
            flag_attr : 1
          }
        ) \
      .end_() \
      .if_("enable_first_screen_mc_s2_discount_by_cid == 1 and page_index == 1 and refreshTimes ~= 0 and gemini_refresh_scene > 0 and gemini_refresh_scene < 4", to_be_delete = "date=2024-05-29;committer=guohao") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "first_screen_mc_s2_discount_by_cid_coef", "as": "boost_discount_coeff"},
          ],
          import_item_attr = [
            {"name": "is_first_screen_discount_by_cid", "as": "need_item_attr"},
            {"name": self._score_attr, "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": self._score_attr}
          ],
          function_name = "BoostOrDiscount",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            flag_attr : 1,
          }
        ) \
      .end_() \
      .if_("enable_first_screen_mc_s2_discount == 1 and page_index >= 1 and page_index <= explore_mc_s2_first_screen_discount_threshold") \
        .if_("(explore_cascading_enable_gemini_refresh_scene_pxtr_adjust == 0) or (gemini_refresh_scene > 0 and gemini_refresh_scene < 4)") \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "first_screen_mc_s2_discount_coef", "as": "boost_discount_coeff"},
            ],
            import_item_attr = [
              {"name": "is_first_screen_discount", "as": "need_item_attr"},
              {"name": self._score_attr, "as": "ensemble_score"},
            ],
            export_item_attr = [
              {"name": "ensemble_score", "as": self._score_attr}
            ],
            function_name = "BoostOrDiscount",
            class_name = "ExploreLightFunctionSetV2",
            target_item = {
              flag_attr : 1,
            }
          ) \
        .end_() \
      .end_() \
      .if_("explore_enable_mc_s2_xhs_target_qualified_photo_boost == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_mc_s2_xhs_target_qualified_photo_boost_coeff", "as": "boost_discount_coeff"},
          ],
          import_item_attr = [
            {"name": "is_xhs_target_qualified_photo", "as": "need_item_attr"},
            {"name": self._score_attr, "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": self._score_attr},
          ],
          function_name = "BoostOrDiscount",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            flag_attr : 1
          }
        ) \
      .end_() \
      .if_("mc_enable_user_intrest_adjust == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "output_intrest_key_list", "as": "intrest_key_list"},
            {"name": "output_intrest_value_list", "as": "intrest_value_list"},
            {"name": "mc_s2_user_intrest_adjust_boost_coef", "as": "boost_coef"},
            {"name": "mc_s2_user_intrest_adjust_discount_coef", "as": "discount_coef"},
            {"name": "explore_enable_hetu1_user_intrest_adjust", "as": "enable_hetu1"}, 
          ],
          import_item_attr = [
            {"name": self._score_attr, "as": "input_score"},
            "hetu_tag_level_info__hetu_level_two",
            "hetu_tag_level_info__hetu_level_one",
          ],
          export_item_attr = [
            {"name": "output_score", "as": self._score_attr}
          ],
          function_name = "IntrestAdjustScore",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            flag_attr : 1
          }
        ) \
      .end_() \
      .if_("mc_enable_high_htr_discount == 1") \
        .enrich_attr_by_light_function(
          import_common_attr=[
            {"name": "mc_high_htr_discount_coef", "as": "high_htr_discount_coef"},
            {"name": "mc_high_htr_threshold", "as": "high_htr_threshold"},
            {"name": "mc_high_htr_discount_power", "as": "htr_discount_power"},
          ],
          import_item_attr=[
            {"name": self._score_attr, "as": "es_score"},
            {"name": "cascade_phtr", "as": "htr_score"},
          ],
          export_item_attr=[
            {"name": "es_score", "as": self._score_attr}
          ],
          function_name="HighHtrMixEsScore",
          class_name="ExploreLightFunctionSetV2",
          target_item = {
            flag_attr : 1
          }
        ) \
      .end_() \
      .if_("mc_ua_reason_boost_fresh_thr >= (refreshTimes or 1000)") \
        .enrich_attr_by_light_function(
          target_reason = [10045],
          import_common_attr = [
            {"name": "mc_boost_ua_reason_weight", "as": "boost_weight"},
            {"name": "mc_weaken_ua_reason_weight", "as": "weaken_weight"},
          ],
          import_item_attr = [
            {"name": self._score_attr, "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": self._score_attr},
          ],
          function_name = "EnsembleScoreBoost",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_cascading_personified_author_boost == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "cascading_personified_author_boost_coef", "as": "personified_author_coeff"},
            {"name": "cascading_blacklist_author_boost_coef", "as": "blacklist_author_coeff"},
            {"name": "explore_personified_author_fans_thre_max", "as": "author_fans_thre_max"},
            {"name": "explore_personified_author_fans_thre_min", "as": "author_fans_thre_min"},
            {"name": "cascading_merchant_cart_boost_coef", "as": "merchant_cart_coeff"},
            {"name": "cascading_young_women_boost_coef", "as": "young_women_coeff"},
            {"name": "cascading_age_segment_18_23_coeff", "as": "age_segment_18_23_coeff"},
            "basic_info_gender_v2",
            "basic_info_age_segment_v2",
          ],
          import_item_attr = [
            {"name": "author__fans_count", "as": "author_fans_count"},
            {"name": "eyeshot_source", "as": "eyeshot_source"},
            {"name": "merchant_photo_cart_relation", "as": "cart_relation"},
            {"name": "merchant_item_info__item_id_list", "as": "cart_itemlist"},
            {"name": "live_photo_info__is_living", "as": "is_living"},
            {"name": self._score_attr, "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": self._score_attr},
          ],
          function_name = "PersonifiedAuthorBoost",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            flag_attr : 1
          }
        ) \
      .end_() \
      .if_("enable_cascading_top_personified_author_boost == 1") \
        .sort(
          score_from_attr = "cascade_prerank_pctr",
          target_item={ flag_attr: 1 , "eyeshot_source": 1}
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "cascading_top_personified_author_boost_coef", "as": "boost_discount_coeff"},
          ],
          import_item_attr = [
            {"name": self._score_attr, "as": "score"},
          ],
          export_item_attr = [
            {"name": "score", "as": self._score_attr},
          ],
          function_name = "BoostOrDiscountV2",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            flag_attr : 1,
            "eyeshot_source" : 1
          },
          range_end = "{{cascading_top_personified_author_boost_cnt}}"
        ) \
      .end_() \
      .if_("enable_cascade_refinement_boost_personified_author == 1") \
        .refinement_boost_personified_author(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_advance_boost_click_count == 1") \
        .boost_click_count(self._score_attr) \
      .end_()
    self._impression_audit_adjust()
    self._merchant_boost_score_by_buyer_type( # 基于用户分层对电商视频调权，新用户降权，老用户提权，整体控电商load
      flag_attr
    )
    self._user_interest_cluster_862_adjust(flag_attr)
    self._beauty_caption_adjust(flag_attr)
    self.flow \
      .if_("explore_mc_ensemble_s2_skip_sort == 0") \
        .sort(
          score_from_attr = self._score_attr,
          target_item = {
            flag_attr : 1
          }
        ) \
      .end_() \
      .if_("enable_fr_refactor_mc_same_author == 1") \
        .deduplicate(
          on_item_attr = "author__id",
          target_item = {
            flag_attr : 1
          }
        ) \
      .end_() \
      .if_("enable_boost_impression_audit == 1  and page_index > impression_audit_page_threshold") \
        .boost_impression_audit(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_deboost_merchant_car_photo == 1 and page_index > 1") \
        .deboost_merchant_car_photo(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_explore_cascading_s2_recent_consume_photo_boost == 1 and user_need_saving_flag == 1") \
        .boost_user_recent_consume_photo(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_explore_cascading_s2_audit_good_photo_boost == 1 and user_need_saving_flag == 1") \
        .boost_audit_good_photo(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_explore_mc_s2_boost_user_short_develop_interest == 1 and uExploreFountainPreferenceTypeKV == 1") \
        .boost_user_short_develop_interest(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_boost_ua_long_view == 1") \
        .boost_ua_long_view(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_mc_high_global_emphtr_discount == 1") \
        .mc_high_global_emphtr_discount(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_mc_search_score_boost == 1") \
        .mc_search_score_boost(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_mc_s2_marketing_compensation_discount == 1") \
        .mc_s2_marketing_compensation_discount(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_mc_s2_marketing_compensation_personal_discount == 1") \
        .mc_s2_marketing_compensation_personal_discount(self._score_attr, flag_attr) \
      .end_() \
      .if_("explore_enable_mc_s2_olympic_latest_boost == 1") \
        .mc_s2_olympic_latest_boost(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_boost_similar_author_reason == 1") \
        .mc_s2_boost_similar_author(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_boost_unbias_interest_reason == 1") \
        .mc_s2_boost_unbias_interest_photo(self._score_attr, flag_attr) \
      .end_() \
      .if_("explore_enable_mc_s2_all_page_interest_boost == 1 and all_page_interest_user_migration_flag == 1") \
        .mc_s2_all_page_interest_boost(self._score_attr, flag_attr) \
      .end_() \
      .if_("explore_enable_mc_s2_sexy_induce_deboost == 1") \
        .mc_s2_sexy_induce_deboost(self._score_attr, flag_attr) \
      .end_() \
      .if_("explore_enable_extract_hetu_info_tag_for_llm == 1")\
        .extract_hetu_info_tag_for_llm(flag_attr) \
      .end_() \
      .if_("explore_enable_mc_llm_negative_photo_personal_adjust == 1") \
        .mc_llm_negative_photo_personal_adjust(self._score_attr, flag_attr) \
      .end_() \
      .if_("explore_enable_mc_s2_sexy_induce_personal_deboost == 1") \
        .mc_s2_sexy_induce_personal_deboost(self._score_attr, flag_attr) \
      .end_() \
      .if_("explore_mc_s2_top_author_new_boost == 1") \
        .mc_s2_top_author_new_boost(self._score_attr, flag_attr) \
      .end_() \
      .if_("explore_mc_s2_new_hot_boost == 1") \
        .mc_s2_new_hot_boost(self._score_attr, flag_attr) \
      .end_() \
      .if_("explore_enable_mc_s2_update_bar_boost == 1") \
        .mc_s2_update_bar_boost(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_mc_s2_unbias_interest_photo_boost == 1") \
        .if_("enable_mc_s2_unbias_interest_photo_boost_vv_adjust == 1") \
          .user_vv_type_weight_adjust("explore_mc_s2_unbias_interest_photo_boost_coeff") \
        .end_() \
        .if_("enable_mc_s2_unbias_interest_photo_boost_cocoon_adjust == 1") \
          .user_cocoon_weight_adjust("explore_mc_s2_unbias_interest_photo_boost_coeff") \
        .end_() \
        .unbias_interest_photo_boost(self._score_attr, "mc_s2") \
      .end_() \
      .if_("enable_mc_s2_hot_list_photo_boost == 1") \
        .hot_list_photo_boost(self._score_attr, flag_attr, "mc_s2") \
      .end_() \
      .if_("enable_mc_s2_unbias_interest_cids_is_in_set == 1") \
        .unbias_interest_cluster_is_in_set() \
      .end_() \
      .if_("enable_mc_s2_unbias_interest_cids_boost == 1") \
        .unbias_interest_cluster_boost(self._score_attr) \
      .end_() \
      .if_("enable_mc_s2_interest_generalization_boost == 1") \
        .interest_generalization_boost(self._score_attr, flag_attr, "mc_s2") \
      .end_() \
      .if_("enable_mc_s2_interest_cid == 1") \
        .if_("enable_mc_s2_use_positive_interest_and_score_list == 1") \
          .mc_cal_interest_cid_coeff("user_postive_interest_score_list") \
        .else_() \
          .mc_cal_interest_cid_coeff() \
        .end_() \
        .mc_interest_score_cids_boost(self._score_attr) \
      .end_() \
      .if_("enable_mc_s2_valid_interest_cid_boost == 1") \
        .mc_cal_valid_interest_cid_coeff() \
        .mc_valid_interest_score_cids_boost(self._score_attr) \
      .end_() \
      .if_("enable_mc_s2_short_valid_interest_first_refresh_boost == 1 and is_first_refresh == 1") \
        .mc_cal_short_valid_interest_first_refresh_coeff() \
        .mc_short_valid_interest_first_refresh_boost(self._score_attr) \
      .end_() \
      .if_("enable_mc_s2_short_uninterest_photo_discount == 1") \
        .short_uninterest_photo_discount(self._score_attr, "mc_s2") \
      .end_() \
      .if_("enable_explore_frist_screen_customization_interest_migration_photo_boost == 1 and is_first_refresh == 1") \
        .explore_frist_screen_customization_interest_migration_photo_boost() \
      .end_() \
      .if_("enable_interest_migration_photo_coef_calculator == 1") \
        .interest_migration_coef_calculator(flag_attr) \
      .end_() \
      .if_("enable_mc_s2_interest_migration_photo_boost == 1") \
        .interest_migration_photo_boost(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_mc_s2_merchant_hetu_tag_discount == 1") \
        .merchant_hetu_tag_discount(self._score_attr) \
      .end_() \
      .if_("explore_enable_mc_s2_high_photo_count_author_adjust == 1") \
        .high_photo_count_author_adjust(self._score_attr, flag_attr) \
      .end_() \
      .if_("explore_enable_mc_s2_llm_negative_photo_adjust == 1") \
        .llm_negative_photo_adjust(self._score_attr, flag_attr) \
      .end_() \
      .if_("explore_enable_partial_time_based_interest_boost_mc_s2 == 1") \
        .partial_time_based_interest_boost(self._score_attr, flag_attr, "mc_s2") \
      .end_() \
      .if_("explore_enable_partial_time_based_tagnex_boost_mc_s2 == 1") \
        .partial_time_based_tagnex_boost(self._score_attr, flag_attr, "mc_s2") \
      .end_() \
      .if_("enable_mc_s2_short_term_photo_tagnex_score_adjust == 1") \
        .short_term_photo_tagnex_score_adjust(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_mc_s2_short_term_photo_cluster_id_score_adjust == 1") \
        .short_term_photo_cluster_id_score_adjust(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_mc_s2_short_term_photo_hetu_level2_score_adjust == 1") \
        .short_term_photo_hetu_level2_score_adjust(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_mc_s2_short_term_photo_hashtag_score_adjust == 1") \
        .short_term_photo_hashtag_score_adjust(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_mc_s2_short_term_photo_hetu_tag_score_adjust == 1") \
        .short_term_photo_hetu_tag_score_adjust(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_mc_s2_short_term_photo_interest_community_tag_score_adjust == 1") \
        .short_term_photo_interest_community_tag_score_adjust(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_mc_s2_short_term_photo_sid_score_adjust == 1") \
        .short_term_photo_sid_score_adjust(self._score_attr, flag_attr) \
      .end_() \
      .if_("explore_enable_mc_s2_interest_card_photo_score_adjust == 1") \
        .explore_cascade_interest_card_photo_score_adjust(self._score_attr, flag_attr) \
      .end_() \
      .if_("explore_enable_mc_s2_ensemble_sort_f1 == 1") \
        .cal_mc_s2_es_score_f1(self._score_attr, flag_attr) \
      .end_() \
      .if_("explore_enable_mc_s2_unaudit_deboost_f1 == 1") \
        .cal_mc_s2_unaudit_deboost_score_f1(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_explore_cs_photo_boost_mc_s2 == 1") \
        .explore_cascade_cs_boost(self._score_attr, flag_attr, "mc_s2") \
      .end_() \
      .if_("enable_explore_not_correlation_deboost_mc_s2 == 1") \
        .explore_cover_video_not_correlation_deboost(self._score_attr, flag_attr, "mc_s2") \
      .end_() \
      .if_("enable_explore_first_refresh_good_boost_mc_s2 == 1 and is_first_refresh == 1") \
        .explore_first_refresh_good_boost(self._score_attr, flag_attr, "mc_s2") \
      .end_()
    self.flow.end_()
    return self

  def _cascading_s2_truncate_by_interest(self, flag_attr):
    self.flow \
    .sort(
       score_from_attr = self._score_attr,
       target_item = {
         flag_attr : 1
       }
    ) \
    .if_("enable_mc_s2_same_author_dedup == 1") \
      .deduplicate(
        on_item_attr = "author__id",
        target_item = {
          flag_attr : 1
        }
      ) \
    .end_() \
    .if_("enable_mc_s2_select_photo_by_interest == 1") \
      .mc_s2_select_photo_by_interest(self._score_attr, flag_attr) \
      .copy_item_meta_info(
        save_item_seq_to_attr = "mc_s2_final_index_photo",
        target_item = {
          flag_attr : 1
        }
      ) \
    .end_()
    return self

  def _caculate_score(self, flag_attr, weight_attr, left_count_attr):
    self._calc_final_score(flag_attr, weight_attr, left_count_attr)

  def _calc_final_score(self, flag_attr, weight_attr, left_count_attr):
      self.flow \
      .if_("explore_enable_upload_xtr_cal_mc_s2 == 1") \
        .explore_cal_upload_xtr_score_mc_s2(flag_attr) \
      .end_() \
      .if_("explore_enable_hetu_one_xtr_debias_cal_mc_s2 == 1") \
        .explore_cal_hetu_one_debias_score_mc_s2(flag_attr) \
      .end_() \
      .if_("enable_explore_mc_s2_cal_svtr_rid_ctr_score == 1") \
        .explore_mc_s2_cal_svtr_rid_ctr_score(flag_attr) \
      .end_() \
      .if_("explore_mc_ensemble_s2_skip_get_score == 0") \
        .explore_calc_ensemble_score(
          use_superscript_rank = "{{explore_mc_ensemble_s2_use_superscript_rank}}",
          user_power_calc_v2 = "{{explore_mc_ensemble_s2_user_power_calc_v2}}",
          value_seq_fusion_status = "{{explore_mc_ensemble_s2_value_seq_fusion_status}}",
          user_info_ptr_attr = "user_info_ptr",
          action_day = "{{mc_variant_weight_action_day_num_s2}}",
          enable_dynamic_weight_by_user_degree = "{{mc_enable_user_differently_by_degree_s2}}",
          rank_smooth = "{{explore_mc2_rank_smooth}}",
          rank_power_weight = "{{explore_mc2_rank_power_weight}}",
          use_fractile_in_ensemble_sort = "{{explore_mc_s2_use_fractile_in_ensemble_sort}}",
          fractile_in_ensemble_sort_type = "{{explore_mc_s2_fractile_in_ensemble_sort_type}}",
          rank_score_calculate_method = "{{explore_mc_s2_rank_score_calculate_method}}",
          hyperbolic_scale = "{{explore_mc_s2_hyperbolic_scale}}",
          hyperbolic_alpha = "{{explore_mc_s2_hyperbolic_alpha}}",
          hyperbolic_beta = "{{explore_mc_s2_hyperbolic_beta}}",
          hyperbolic_min_num = "{{explore_mc_s2_hyperbolic_min_num}}",
          queues = final_channel_sort_queues,
          save_score_to_attr = self._score_attr,
          target_item = {
            flag_attr : 1
          }
        ) \
      .end_() \
      .if_("explore_enable_short_term_item_adjust == 1") \
        .explore_short_item_adjust_enricher(
          realshow_list_attr = "standard_explore_realshow_pid_list",
          realshow_list_timestamp_attr = "uStandardExploreRealshowTimestampList",
          realshow_list_label_attr = "uStandardExploreRealshowLabelList",
          click_list_attr = "explore_user_recent_click_list",
          valid_interest_list_attr = "user_valid_interest_cid_list",
          time_window = "{{explore_short_realshow_timestamp_threshold}}",
          attr_min = "{{explore_tagnex_id_min}}",
          attr_max = "{{explore_tagnex_id_max}}",
          tagnex_adjust_alpha_coeff = "{{explore_short_term_item_tagnex_adjust_alpha_coeff}}",
          tagnex_adjust_beta_coeff = "{{explore_short_term_item_tagnex_adjust_beta_coeff}}",
          interest_community_tag_adjust_alpha_coeff = "{{explore_short_term_item_interest_community_tag_adjust_alpha_coeff}}",
          interest_community_tag_adjust_beta_coeff = "{{explore_short_term_item_interest_community_tag_adjust_beta_coeff}}",
          cluster_id_adjust_alpha_coeff = "{{explore_short_term_item_cluster_id_adjust_alpha_coeff}}",
          cluster_id_adjust_beta_coeff = "{{explore_short_term_item_cluster_id_adjust_beta_coeff}}",
          hetu_level2_adjust_alpha_coeff = "{{explore_short_term_item_hetu_level2_adjust_alpha_coeff}}",
          hetu_level2_adjust_beta_coeff = "{{explore_short_term_item_hetu_level2_adjust_beta_coeff}}",
          hashtag_adjust_alpha_coeff = "{{explore_short_term_item_hashtag_adjust_alpha_coeff}}",
          hashtag_adjust_beta_coeff = "{{explore_short_term_item_hashtag_adjust_beta_coeff}}",
          hetu_tag_adjust_alpha_coeff = "{{explore_short_term_item_hetu_tag_adjust_alpha_coeff}}",
          hetu_tag_adjust_beta_coeff = "{{explore_short_term_item_hetu_tag_adjust_beta_coeff}}",
          sid_adjust_alpha_coeff = "{{explore_short_term_item_sid_adjust_alpha_coeff}}",
          sid_adjust_beta_coeff = "{{explore_short_term_item_sid_adjust_beta_coeff}}",
          valid_interest_coeff = "{{explore_short_term_item_valid_interest_coeff}}",
          invalid_interest_coeff = "{{explore_short_term_item_invalid_interest_coeff}}",
          ratio_positive_tagnex_coeff = "{{explore_short_term_item_ratio_positive_tagnex_coeff}}",
          ratio_negative_tagnex_coeff = "{{explore_short_term_item_ratio_negative_tagnex_coeff}}",
          ratio_positive_interest_community_tag_coeff = "{{explore_short_term_item_ratio_positive_interest_community_tag_coeff}}",
          ratio_negative_interest_community_tag_coeff = "{{explore_short_term_item_ratio_negative_interest_community_tag_coeff}}",
          ratio_positive_cluster_id_coeff = "{{explore_short_term_item_ratio_positive_cluster_id_coeff}}",
          ratio_negative_cluster_id_coeff = "{{explore_short_term_item_ratio_negative_cluster_id_coeff}}",
          ratio_positive_hetu_level2_coeff = "{{explore_short_term_item_ratio_positive_hetu_level2_coeff}}",
          ratio_negative_hetu_level2_coeff = "{{explore_short_term_item_ratio_negative_hetu_level2_coeff}}",
          ratio_positive_hashtag_coeff = "{{explore_short_term_item_ratio_positive_hashtag_coeff}}",
          ratio_negative_hashtag_coeff = "{{explore_short_term_item_ratio_negative_hashtag_coeff}}",
          ratio_positive_hetu_tag_coeff = "{{explore_short_term_item_ratio_positive_hetu_tag_coeff}}",
          ratio_negative_hetu_tag_coeff = "{{explore_short_term_item_ratio_negative_hetu_tag_coeff}}",
          ratio_positive_sid_coeff = "{{explore_short_term_item_ratio_positive_sid_coeff}}",
          ratio_negative_sid_coeff = "{{explore_short_term_item_ratio_negative_sid_coeff}}",
          min_ratio_coeff = "{{explore_short_term_item_min_ratio_coeff}}",
          max_ratio_coeff = "{{explore_short_term_item_max_ratio_coeff}}",
          tagnex_no_click_ratio_threshold = "{{explore_short_term_item_tagnex_no_click_ratio_threshold}}",
          interest_community_tag_no_click_ratio_threshold = "{{explore_short_term_item_interest_community_tag_no_click_ratio_threshold}}",
          cluster_id_no_click_ratio_threshold = "{{explore_short_term_item_cluster_id_no_click_ratio_threshold}}",
          hetu_level2_no_click_ratio_threshold = "{{explore_short_term_item_hetu_level2_no_click_ratio_threshold}}",
          hashtag_no_click_ratio_threshold = "{{explore_short_term_item_hashtag_no_click_ratio_threshold}}",
          hetu_tag_no_click_ratio_threshold = "{{explore_short_term_item_hetu_tag_no_click_ratio_threshold}}",
          sid_no_click_ratio_threshold = "{{explore_short_term_item_sid_no_click_ratio_threshold}}",
          enable_tagnex_score = "{{explore_enable_cal_short_term_item_tagnex_score}}",
          enable_interest_community_tag_score = "{{explore_enable_cal_short_term_item_interest_community_tag_score}}",
          enable_cluster_score = "{{explore_enable_cal_short_term_item_cluster_score}}",
          enable_hetu2_score = "{{explore_enable_cal_short_term_item_hetu2_score}}",
          enable_hashtag_score = "{{explore_enable_cal_short_term_item_hashtag_score}}",
          enable_hetu_tag_score = "{{explore_enable_cal_short_term_item_hetu_tag_score}}",
          enable_sid_score = "{{explore_enable_cal_short_term_item_sid_score}}",
          enable_use_set_tagnex_ratio = "{{explore_enable_short_term_item_use_set_tagnex_ratio}}",
          enable_use_set_interest_community_tag_ratio = "{{explore_enable_short_term_item_use_set_interest_community_tag_ratio}}",
          enable_use_set_cluster_id_ratio = "{{explore_enable_short_term_item_use_set_cluster_id_ratio}}",
          enable_use_set_hetu_level2_ratio = "{{explore_enable_short_term_item_use_set_hetu_level2_ratio}}",
          enable_use_set_hashtag_ratio = "{{explore_enable_short_term_item_use_set_hashtag_ratio}}",
          enable_use_set_hetu_tag_ratio = "{{explore_enable_short_term_item_use_set_hetu_tag_ratio}}",
          enable_use_set_sid_ratio = "{{explore_enable_short_term_item_use_set_sid_ratio}}",
          enable_tagnex_use_threshold_adjust_score = "{{explore_enable_tagnex_use_threshold_adjust_short_term_item_score}}",
          enable_interest_community_tag_use_threshold_adjust_score = "{{explore_enable_interest_community_tag_use_threshold_adjust_short_term_item_score}}",
          enable_cluster_id_use_threshold_adjust_score = "{{explore_enable_cluster_id_use_threshold_adjust_short_term_item_score}}",
          enable_hetu_level2_use_threshold_adjust_score = "{{explore_enable_hetu_level2_use_threshold_adjust_short_term_item_score}}",
          enable_hashtag_use_threshold_adjust_score = "{{explore_enable_hashtag_use_threshold_adjust_short_term_item_score}}",
          enable_hetu_tag_use_threshold_adjust_score = "{{explore_enable_hetu_tag_use_threshold_adjust_short_term_item_score}}",
          enable_sid_use_threshold_adjust_score = "{{explore_enable_sid_use_threshold_adjust_short_term_item_score}}",
          tagnex_attr = "hetu_tag_level_info__hetu_tag",
          interest_community_tag_attr = "interest_community_tag_id",
          cluster_id_attr = "cluster_id_632",
          hetu_level2_attr = "hetu_tag_level_info__hetu_level_two",
          hashtag_attr = "user_hash_tag_id",
          sid_attr = "explore_sid",
          output_tagnex_score_attr = "short_term_item_tagnex_score",
          output_interest_community_tag_score_attr = "short_term_item_interest_community_tag_score",
          output_cluster_score_attr = "short_term_item_cluster_id_score",
          output_hetu2_score_attr = "short_term_item_hetu_level2_score",
          output_hashtag_score_attr = "short_term_item_hashtag_score",
          output_hetu_tag_score_attr = "short_term_item_hetu_tag_score",
          output_sid_score_attr = "short_term_item_sid_score",
          target_item = {
            flag_attr : 1
          }
        ) \
      .end_() \
      .if_("explore_enable_interest_card_adjust == 1") \
        .explore_interest_card_adjust_enricher(
          click_list_attr = "videoPlayingPid",
          click_list_exptags_attr = "playstat_reasons",
          click_list_hetutags_attr = "playstat_hetutags",
          tagnex_level_three_attr = "hetu_tag_level_info__hetu_tag",
          output_score_attr = "interest_card_adjust_score",
          score_alpha_coeff = "{{explore_interest_card_score_alpha_coeff}}",
          tagnex_min = "{{explore_tagnex_lv3_min}}",
          tagnex_max = "{{explore_tagnex_lv3_max}}",
          click_list_num_threshold = "{{explore_interest_card_click_list_num_threshold}}",
          target_item = {
            flag_attr : 1
          }
        ) \
      .end_()
      self._cascading_s2_boost_logic(flag_attr)
      self._cascading_s2_truncate_by_interest(flag_attr)
      

