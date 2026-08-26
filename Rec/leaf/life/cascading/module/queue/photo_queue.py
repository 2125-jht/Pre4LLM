from cascading.module.queue.cascading_channel_sort_queue import ChannelSortQueuePartitioner
from cascading.module.queue.cascading_channel_sort_queue import ChannelSortQueueScorer
from cascading.module.queue.cascade_queues import cluster_variant_sort_queue
from cascading.module.queue.cascade_prerank_queues import prerank_ensemble_sort_queues
from cascading.module.queue.cascade_final_queues import final_channel_sort_queues,final_channel_sort_queues_new

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
    self.flow.set_attr_value(
      no_overwrite=True,
      item_attrs=[
        {
          "name": "prerank_action_once_score",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "prerank_duration_score",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "prerank_mc_ensemble_cascade_score",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "prerank_mc_ensemble_pcltr",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "prerank_mc_ensemble_pcmtr",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "prerank_mc_ensemble_pctr",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "prerank_mc_ensemble_pepstr",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "prerank_mc_ensemble_pftr",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "prerank_mc_ensemble_pltr",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "prerank_mc_ensemble_plvtr",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "prerank_mc_ensemble_ptr",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "prerank_mc_ensemble_pwatch_time",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "prerank_mc_ensemble_pwtr",
          "type": "double",
          "value": 0.0
        }
      ]
    )
    self.flow.enrich_attr_by_light_function(
      import_common_attr = [
        "prerank_duration_debias_bucket",
        "prerank_duration_debias_prefix",
        "prerank_short_duration_debias_interval",
        "prerank_mid_duration_debias_interval",
        "prerank_long_duration_debias_interval",
      ],
      import_item_attr = [
        "duration_ms",
        "explore_stat__view_length_sum",
        "explore_stat__click_count",
      ],
      export_item_attr = [
        "cascade_emp_watchtime_score",
      ],
      function_name = "CalEmpWatchTimeScore",
      class_name = "ExploreLightFunctionSetV2",
      target_item={ flag_attr: 1 }
    ) \
    .split_string(
      input_common_attr = "hot_prerank_mc_pxtr_weight",
      output_common_attr = "prerank_mc_pxtr_weight_list",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_double = True,
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        "cascade_prerank_pctr_weight",
        "cascade_prerank_pltr_weight",
        "cascade_prerank_prstr_weight",
        "cascade_emp_watchtime_score_weight",
        "cascade_prerank_calc_type",
        "prerank_ltr_weight",
        "prerank_ctr_weight",
        "prerank_wtd_weight",
        "prerank_life_ctr_weight",
        "prerank_duration_weight",
        "prerank_fountain_efficiency_vv_weight",
        "prerank_mc_pxtr_weight_list",
        "prerank_action_once_weight",
      ],
      import_item_attr = [
        "cascade_prerank_pctr",
        "cascade_prerank_pltr",
        "cascade_prerank_prstr",
        "cascade_emp_watchtime_score",
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
        "prerank_action_once_score"
        # "prerank_mc_pic_wtd",
        # "prerank_mc_pic_lvtr",
        # "prerank_mc_pic_cpr",
      ],
      export_item_attr = [
        {"name": "cascade_prerank_score", "as": self._score_attr}
      ],
      function_name = "CalPreRankScore",
      class_name = "ExploreLightFunctionSetV2",
      target_item={ flag_attr: 1 }
    ) \
    .if_("enable_prerank_search_retri_boost == 1") \
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
    .if_("enable_cascading_prerank_female_porn_discount == 1") \
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
    .if_("enable_mc_calc_search_score == 1") \
      .mc_calc_search_score(flag_attr) \
    .end_() \
    .if_("enable_prerank_search_score_boost == 1") \
      .prerank_search_score_boost(self._score_attr, flag_attr) \
    .end_() \
    .if_("life_enable_prerank_search_topk_boost == 1") \
      .prerank_search_topk_boost(self._score_attr, flag_attr) \
    .end_() \
    .if_("enable_life_prerank_user_pos_hetu_boost == 1 and page == 1 and (life_user_pos_hetu_boost_limit_low_active ~= 1 or uIsLifeHighActive ~= 1)") \
      .prerank_user_pos_hetu_boost(self._score_attr, flag_attr) \
    .end_() \
    .if_("enable_life_prerank_hotfire_yellow_boost == 1 and (life_hotfire_yellow_boost_limit_low_active ~= 1 or uIsLifeHighActive ~= 1)") \
      .prerank_hotfire_yellow_boost(self._score_attr, flag_attr) \
    .end_() \
    .if_("enable_life_prerank_low_cost_photo_discount == 1") \
      .prerank_low_cost_photo_discount(self._score_attr, flag_attr) \
    .end_() \
    .if_("enable_life_direct_tab_boost == 1") \
      .set_attr_value(
        item_attrs=[
          {
            "name": self._score_attr,
            "type": "double",
            "value": 100000.0
          }
        ],
        target_item = {
          "reason": 2416,
          flag_attr : 1
        }
      ) \
    .end_() \
    .if_("is_fresh_request == 1 and enable_life_active_interest_boost == 1 and (life_active_interest_boost_limit_low_active ~= 1 or uIsLifeHighActive ~= 1)") \
      .sort(
        score_from_attr = self._score_attr,
        target_item = {
          "reason": [2422],
          flag_attr : 1
        }
      ) \
      .limit(
        size = 50,
        target_item = {
          "reason": [2422],
          flag_attr : 1
        }
      ) \
      .set_attr_value(
        item_attrs=[
          {
            "name": self._score_attr,
            "type": "double",
            "value": 100000.0
          }
        ],
        target_item = {
          "reason": [2422],
          flag_attr : 1
        }
      ) \
    .end_() \
    .copy_attr(
      attrs=[{
        "from_item": self._score_attr,
        "to_item": "cascade_prerank_score"
      }]
    )  # copy_attr 放在 prerank 算分最后

class PhotoQueueCascadingScorer(ChannelSortQueueScorer):
  def __init__(self, name, flow, config):
    super().__init__(name, flow, config)

  def _caculate_score(self, flag_attr, weight_attr, left_count_attr):
    self._calc_cascading_score(flag_attr, weight_attr, left_count_attr)

  def _calc_cascading_score(self, flag_attr, weight_attr, left_count_attr):
    self.flow \
    .set_attr_value(
      no_overwrite=True,
      item_attrs=[
        {
          "name": "empctr_cluster",
          "type": "int",
          "value": 0
        },
        {
          "name": "young_age_boost_rate",
          "type": "double",
          "value": 0.0
        }
      ]
    ) \
    .if_("enable_life_cluster_by_hetu == 1") \
      .explore_life_cluster_rule_enricher(
        skip = 0,
        use_extra_page = "{{explore_use_more_page_photos}}",
        user_info_ptr_attr = "user_info_ptr",
        save_cluster_id_to_attr = "cascade_cluster_id",
        enable_user_profile_top_hetu_level_one_cluster = "{{explore_enable_use_hetu_level1_id}}",
        enable_user_profile_top_hetu_level_two_cluster = "{{explore_enable_use_hetu_level2_id}}",
        enable_user_profile_top_hetu_level_three_cluster = "{{explore_enable_use_hetu_level3_id}}",
        enable_use_photo_age_cluster = "{{explore_enable_use_photo_age_cluster_dryrun}}",
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
        enable_use_download_list = "{{explore_mc_cluster_use_download_list}}",
        enable_use_enter_profile_list = "{{explore_mc_cluster_use_enter_profile_list}}",
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
        download_weight = "{{explore_mc_cluster_download_weight}}",
        enter_profile_weight = "{{explore_mc_cluster_enter_profile_weight}}",
        video_play_weight = "{{explore_mc_cluster_video_play_weight}}",
        min_effective_play_length = "{{explore_mc_cluster_effective_play_min_length}}",
        fountain_real_show_weight = "{{fountain_real_show_weight}}", # 1.0
        fountain_click_weight = "{{fountain_click_weight}}", #2.0,
        fountain_like_weight = "{{fountain_like_weight}}", #3.0,
        fountain_follow_weight = "{{fountain_follow_weight}}", # 3.0
        fountain_forward_weight = "{{fountain_forward_weight}}",
        enable_mc_use_realshow_no_click = "{{enable_mc_use_realshow_no_click_dryrun}}",
        enable_uninterested_hetu_level_one_cluster = "{{mc_cluster_uninterested_use_hetu_level1_id}}",
        enable_uninterested_hetu_level_two_cluster = "{{mc_cluster_uninterested_use_hetu_level2_id}}",
        enable_uninterested_hetu_level_three_cluster = "{{mc_cluster_uninterested_use_hetu_level3_id}}",
        uninterested_tag_score_limit = "{{mc_cluster_uninterested_tag_score_limit}}", # 4
        uninterested_limit_num = "{{mc_cluster_uninterested_limit_num}}",
        real_show_no_click_weight = "{{mc_cluster_real_show_no_click_weight}}",
        uninterested_expired_gap_second = "{{mc_cluster_uninterested_expired_gap_second}}",
        enable_mc_use_xhs_tag = "{{enable_mc_use_xhs_tag_dryrun}}",
        enable_colossus_cluster = "{{enable_use_colossus_cluster}}",
        enable_mc_explore_cluster = "{{enable_mc_explore_cluster_dryrun}}",
        input_colossus_attr_one = "sim_one_tags",
        input_colossus_attr_two = "sim_two_tags",
        input_colossus_attr_three = "sim_three_tags",
        input_colossus_attr_explore = "sim_explore_tags",
        enable_mc_interact_cluster = "{{enable_mc_interact_cluster_dryrun}}",
        input_colossus_attr_interact = "sim_interact_tags",
        hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
        hetu_level_two_attr = "hetu_tag_level_info__hetu_level_two",
        hetu_level_three_attr = "hetu_tag_level_info__hetu_level_three",
        hetu_level_four_attr = "hetu_tag_level_info__hetu_level_four",
        age_cluster_attr = "upload_time",
        mc_age_gap_str = "{{mc_mc_age_gap_str}}",
        input_xhs_hetu_tags_attr = "xhs_hetu_tags",
        enable_mc_follow_author_cluster = "{{enable_mc_follow_author_cluster_dryrun}}",
        enable_mc_follow_author_cluster_first = "{{enable_mc_follow_author_cluster_first_dryrun}}",
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
        enable_mc_empctr_cluster = "{{enable_mc_empctr_cluster_dryrun}}",
        empctr_cluster_flag_attr = "empctr_cluster",
        mc_realtime_bucket_limit_num_ratio = "{{mc_realtime_bucket_limit_num_ratio}}",
        target_item={ flag_attr: 1 },
        perf_checkpoint = "cascade",
        enable_shortterm_interest_cluster_opt = "{{enable_shortterm_interest_cluster_opt}}",
        shortterm_hetu_attr = "interest_explore_shortterm_hetu",
        enable_short_interest_cluster = "{{enable_xlife_short_interest_cluster}}",
        enable_default_hetu_cluster = "{{enable_xlife_default_hetu_cluster}}",
        default_hetu_use_level_one = "{{enable_xlife_default_hetu_one_cluster}}",
        enable_unbias_interest_cluster = "{{life_enable_unbias_interest_cluster}}",
        input_unbias_interest_attr = "life_unbias_interest_list",
        unbias_interest_hetu_attr = "hetu_tag_level_info_v2__hetu_level_two",
        enable_mmu_interest_hetu_l2_cluster = "{{life_enable_mmu_interest_hetu_l2_cluster}}",
        mmu_interest_hetu_l2_score_thr = "{{life_mmu_interest_hetu_l2_score_thr}}",
        mmu_interest_hetu_l2_cluster_max_num = "{{life_mmu_interest_hetu_l2_cluster_max_num}}",
        input_mmu_interest_hetu_l2_ids_attr = "uHetuCategoryInterestlv2IdList",
        input_mmu_interest_hetu_l2_scores_attr = "uHetuCategoryInterestlv2ScoreList",
        enable_mmu_interest_hetu_l1_cluster = "{{life_enable_mmu_interest_hetu_l1_cluster}}",
        mmu_interest_hetu_l1_score_thr = "{{life_mmu_interest_hetu_l1_score_thr}}",
        mmu_interest_hetu_l1_cluster_max_num = "{{life_mmu_interest_hetu_l1_cluster_max_num}}",
        input_mmu_interest_hetu_l1_ids_attr = "uHetuCategoryInterestlv1IdList",
        input_mmu_interest_hetu_l1_scores_attr = "uHetuCategoryInterestlv1ScoreList"
      ) \
    .else_() \
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
        shortterm_hetu_attr = "interest_explore_shortterm_hetu"
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
    .if_("enable_life_direct_tab_boost == 1") \
      .set_attr_value(
        item_attrs = [
          {
            "name": "mc_ensemble_pctr",
            "type": "double",
            "value": 1000.0
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
            "name": "mc_ensemble_pctr",
            "type": "double",
            "value": 100000.0
          }
        ],
        target_item = {
          "reason": [2422]
        }
      ) \
    .end_() \
    .if_("enable_life_unbias_interest_adjust_low_active == 1 and uIsLifeHighActive ~= 1") \
      .gen_common_attr_by_lua(
        attr_map={
          "explore_cascade_variant_cut_ratio": "explore_cascade_variant_cut_ratio_low_active",
        }
      ) \
    .end_() \
    .if_("enable_life_cluster_sort_interest_weight_adjust_low_active == 1 and uIsLifeHighActive ~= 1") \
      .gen_common_attr_by_lua(
        attr_map={
          "life_cluster_sort_interest_weight_str": "life_cluster_sort_interest_weight_str_low_active",
        }
      ) \
    .end_() \
    .if_("enable_life_mc_s1_hetu_debias_pctr == 1") \
      .explore_life_uninterest_hetu_exit_enricher(
        user_info_ptr_attr = "user_info_ptr",
        realshow_num_threshold = "{{life_mc_s1_hetu_debias_pctr_realshow_num_threshold}}",
        time_gap_s = "{{life_mc_s1_hetu_debias_pctr_time_gap_s}}",
        hetu_tag_attr = "hetu_tag_level_info__hetu_level_two",
        input_pctr_attr = "mc_ensemble_pctr",
        output_pctr_attr = "mc_ensemble_pctr",
        calculate_mode = "{{life_mc_s1_hetu_debias_pctr_calculate_mode}}",
        discount_coef = "{{life_mc_s1_hetu_debias_pctr_discount_coef}}",
        realshow_unclick_num_thr = "{{life_mc_s1_hetu_debias_pctr_realshow_unclick_num_thr}}",
      ) \
    .end_() \
    .enrich_attr_by_lua(
        import_item_attr = ["hetu_tag_level_info__hetu_level_one"],
        export_item_attr = ["hetu_tag_level1"],
        function_for_item = "get_hetu_one",
        lua_script = """
              function get_hetu_one()
                local hetu_one_list = hetu_tag_level_info__hetu_level_one or {}
                local hetu_tag_level1 = -1
                  
                if #hetu_one_list > 0 then
                    hetu_tag_level1 = hetu_one_list[1]
                end
              return hetu_tag_level1
          end
          """
    ) \
    .if_("life_enable_f1_mc_first_page_adjust_score == 1") \
      .calc_by_formula1(
        kconf_key = "formula.scenarioKey38.LifeMcFirstPageAdjustScore",
        import_common_attr = [
          "page",
          "refreshTimes",
          "uNebulaXlifeVisitDays30dKV", 
          "uNebulaDoubleFindVisitDays30dKV",
        ],
        import_item_attr = [
          "hetu_tag_level1",
          "cascade_pctr", 
          "cascade_pltr",  
          "cascade_plvtr",
          "cascade_plvtr2",
        ],
        export_formula_value = [
          "mc_first_page_adjust_score"
        ],
        abtest_biz_name = "KUAISHOU_APPS"
      ) \
    .end_() \
    .switch_("enable_enable_life_cluster_sort_version") \
      .case_(1) \
        .explore_life_cluster_variant_sort_v2_enrich(
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
          two_times_sort = "{{explore_enable_mc_two_times_sort}}",
          first_time_cut_ratio = "{{explore_mc_first_time_sort_cut_ratio}}",
          use_fractile_in_ensemble_sort = "{{explore_mc_s1_use_fractile_in_ensemble_sort}}",
          fractile_in_ensemble_sort_type = "{{explore_mc_s1_fractile_in_ensemble_sort_type}}",
          queues = cluster_variant_sort_queue,
          save_cluster_id_common_attr = "mc_s1_cluster_id",
          save_cluster_cnt_common_attr = "mc_s1_cluster_cnt",
          enable_dynamic_interest_ratio = "{{life_cluster_sort_enable_dynamic_interest_ratio}}",
          keep_num = "{{life_cluster_sort_keep_num}}",
          skip_fillback_result = "{{life_cluster_sort_skip_fillback_result}}",
          interest_weight_str = "{{life_cluster_sort_interest_weight_str}}",
          target_item = { # 这个 processor 会对视频做逻辑截断，这里只对打上标签的 item 进行处理
            flag_attr : 1
          }
        ) \
        .log_debug_info(
          item_attrs = [
            "empirical_svtr",
            # "interest_migration_photo_coef",
            "long_term_nature_score",
            "cascade_score",
            self._score_attr
          ],
          for_debug_request_only = True,
          respect_sample_loggging = True,
        ) \
      .case_(2) \
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
          two_times_sort = "{{explore_enable_mc_two_times_sort}}",
          first_time_cut_ratio = "{{explore_mc_first_time_sort_cut_ratio}}",
          use_fractile_in_ensemble_sort = "{{explore_mc_s1_use_fractile_in_ensemble_sort}}",
          fractile_in_ensemble_sort_type = "{{explore_mc_s1_fractile_in_ensemble_sort_type}}",
          queues = cluster_variant_sort_queue,
          save_cluster_id_common_attr = "mc_s1_cluster_id",
          save_cluster_cnt_common_attr = "mc_s1_cluster_cnt",
          target_item = { # 这个 processor 会对视频做逻辑截断，这里只对打上标签的 item 进行处理
            flag_attr : 1
          }
        ) \
      .case_(3) \
        .explore_life_cluster_variant_sort_v3_enrich(
          check_point = "cascade",
          use_superscript_rank = True,
          cluster_attr_name = "cascade_cluster_id",
          hetu_level_one_name = "hetu_tag_level_info__hetu_level_one",
          global_cut_ratio = "{{" + weight_attr + "}}",  #
          min_survival = "{{mc_bucket_shrink_min_num}}",
          use_reciprocal = "{{explore_use_multiply_fusion_s1}}",
          use_reciprocal_new_value_seq_fusion = "{{mc_use_new_value_seq_fusion}}",
          topk_size_limit = "{{mc_topk_candidate_num}}",
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
          two_times_sort = "{{explore_enable_mc_two_times_sort}}",
          first_time_cut_ratio = "{{explore_mc_first_time_sort_cut_ratio}}",
          use_fractile_in_ensemble_sort = "{{explore_mc_s1_use_fractile_in_ensemble_sort}}",
          fractile_in_ensemble_sort_type = "{{explore_mc_s1_fractile_in_ensemble_sort_type}}",
          queues = cluster_variant_sort_queue,
          save_cluster_id_common_attr = "mc_s1_cluster_id",
          save_cluster_cnt_common_attr = "mc_s1_cluster_cnt",
          enable_dynamic_interest_ratio = "{{life_cluster_sort_enable_dynamic_interest_ratio}}",
          keep_num = "{{life_cluster_sort_keep_num}}",
          skip_fillback_result = "{{life_cluster_sort_skip_fillback_result}}",
          interest_weight_str = "{{life_cluster_sort_interest_weight_str}}",
          target_item = { # 这个 processor 会对视频做逻辑截断，这里只对打上标签的 item 进行处理
            flag_attr : 1
          }
        ) \
      .default_() \
        .do_nothing() \
    .end_() \
    .copy_attr(
      attrs=[{
        "from_item": self._score_attr,
        "to_item": "cascade_s1_score"
      }]
    )

    self.flow.log_debug_info(
      common_attrs=[],
      item_attrs=['cascade_score', 'mc_ensemble_pctr', self._score_attr, "cascade_s1_score"],
      item_num_limit=10,
      target_item = {
        flag_attr : 1
      },
      for_debug_request_only = True,
      respect_sample_logging = True,
    )

class PhotoQueueFinalScorer(ChannelSortQueueScorer):
  def __init__(self, name, flow, config):
    super().__init__(name, flow, config)

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

  def _mc_s2_diversity_control(self, flag_attr, score_attr):
    self.flow \
      .sort(
       score_from_attr = score_attr,
       target_item = {
         flag_attr : 1
       }
    ) \
    .explore_control_hetu_count_enricher(
      user_hetu_stat_attr = "colossus_hetu_distribution_hetu_stat",
      hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
      hetu_level_two_attr = "hetu_tag_level_info__hetu_level_two",
      hetu_level_five_attr = "hetu_tag_level_info__hetu_level_five",
      cluster_id_attr = "hetu_sim_cluster_id862",
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
      target_item = {
        flag_attr : 1
      }
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "mc_s2_diversity_select_flag", "as": "flag"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr},
      ],
      function_name = "SetMinimumScoreByFlag",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1
      }
    )
  
  def _caculate_score(self, flag_attr, weight_attr, left_count_attr):
    self.flow.if_("enable_skip_life_mc_s2_es == 1")
    self.flow.copy_attr(
      attrs=[{
        "from_item": "cascade_s1_score",
        "to_item": self._score_attr,
      }],
      target_item = {
        flag_attr : 1
      }
    ) 
    self.flow.else_()
    self._calc_final_score(flag_attr, weight_attr, left_count_attr)
    self.flow.end_()

  def _calc_final_score(self, flag_attr, weight_attr, left_count_attr):
      self.flow \
      .if_("enable_life_mc_s2_hetu_debias_pctr == 1") \
        .explore_life_uninterest_hetu_exit_enricher(
          user_info_ptr_attr = "user_info_ptr",
          realshow_num_threshold = "{{life_mc_s2_hetu_debias_pctr_realshow_num_threshold}}",
          time_gap_s = "{{life_mc_s2_hetu_debias_pctr_time_gap_s}}",
          hetu_tag_attr = "hetu_tag_level_info__hetu_level_two",
          input_pctr_attr = "mc_ensemble_pctr",
          output_pctr_attr = "mc_ensemble_pctr",
          calculate_mode = "{{life_mc_s2_hetu_debias_pctr_calculate_mode}}",
          discount_coef = "{{life_mc_s2_hetu_debias_pctr_discount_coef}}",
          realshow_unclick_num_thr = "{{life_mc_s2_hetu_debias_pctr_realshow_unclick_num_thr}}",
        ) \
      .end_() \
      .if_("enable_life_mc_s2_diversity_weight_adjust == 1") \
        .life_mc_s2_diversity_weight_adjust() \
      .end_() \
      .explore_calc_ensemble_score(
        use_superscript_rank = True,
        user_power_calc_v2 = "{{explore_mc_ensemble_s2_user_power_calc_v2}}",
        value_seq_fusion_status = "{{explore_mc_ensemble_s2_value_seq_fusion_status}}",
        user_info_ptr_attr = "user_info_ptr",
        action_day = "{{mc_variant_weight_action_day_num_s2}}",
        enable_dynamic_weight_by_user_degree = "{{mc_enable_user_differently_by_degree_s2}}",
        rank_smooth = "{{explore_mc2_rank_smooth}}",
        rank_power_weight = "{{explore_mc2_rank_power_weight}}",
        use_fractile_in_ensemble_sort = "{{explore_mc_s2_use_fractile_in_ensemble_sort}}",
        fractile_in_ensemble_sort_type = "{{explore_mc_s2_fractile_in_ensemble_sort_type}}",
        queues = final_channel_sort_queues,
        save_score_to_attr = self._score_attr,
        target_item = {
          flag_attr : 1
        }
      ) \
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
      .if_("cascade_final_enable_follow_author_pic_boost == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "cascading_final_follow_author_pic_boost_coef", "as": "boost_discount_coeff"},
          ],
          import_item_attr = [
            {"name": "is_picture_follow_author", "as": "need_item_attr"},
            {"name": self._score_attr, "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": self._score_attr}
          ],
          function_name = "BoostOrDiscount",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            flag_attr : 1,
            "is_picture": 1
          }
        ) \
      .end_() \
      .if_("enable_cascade_target_hetu_pic_mc_s2_boost == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "cascade_target_hetu_pic_mc_s2_boost_coeff", "as": "boost_discount_coeff"},
          ],
          import_item_attr = [
            {"name": "is_boost_hetu_pic", "as": "need_item_attr"},
            {"name": self._score_attr, "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": self._score_attr},
          ],
          function_name = "BoostOrDiscount",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            flag_attr : 1,
            "is_picture": 1
          }
        ) \
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
          import_common_attr=[
            {"name": "output_intrest_key_list", "as": "intrest_key_list"},
            {"name": "output_intrest_value_list", "as": "intrest_value_list"},
        ],
        import_item_attr=[
          {"name": self._score_attr, "as": "input_score"},
          "hetu_tag_level_info__hetu_level_two"
        ],
        export_item_attr=[
          {"name": "output_score", "as": self._score_attr}
        ],
        function_name="IntrestAdjustScore",
        class_name="ExploreLightFunctionSetV2",
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
      .if_("mc_enable_la_follow_boost_fresh_thr >= (refreshTimes or 1000)") \
        .enrich_attr_by_light_function(
          target_item = {"is_long_view_author": 1},
          import_common_attr = [
            {"name": "mc_boost_follow_author_weight", "as": "boost_weight"},
            {"name": "mc_weaken_follow_author_weight", "as": "weaken_weight"},
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
      .end_if_() \
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
      .if_("life_enable_mc_uninterest_deboost == 1 and page > 2") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "life_mc_uninterest_deboost_coeff", "as": "boost_discount_coeff"},
          ],
          import_item_attr = [
            {"name": "is_uninterest_depress", "as": "need_item_attr"},
            {"name": self._score_attr, "as": "score"},
          ],
          export_item_attr = [
            {"name": "score", "as": self._score_attr},
          ],
          function_name = "BoostOrDiscount",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_() \
      .if_("enable_boost_click_count == 1") \
        .boost_click_count(self._score_attr) \
      .end_() \
      .if_("enable_cascade_refinement_boost_personified_author == 1") \
        .refinement_boost_personified_author(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_mc_search_score_boost == 1") \
        .mc_search_score_boost(self._score_attr, flag_attr) \
      .end_() \
      .if_("life_enable_mc_search_topk_boost == 1") \
        .mc_search_topk_boost(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_life_mc_user_pos_hetu_boost == 1 and page == 1 and (life_user_pos_hetu_boost_limit_low_active ~= 1 or uIsLifeHighActive ~= 1)") \
        .mc_user_pos_hetu_boost(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_life_mc_hotfire_yellow_boost == 1 and (life_hotfire_yellow_boost_limit_low_active ~= 1 or uIsLifeHighActive ~= 1)") \
        .mc_hotfire_yellow_boost(self._score_attr, flag_attr) \
      .end_() \
      .if_("life_enable_mc_s2_marketing_compensation_discount == 1") \
        .mc_s2_marketing_compensation_discount(self._score_attr, flag_attr) \
      .end_() \
      .if_("life_enable_mc_s2_llm_negative_photo_adjust == 1") \
        .llm_negative_photo_adjust(self._score_attr, flag_attr) \
      .end_()
      self._impression_audit_adjust()
      self.flow.if_("enable_mc_s2_select_photo_by_interest == 1")
      self._mc_s2_diversity_control(flag_attr, self._score_attr)
      self.flow.end_()
      self.flow \
      .if_("enable_life_direct_tab_boost == 1") \
        .limit(
          size = 10,
          target_item = {
            "reason": 2416,
            flag_attr : 1
          }
        ) \
        .set_attr_value(
          item_attrs=[
            {
              "name": self._score_attr,
              "type": "double",
              "value": 100.0
            }
          ],
          target_item = {
            "reason": 2416,
            flag_attr : 1
          }
        ) \
      .end_() \
      .if_("is_fresh_request == 1 and enable_life_active_interest_boost == 1 and (life_active_interest_boost_limit_low_active ~= 1 or uIsLifeHighActive ~= 1)") \
        .limit(
          size = 10,
          target_item = {
            "reason": [2422],
            flag_attr : 1
          }
        ) \
        .set_attr_value(
          item_attrs=[
            {
              "name": self._score_attr,
              "type": "double",
              "value": 100.0
            }
          ],
          target_item = {
            "reason": [2422],
            flag_attr : 1
          }
        ) \
      .end_() \
      .set_attr_value(
        item_attrs = [
          {
            "name": "is_direct_tab_photo",
            "type": "int",
            "value": 0
          }
        ]
      ) \
      .set_attr_value(
        item_attrs = [
          {
            "name": "is_direct_tab_photo",
            "type": "int",
            "value": 1
          }
        ],
        target_item = {
          "reason": [2416, 2422]
        }
      ) \
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
            flag_attr : 1,
            "is_direct_tab_photo" : 0
          }
        ) \
      .end_() \
      .log_debug_info(
        item_attrs = [
          'xhs_install_find_click_value', 'xhs_install_find_outflow_click_value', 'da_1_2_city_vv_rate', 'da_young_18_30_vv_rate', 'young_photo_18_23_prob', 'young_photo_24_30_prob', 'questionnaire_score', 'is_high_quality_merchant_cart'
        ],
        common_attrs = [
          'cascade_prerank_enable_comment_boost', 'cascade_prerank_enable_comment_boost__god__coeff_max_w', 'cascade_prerank_enable_comment_boost__god__coeff_min_w', 'cascade_prerank_enable_comment_boost__god__coeff_p', 
          'cascade_prerank_enable_comment_boost__god__coeff_w', 'cascade_prerank_enable_comment_boost__hot__coeff_max_w', 'cascade_prerank_enable_comment_boost__hot__coeff_min_w', 
          'cascade_prerank_enable_comment_boost__hot__coeff_p', 'cascade_prerank_enable_comment_boost__hot__coeff_w', 'enable_action_once_score_prerank', 'enable_calc_prerank_wtd_score', 'enable_prerank_calculate_duration_score', 
          'enable_prerank_cluster_sort', 'enable_prerank_emp_watchtime_boost', 'enable_prerank_interest_explore_boost', 'enable_prerank_quantile_score_filter', 'explore_prerank_ensemble_action_once_power_weight', 
          'explore_prerank_ensemble_action_once_raw_power_weight', 'explore_prerank_ensemble_action_once_raw_weight', 'explore_prerank_ensemble_cascade_prerank_pctr_raw_power_weight', 
          'explore_prerank_ensemble_cascade_prerank_pctr_raw_weight', 'explore_prerank_ensemble_cascade_prerank_pltr_raw_power_weight', 'explore_prerank_ensemble_cascade_prerank_pltr_raw_weight', 
          'explore_prerank_ensemble_cascade_score_power_weight', 'explore_prerank_ensemble_cascade_score_raw_power_weight', 'explore_prerank_ensemble_cascade_score_raw_weight', 'explore_prerank_ensemble_life_ctr_power_weight', 
          'explore_prerank_ensemble_pcltr_power_weight', 'explore_prerank_ensemble_pcltr_raw_power_weight', 'explore_prerank_ensemble_pcltr_raw_weight', 'explore_prerank_ensemble_pcmtr_power_weight', 
          'explore_prerank_ensemble_pcmtr_raw_power_weight', 'explore_prerank_ensemble_pcmtr_raw_weight', 'explore_prerank_ensemble_pcptr_power_weight', 'explore_prerank_ensemble_pcptr_raw_power_weight', 
          'explore_prerank_ensemble_pcptr_raw_weight', 'explore_prerank_ensemble_pctr_power_weight', 'explore_prerank_ensemble_pctr_raw_power_weight', 'explore_prerank_ensemble_pctr_raw_weight', 
          'explore_prerank_ensemble_pefctr_power_weight', 'explore_prerank_ensemble_peftr_power_weight', 'explore_prerank_ensemble_peftr_raw_power_weight', 'explore_prerank_ensemble_peftr_raw_weight', 
          'explore_prerank_ensemble_pepstr_power_weight', 'explore_prerank_ensemble_pepstr_raw_power_weight', 'explore_prerank_ensemble_pepstr_raw_weight', 'explore_prerank_ensemble_pftr_power_weight', 
          'explore_prerank_ensemble_pftr_raw_power_weight', 'explore_prerank_ensemble_pftr_raw_weight', 'explore_prerank_ensemble_phtr_power_weight', 'explore_prerank_ensemble_phtr_rank_height', 
          'explore_prerank_ensemble_pltr_power_weight', 'explore_prerank_ensemble_pltr_raw_power_weight', 'explore_prerank_ensemble_pltr_raw_weight', 'explore_prerank_ensemble_plvtr2_power_weight', 
          'explore_prerank_ensemble_plvtr2_raw_power_weight', 'explore_prerank_ensemble_plvtr2_raw_weight', 'explore_prerank_ensemble_plvtr_power_weight', 'explore_prerank_ensemble_plvtr_raw_power_weight', 
          'explore_prerank_ensemble_plvtr_raw_weight', 'explore_prerank_ensemble_prerank_pctr_power_weight', 'explore_prerank_ensemble_prerank_pltr_power_weight', 'explore_prerank_ensemble_psvtr_power_weight', 
          'explore_prerank_ensemble_ptr_power_weight', 'explore_prerank_ensemble_ptr_raw_power_weight', 'explore_prerank_ensemble_ptr_raw_weight', 'explore_prerank_ensemble_pwatch_time_power_weight', 
          'explore_prerank_ensemble_pwatch_time_raw_power_weight', 'explore_prerank_ensemble_pwatch_time_raw_weight', 'explore_prerank_ensemble_pwtd_power_weight', 'explore_prerank_ensemble_pwtd_raw_power_weight', 
          'explore_prerank_ensemble_pwtd_raw_weight', 'explore_prerank_ensemble_pwtr_power_weight', 'explore_prerank_ensemble_pwtr_raw_power_weight', 'explore_prerank_ensemble_pwtr_raw_weight', 
          'explore_prerank_ensemble_user_power_calc_v2', 'explore_prerank_power_weight', 'explore_prerank_smooth', 'explore_prerank_use_ensembe_score', 'prerank_act_fusion_score_cltr_weight', 
          'prerank_act_fusion_score_cmtr_weight', 'prerank_act_fusion_score_cptr_weight', 'prerank_act_fusion_score_ctr_weight', 'prerank_act_fusion_score_eftr_weight', 'prerank_act_fusion_score_epstr_weight', 
          'prerank_act_fusion_score_ftr_weight', 'prerank_act_fusion_score_htr_weight', 'prerank_act_fusion_score_ltr_weight', 'prerank_act_fusion_score_ptr_weight', 'prerank_act_fusion_score_wtr_weight', 
          'prerank_ctr_filter_limit_num', 'prerank_ctr_filter_threshold', 'prerank_duration_score_duration_for_pic', 'prerank_duration_score_duration_max', 'prerank_duration_score_duration_min', 
          'prerank_duration_score_duration_power', 'prerank_duration_score_duration_scaler', 'prerank_duration_score_duration_smooth', 'prerank_duration_score_seperate_point_str', 'prerank_duration_score_seperate_weight_str', 
          'prerank_duration_seperate_point_str', 'prerank_duration_seperate_ratio', 'prerank_emp_watchtime_boost_coef', 'prerank_emp_watchtime_boost_thres', 'prerank_interest_explore_boost_coef', 'prerank_ltr_filter_limit_num', 
          'prerank_ltr_filter_threshold', 'prerank_wtd_filter_limit_num', 'prerank_wtd_filter_threshold', 'prerank_wtd_finish_max', 'prerank_wtd_finish_min', 'prerank_wtd_finish_pow_weight', 'prerank_wtd_max_score', 
          'prerank_wtd_min_score', 'prerank_wtd_table_0', 'prerank_wtd_table_1', 'prerank_wtd_table_2', 'prerank_wtd_table_3', 'prerank_wtd_table_4', 'prerank_wtd_table_5', 'prerank_wtd_table_6', 'prerank_wtd_table_7', 
          'prerank_wtd_table_8', 'prerank_wtd_table_seg', 'boost_click_count_alpha', 'boost_click_count_beta', 'boost_click_count_omega', 'boost_click_val_max', 'boost_click_val_min', 'cascade_absolute_score_weight', 
          'cascade_alpha_for_top', 'cascade_cltr_boost_top_num', 'cascade_cmtr_boost_top_num', 'cascade_relative_score_weight', 'cascade_wtr_boost_top_num', 'cascading_cocoon_discount_coef', 
          'cascading_final_photo_hetu_distribution_colossus_total_count_threshold', 'cascading_final_photo_hetu_distribution_global_fuse_corr', 'cascading_final_photo_hetu_distribution_hetu_coef_alpha', 
          'cascading_final_photo_hetu_distribution_hetu_coef_beta', 'cascading_final_photo_hetu_distribution_hetu_discount_threshold', 'cascading_final_photo_hetu_distribution_hetu_encourage_threshold', 
          'cascading_final_photo_hetu_distribution_max_count', 'cascading_support_author_boost_coef', 'click_thred', 'enable_advance_boost_click_count', 'enable_boost_click_count', 'enable_cascade_top_interaction_boost', 
          'enable_cascading_cocoon_discount', 'enable_cascading_final_photo_sort_hetu_distribution_adjust', 'enable_cascading_support_author_boost', 'enable_explore_cascade_score_multiply_gate', 'enable_explore_mc_click_boost_v2', 
          'enable_explore_mc_follow_aid_followtime_boost', 'enable_explore_mc_xhs_install_click_boost', 'enable_life_young_age_boost', 'enable_mc_s2_hot_content_retr_boost', 'enable_mc_s2_select_photo_by_interest', 
          'explore_mc_enable_merchant_live_boost', 'explore_mc_enable_merchant_photo_boost', 'explore_mc_ensemble_s2_pcltr_normalize_alpha', 'explore_mc_ensemble_s2_pcmtr_normalize_alpha', 'explore_mc_ensemble_s2_pcptr_power_weight', 
          'explore_mc_ensemble_s2_pcptr_raw_power_weight', 'explore_mc_ensemble_s2_pcptr_raw_weight', 'explore_mc_ensemble_s2_pctr_normalize_alpha', 'explore_mc_ensemble_s2_pftr_normalize_alpha', 
          'explore_mc_ensemble_s2_pltr_normalize_alpha', 'explore_mc_ensemble_s2_plvtr2_normalize_alpha', 'explore_mc_ensemble_s2_plvtr_normalize_alpha', 'explore_mc_ensemble_s2_pwatch_time_normalize_alpha', 
          'explore_mc_ensemble_s2_pwtd_raw_power_weight', 'explore_mc_ensemble_s2_pwtd_raw_weight', 'explore_mc_ensemble_s2_pwtr_normalize_alpha', 'explore_mc_ensemble_s2_skip_get_score', 'explore_mc_ensemble_s2_use_new_score', 
          'explore_mc_media_follow_aid_followtime_boost_coeff', 'explore_mc_merchant_live_boost_coef', 'explore_mc_merchant_photo_boost_coef', 'explore_mc_new_follow_aid_followtime_boost_coeff', 
          'explore_mc_outflow_boost_click_count_alpha', 'explore_mc_outflow_boost_click_count_beta', 'explore_mc_outflow_boost_click_count_omega', 'explore_mc_whole_boost_click_count_alpha', 'explore_mc_whole_boost_click_count_beta', 
          'explore_mc_whole_boost_click_count_omega', 'explore_mc_xhs_install_outflow_click_weight', 'explore_mc_xhs_install_whole_click_weight', 'hot_cascade_control_duration_0_7s_max_size', 
          'hot_cascade_control_duration_0s_max_size', 'hot_cascade_control_duration_12_17s_max_size', 'hot_cascade_control_duration_17_20s_max_size', 'hot_cascade_control_duration_7_9s_max_size', 
          'hot_cascade_control_duration_9_12s_max_size', 'hot_cascade_control_hetu1_max_size', 'hot_cascade_control_hetu2_max_size', 'hot_cascade_control_hetu5_max_size', 'hot_cascade_duration_control_diversity_start', 
          'hot_cascade_enable_duration_control_diversity', 'hot_cascade_enable_hetu_control_diversity', 'hot_cascade_enable_hetu_control_interest', 'hot_cascade_hetu_control_diversity_start', 
          'hot_cascade_hetu_control_interest_start', 'hot_cascade_pctr_gate_alpha', 'hot_cascade_pctr_gate_beta', 'hot_cascade_psvtr_gate_alpha', 'hot_cascade_psvtr_gate_beta', 'mc_age_0_12_score_cliff_ratio', 
          'mc_age_12_17_score_cliff_ratio', 'mc_age_18_23_score_cliff_ratio', 'mc_age_24_30_score_cliff_ratio', 'mc_age_31_40_score_cliff_ratio', 'mc_age_41_49_score_cliff_ratio', 'mc_age_greater_50_score_cliff_ratio', 
          'mc_boost_ua_reason_weight', 'mc_city_vv_boost_coeff', 'mc_city_vv_boost_threshold', 'mc_enable_high_quality_merchant_boost', 'mc_enable_personal_cliff_ratio', 'mc_enable_questionnaire_boost', 
          'mc_enable_young_photo_boost_rate_threshold', 'mc_high_quality_merchant_boost_coef', 'mc_pcptr_ensemble_sort_weight_s2', 'mc_questionnaire_boost_coef', 'mc_questionnaire_boost_thres', 'mc_s2_hot_content_retr_boost_coef', 
          'mc_ua_reason_boost_fresh_thr', 'mc_weaken_ua_reason_weight', 'mc_young_age_score_cliff_ratio', 'mc_young_photo_18_23_prob_boost_coeff', 'mc_young_photo_18_23_prob_boost_threshold', 'mc_young_photo_24_30_prob_boost_coeff', 
          'mc_young_photo_24_30_prob_boost_threshold', 'mc_young_photo_boost_rate_threshold', 'mc_young_vv_boost_coeff', 'mc_young_vv_photo_threshold', 'mc_young_vv_pic_threshold', 'mc_enable_opt_card_adjust', 'output_opt_card_key_list', 
          'output_opt_card_value_list', 'output_opt_card_key_list', 'output_opt_card_value_list', 'colossus_hetu_distribution_total_count', 'colossus_hetu_distribution_hetu_stat'
        ]
      )