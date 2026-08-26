from cascading.module.queue.cascading_channel_sort_queue import ChannelSortQueuePartitioner
from cascading.module.queue.cascading_channel_sort_queue import ChannelSortQueueScorer
from cascading.module.queue.cascade_queues import *
from cascading.module.queue.cascade_prerank_queues import pic_prerank_ensemble_sort_queues

class PictureQueueParitioner(ChannelSortQueuePartitioner):
  def __init__(self, name, flow, config):
    super().__init__(name, flow, config)

  def _calculate_flag(self):
    self.flow \
      .if_("enable_cascading_use_longpic_picset == 1", to_be_delete = "date=2024-05-29;committer=liuyanlei") \
        .copy_attr(
          attrs=[{
            "from_item": "is_longpic_picset",
            "to_item": self._flag_attr,
          }]
        ) \
      .else_() \
        .copy_attr(
          attrs=[{
            "from_item": "is_picture",
            "to_item": self._flag_attr,
          }]
        ) \
      .end_() \

class PictureQueuePrerankScorer(ChannelSortQueueScorer):
  def __init__(self, name, flow, config):
    super().__init__(name, flow, config)

  def _caculate_score(self, flag_attr, weight_attr, left_count_attr):
    self._calc_prerank_score(flag_attr, weight_attr)

  def _calc_prerank_score(self, flag_attr, weight_attr):
    # 先与视频score对齐
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
    .if_("explore_pic_prerank_tower_model_infer_v2_skip == 0", to_be_delete = "date=2024-05-29;committer=liuyanlei") \
      .enrich_attr_by_light_function(
        target_item={ flag_attr: 1 },
        import_common_attr=[
          "explore_cascade_pic_xtr_calc_with_ctr",
          "explore_cascade_pic_ctr_pow_weight",
        ],
        import_item_attr=[
          "pic_hot_click",
          "pic_hot_long_view",
          "pic_hot_finish_view",
          "pic_hot_pos_wtd",
          "pic_hot_action",
          "pic_hot_collect",
          "pic_hot_scroll",
        ],
        export_item_attr=[
          "pic_ensemble_pctr",
          "pic_ensemble_pltr",
          "pic_ensemble_pcptr",
          "pic_ensemble_pwtd",
          "pic_ensemble_action",
          "pic_ensemble_collect",
          "pic_ensemble_scroll",
        ],
        function_name="CalcPictureXTR",
        class_name="ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .if_("enable_explore_pic_prerank_queue == 0") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name":"pic_cascade_prerank_pctr_weight", "as":"cascade_prerank_pctr_weight"},
          {"name":"pic_cascade_prerank_pltr_weight", "as":"cascade_prerank_pltr_weight"},
          {"name":"pic_cascade_prerank_calc_type", "as":"cascade_prerank_calc_type"},
          {"name":"pic_cascade_prerank_rand_weight", "as":"cascade_prerank_rand_weight"},
          {"name":"pic_cascade_prerank_emp_ctr_weight", "as":"cascade_prerank_emp_ctr_weight"},
          {"name":"pic_cascade_prerank_emp_ltr_weight", "as":"cascade_prerank_emp_ltr_weight"},
          {"name":"pic_cascade_prerank_emp_wtr_weight", "as":"cascade_prerank_emp_wtr_weight"},
          {"name":"pic_cascade_prerank_emp_cmtr_weight", "as":"cascade_prerank_emp_cmtr_weight"},
          {"name":"pic_cascade_emp_watchtime_score_weight", "as":"cascade_emp_watchtime_score_weight"},
          {"name":"pic_cascade_hot_action_score_weight", "as":"prerank_ltr_weight"},
          "prerank_ctr_weight",
          "prerank_wtd_weight",
          "prerank_life_ctr_weight",
        ],
        import_item_attr = [
          {"name":"pic_ensemble_action", "as":"prerank_ltr"},
          "cascade_prerank_pctr",
          "cascade_prerank_pltr",
          "cascade_emp_watchtime_score",
          "prerank_ctr",
          "prerank_wtd",
          "prerank_life_ctr",
          "empirical_ctr",
          "empirical_ltr",
          "empirical_wtr",
          "empirical_cmtr",
          "prerank_rand_score",
        ],
        export_item_attr = [
          {"name": "cascade_prerank_score", "as": self._score_attr}
        ],
        function_name = "CalPreRankScore",
        class_name = "ExploreLightFunctionSetV2",
        target_item={ flag_attr: 1 }
      ) \
    .else_() \
      .explore_calc_ensemble_score(
        target_item={ flag_attr: 1 },
        use_superscript_rank = False,
        user_power_calc_v2 = 1,
        user_info_ptr_attr = "user_info_ptr",
        queues = pic_prerank_ensemble_sort_queues,
        save_score_to_attr = self._score_attr,
      ) \
    .end_() \
    .if_("enable_prerank_key_target_hetu_pic_boost == 1") \
      .enrich_attr_by_light_function(
        import_common_attr=[
          {"name": "prerank_key_target_hetu_pic_boost_coef", "as": "boost_discount_coeff"}
        ],
        import_item_attr=[
          {"name": self._score_attr, "as": "score"},
        ],
        export_item_attr=[
          {"name": "score", "as": self._score_attr},
        ],
        function_name="BoostOrDiscountV2",
        class_name="ExploreLightFunctionSetV2",
        target_item={
          flag_attr: 1,
          "is_key_target_hetu_pic": 1
        }
      ) \
    .end_() \
    .if_("prerank_enable_hetu_ratio_decay == 1") \
      .sort(
        score_from_attr = self._score_attr,
        target_item={ flag_attr: 1 }
      ) \
      .enrich_attr_by_light_function(
        import_common_attr=[
          {"name": "prerank_hetu_decay_coeff", "as": "decay_coeff"},
          {"name": "prerank_hetu_decay_keep_size_coeff", "as": "decay_keep_size_coeff"},
        ],
        import_item_attr=[
          {"name": self._score_attr, "as": "score"},
          {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_level_one_list"},
          {"name": "is_key_target_hetu_pic", "as": "is_target_hetu"},
        ],
        export_item_attr=[
          {"name": "score", "as": self._score_attr},
        ],
        function_name="HetuRatioDecay",
        class_name="ExploreLightFunctionSetV2",
        target_item={ flag_attr: 1 }
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
    .if_("enable_explore_prerank_pic_search_boost == 1 and (pic_search_boost_user_degree or 0) >= explore_pic_search_boost_user_degree_thresh") \
      .if_("enable_explore_pic_search_candicate_expand_by_cluster == 1") \
        .prerank_pic_search_boost(self._score_attr, flag_attr, "is_pic_search_cluster") \
      .else_() \
        .prerank_pic_search_boost(self._score_attr, flag_attr, "is_pic_search") \
      .end_() \
    .end_() \
    .if_("enable_explore_prerank_pic_recent_search_boost == 1 and (pic_search_boost_user_degree or 0) >= explore_pic_search_boost_user_degree_thresh") \
      .if_("enable_explore_pic_search_candicate_expand_by_cluster == 1") \
        .prerank_pic_recent_search_boost(self._score_attr, flag_attr, "is_pic_recent_search_cluster") \
      .else_() \
        .prerank_pic_recent_search_boost(self._score_attr, flag_attr, "is_pic_recent_search") \
      .end_() \
    .end_() \
    .if_("enable_explore_prerank_pic_valid_interest_cluster_boost == 1") \
      .prerank_pic_valid_interest_cluster_boost(self._score_attr, flag_attr) \
    .end_() \
    .if_("enable_explore_prerank_pic_long_interest_cluster_boost == 1") \
      .prerank_pic_long_interest_cluster_boost(self._score_attr, flag_attr) \
    .end_() \
    .if_("explore_enable_user_pic_growth_cluster_boost == 1 and ((uDoubleOutsideValidPicClusterCnt7dKV or 0) < explore_user_pic_growth_cluster_boost_interest_thresh)") \
      .mc_pic_boost_coef_with_flag(
        coef_attr = "explore_prerank_pic_growth_cluster_boost_coef",
        score_attr = self._score_attr,
        flag_attrs = [flag_attr, "is_pic_growth_cluster"],
        boost_num_max_attr = "explore_prerank_pic_growth_cluster_boost_num_max",
        boost_num_ratio_attr = "explore_prerank_pic_growth_cluster_boost_num_ratio"
      ) \
    .end_() \
    .if_("enable_explore_prerank_pic_valid_interest_cluster_boost_first_screen_adjust == 1 and page_index == 1") \
      .gen_common_attr_by_lua(
        attr_map={
          "explore_pic_double_valid_interest_cluster_boost_interest_thresh": "explore_pic_double_valid_interest_cluster_boost_interest_thresh_first_screen",
          "explore_prerank_pic_double_valid_interest_cluster_boost_coef": "explore_prerank_pic_double_valid_interest_cluster_boost_coef_first_screen",
          "explore_pic_single_valid_interest_cluster_boost_interest_thresh": "explore_pic_single_valid_interest_cluster_boost_interest_thresh_first_screen",
          "explore_prerank_pic_single_valid_interest_cluster_boost_coef": "explore_prerank_pic_single_valid_interest_cluster_boost_coef_first_screen"
        }
      ) \
    .end_() \
    .if_("enable_explore_prerank_pic_double_valid_interest_cluster_boost == 1 and (uDoubleOutsideValidPicClusterCnt7dKV or 0) < explore_pic_double_valid_interest_cluster_boost_interest_thresh") \
      .prerank_pic_double_valid_interest_cluster_boost(self._score_attr, flag_attr) \
    .end_() \
    .if_("enable_explore_prerank_pic_single_valid_interest_cluster_boost == 1 and (uDoubleOutsideValidPicClusterCnt7dKV or 0) < explore_pic_single_valid_interest_cluster_boost_interest_thresh") \
      .prerank_pic_single_valid_interest_cluster_boost(self._score_attr, flag_attr) \
    .end_() \
    .if_("enable_explore_prerank_pic_recent_interest_cluster_boost == 1") \
      .prerank_pic_recent_interest_cluster_boost(self._score_attr, flag_attr) \
    .end_() \
    .if_("enable_explore_prerank_operation_pic_boost == 1 and user_has_pic_crowd_show == 0") \
      .prerank_operation_pic_boost(self._score_attr, flag_attr, "operation_pic") \
    .end_() \
    .if_("enable_explore_prerank_pic_type_boost == 1") \
      .prerank_pic_type_boost(self._score_attr, flag_attr) \
    .end_() \
    .if_("enable_explore_prerank_pic_recent_repeated_realshow_deboost == 1") \
      .prerank_pic_recent_repeated_realshow_deboost(self._score_attr, flag_attr) \
    .end_()

     # 动态放弃channel槽位, 放在 _caculate_score 最后
    self.flow.if_("skip_cascade_prerank_pic_channel_dynamic_shrink == 0") \
      .sort(
        score_from_attr = self._score_attr,
        target_item={ flag_attr: 1 }
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": weight_attr, "as": "pic_channel_weight"},
          "dynamic_pic_quota",
          # ab param
          {"name": "cascade_prerank_fixed_final_size", "as": "mc_candidate_num"},
          {"name": "cascade_prerank_pic_channel_keep_min", "as": "pic_channel_keep_min"},
          {"name": "cascade_prerank_pic_quota_threshold", "as": "pic_prerank_quota_threshold"},
          {"name": "cascade_prerank_pic_quota_mode", "as": "pic_quota_mode"},
          {"name": "enable_cascade_prerank_quota_v2_limit", "as" : "enable_quota_limit"}
        ],
        export_item_attr = [
          {"name": "score_attr", "as": self._score_attr},
        ],
        export_common_attr = [
          "pic_prerank_quota"
        ],
        function_name = "PrerankChannelSortPicQueueDynamicShrink",
        class_name = "ExploreLightFunctionSetV2",
        target_item={ flag_attr: 1 }
      ) \
    .end_() \
    .if_("enable_pic_prerank_diversity_control == 1") \
      .explore_pic_diversity_control_enricher(
        enable_interest_control = "{{enable_pic_prerank_interest_control}}",
        enable_hetu_control = "{{enable_pic_prerank_hetu_control}}",
        enable_cluster_control = "{{enable_pic_prerank_cluster_control}}",
        enable_actual_hetu_control = "{{enable_pic_prerank_actual_hetu_adjust}}",
        keep_size = "pic_prerank_quota",
        enable_quota_complete = "{{pic_prerank_diversity_quota_complete}}",
        quota_complete_adjust_coeff = "{{pic_prerank_diversity_quota_complete_adjust}}",
        final_quota_adjust = "{{pic_prerank_diversity_quota_adjust}}",
        user_hetu_distribution_attr = "colossus_hetu_distribution_hetu_stat",
        user_actual_distribution_attr = "colossus_actual_reward_hetu_stat",
        old_cluster_id_interest_list_attr = "uOldMmuClusterId300ListList",
        cluster_id_attr = "hetu_sim_cluster_id",
        hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
        hetu_level_two_attr = "hetu_tag_level_info__hetu_level_two",
        hetu_level_five_attr = "hetu_tag_level_info__hetu_level_five",
        cluster_control_start = "{{pic_prerank_cluster_control_start}}",
        interest_control_start = "{{pic_prerank_interest_control_start}}",
        hetu_control_start = "{{pic_prerank_hetu_control_start}}",
        cluster_quota_coeff = "{{pic_prerank_cluster_quota_coeff}}",
        hetu1_quota_coeff = "{{pic_prerank_hetu1_quota_coeff}}",
        hetu2_quota_coeff = "{{pic_prerank_hetu2_quota_coeff}}",
        hetu5_quota_coeff = "{{pic_prerank_hetu5_quota_coeff}}",
        hetu_adjust_coef = "{{pic_prerank_hetu_adjust_coef}}",
        hetu_adjust_min_value = "{{pic_prerank_hetu_adjust_min_value}}",
        hetu_adjust_max_value = "{{pic_prerank_hetu_adjust_max_value}}",
        enable_dynamic_hetu_control_start = "{{enable_pic_prerank_dynamic_hetu_control_start}}",
        dynamic_hetu_control_start_alpha = "{{pic_prerank_dynamic_hetu_control_start_alpha}}",
        dynamic_hetu_control_start_bias = "{{pic_prerank_dynamic_hetu_control_start_bias}}",
        dynamic_hetu_control_start_pow = "{{pic_prerank_dynamic_hetu_control_start_pow}}",
        dynamic_hetu_control_start_min = "{{pic_prerank_dynamic_hetu_control_start_min}}",
        dynamic_hetu_control_start_max = "{{pic_prerank_dynamic_hetu_control_start_max}}",
        old_cluster_id_interest_coef = "{{pic_prerank_cluster_interest_boost_coef}}",
        es_score_attr = self._score_attr,
        target_item={ flag_attr: 1 }
      ) \
    .end_()
    self.flow.if_("enable_explore_pic_target_cluster_sort == 1", to_be_delete = "date=2024-05-29;committer=xubaoquan")
    self._explore_pic_prerank_cluster_sort(flag_attr, weight_attr)
    self.flow.end_()
    self.flow.log_debug_info(
      common_attrs=[
        "dynamic_pic_quota",
        "cascade_prerank_pctr_weight",
        "cascade_prerank_pltr_weight",
        "explore_pic_prerank_tower_model_infer_v2_skip",
        "explore_pic_prerank_ensemble_ori_pr_score_weight",
        "explore_pic_prerank_ensemble_pctr_weight",
        "explore_pic_prerank_ensemble_pltr_weight",
        "explore_pic_prerank_ensemble_pcptr_weight",
        "explore_pic_prerank_ensemble_pwtd_weight",
        "explore_pic_prerank_ensemble_action_weight",
        "explore_pic_prerank_ensemble_collect_weight",
        "explore_pic_prerank_ensemble_pscroll_weight",
      ],
      item_attrs=[
        "cascade_prerank_pctr",
        "cascade_prerank_pltr",
        self._score_attr,
        "pic_hot_click",
        "pic_hot_long_view",
        "pic_hot_finish_view",
        "pic_hot_pos_wtd",
        "pic_hot_action",
        "pic_hot_collect",
        "pic_hot_scroll",
        "pic_ensemble_pctr",
        "pic_ensemble_pltr",
        "pic_ensemble_pcptr",
        "pic_ensemble_pwtd",
        "pic_ensemble_action",
        "pic_ensemble_collect",
        "pic_ensemble_scroll",
      ],
      # for_debug_request_only=False,
      item_num_limit=10,
      target_item={ flag_attr: 1 }) \
    .copy_attr(
      attrs=[{
        "from_item": self._score_attr,
        "to_item": "cascade_prerank_score"
      }]
    )  # copy_attr 放在 prerank 算分最后

  def _explore_pic_prerank_cluster_sort(self, flag_attr, weight_attr):
    self.flow.explore_pic_calc_cluster(
      user_info_ptr_attr = "user_info_ptr",
      save_cluster_id_to_attr = "prerank_pic_cluster_id",
      perf_checkpoint = "prerank_pic_calc_cluster",
      enable_multi_hit_independent_bucket = False,
      clusters = [
        {
          "name": "privilege_tag",
          "cluster_type_id": 10000,
          "enable": "{{explore_pic_prerank_cluster_sort_privilege_tag_enable}}",
          "privilege_tags_attr": "pic_mc_cluster_sort_privilege_tags",
          "hetu_level_one_attr": "hetu_tag_level_info__hetu_level_one",
          "priority_num": '{{explore_prerank_explore_pic_privilege_tag_priority_num}}'
        },
        {
          "name": "long_caption",
          "cluster_type_id": 20000,
          "enable": "{{explore_pic_prerank_cluster_sort_long_caption_enable}}",
          "caption_length_attr": "caption_length",
          "is_xhs_type_photo_attr": "is_xhs_type_photo",
          "caption_length_threshold": 50,
          "priority_num": '{{explore_prerank_explore_pic_long_caption_priority_num}}'
        },
        {
          "name": "follow_author",
          "cluster_type_id": 20000,
          "enable": "{{explore_pic_prerank_cluster_sort_follow_author_enable}}",
          "is_follow_author_attr": "is_follow_author",
          "priority_num": '{{explore_prerank_explore_pic_follow_author_priority_num}}'
        },
        {
          "name": "long_pic_and_pic_set",
          "cluster_type_id": 30000,
          "enable": "{{explore_prerank_explore_long_pic_and_pic_set_enable}}",
          "upload_type_attr": "upload_type",
          "picture_type_attr": "picture_type",
          "priority_num": '{{explore_prerank_explore_long_pic_and_pic_set_priority_num}}'
        },
        {
          "name": "pic_cnt",
          "cluster_type_id": 40000,
          "enable": "{{explore_pic_prerank_cluster_sort_pic_cnt_enable}}",
          "picture_count_attr": "photo_picture_count",
          "priority_num": '{{explore_prerank_explore_pic_cnt_priority_num}}'
        },
        {
          "name": "high_value_pic",
          "cluster_type_id": 50000,
          "enable": "{{explore_pic_prerank_cluster_sort_high_value_pic_enable}}",
          "high_value_pic_flag_attr": "high_value_pic_flag",
          "priority_num": '{{explore_prerank_pic_high_value_pic_priority_num}}'
        },
        {
          "name": "pic_default",
          "cluster_type_id": 60000,
          "enable": "{{explore_pic_prerank_cluster_sort_pic_default_enable}}",
          "upload_type_attr": "upload_type",
          "picture_type_attr": "picture_type",
          "picture_count_attr": "photo_picture_count",
          "priority_num": '{{explore_prerank_pic_pic_default_priority_num}}'
        },
      ],
      target_item = {
        flag_attr : 1
      }
    )
    self.flow.explore_cluster_variant_sort_v2_enrich(
      check_point = "prerank_pic_calc_cluster",
      use_superscript_rank = False,
      cluster_attr_name = "prerank_pic_cluster_id",
      hetu_level_one_name = "hetu_tag_level_info__hetu_level_one",
      global_cut_ratio = "{{explore_prerank_pic_global_cut_ratio}}",  #
      min_survival = "{{explore_prerank_explore_pic_bucket_shrink_min_num}}",
      size_limit = "{{explore_prerank_explore_pic_final_candidate_num}}",
      user_info_ptr_attr = "user_info_ptr",
      enable_variant_cut_ratio = "{{explore_prerank_explore_pic_cascade_enable_variant_cut_ratio}}",
      variant_cut_ratio = "{{explore_pic_prerank_cascade_variant_cut_ratio}}",
      save_score_to_attr = self._score_attr,
      explore_pic_es_score_attr_name = self._score_attr,
      enable_explore_pic_user_es_score = True,
      queues = self._get_prerank_queue(),
      target_item = {
        flag_attr : 1
      }
    )

  def _get_prerank_queue(self):
    return [
        {
          "name" : self._score_attr,
          "weight" : 1.0,
          "power_weight_attr" : "explore_pic_prerank_ensemble_ori_pr_score_weight",
        },
        {
          "name": "pic_ensemble_pctr",
          "weight": 0.0,
          "power_weight_attr": "explore_pic_prerank_ensemble_pctr_weight"
        },
        {
          "name": "pic_ensemble_pltr",  # 长播
          "weight": 0.0,
          "power_weight_attr": "explore_pic_prerank_ensemble_pltr_weight"
        },
        {
          "name": "pic_ensemble_pcptr",  # 完播
          "weight": 0.0,
          "power_weight_attr": "explore_pic_prerank_ensemble_pcptr_weight"
        },
        {
          "name": "pic_ensemble_pwtd",  # plvtr_wtd
          "weight": 0.0,
          "power_weight_attr": "explore_pic_prerank_ensemble_pwtd_weight"
        },
        {
          "name": "pic_ensemble_action",
          "weight": 0.0,
          "power_weight_attr": "explore_pic_prerank_ensemble_action_weight"
        },
        {
          "name": "pic_ensemble_collect",
          "weight": 0.0,
          "power_weight_attr": "explore_pic_prerank_ensemble_collect_weight"
        },
        {
          "name": "pic_ensemble_scroll",
          "weight": 0.0,
          "power_weight_attr": "explore_pic_prerank_ensemble_pscroll_weight"
        },
        {
          "name": "pic_longterm_click",
          "weight": 0.0,
          "power_weight_attr": "explore_pic_prerank_ensemble_weight__pic_longterm_click"
        },
        {
          "name": "pic_longterm_collect",
          "weight": 0.0,
          "power_weight_attr": "explore_pic_prerank_ensemble_weight__pic_longterm_collect"
        },
        {
          "name": "pic_longterm_revisit",
          "weight": 0.0,
          "power_weight_attr": "explore_pic_prerank_ensemble_weight__pic_longterm_revisit"
        },
        {
          "name": "prerank_life_ctr",
          "weight": 0.0,
          "power_weight_attr": "explore_pic_prerank_ensemble_weight_pic_life_ctr"
        }
      ]

  def _get_prerank_pure_value_queue(self):
    return [
        {
          "name" : self._score_attr,
          "use_mapping" : "explore_pic_prerank_ensemble_ori_score_use_mp",
          "weight_attr" : "explore_pic_prerank_ensemble_ori_score_weight",
        },
        {
          "name": "pic_ensemble_pctr",
          "use_mapping" : "explore_pic_prerank_ensemble_pctr_score_use_mp",
          "weight_attr" : "explore_pic_prerank_ensemble_pctr_score_weight",
        },
        {
          "name": "pic_ensemble_action",
          "use_mapping" : "explore_pic_prerank_ensemble_action_score_use_mp",
          "weight_attr" : "explore_pic_prerank_ensemble_action_score_weight",
        },
        {
          "name": "pic_ensemble_collect",
          "use_mapping" : "explore_pic_prerank_ensemble_collect_score_use_mp",
          "weight_attr" : "explore_pic_prerank_ensemble_collect_score_weight",
        }
      ]

class PictureQueueCascadingScorer(ChannelSortQueueScorer):
  def __init__(self, name, flow, config):
    super().__init__(name, flow, config)

  def _caculate_score(self, flag_attr, weight_attr, left_count_attr):
    self._calc_cascading_score(flag_attr, weight_attr)
    self.flow.copy_attr(
      attrs=[{
        "from_item": self._score_attr,
        "to_item": 'picture_mc_stage1_score'
      }],
      target_item={ flag_attr: 1 }
    )

  def _calc_cascading_score(self, flag_attr, weight_attr):
    self.flow \
      .if_("enable_pic_cascade_variety == 1") \
        .calc_pic_set_variety_score(flag_attr) \
      .end_() \
      .if_("enable_explore_pic_mc_min_act_rank_score == 1", to_be_delete = "date=2024-05-29;committer=zhuwenyong") \
        .explore_min_act_rank_score_enricher(
          target_item = { flag_attr: 1 },
          max_rank_ratio = "{{explore_pic_mc_max_rank_ratio}}",
          queues = self._get_min_rank_queue(),
          save_score_to_attr = "pic_mc_min_act_rank_score"
        ) \
      .end_() \
      .if_("skip_explore_cascade_pic_xtr_calc == 0", to_be_delete = "date=2024-05-29;committer=caozhong") \
        .enrich_attr_by_light_function(
          target_item={ flag_attr: 1 },
          import_common_attr=[
            "explore_cascade_pic_xtr_calc_with_ctr",
            "explore_cascade_pic_ctr_pow_weight",
          ],
          import_item_attr=[
            "pic_hot_click",
            "pic_hot_long_view",
            "pic_hot_finish_view",
            "pic_hot_pos_wtd",
            "pic_hot_action",
            "pic_hot_collect",
            "pic_hot_enter_comment",
            "pic_hot_comment_effctive_stop",
            "pic_hot_scroll",
          ],
          export_item_attr=[
            "pic_ensemble_pctr",
            "pic_ensemble_pltr",
            "pic_ensemble_pcptr",
            "pic_ensemble_pwtd",
            "pic_ensemble_action",
            "pic_ensemble_collect",
            "pic_ensemble_scroll",
          ],
          function_name="CalcPictureXTR",
          class_name="ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_pic_action_once_cascade_s1 == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "pic_cascade_s1_action_once_ctr_weight", "as": "act_fusion_score_ctr_weight"},
            {"name": "pic_cascade_s1_action_once_ltr_weight", "as": "act_fusion_score_ltr_weight"},
            {"name": "pic_cascade_s1_action_once_wtr_weight", "as": "act_fusion_score_wtr_weight"},
            {"name": "pic_cascade_s1_action_once_ftr_weight", "as": "act_fusion_score_ftr_weight"},
            {"name": "pic_cascade_s1_action_once_cmtr_weight", "as": "act_fusion_score_cmtr_weight"},
            {"name": "pic_cascade_s1_action_once_cltr_weight", "as": "act_fusion_score_cltr_weight"},
            {"name": "pic_cascade_s1_action_once_epstr_weight", "as": "act_fusion_score_epstr_weight"},
          ],
          import_item_attr = [
            {"name": "mc_ensemble_pctr", "as": "pctr"},
            {"name": "mc_ensemble_pltr", "as": "pltr"},
            {"name": "mc_ensemble_pwtr", "as": "pwtr"},
            {"name": "mc_ensemble_pftr", "as": "pftr"},
            {"name": "mc_ensemble_pcmtr", "as": "pcmtr"},
            {"name": "mc_ensemble_pcltr", "as": "pcltr"},
            {"name": "mc_ensemble_pepstr", "as": "pepstr"},
          ],
          export_item_attr = [
            {"name": "action_once_score", "as": "pic_action_once_score"},
          ],
          function_name = "CalActionOnceScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_explore_hv_pic_mc_pxtr_calib_by_emp == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "explore_pic_xtr_emp_debias_map_ptr",
            "basic_info_gender_v2",
            {"name": "basic_info_age_segment_v2", "as":"age_segment"},
            {"name": "explore_hv_pic_mc_emp_debias_thresh", "as": "debias_thresh"},
            {"name": "explore_hv_pic_mc_emp_debias_redis_prefix", "as": "redis_prefix"}
          ],
          import_item_attr = [
            {"name": "mc_ensemble_pctr", "as": "pctr"},
            {"name": "mc_ensemble_pltr", "as": "pltr"},
            {"name": "mc_ensemble_pwtr", "as": "pwtr"},
            {"name": "mc_ensemble_pftr", "as": "pftr"},
            {"name": "mc_ensemble_pcmtr", "as": "pcmtr"},
            {"name": "mc_ensemble_pcltr", "as": "pcltr"},
            "high_value_pic_flag",
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_level_one_list"},
          ],
          export_item_attr = [
            "pic_mc_emp_debias_pwtr",
            "pic_mc_emp_debias_pcmtr",
            "pic_mc_emp_debias_pctr",
            "pic_mc_emp_debias_pltr",
            "pic_mc_emp_debias_pcltr",
          ],
          function_name = "HvPicPxtrEmpDebias",
          class_name = "ExploreLightFunctionSetV2",
          target_item={ flag_attr: 1 }
        ) \
      .end_() \
      .if_("enable_expl_pic_mc_queue_weight_adjust == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_pic_mc_pxtr_attr_config_str", "as": "pxtr_attr_config_str"},
            {"name": "explore_pic_mc_avg_top_num", "as": "avg_top_num"},
          ],
          export_common_attr = [
            {"name": "pxtr_topn_avg_mc_ensemble_pctr", "as": "pic_mc_pxtr_topn_avg_pctr"},
            {"name": "pxtr_topn_avg_cascade_pltr", "as": "pic_mc_pxtr_topn_avg_pltr"},
            {"name": "pxtr_topn_avg_cascade_pwtr", "as": "pic_mc_pxtr_topn_avg_pwtr"},
            {"name": "pxtr_topn_avg_cascade_pcltr", "as": "pic_mc_pxtr_topn_avg_pcltr"},
            {"name": "pxtr_topn_avg_cascade_pcmtr", "as": "pic_mc_pxtr_topn_avg_pcmtr"},
          ],
          import_item_attr = [
            "mc_ensemble_pctr",
            "cascade_pltr",
            "cascade_pwtr",
            "cascade_pcltr",
            "cascade_pcmtr",
          ],
          function_name = "CalcPxtrStatScore",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture": 1
          }
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "expl_pic_mc_q_w_adj_action_benefit_fac_max", "as": "action_benefit_fac_max"},
            {"name": "expl_pic_mc_q_w_adj_action_explore_prob", "as": "action_explore_prob"},
            {"name": "expl_pic_mc_q_w_adj_pctr_power", "as": "pctr_power"},
            {"name": "expl_pic_mc_q_w_adj_pxtr_power", "as": "pxtr_power"},
            {"name": "expl_pic_mc_q_w_adj_pctr_benefit", "as": "pctr_benefit"},
            {"name": "expl_pic_mc_q_w_adj_pltr_benefit", "as": "pltr_benefit"},
            {"name": "expl_pic_mc_q_w_adj_pwtr_benefit", "as": "pwtr_benefit"},
            {"name": "expl_pic_mc_q_w_adj_pcltr_benefit", "as": "pcltr_benefit"},
            {"name": "expl_pic_mc_q_w_adj_pcmtr_benefit", "as": "pcmtr_benefit"},
            {"name": "expl_pic_mc_q_w_adj_pctr_risk", "as": "pctr_risk"},
            {"name": "expl_pic_mc_q_w_adj_pltr_risk", "as": "pltr_risk"},
            {"name": "expl_pic_mc_q_w_adj_pwtr_risk", "as": "pwtr_risk"},
            {"name": "expl_pic_mc_q_w_adj_pcltr_risk", "as": "pcltr_risk"},
            {"name": "expl_pic_mc_q_w_adj_pcmtr_risk", "as": "pcmtr_risk"},
            {"name": "expl_pic_mc_q_w_adj_pctr_coef_min", "as": "pctr_coef_min"},
            {"name": "expl_pic_mc_q_w_adj_pltr_coef_min", "as": "pltr_coef_min"},
            {"name": "expl_pic_mc_q_w_adj_pwtr_coef_min", "as": "pwtr_coef_min"},
            {"name": "expl_pic_mc_q_w_adj_pcltr_coef_min", "as": "pcltr_coef_min"},
            {"name": "expl_pic_mc_q_w_adj_pcmtr_coef_min", "as": "pcmtr_coef_min"},
            {"name": "expl_pic_mc_q_w_adj_pctr_coef_max", "as": "pctr_coef_max"},
            {"name": "expl_pic_mc_q_w_adj_pltr_coef_max", "as": "pltr_coef_max"},
            {"name": "expl_pic_mc_q_w_adj_pwtr_coef_max", "as": "pwtr_coef_max"},
            {"name": "expl_pic_mc_q_w_adj_pcltr_coef_max", "as": "pcltr_coef_max"},
            {"name": "expl_pic_mc_q_w_adj_pcmtr_coef_max", "as": "pcmtr_coef_max"},
            {"name": "pic_mc_pxtr_topn_avg_pctr", "as": "pctr_avg"},
            {"name": "pic_mc_pxtr_topn_avg_pltr", "as": "pltr_avg"},
            {"name": "pic_mc_pxtr_topn_avg_pwtr", "as": "pwtr_avg"},
            {"name": "pic_mc_pxtr_topn_avg_pcmtr", "as": "pcltr_avg"},
            {"name": "pic_mc_pxtr_topn_avg_pcltr", "as": "pcmtr_avg"},
            "pic_stat_pic_like_cnt",
            "pic_stat_pic_follow_cnt",
            "pic_stat_pic_forward_cnt",
            "pic_stat_pic_comment_cnt",
            "explore_mc_ensemble_pic_pctr_power_weight",
            "explore_mc_ensemble_pic_pltr_power_weight",
            "explore_mc_ensemble_pic_pwtr_power_weight",
            "explore_mc_ensemble_pic_pcmtr_power_weight",
            "explore_mc_ensemble_pic_pcltr_power_weight",
          ],
          export_common_attr = [
            "explore_mc_ensemble_pic_pctr_power_weight",
            "explore_mc_ensemble_pic_pltr_power_weight",
            "explore_mc_ensemble_pic_pwtr_power_weight",
            "explore_mc_ensemble_pic_pcmtr_power_weight",
            "explore_mc_ensemble_pic_pcltr_power_weight",
          ],
          function_name = "AdjustQueueWeightsByBenefitRisk",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture": 1
          }
        ) \
      .end_() \
      .if_("enable_explore_pic_vv_control_mc_queue_weight_adjust == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_mc_top_pic_vv_recent_thre", "as": "top_pic_vv_recent_thre"},
            {"name": "explore_mc_tail_pic_vv_recent_thre", "as": "tail_pic_vv_recent_thre"},
            {"name": "pic_stat_pic_recent_play_cnt", "as": "pic_recent_play_cnt"},
            {"name": "explore_mc_ensemble_pic_diversity_mgs_score_power_weight", "as": "diversity_mgs_score_power_weight"},
            {"name": "explore_mc_pic_diversity_mgs_score_power_coeff", "as": "power_coeff"},
          ],
          export_common_attr = [
            {"name": "diversity_mgs_score_power_weight", "as": "explore_mc_ensemble_pic_diversity_mgs_score_power_weight"},
          ],
          function_name = "CalcPicVVControlMgsWeight",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture": 1
          }
        ) \
      .end_() \
      .if_("enable_explore_pic_vid2pic_boost_few_cluster_user == 1 and (uDoubleOutsideValidPicClusterCnt7dKV or 0) < explore_pic_vid2pic_boost_user_cluster_cnt_thresh") \
        .gen_common_attr_by_lua(
          attr_map = {
            "explore_mc_ensemble_vid2pic_sim_score_power_weight": "explore_mc_ensemble_vid2pic_sim_score_power_weight * explore_mc_vid2pic_boost_weight * (explore_pic_vid2pic_boost_user_cluster_cnt_thresh - (uDoubleOutsideValidPicClusterCnt7dKV or 0)) ^ explore_mc_vid2pic_boost_pow_weight",
          }
        ) \
      .end_() \
      .if_("enable_explore_pic_u2c_boost_on_low_interest_user == 1 and (uDoubleOutsideValidPicClusterCnt7dKV or 0) < explore_pic_u2c_boost_interest_thresh") \
        .copy_attr(
          attrs = [
            {"from_common": "explore_mc_ensemble_pic_u2c_ensemble_score_power_weight_low_interest_user", "to_common": "explore_mc_ensemble_pic_u2c_ensemble_score_power_weight"},
            {"from_common": "explore_mc_ensemble_pic_u2c_ensemble_score_raw_weight_low_interest_user", "to_common": "explore_mc_ensemble_pic_u2c_ensemble_score_raw_weight"},
            {"from_common": "explore_mc_ensemble_pic_u2c_ensemble_score_raw_power_weight_low_interest_user", "to_common": "explore_mc_ensemble_pic_u2c_ensemble_score_raw_power_weight"},
          ],
        ) \
      .end_() \
      .if_("enable_explore_pic_mc_real_pctr > 0 and (uDoubleOutsideValidPicClusterCnt7dKV or 0) < explore_pic_mc_real_pctr_interest_thresh") \
        .calc_pic_cascade_s1_real_pctr() \
      .end_() \
      .if_("enable_explore_pic_mc_boost_pctr_on_not_click_user == 1") \
        .mc_pic_boost_pctr_on_not_click_user() \
      .end_() \
      .explore_calc_ensemble_score(
        target_item={ flag_attr: 1 },
        use_superscript_rank = False,
        user_power_calc_v2 = 1,
        user_info_ptr_attr = "user_info_ptr",
        queues = self._get_queue(),
        save_score_to_attr = self._score_attr,
        use_queue_smooth_as_rank_smooth = "{{explore_pic_mc_use_queue_smooth_as_rank_smooth}}",
        value_seq_fusion_status = "{{explore_pic_value_seq_fusion_status}}",
        use_rank_with_absolute_score = "{{explore_pic_mc_use_rank_with_absolute_score}}",
        rank_score_calculate_method = "{{explore_pic_mc_rank_score_calculate_method}}",
        queue_max_raw_score = "{{explore_mc_pic_rerank_queue_max_raw_score}}",
        queue_min_raw_score = "{{explore_mc_pic_rerank_queue_min_raw_score}}",
        enable_normalization_item_score = "{{explore_mc_pic_rerank_enable_normalization_item_score}}",
      )\
      .if_("cascade_enable_follow_author_pic_mc_boost == 1") \
        .boost_pic_cascade_s1_es_by_follow_author(self._score_attr, flag_attr) \
      .end_if_() \
      .if_("enable_cascade_channel_caption_boost == 1") \
         .boost_pic_cascade_s1_es_by_caption(self._score_attr, flag_attr) \
      .end_if_() \
      .if_("enable_cascade_target_hetu_pic_mc_s1_boost == 1") \
         .boost_pic_cascade_s1_es_by_target_hetu(self._score_attr, flag_attr) \
      .end_if_() \
      .if_("cascade_s1_enable_hetu_ratio_decay == 1") \
         .boost_pic_cascade_s1_es_by_hetu_ratio(self._score_attr, flag_attr) \
      .end_if_() \
      .if_("enable_explore_mc_pic_valid_interest_cluster_boost == 1") \
        .mc_pic_valid_interest_cluster_boost(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_explore_mc_pic_long_interest_cluster_boost == 1") \
        .mc_pic_long_interest_cluster_boost(self._score_attr, flag_attr) \
      .end_() \
      .if_("explore_enable_user_pic_growth_cluster_boost == 1 and ((uDoubleOutsideValidPicClusterCnt7dKV or 0) < explore_user_pic_growth_cluster_boost_interest_thresh)") \
        .mc_pic_boost_coef_with_flag(
          coef_attr = "explore_mc_s1_pic_growth_cluster_boost_coef",
          score_attr = self._score_attr,
          flag_attrs = [flag_attr, "is_pic_growth_cluster"],
          boost_num_max_attr = "explore_mc_pic_growth_cluster_boost_num_max",
          boost_num_ratio_attr = "explore_mc_pic_growth_cluster_boost_num_ratio"
        ) \
      .end_() \
      .if_("enable_explore_mc_pic_search_boost == 1 and (pic_search_boost_user_degree or 0) >= explore_pic_search_boost_user_degree_thresh") \
        .if_("enable_explore_pic_search_candicate_expand_by_cluster == 1") \
          .mc_pic_search_boost(self._score_attr, flag_attr, "is_pic_search_cluster") \
        .else_() \
          .mc_pic_search_boost(self._score_attr, flag_attr, "is_pic_search") \
        .end_() \
      .end_() \
      .if_("enable_explore_mc_pic_recent_search_boost == 1 and (pic_search_boost_user_degree or 0) >= explore_pic_search_boost_user_degree_thresh") \
        .if_("enable_explore_pic_search_candicate_expand_by_cluster == 1") \
          .mc_pic_recent_search_boost(self._score_attr, flag_attr, "is_pic_recent_search_cluster") \
        .else_() \
          .mc_pic_recent_search_boost(self._score_attr, flag_attr, "is_pic_recent_search") \
        .end_() \
      .end_() \
      .if_("enable_explore_mc_pic_valid_interest_cluster_boost_first_screen_adjust == 1 and page_index == 1") \
        .gen_common_attr_by_lua(
          attr_map={
            "explore_pic_double_valid_interest_cluster_boost_interest_thresh": "explore_pic_double_valid_interest_cluster_boost_interest_thresh_first_screen",
            "explore_mc_pic_double_valid_interest_cluster_boost_coef": "explore_mc_pic_double_valid_interest_cluster_boost_coef_first_screen",
            "explore_pic_single_valid_interest_cluster_boost_interest_thresh": "explore_pic_single_valid_interest_cluster_boost_interest_thresh_first_screen",
            "explore_mc_pic_single_valid_interest_cluster_boost_coef": "explore_mc_pic_single_valid_interest_cluster_boost_coef_first_screen"
          }
        ) \
      .end_() \
      .if_("enable_explore_mc_pic_double_valid_interest_cluster_boost == 1 and (uDoubleOutsideValidPicClusterCnt7dKV or 0) < explore_pic_double_valid_interest_cluster_boost_interest_thresh") \
        .mc_pic_double_valid_interest_cluster_boost(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_explore_mc_pic_single_valid_interest_cluster_boost == 1 and (uDoubleOutsideValidPicClusterCnt7dKV or 0) < explore_pic_single_valid_interest_cluster_boost_interest_thresh") \
        .mc_pic_single_valid_interest_cluster_boost(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_explore_mc_pic_recent_interest_cluster_boost == 1") \
        .mc_pic_recent_interest_cluster_boost(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_explore_mc_pic_hot_content_topk_boost == 1 and (user_pic_recent_show_cnt or 0) < explore_mc_pic_recent_low_show_boost_thresh and (pic_stat_pic_eff_play_cnt or 0) > explore_mc_pic_recent_low_show_boost_history_play_thresh") \
        .mc_pic_hot_content_topk_boost(self._score_attr, flag_attr, "is_pic_hot_content") \
      .end_() \
      .if_("enable_explore_mc_operation_pic_boost == 1 and user_has_pic_crowd_show == 0") \
        .mc_operation_pic_boost(self._score_attr, flag_attr, "operation_pic") \
      .end_() \
      .if_("enable_explore_mc_pic_type_boost == 1") \
        .mc_pic_type_boost(self._score_attr, flag_attr) \
      .end_()

    self.flow.if_("enable_explore_pic_s1_cluster_sort_v1 == 1 and (uDoubleOutsideValidPicClusterCnt7dKV or 0) < explore_pic_s1_cluster_sort_v1_user_cluster_cnt_thresh")
    self._explore_pic_s1_cluster_sort_v1(flag_attr, weight_attr) \
    .end_()

  def _get_min_rank_queue(self):
    return [
        {
          "name" : "mc_ensemble_pctr",
          "enable_attr": "explore_pic_mc_min_rank_score_enable_pctr"
        },
        {
          "name" : "mc_ensemble_pltr",
          "enable_attr": "explore_pic_mc_min_rank_score_enable_pltr"
        },
        {
          "name" : "mc_ensemble_pwtr",
          "enable_attr": "explore_pic_mc_min_rank_score_enable_pwtr"
        },
        {
          "name" : "mc_ensemble_pcmtr",
          "enable_attr": "explore_pic_mc_min_rank_score_enable_pcmtr"
        },
        {
          "name" : "mc_ensemble_pcltr",
          "enable_attr": "explore_pic_mc_min_rank_score_enable_pcltr"
        },
        {
          "name" : "mc_ensemble_pftr",
          "enable_attr": "explore_pic_mc_min_rank_score_enable_pftr"
        },
        {
          "name": "mc_ensemble_pic_wtd",
          "enable_attr": "explore_pic_mc_min_rank_score_enable_pic_wtd"
        },
        {
          "name" : "mc_ensemble_plvtr",
          "enable_attr": "explore_pic_mc_min_rank_score_enable_plvtr"
        },
        {
          "name" : "mc_ensemble_plvtr2",
          "enable_attr": "explore_pic_mc_min_rank_score_enable_plvtr2"
        }
      ]

  def _get_queue(self):
    return [
        {
          "name" : "mc_ensemble_pctr",
          "weight" : 0.1,
          "power_weight_attr" : "explore_mc_ensemble_pic_pctr_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_pctr_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_pctr_raw_power_weight",
          "raw_bias_attr": 'explore_mc_pic_rerank_mc_ensemble_pctr_raw_bias',
          "smooth_attr": "explore_mc_pic_rerank_mc_ensemble_pctr_rank_smooth",
          "score_threshold": "explore_mc_ensemble_pic_pctr_rank_cliff_threshold",
          "rank_height_attr": "explore_mc_ensemble_pic_pctr_rank_height",
        },
        {
          "name" : "mc_ensemble_pltr",
          "weight" : 0.2,
          "power_weight_attr" : "explore_mc_ensemble_pic_pltr_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_pltr_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_pltr_raw_power_weight",
          "raw_bias_attr": 'explore_mc_pic_rerank_mc_ensemble_pltr_raw_bias',
          "smooth_attr": "explore_mc_pic_rerank_mc_ensemble_pltr_rank_smooth",
          "score_threshold": "explore_mc_ensemble_pic_pltr_rank_cliff_threshold",
          "rank_height_attr": "explore_mc_ensemble_pic_pltr_rank_height",
        },
        {
          "name" : "mc_ensemble_pwtr",
          "weight" : 0.45,
          "power_weight_attr" : "explore_mc_ensemble_pic_pwtr_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_pwtr_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_pwtr_raw_power_weight",
          "raw_bias_attr": 'explore_mc_pic_rerank_mc_ensemble_pwtr_raw_bias',
          "smooth_attr": "explore_mc_pic_rerank_mc_ensemble_pwtr_rank_smooth",
          "score_threshold": "explore_mc_ensemble_pic_pwtr_rank_cliff_threshold",
          "rank_height_attr": "explore_mc_ensemble_pic_pwtr_rank_height",
        },
        {
          "name" : "mc_ensemble_pftr",
          "weight" : 0.05,
          "power_weight_attr" : "explore_mc_ensemble_pic_pftr_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_pftr_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_pftr_raw_power_weight",
          "raw_bias_attr": 'explore_mc_pic_rerank_mc_ensemble_pftr_raw_bias',
          "smooth_attr": "explore_mc_pic_rerank_mc_ensemble_pftr_rank_smooth",
          "score_threshold": "explore_mc_ensemble_pic_pftr_rank_cliff_threshold",
          "rank_height_attr": "explore_mc_ensemble_pic_pftr_rank_height",
        },
        {
          "name" : "mc_ensemble_plvtr",
          "weight" : 0.2,
          "power_weight_attr" : "explore_mc_ensemble_pic_plvtr_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_plvtr_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_plvtr_raw_power_weight"
        },
        {
          "name" : "mc_ensemble_plvtr2",
          "weight" : 0.12,
          "power_weight_attr" : "explore_mc_ensemble_pic_plvtr2_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_plvtr2_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_plvtr2_raw_power_weight",
          "raw_bias_attr": 'explore_mc_pic_rerank_mc_ensemble_plvtr2_raw_bias',
          "smooth_attr": "explore_mc_pic_rerank_mc_ensemble_plvtr2_rank_smooth",
          "score_threshold": "explore_mc_ensemble_pic_plvtr2_rank_cliff_threshold",
          "rank_height_attr": "explore_mc_ensemble_pic_plvtr2_rank_height",
        },
        {
          "name" : "mc_ensemble_psvtr",
          "weight" : -0.1,
          "power_weight_attr" : "explore_mc_ensemble_pic_psvtr_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_psvtr_raw_weight"
        },
        {
          "name": "mc_ensemble_ptr",
          "weight": 0.05,
          "power_weight_attr": "explore_mc_ensemble_pic_ptr_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_ptr_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_ptr_raw_power_weight"
        },
        {
          "name" : "mc_ensemble_pepstr",
          "weight" : 0.3,
          "power_weight_attr" : "explore_mc_ensemble_pic_pepstr_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_pepstr_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_pepstr_raw_power_weight"
        },
        {
          "name" : "mc_ensemble_pcestr",
          "weight" : 0.18,
          "power_weight_attr" : "explore_mc_ensemble_pic_pcestr_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_pcestr_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_pcestr_raw_power_weight"
        },
        {
          "name" : "mc_ensemble_pcmtr",
          "weight" : 0.18,
          "power_weight_attr" : "explore_mc_ensemble_pic_pcmtr_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_pcmtr_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_pcmtr_raw_power_weight",
          "raw_bias_attr": 'explore_mc_pic_rerank_mc_ensemble_pcmtr_raw_bias',
          "smooth_attr": "explore_mc_pic_rerank_mc_ensemble_pcmtr_rank_smooth",
          "score_threshold": "explore_mc_ensemble_pic_pcmtr_rank_cliff_threshold",
          "rank_height_attr": "explore_mc_ensemble_pic_pcmtr_rank_height",
        },
        {
          "name": "cascade_phtr",  #粗排 htr
          "weight": 0.0,
          "reverse_order": True,
          "power_weight_attr": "explore_mc_ensemble_pic_phtr_power_weight",
          "weight_attr": "explore_mc_ensemble_pic_phtr_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_phtr_raw_power_weight"
        },
        {
          "name" : "mc_ensemble_pwatch_time",
          "weight" : 0.45,
          "power_weight_attr" : "explore_mc_ensemble_pic_pwatch_time_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_pwatch_time_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_pwatch_time_raw_power_weight",
          "raw_bias_attr": 'explore_mc_pic_rerank_mc_ensemble_pwatch_time_raw_bias',
          "smooth_attr": "explore_mc_pic_rerank_mc_ensemble_pwatch_time_rank_smooth",
          "score_threshold": "explore_mc_ensemble_pic_pwatch_time_rank_cliff_threshold",
          "rank_height_attr": "explore_mc_ensemble_pic_pwatch_time_rank_height",
        },
        {
          "name" : "mc_ensemble_pcltr",
          "weight" : 0.0,
          "power_weight_attr" : "explore_mc_ensemble_pic_pcltr_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_pcltr_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_pcltr_raw_power_weight",
          "raw_bias_attr": 'explore_mc_pic_rerank_mc_ensemble_pcltr_raw_bias',
          "smooth_attr": "explore_mc_pic_rerank_mc_ensemble_pcltr_rank_smooth",
          "score_threshold": "explore_mc_ensemble_pic_pcltr_rank_cliff_threshold",
          "rank_height_attr": "explore_mc_ensemble_pic_pcltr_rank_height",
        },
        {
          "name" : "mc_ensemble_pwtd",
          "weight" : 0.0,
          "power_weight_attr" : "explore_mc_ensemble_pic_pwtd_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_pwtd_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_pwtd_raw_power_weight"
        },
        {
          "name": "mc_ensemble_pfptr",  #粗排播放完成度队列
          "weight": 1.0,
          "power_weight_attr": "explore_mc_ensemble_pic_pfptr_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_pfptr_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_pfptr_raw_power_weight"
        },
        {
          "name": "pic_ensemble_pctr",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_ensemble_pctr_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_ensemble_pctr_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_ensemble_pctr_raw_power_weight"
        },
        {
          "name": "pic_ensemble_pltr",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_ensemble_pltr_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_ensemble_pltr_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_ensemble_pltr_raw_power_weight"
        },
        {
          "name": "pic_ensemble_pcptr",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_ensemble_pcptr_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_ensemble_pcptr_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_ensemble_pcptr_raw_power_weight"
        },
        {
          "name": "pic_ensemble_pwtd",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_ensemble_pwtd_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_ensemble_pwtd_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_ensemble_pwtd_raw_power_weight"
        },
        {
          "name": "pic_ensemble_action",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_ensemble_paction_weight"
        },
        {
          "name": "pic_ensemble_collect",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_ensemble_pcltr_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_ensemble_pcltr_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_ensemble_pcltr_raw_power_weight"
        },
        {
          "name": "pic_ensemble_scroll",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_ensemble_pscroll_weight"
        },
        {
          "name": "mc_ensemble_pic_wtd",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_ensemble_pic_wtd_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_ensemble_pic_wtd_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_ensemble_pic_wtd_raw_power_weight",
          "raw_bias_attr": 'explore_mc_pic_rerank_mc_ensemble_pic_wtd_raw_bias',
          "smooth_attr": "explore_mc_pic_rerank_mc_ensemble_pic_wtd_rank_smooth",
          "score_threshold": "explore_mc_ensemble_pic_ensemble_pic_wtd_rank_cliff_threshold",
          "rank_height_attr": "explore_mc_ensemble_pic_ensemble_pic_wtd_rank_height",
        },
        {
          "name": "mc_ensemble_pic_lvtr",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_ensemble_pic_lvtr_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_ensemble_pic_lvtr_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_ensemble_pic_lvtr_raw_power_weight",
          "raw_bias_attr": 'explore_mc_pic_rerank_mc_ensemble_pic_lvtr_raw_bias',
          "smooth_attr": "explore_mc_pic_rerank_mc_ensemble_pic_lvtr_rank_smooth",
          "score_threshold": "explore_mc_ensemble_pic_ensemble_pic_lvtr_rank_cliff_threshold",
          "rank_height_attr": "explore_mc_ensemble_pic_ensemble_pic_lvtr_rank_height",
        },
        {
          "name": "mc_ensemble_pic_cpr",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_ensemble_pic_cpr_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_ensemble_pic_cpr_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_ensemble_pic_cpr_raw_power_weight"
        },
        {
          "name": "mc_ensemble_prerank_er",  #粗排LTR
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_prerank_er_power_weight",
        },
        {
          "name": "revisited_rate_1d",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_revisited_rate_1d_power_weight",
        },
        {
          "name": "revisited_rate_3d",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_revisited_rate_3d_power_weight",
        },
        {
          "name": "revisited_rate_7d",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_revisited_rate_7d_power_weight",
        },
        {
          "name" : "cascase_life_ctr",
          "weight" : 0.0,
          "power_weight_attr" : "explore_mc_ensemble_pic_lifetab_pctr_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_lifetab_pctr_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_lifetab_pctr_raw_power_weight"
        },
        {
          "name": "cascade_prerank_score",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_prerank_score_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_prerank_score_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_prerank_score_raw_power_weight",
        },
        {
          "name": "pic_action_once_score",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_action_once_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_action_once_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_action_once_raw_power_weight",
          "raw_bias_attr": 'explore_mc_pic_rerank_pic_action_once_score_raw_bias',
          "smooth_attr": "explore_mc_pic_rerank_pic_action_once_score_rank_smooth",
        },
        {
          "name": "pic_mc_emp_debias_pwtr",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_emp_debias_pwtr_power_weight",
        },
        {
          "name": "pic_mc_emp_debias_pctr",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_emp_debias_pctr_power_weight",
        },
        {
          "name": "pic_mc_emp_debias_pcltr",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_emp_debias_pcltr_power_weight",
        },
        {
          "name": "pic_mc_emp_debias_pcmtr",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_emp_debias_pcmtr_power_weight",
        },
        {
          "name": "pic_mc_emp_debias_pltr",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_emp_debias_pltr_power_weight",
        },
        {
          "name": "pic_variety_score",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_pic_s1_pic_variety_score_power_weight",
          "raw_weight_attr": "explore_mc_pic_s1_pic_variety_score_raw_weight",
          "raw_power_weight_attr": "explore_mc_pic_s1_pic_variety_score_raw_power_weight",
          "raw_bias_attr": 'explore_mc_pic_rerank_pic_variety_score_raw_bias',
          "smooth_attr": "explore_mc_pic_rerank_pic_variety_score_rank_smooth",
        },
        {
          "name": "pic_diversity_mgs_score",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_diversity_mgs_score_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_diversity_mgs_score_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_diversity_mgs_score_raw_power_weight",
          "raw_bias_attr": 'explore_mc_pic_rerank_pic_diversity_mgs_score_raw_bias',
          "smooth_attr": "explore_mc_pic_rerank_pic_diversity_mgs_score_rank_smooth",
        },
        {
          "name": "vid2pic_sim_score",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_vid2pic_sim_score_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_vid2pic_sim_score_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_vid2pic_sim_score_raw_power_weight",
          "raw_bias_attr": "explore_mc_ensemble_vid2pic_sim_score_raw_bias",
          "smooth_attr": "explore_mc_ensemble_vid2pic_sim_score_rank_smooth",
        },
        {
          "name": "pic_unbias_interset_score",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_unbias_interset_score_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_unbias_interset_score_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_unbias_interset_score_raw_power_weight",
          "raw_bias_attr": "explore_mc_ensemble_pic_unbias_interset_score_raw_bias",
          "smooth_attr": "explore_mc_ensemble_pic_unbias_interset_score_rank_smooth",
        },
        {
          "name": "pic_search_interest_cluster_score",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_search_interest_cluster_score_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_search_interest_cluster_score_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_search_interest_cluster_score_raw_power_weight",
          "raw_bias_attr": "explore_mc_ensemble_pic_search_interest_cluster_score_raw_bias",
          "smooth_attr": "explore_mc_ensemble_pic_search_interest_cluster_score_rank_smooth",
        },
        {
          "name": "mc_pic_search_score",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_search_score_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_search_score_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_search_score_raw_power_weight",
          "raw_bias_attr": "explore_mc_ensemble_pic_search_score_raw_bias",
          "smooth_attr": "explore_mc_ensemble_pic_search_score_rank_smooth",
        },
        {
          "name": "pic_mc_min_act_rank_score",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_min_act_rank_score_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_min_act_rank_score_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_min_act_rank_score_raw_power_weight",
        },
        {
          "name": "pic_cascade_fc_pctr",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_pic_s1_pic_cascade_fc_pctr_power_weight",
        },
        {
          "name": "pic_cascade_fc_interact_score",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_pic_s1_pic_cascade_fc_interact_score_power_weight",
        },
        {
          "name": "pic_cascade_fc_ltr",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_pic_s1_pic_cascade_fc_ltr_power_weight",
        },
        {
          "name": "pic_cascade_fc_wtr",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_pic_s1_pic_cascade_fc_wtr_power_weight",
        },
        {
          "name": "pic_cascade_fc_cmtr",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_pic_s1_pic_cascade_fc_cmtr_power_weight",
        },
        {
          "name": "pic_cascade_fc_d2q",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_pic_s1_pic_cascade_fc_d2q_power_weight",
        },
        {
          "name": "is_same_location",
          "value_type": "int",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_pic_s1_pic_same_location_power_weight",
        },
        {
          "name": "high_value_pic_flag",
          "value_type": "int",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_pic_s1_high_value_pic_flag_power_weight",
        },
        {
          "name": "longpic_picset_score",
          "weight": 0.0,
          "default": 0.1,
          "power_weight_attr": "explore_mc_pic_s1_longpic_picset_power_weight",
        },
        {
          "name": "cascade_phtr",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_phtr_order_power_weight",
          "weight_attr": "explore_mc_ensemble_pic_phtr_order_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_phtr_order_raw_power_weight",
          "raw_bias_attr": 'explore_mc_pic_rerank_cascade_phtr_raw_bias',
          "smooth_attr": "explore_mc_pic_rerank_cascade_phtr_rank_smooth",
          "score_threshold": "explore_mc_ensemble_pic_phtr_order_rank_cliff_threshold",
          "rank_height_attr": "explore_mc_ensemble_pic_phtr_order_rank_height",
        },
        {
          "name": "cascade_prerank_pctr",
          "weight": 0.0,
          "power_weight_attr" : "explore_mc_ensemble_s1_cascade_prerank_pic_pctr_power_weight",
        },
        {
          "name": "cascade_prerank_pltr",
          "weight": 0.0,
          "power_weight_attr" : "explore_mc_ensemble_s1_cascade_prerank_pic_pltr_power_weight",
        },
        {
          "name": "pic_mc_ltr_ctr",
          "weight": 0.0,
          "power_weight_attr": "explore_cascade_es_pic_mc_ltr_ctr_power",
        },
        {
          "name": "pic_mc_ltr_fvtr",
          "weight": 0.0,
          "power_weight_attr": "explore_cascade_es_pic_mc_ltr_fvtr_power",
        },
        {
          "name": "pic_mc_ltr_wtd",
          "weight": 0.0,
          "power_weight_attr": "explore_cascade_es_pic_mc_ltr_wtd_power",
        },
        {
          "name" : "cascade_real_pctr",
          "weight" : 0.0,
          "power_weight_attr" : "explore_pic_mc_real_pctr_pow_weight",
          "raw_weight_attr": "explore_pic_mc_real_pctr_raw_weight",
          "raw_power_weight_attr": "explore_pic_mc_real_pctr_raw_pow_weight",
          "raw_bias_attr": 'explore_pic_mc_real_pctr_raw_bias',
          "smooth_attr": "explore_pic_mc_real_pctr_smooth",
        },
        {
          "name": "pic_valid_interest_tag_score",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_valid_interest_tag_score_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_valid_interest_tag_score_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_valid_interest_tag_score_raw_power_weight",
          "raw_bias_attr": "explore_mc_ensemble_pic_valid_interest_tag_score_raw_bias",
          "smooth_attr": "explore_mc_ensemble_pic_valid_interest_tag_score_rank_smooth",
        },
        {
          "name" : "cascading_explore_gamora_interest_ptr",
          "weight" : 0.0,
          "power_weight_attr" : "explore_pic_mc_gamora_interest_ptr_power_weight",
          "raw_weight_attr": "explore_pic_mc_gamora_interest_ptr_raw_weight",
          "raw_power_weight_attr": "explore_pic_mc_gamora_interest_ptr_raw_power_weight",
          "raw_bias_attr": 'explore_pic_mc_gamora_interest_ptr_raw_bias',
          "smooth_attr": "explore_pic_mc_gamora_interest_ptr_rank_smooth",
        },
        {
          "name" : "cascading_explore_gamora_interest_ltr",
          "weight" : 0.0,
          "power_weight_attr" : "explore_pic_mc_gamora_interest_ltr_power_weight",
          "raw_weight_attr": "explore_pic_mc_gamora_interest_ltr_raw_weight",
          "raw_power_weight_attr": "explore_pic_mc_gamora_interest_ltr_raw_power_weight",
          "raw_bias_attr": 'explore_pic_mc_gamora_interest_ltr_raw_bias',
          "smooth_attr": "explore_pic_mc_gamora_interest_ltr_rank_smooth",
        },
        {
          "name": "mc_pic_u2c_ensemble_score",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_u2c_ensemble_score_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_u2c_ensemble_score_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_u2c_ensemble_score_raw_power_weight",
          "smooth_attr": "explore_mc_ensemble_pic_u2c_ensemble_score_rank_smooth",
        },
        {
          "name": "pic_search_interest_tagnex_score",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_search_interest_tagnex_score_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_search_interest_tagnex_score_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_search_interest_tagnex_score_raw_power_weight",
          "smooth_attr": "explore_mc_ensemble_pic_search_interest_tagnex_score_rank_smooth",
        },
        {
          "name": "pic_mc_comment_quality_score",
          "weight": 0.0,
          "power_weight_attr": "explore_pic_mc_comment_quality_score_power_weight",
          "raw_weight_attr": "explore_pic_mc_comment_quality_score_raw_weight",
          "raw_power_weight_attr": "explore_pic_mc_comment_quality_score_raw_power_weight",
          "smooth_attr": "explore_pic_mc_comment_quality_score_rank_smooth",
        },
        {
          "name": "pic_u2c_collaborative_score",
          "weight": 0.0,
          "power_weight_attr": "explore_pic_mc_u2c_collaborative_score_power_weight",
          "raw_weight_attr": "explore_pic_mc_u2c_collaborative_score_raw_weight",
          "raw_power_weight_attr": "explore_pic_mc_u2c_collaborative_score_raw_power_weight",
          "smooth_attr": "explore_pic_mc_u2c_collaborative_score_rank_smooth",
        },
        {
          "name": "pic_career_interest_tagnex_tgi_score", # 2026-06-03 by zhangziqian03
          "weight": 0.0,
          "power_weight_attr": "explore_pic_mc_career_tgi_score_power_weight",
        },
        {
          "name": "pic_age_interest_tagnex_tgi_score", # 2026-06-03 by zhangziqian03
          "weight": 0.0,
          "power_weight_attr": "explore_pic_mc_age_tgi_score_power_weight",
        },
      ]

  def _explore_pic_s1_cluster_sort_v1(self, flag_attr, weight_attr):
    self.flow \
      .explore_pic_calc_cluster(
        user_info_ptr_attr = "user_info_ptr",
        save_cluster_id_to_attr = "cascade_cluster_id",
        perf_checkpoint = "pic_calc_cluster",
        enable_multi_hit_independent_bucket = "{{mc_s1_enable_multi_hit_independent_bucket}}",
        clusters = [
          {
            "name": "pic_default",
            "cluster_type_id": 10000,
            "enable": "{{explore_pic_s1_cluster_sort_pic_default_enable}}",
            "upload_type_attr": "upload_type",
            "picture_type_attr": "picture_type",
            "picture_count_attr": "photo_picture_count",
            "priority_num": '{{mc_s1_explore_pic_pic_default_priority_num}}'
          },
          {
            "name": "short_term",
            "cluster_type_id": 20000,
            "enable": "{{explore_pic_s1_cluster_sort_short_term_enable}}",
            "realshow_pic_list_attr": "uStandardRealShowPicAllIdList",
            "hetu_level_one_attr": "hetu_tag_level_info__hetu_level_one",
            "valid_play_weight_attr": "explore_pic_mc_s1_cluster_sort_short_term_valid_play_weight",
            "interact_weight_attr": "explore_pic_mc_s1_cluster_sort_short_term_interact_weight",
            "time_range_sec_attr": "explore_pic_mc_s1_cluster_sort_short_term_time_range_sec",
            "only_pic_attr": "explore_pic_mc_s1_cluster_sort_short_term_only_pic",
            "limit_num_attr": "explore_pic_mc_s1_cluster_sort_short_term_limit_num",
            "score_thresh_attr": "explore_pic_mc_s1_cluster_sort_short_term_score_thresh",
            "priority_num": "{{mc_s1_explore_pic_short_term_priority_num}}"
          },
          {
            "name": "long_term",
            "cluster_type_id": 30000,
            "enable": "{{explore_pic_s1_cluster_sort_long_term_enable}}",
            "colossus_hetu_l1_tags_attr": "pic_hetu_l1_cnt2",
            "enable_user_longterm_hetu_distr_attr": "explore_pic_mc_enable_user_longterm_hetu_distr",
            "user_longterm_hetu_distr_attr": "user_pic_interest_hetu_distr",
            "hetu_level_one_attr": "hetu_tag_level_info__hetu_level_one",
            "limit_num_attr": "explore_pic_mc_s1_cluster_sort_long_term_limit_num",
            "score_thresh_attr": "explore_pic_mc_s1_cluster_sort_long_term_score_thresh",
            "priority_num": "{{mc_s1_explore_pic_long_term_priority_num}}"
          },
          {
            "name": "privilege_tag",
            "cluster_type_id": 40000,
            "enable": "{{explore_pic_s1_cluster_sort_privilege_tag_enable}}",
            "privilege_tags_attr": "pic_mc_cluster_sort_privilege_tags",
            "hetu_level_one_attr": "hetu_tag_level_info__hetu_level_one",
            "priority_num": '{{mc_s1_explore_pic_privilege_tag_priority_num}}'
          },
          {
            "name": "long_caption",
            "cluster_type_id": 50000,
            "enable": "{{explore_pic_s1_cluster_sort_long_caption_enable}}",
            "caption_length_attr": "caption_length",
            "is_xhs_type_photo_attr": "is_xhs_type_photo",
            "caption_length_threshold": 50,
            "priority_num": '{{mc_s1_explore_pic_long_caption_priority_num}}'
          },
          {
            "name": "follow_author",
            "cluster_type_id": 50000,
            "enable": "{{explore_pic_s1_cluster_sort_follow_author_enable}}",
            "is_follow_author_attr": "is_follow_author",
            "priority_num": '{{mc_s1_explore_pic_follow_author_priority_num}}'
          },
          {
            "name": "high_value_pic",
            "cluster_type_id": 60000,
            "enable": "{{explore_pic_s1_cluster_sort_high_value_pic_enable}}",
            "high_value_pic_flag_attr": "high_value_pic_flag",
            "priority_num": '{{mc_s1_explore_pic_high_value_pic_priority_num}}'
          },
          {
            "name": "interest_explore",
            "cluster_type_id": 70000,
            "enable": "{{explore_pic_s1_cluster_sort_interest_explore_enable}}",
            "interest_explore_hetu_list_attr": "pic_interest_explore_hetu_list",
            "hetu_level_one_attr": "hetu_tag_level_info__hetu_level_one",
            "priority_num": '{{mc_s1_explore_pic_interest_explore_priority_num}}'
          },
          {
            "name": "long_interest_hetu",
            "cluster_type_id": 80000,
            "enable": "{{explore_pic_mc_s1_cluster_sort_long_interest_hetu_enable}}",
            "long_interest_hetu_ids_attr": "uHetuCategoryInterestlv1IdList",
            "long_interest_hetu_scores_attr": "uHetuCategoryInterestlv1ScoreList",
            "hetu_level_one_attr": "hetu_tag_level_info__hetu_level_one",
            "limit_num_attr": "explore_pic_mc_s1_cluster_sort_long_interest_hetu_limit_num",
            "score_thresh_attr": "explore_pic_mc_s1_cluster_sort_long_interest_hetu_score_thresh",
            "priority_num": '{{explore_pic_mc_s1_cluster_sort_long_interest_hetu_priority_num}}'
          },
          {
            "name": "valid_interest_cluster",
            "cluster_type_id": 90000,
            "enable": "{{explore_pic_mc_s1_cluster_sort_valid_interest_cluster_enable}}",
            "interest_cluster_ids_attr": "uPicValidInterestClusterIdList",
            "cluster_id_632_attr": "cluster_id_632",
            "one_bucket_attr": "explore_pic_mc_s1_cluster_sort_valid_interest_cluster_one_bucket",
            "priority_num": "{{explore_pic_mc_s1_cluster_sort_valid_interest_cluster_priority_num}}"
          },
          {
            "name": "long_interest_cluster",
            "cluster_type_id": 100000,
            "enable": "{{explore_pic_mc_s1_cluster_sort_long_interest_cluster_enable}}",
            "interest_cluster_ids_attr": "uPicLongInterestClusterIdList",
            "cluster_id_632_attr": "cluster_id_632",
            "one_bucket_attr": "explore_pic_mc_s1_cluster_sort_long_interest_cluster_one_bucket",
            "priority_num": '{{explore_pic_mc_s1_cluster_sort_long_interest_cluster_priority_num}}'
          },
          {
            "name": "search_interest_cluster",
            "cluster_type_id": 110000,
            "enable": "{{explore_pic_mc_s1_cluster_sort_search_interest_cluster_enable}}",
            "interest_cluster_ids_attr": "uPicSearchInterestClusterIdList",
            "cluster_id_632_attr": "cluster_id_632",
            "one_bucket_attr": "explore_pic_mc_s1_cluster_sort_search_interest_cluster_one_bucket",
            "priority_num": '{{explore_pic_mc_s1_cluster_sort_search_interest_cluster_priority_num}}'
          },
        ],
        target_item = {
          flag_attr : 1
        }
      ) \
      .explore_cluster_variant_sort_v2_enrich(
        check_point = "pic_calc_cluster",
        use_superscript_rank = False,
        weight_check_skip = True,
        cluster_attr_name = "cascade_cluster_id",
        hetu_level_one_name = "hetu_tag_level_info__hetu_level_one",
        global_cut_ratio = "{{explore_pic_global_cut_ratio}}",  #
        min_survival = "{{mc_explore_pic_bucket_shrink_min_num}}",
        use_reciprocal = "{{explore_pic_use_multiply_fusion_s1}}",
        explore_pic_user_power_calc_v2 = "{{explore_pic_user_power_calc_v2_enable}}",
        size_limit = "{{mc_explore_pic_final_candidate_num}}",
        user_info_ptr_attr = "user_info_ptr",
        action_day = "{{mc_variant_weight_action_day_num}}",
        enable_dynamic_weight_by_user_degree = "{{mc_explore_pic_enable_user_differently_by_degree_s1}}",
        enable_variant_cut_ratio = "{{explore_explore_pic_cascade_enable_variant_cut_ratio}}",
        variant_cut_ratio = "{{explore_pic_cascade_variant_cut_ratio}}",
        save_score_to_attr = self._score_attr,
        rank_smooth = "{{explore_pic_mc1_rank_smooth}}",
        use_rank_as_score = "{{explore_pic_cascade_channel_sort_use_rank_as_score}}",
        queues = self._get_queue(),
        explore_pic_es_score_attr_name = self._score_attr,
        enable_explore_pic_user_es_score = True,
        target_item = { # 这个 processor 会对视频做逻辑截断，这里只对打上标签的 item 进行处理
          flag_attr : 1
        }
      )
    return self.flow

class PictureQueueFinalScorer(ChannelSortQueueScorer):
  def __init__(self, name, flow, config):
    super().__init__(name, flow, config)

  def _caculate_score(self, flag_attr, weight_attr, left_count_attr):
    self.flow \
    .copy_attr(
      attrs=[{
        "from_item": 'picture_mc_stage1_score',
        "to_item": self._score_attr
      }],
      target_item={ flag_attr: 1 }
    ) \
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
        }
      ) \
    .end_() \
    .if_("enable_cascade_refinement_boost_personified_author == 1") \
      .refinement_boost_personified_author(self._score_attr, flag_attr) \
    .end_() \
    .if_("enable_explore_pic_hack_act_mc_es_decay == 1", to_be_delete = "date=2024-05-29;committer=zhuwenyong") \
      .hack_act_pic_es_decay(self._score_attr, flag_attr) \
    .end_() \
    .if_("enable_fr_refactor_pic_mc_same_author == 1") \
      .deduplicate(
        on_item_attr = "author__id",
        target_item = {
          flag_attr : 1
        }
      ) \
    .end_() \
    .if_("enable_pic_mc_phtr_control == 1") \
      .pic_phtr_mc_filter(flag_attr) \
    .end_() \

    # 动态放弃channel槽位, 放在 _caculate_score 最后
    self.flow.if_("skip_cascade_s2_pic_channel_dynamic_shrink == 0") \
      .sort(
        score_from_attr = self._score_attr,
        target_item={ flag_attr: 1 }
      ) \
      .if_("enable_cascade_s2_pic_quota_pxtr_adjust == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "cascade_s2_pic_channel_pctr_weight", "as": "pctr_weight"},
            {"name": "cascade_s2_pic_channel_pltr_weight", "as": "pltr_weight"},
            {"name": "cascade_s2_pic_channel_pwtr_weight", "as": "pwtr_weight"},
            {"name": "cascade_s2_pic_channel_pftr_weight", "as": "pftr_weight"},
            {"name": "cascade_s2_pic_channel_pcltr_weight", "as": "pcltr_weight"},
            {"name": "cascade_s2_pic_channel_pxtr_avg_topn", "as": "avg_top_num"},
            {"name": "cascade_s2_pic_channel_pos_decay_coeff", "as": "pos_decay_coeff"},
            {"name": "cascade_s2_pic_channel_adjust_max_limit", "as": "adjust_max_limit"},
            {"name": "cascade_s2_pic_channel_adjust_smooth_coeff", "as": "reward_smooth_coeff"},
            {'name': "cascade_s2_pic_channel_adjust_smooth_frac", "as": "reward_smooth_frac"},
            {"name": "cascade_s2_pic_channel_adjust_smooth_bias", "as": "reward_smooth_bias"}
          ],
          import_item_attr = [
            "mc_ensemble_pctr",
            "mc_ensemble_pltr",
            "mc_ensemble_pwtr",
            "mc_ensemble_pftr",
            "mc_ensemble_pcltr"
          ],
          export_common_attr = [
            "cascade_s2_pxtr_topn_avg_score"
          ],
          function_name = "CalcFinalChannelPicQuotaAdjust",
          class_name = "ExploreLightFunctionSetV2",
          target_item = { flag_attr: 1 }
        ) \
      .end_() \
      .if_("enable_cascade_s2_pic_quota_v2 == 1") \
        .split_string( # 打压参数
            input_common_attr="cascade_s2_quota_prefer_score_weights_str",
            output_common_attr="cascade_s2_quota_prefer_score_weights",
            delimiters=",",
            parse_to_double=True
        ) \
        .split_string( # 打压参数
            input_common_attr="cascade_s2_quota_colossus_score_weights_str",
            output_common_attr="cascade_s2_quota_colossus_score_weights",
            delimiters=",",
            parse_to_double=True
        ) \
        .split_string( # 打压参数
            input_common_attr="cascade_s2_quota_recent_decay_weights_str",
            output_common_attr="cascade_s2_quota_recent_decay_weights",
            delimiters=",",
            parse_to_double=True
        ) \
        .split_string( # 打压参数
            input_common_attr="cascade_s2_quota_pxtr_score_weights_str",
            output_common_attr="cascade_s2_quota_pxtr_score_weights",
            delimiters=",",
            parse_to_double=True
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "basic_info_age_segment_v2",
            {"name": weight_attr, "as": "pic_channel_weight"},
            {"name": "pic_stat_pic_play_cnt", "as": "user_colossus_pic_play_count"},
            {"name": "pic_stat_video_play_cnt", "as": "user_colossus_video_play_count"},
            {"name": "explore_cascading_s2_pic_full", "as": "cascading_s2_pic_full"},
            "cascade_s2_quota_longterm_interest_adjust_map",
            "cascade_s2_pxtr_topn_avg_score",
            "dynamic_pic_quota",
            "enable_pic_explore_flag",
            "user_pic_recent_ctr_score",
            "cascade_s2_explore_bias",
            "cascade_s2_explore_keep_min",
            "cascade_s2_quota_adjust_coeff",
            "cascade_s2_quota_range",
            "enable_cascade_s2_quota_v2_limit",
            # 打分因子参数
            "cascade_s2_quota_prefer_score_weights",
            "cascade_s2_quota_colossus_score_weights",
            "cascade_s2_quota_recent_decay_weights",
            "cascade_s2_quota_pxtr_score_weights",
            # 外流偏好用户图文quota摸高
            "external_prefer_user_flag",
          ],
          export_item_attr = [
            {"name": "score_attr", "as": self._score_attr},
          ],
          export_common_attr = [
            "pic_final_quota"
          ],
          function_name = "FinalChannelSortPicQueueDynamicShrinkV2",
          class_name = "ExploreLightFunctionSetV2",
          target_item={ flag_attr: 1 }
        ) \
      .else_if_("enable_cascade_s2_pic_quota_emp_ctr == 1") \
        .split_string( # 打压参数
          input_common_attr="cascade_s2_quota_emp_ctr_score_weights_str",
          output_common_attr="cascade_s2_quota_emp_ctr_score_weights",
          delimiters=",",
          parse_to_double=True
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": weight_attr, "as": "pic_channel_weight"},
            "pic_ctr_preference_coeff",
            "enable_pic_explore_flag",
            "cascade_s2_explore_keep_min",
            "cascade_s2_quota_emp_ctr_score_weights",
            "cascade_s2_quota_range",
          ],
          export_common_attr = [
            "pic_final_quota"
          ],
          function_name = "FinalChannelSortPicQueueDynamicShrinkByEmpCtr",
          class_name = "ExploreLightFunctionSetV2",
          target_item={ flag_attr: 1 }
        ) \
      .else_() \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": weight_attr, "as": "pic_channel_weight"},
            {"name": left_count_attr, "as": "pic_left_count"},
            # ab param
            "mc_final_candidate_num",
            {"name": "cascade_s2_pic_channel_keep_min", "as": "pic_channel_keep_min"},
            {"name": "cascade_s2_pic_channel_pow_weight", "as": "pic_channel_pow_weight"},
            {"name": "cascade_s2_pic_channel_recent_weight", "as": "pic_channel_recent_weight"},
            {"name": "cascade_s2_pic_channel_colossus_weight", "as": "pic_channel_colossus_weight"},
            {"name": "cascade_s2_pic_channel_weight_mode", "as": "pic_channel_weight_mode"},
            {"name": "cascade_s2_pic_channel_enable_thompson", "as": "pic_channel_enable_thompson"},
            {"name": "cascade_s2_pic_channel_thompson_avg_ratio", "as": "pic_channel_thompson_avg_ratio"},
            {"name": "cascade_s2_pic_channel_thompson_scale", "as": "pic_channel_thompson_scale"},
            {"name": "cascade_s2_enable_pic_channel_thompson_scale_quota", "as": "enable_pic_channel_thompson_scale_quota"},
            {"name": "cascade_s2_pic_channel_thompson_scale_quota_base", "as": "pic_channel_thompson_scale_quota_base"},
            {"name": "cascade_s2_pic_channel_thompson_range", "as": "pic_channel_thompson_range"},
            {"name": "cascade_s2_pic_channel_prefer_score_weight", "as": "prefer_score_weight"},
            {"name": "cascade_s2_pic_channel_prefer_score_coeff", "as": "prefer_score_coeff"},
            # recent
            "user_pic_play_count",
            "user_photo_play_count", # 视频 + 图文
            "user_pic_eff_play_count",
            "user_photo_eff_play_count", # 视频 + 图文
            # colossus
            {"name": "pic_stat_pic_play_cnt", "as": "user_colossus_pic_play_count"},
            {"name": "pic_stat_video_play_cnt", "as": "user_colossus_video_play_count"}, # 视频
            {"name": "pic_stat_pic_eff_play_cnt", "as": "user_colossus_pic_eff_play_count"},
            {"name": "pic_stat_video_eff_play_cnt", "as": "user_colossus_video_eff_play_count"}, # 视频
            # pxtr adjust
            "cascade_s2_pxtr_topn_avg_score",
            "dynamic_pic_quota",
            # uv explore adjust
            "enable_pic_explore_flag",
            "enable_cascade_s2_pic_quota_explore",
            "cascade_s2_quota_explore_keep_hold",
            "cascade_s2_quota_explore_keep_thres",
            "enable_cascade_s2_quota_limit",
          ],
          export_item_attr = [
            {"name": "score_attr", "as": self._score_attr},
          ],
          export_common_attr = [
            "pic_final_quota"
          ],
          function_name = "FinalChannelSortPicQueueDynamicShrink",
          class_name = "ExploreLightFunctionSetV2",
          target_item={ flag_attr: 1 }
        ) \
      .end_() \
      .if_("enable_cascade_s2_pic_quota_uplift_adjust == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "uExplorePicUpliftValuesKV",
            "cascade_s2_pic_quota_uplift_task_num",
            "cascade_s2_pic_quota_uplift_thresholds",
            "cascade_s2_pic_quota_uplift_coeff_alphas",
            "cascade_s2_pic_quota_uplift_coeff_betas",
            "cascade_s2_pic_quota_uplift_coeff_power_weights",
            "cascade_s2_pic_quota_uplift_upper_bound",
            "cascade_s2_pic_quota_uplift_lower_bound",
            "pic_final_quota"
          ],
          export_common_attr = [
            "pic_final_quota"
          ],
          function_name = "RankPicQuotaUpliftAdjust",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_pic_mc_diversity_control == 1") \
        .explore_pic_diversity_control_enricher(
          enable_interest_control = "{{enable_pic_mc_interest_control}}",
          enable_hetu_control = "{{enable_pic_mc_hetu_control}}",
          enable_cluster_control = "{{enable_pic_mc_cluster_control}}",
          enable_actual_hetu_control = "{{enable_pic_mc_actual_hetu_adjust}}",
          enable_phtr_control = "{{enable_pic_mc_phtr_control}}",
          keep_size = "pic_final_quota",
          enable_quota_complete = "{{pic_mc_diversity_quota_complete}}",
          quota_complete_adjust_coeff = "{{pic_mc_diversity_quota_complete_adjust}}",
          final_quota_adjust = "{{pic_mc_diversity_final_quota_adjust}}",
          user_hetu_distribution_attr = "colossus_hetu_distribution_hetu_stat",
          user_actual_distribution_attr = "colossus_actual_reward_hetu_stat",
          old_cluster_id_interest_list_attr = "uOldMmuClusterId300ListList",
          cluster_id_attr = "hetu_sim_cluster_id",
          hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
          hetu_level_two_attr = "hetu_tag_level_info__hetu_level_two",
          hetu_level_five_attr = "hetu_tag_level_info__hetu_level_five",
          cluster_control_start = "{{pic_mc_cluster_control_start}}",
          interest_control_start = "{{pic_mc_interest_control_start}}",
          hetu_control_start = "{{pic_mc_hetu_control_start}}",
          cluster_quota_coeff = "{{pic_mc_cluster_quota_coeff}}",
          hetu1_quota_coeff = "{{pic_mc_hetu1_quota_coeff}}",
          hetu2_quota_coeff = "{{pic_mc_hetu2_quota_coeff}}",
          hetu5_quota_coeff = "{{pic_mc_hetu5_quota_coeff}}",
          hetu_adjust_coef = "{{pic_mc_hetu_adjust_coef}}",
          hetu_adjust_min_value = "{{pic_mc_hetu_adjust_min_value}}",
          hetu_adjust_max_value = "{{pic_mc_hetu_adjust_max_value}}",
          old_cluster_id_interest_coef = "{{pic_mc_cluster_interest_boost_coef}}",
          # 动态保消费
          enable_dynamic_hetu_control_start = "{{enable_pic_mc_dynamic_hetu_control_start}}",
          dynamic_hetu_control_start_alpha = "{{pic_mc_dynamic_hetu_control_start_alpha}}",
          dynamic_hetu_control_start_bias = "{{pic_mc_dynamic_hetu_control_start_bias}}",
          dynamic_hetu_control_start_pow = "{{pic_mc_dynamic_hetu_control_start_pow}}",
          dynamic_hetu_control_start_min = "{{pic_mc_dynamic_hetu_control_start_min}}",
          dynamic_hetu_control_start_max = "{{pic_mc_dynamic_hetu_control_start_max}}",
          # 图文类型 quota 限制
          picture_type_attr = "picture_type",
          enable_pic_type_control = "{{enable_explore_pic_mc_pic_type_control}}",
          pic_type_control_start = "{{explore_pic_mc_pic_type_control_start}}",
          pic_type_control_single_pic_max_ratio = "{{explore_pic_mc_pic_type_control_single_pic_max_ratio}}",
          pic_type_control_pic_set_max_ratio = "{{explore_pic_mc_pic_type_control_pic_set_max_ratio}}",
          pic_type_control_long_pic_max_ratio = "{{explore_pic_mc_pic_type_control_long_pic_max_ratio}}",
          # 动态 quota 按占比分配
          enable_dynamic_quota_by_hetu = "{{enable_pic_mc_dynamic_quota_by_hetu}}",
          enable_dynamic_quota_by_cid = "{{enable_pic_mc_dynamic_quota_by_cid}}",
          relax_factor_hetu = "{{pic_mc_relax_factor_hetu}}",
          relax_factor_cid = "{{pic_mc_relax_factor_cid}}",
          cluster632_id_attr = "cluster_id_632",
          # 其他
          es_score_attr = self._score_attr,
          phtr_filter_attr = "is_phtr_mc_filter_pic",
          target_item={ flag_attr: 1 }
        ) \
      .end_() \
    .end_()