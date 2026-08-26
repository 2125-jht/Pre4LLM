from cascading.module.queue.cascading_channel_sort_queue import ChannelSortQueuePartitioner
from cascading.module.queue.cascading_channel_sort_queue import ChannelSortQueueScorer
from cascading.module.queue.cascade_queues import *
from cascading.module.queue.cascade_prerank_queues import pic_prerank_ensemble_sort_queues

class PictureQueueParitioner(ChannelSortQueuePartitioner):
  def __init__(self, name, flow, config):
    super().__init__(name, flow, config)

  def _calculate_flag(self):
    self.flow \
      .if_("enable_cascading_use_longpic_picset == 1") \
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
    self.flow.set_attr_value(
      no_overwrite=True,
      item_attrs=[
        {
          "name": "pic_ensemble_action",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "pic_ensemble_collect",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "pic_ensemble_key_target_hetu_score",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "pic_ensemble_pcptr",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "pic_ensemble_pctr",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "pic_ensemble_pltr",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "pic_ensemble_pwtd",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "pic_ensemble_scroll",
          "type": "double",
          "value": 0.0
        },
      ]
    )
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
          "prerank_ltr_weight",
          "prerank_ctr_weight",
          "prerank_wtd_weight",
          "prerank_life_ctr_weight",
        ],
        import_item_attr = [
          "cascade_prerank_pctr",
          "cascade_prerank_pltr",
          "cascade_emp_watchtime_score",
          "prerank_ltr",
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

    self.flow.if_("enable_explore_pic_target_cluster_sort == 1")
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
    self.flow.explore_life_pic_calc_cluster(
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
          "caption_length_attr_name": "caption_length",
          "is_xhs_type_photo_attr_name": "is_xhs_type_photo",
          "caption_length_threshold": 50,
          "priority_num": '{{explore_prerank_explore_pic_long_caption_priority_num}}'
        },
        {
          "name": "follow_author",
          "cluster_type_id": 20000,
          "enable": "{{explore_pic_prerank_cluster_sort_follow_author_enable}}",
          "is_follow_author_attr_name": "is_follow_author",
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
        {
          "name": "short_term",
          "cluster_type_id": 70000,
          "enable": "{{explore_pic_prerank_cluster_sort_short_term_enable}}",
          "tag_score_min": 2,
          "limit_num": 10,
          "time_range_sec": 86400,
          "video_play_weight": 1,
          "video_play_ev_ms_thd": 3000,
          "hetu_level": 1,
          "hetu_level_attr": "hetu_tag_level_info__hetu_level_one",
          "priority_num": '{{explore_prerank_pic_short_term_priority_num}}',
          "only_pic": "pic_cascading_s1_short_term_only_pic"
        },
        {
          "name": "long_term",
          "cluster_type_id": 80000,
          "enable": "{{explore_pic_prerank_cluster_sort_long_term_enable}}",
          "enable_user_longterm_hetu_distr": "explore_pic_mc_enable_user_longterm_hetu_distr",
          "user_longterm_hetu_distr_attr": "user_pic_interest_hetu_distr",
          "enable_longterm_default_hetu": "explore_pic_mc_enable_longterm_default_hetu",
          "tag_score_min": "pic_cascading_s1_tag_score_min",
          "limit_num": 5,
          "colossus_hetu_l1_tags_attr": "pic_hetu_l1_cnt2",
          "hetu_level_one_attr": "hetu_tag_level_info__hetu_level_one",
          "priority_num": '{{explore_prereank_pic_long_term_priority_num}}'
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
          "name": "pic_ensemble_key_target_hetu_score",
          "weight": 0.0,
          "power_weight_attr": "explore_pic_prerank_ensemble_key_target_hetu_weight"
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
    self.flow.set_attr_value(
      no_overwrite=True,
      item_attrs=[
        {
          "name": "mc_ensemble_pic_oppo_cost_score",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "pic_cascade_interact_fusion_score",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "pic_cascade_watch_time_fusion_score",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "pic_ensemble_comment_effctive_stop",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "pic_ensemble_enter_comment",
          "type": "double",
          "value": 0.0
        }
      ]
    )
    self.flow \
      .if_("enable_pic_cascade_variety == 1") \
        .calc_pic_set_variety_score(flag_attr) \
      .end_() \
      .if_("enable_explore_pic_mc_min_act_rank_score == 1") \
        .explore_min_act_rank_score_enricher(
          target_item = { flag_attr: 1 },
          max_rank_ratio = "{{explore_pic_mc_max_rank_ratio}}",
          queues = self._get_min_rank_queue(),
          save_score_to_attr = "pic_mc_min_act_rank_score"
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
      .explore_calc_ensemble_score(
        target_item={ flag_attr: 1 },
        use_superscript_rank = False,
        user_power_calc_v2 = 1,
        user_info_ptr_attr = "user_info_ptr",
        queues = self._get_queue(),
        save_score_to_attr = self._score_attr,
        value_seq_fusion_status = "{{explore_pic_value_seq_fusion_status}}"
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

    self.flow.if_("enable_explore_pic_s1_cluster_sort_v1 == 1")
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
          "raw_power_weight_attr": "explore_mc_ensemble_pic_pctr_raw_power_weight"
        },
        {
          "name" : "mc_ensemble_pltr",
          "weight" : 0.2,
          "power_weight_attr" : "explore_mc_ensemble_pic_pltr_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_pltr_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_pltr_raw_power_weight"
        },
        {
          "name" : "mc_ensemble_pwtr",
          "weight" : 0.45,
          "power_weight_attr" : "explore_mc_ensemble_pic_pwtr_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_pwtr_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_pwtr_raw_power_weight"
        },
        {
          "name" : "mc_ensemble_pftr",
          "weight" : 0.05,
          "power_weight_attr" : "explore_mc_ensemble_pic_pftr_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_pftr_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_pftr_raw_power_weight"
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
          "raw_power_weight_attr": "explore_mc_ensemble_pic_plvtr2_raw_power_weight"
        },
        {
          "name" : "mc_ensemble_psvtr",
          "weight" : -0.1,
          "power_weight_attr" : "explore_mc_ensemble_pic_psvtr_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_psvtr_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_psvtr_raw_power_weight"
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
          "raw_power_weight_attr": "explore_mc_ensemble_pic_pcmtr_raw_power_weight"
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
          "raw_power_weight_attr": "explore_mc_ensemble_pic_pwatch_time_raw_power_weight"
        },
        {
          "name" : "mc_ensemble_pcltr",
          "weight" : 0.0,
          "power_weight_attr" : "explore_mc_ensemble_pic_pcltr_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_pcltr_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_pcltr_raw_power_weight"
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
          "name": "mc_ensemble_pic_oppo_cost_score",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_mc_ensemble_pic_oppo_cost_score",
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
          "name": "pic_ensemble_enter_comment",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_ensemble_pecmtr_weight"
        },
        {
          "name": "pic_ensemble_comment_effctive_stop",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_ensemble_pcmeftr_weight"
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
          "raw_power_weight_attr": "explore_mc_ensemble_pic_ensemble_pic_wtd_raw_power_weight"
        },
        {
          "name": "mc_ensemble_pic_lvtr",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_ensemble_pic_lvtr_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_ensemble_pic_lvtr_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_ensemble_pic_lvtr_raw_power_weight"
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
        },
        {
          "name": "pic_cascade_interact_fusion_score",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_interact_fusion_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_interact_fusion_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_interact_fusion_raw_power_weight",
        },
        {
          "name": "pic_cascade_watch_time_fusion_score",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_time_fusion_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_time_fusion_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_time_fusion_raw_power_weight",
        },  
        {
          "name": "pic_variety_score",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_pic_s1_pic_variety_score_power_weight",
          "raw_weight_attr": "explore_mc_pic_s1_pic_variety_score_raw_weight",
          "raw_power_weight_attr": "explore_mc_pic_s1_pic_variety_score_raw_power_weight",
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
          "name": "pic_diversity_mgs_score",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_diversity_mgs_score_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_diversity_mgs_score_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_diversity_mgs_score_raw_power_weight",
        },
        {
          "name": "pic_mc_min_act_rank_score",
          "weight": 0.0,
          "power_weight_attr": "explore_mc_ensemble_pic_min_act_rank_score_power_weight",
          "raw_weight_attr": "explore_mc_ensemble_pic_min_act_rank_score_raw_weight",
          "raw_power_weight_attr": "explore_mc_ensemble_pic_min_act_rank_score_raw_power_weight",
        }
      ]

  def _explore_pic_s1_cluster_sort_v1(self, flag_attr, weight_attr):
    self.flow \
      .explore_life_pic_calc_cluster(
        user_info_ptr_attr = "user_info_ptr",
        save_cluster_id_to_attr = "cascade_cluster_id",
        perf_checkpoint = "pic_calc_cluster",
        enable_multi_hit_independent_bucket = "{{mc_s1_enable_multi_hit_independent_bucket}}",
        clusters = [
          {
            "name": "high_value_pic",
            "cluster_type_id": 70000,
            "enable": "{{explore_pic_s1_cluster_sort_high_value_pic_enable}}",
            "high_value_pic_flag_attr": "high_value_pic_flag",
            "priority_num": '{{mc_s1_explore_pic_high_value_pic_priority_num}}'
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
            "name": "short_term",
            "cluster_type_id": 20000,
            "enable": "{{explore_pic_s1_cluster_sort_short_term_enable}}",
            "tag_score_min": 2,
            "limit_num": 10,
            "time_range_sec": 86400,
            "video_play_weight": 1,
            "video_play_ev_ms_thd": 3000,
            "hetu_level": 1,
            "hetu_level_attr": "hetu_tag_level_info__hetu_level_one",
            "priority_num": '{{mc_s1_explore_pic_short_term_priority_num}}',
            "only_pic": "pic_cascading_s1_short_term_only_pic"
          },
          {
            "name": "long_term",
            "cluster_type_id": 30000,
            "enable": "{{explore_pic_s1_cluster_sort_long_term_enable}}",
            "enable_user_longterm_hetu_distr": "explore_pic_mc_enable_user_longterm_hetu_distr",
            "user_longterm_hetu_distr_attr": "user_pic_interest_hetu_distr",
            "enable_longterm_default_hetu": "explore_pic_mc_enable_longterm_default_hetu",
            "tag_score_min": "pic_cascading_s1_tag_score_min",
            "limit_num": 5,
            "colossus_hetu_l1_tags_attr": "pic_hetu_l1_cnt2",
            "hetu_level_one_attr": "hetu_tag_level_info__hetu_level_one",
            "priority_num": '{{mc_s1_explore_pic_long_term_priority_num}}'
          },
          {
            "name": "long_caption",
            "cluster_type_id": 50000,
            "enable": "{{explore_pic_s1_cluster_sort_long_caption_enable}}",
            "caption_length_attr_name": "caption_length",
            "is_xhs_type_photo_attr_name": "is_xhs_type_photo",
            "caption_length_threshold": 50,
            "priority_num": '{{mc_s1_explore_pic_long_caption_priority_num}}'
          },
          {
            "name": "follow_author",
            "cluster_type_id": 50000,
            "enable": "{{explore_pic_s1_cluster_sort_follow_author_enable}}",
            "is_follow_author_attr_name": "is_follow_author",
            "priority_num": '{{mc_s1_explore_pic_follow_author_priority_num}}'
          },
          {
            "name": "pic_default",
            "cluster_type_id": 10000,
            "enable": "{{explore_pic_s1_cluster_sort_pic_default_enable}}",
            "upload_type_attr": "upload_type",
            "picture_type_attr": "picture_type",
            "picture_count_attr": "photo_picture_count",
            "priority_num": '{{mc_s1_explore_pic_pic_default_priority_num}}'
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
    .if_("enable_fr_refactor_pic_mc_same_author == 1") \
      .deduplicate(
        on_item_attr = "author__id",
        target_item = {
          flag_attr : 1
        }
      ) \
    .end_() \

    # 动态放弃channel槽位, 放在 _caculate_score 最后
    self.flow.if_("skip_cascade_s2_pic_channel_dynamic_shrink == 0") \
      .sort(
        score_from_attr = self._score_attr,
        target_item={ flag_attr: 1 }
      ) \
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
          "enable_cascade_s2_quota_limit",
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
    .if_("enable_pic_mc_diversity_control == 1") \
      .explore_pic_diversity_control_enricher(
        enable_interest_control = "{{enable_pic_mc_interest_control}}",
        enable_hetu_control = "{{enable_pic_mc_hetu_control}}",
        enable_actual_hetu_control = "{{enable_pic_mc_actual_hetu_adjust}}",
        keep_size = "pic_final_quota",
        enable_quota_complete = "{{pic_mc_diversity_quota_complete}}",
        quota_complete_adjust_coeff = "{{pic_mc_diversity_quota_complete_adjust}}",
        final_quota_adjust = "{{pic_mc_diversity_final_quota_adjust}}",
        user_hetu_distribution_attr = "colossus_hetu_distribution_hetu_stat",
        user_actual_distribution_attr = "colossus_actual_reward_hetu_stat",
        hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
        hetu_level_two_attr = "hetu_tag_level_info__hetu_level_two",
        hetu_level_five_attr = "hetu_tag_level_info__hetu_level_five",
        interest_control_start = "{{pic_mc_interest_control_start}}",
        hetu_control_start = "{{pic_mc_hetu_control_start}}",
        hetu1_quota_coeff = "{{pic_mc_hetu1_quota_coeff}}",
        hetu2_quota_coeff = "{{pic_mc_hetu2_quota_coeff}}",
        hetu5_quota_coeff = "{{pic_mc_hetu5_quota_coeff}}",
        hetu_adjust_coef = "{{pic_mc_hetu_adjust_coef}}",
        hetu_adjust_min_value = "{{pic_mc_hetu_adjust_min_value}}",
        hetu_adjust_max_value = "{{pic_mc_hetu_adjust_max_value}}",
        es_score_attr = self._score_attr,
        target_item={ flag_attr: 1 }
      ) \
    .end_() \
    .log_debug_info(
      item_attrs = [
        'pic_hot_action', 'pic_hot_click', 'pic_hot_collect', 'pic_hot_finish_view', 'pic_hot_long_view', 'pic_hot_pos_wtd', 'pic_hot_scroll', 'explore_rank_xtr_list', 'pic_hot_comment_effctive_stop', 'pic_hot_enter_comment'
      ],
      common_attrs = [
        'cascade_prerank_pic_channel_keep_min', 'enable_pic_action_once_score_prerank', 'enable_prerank_caption_pic_queue', 'enable_prerank_key_target_hetu_pic_queue', 'enable_revisited_retrieval_item', 
        'explore_pic_prerank_ensemble_action_score_use_mp', 'explore_pic_prerank_ensemble_action_score_weight', 'explore_pic_prerank_ensemble_collect_score_use_mp', 'explore_pic_prerank_ensemble_collect_score_weight', 
        'explore_pic_prerank_ensemble_key_target_hetu_score_use_mp', 'explore_pic_prerank_ensemble_key_target_hetu_score_weight', 'explore_pic_prerank_ensemble_long_caption_score_use_mp', 
        'explore_pic_prerank_ensemble_long_caption_score_weight', 'explore_pic_prerank_ensemble_long_caption_weight', 'explore_pic_prerank_ensemble_ori_score_use_mp', 'explore_pic_prerank_ensemble_ori_score_weight', 
        'explore_pic_prerank_ensemble_pctr_score_use_mp', 'explore_pic_prerank_ensemble_pctr_score_weight', 'explore_pic_prerank_ensemble_sort_type', 'explore_pic_prerank_score_use_power_calc', 
        'explore_pic_prerank_use_ensembe_score', 'pic_xtr_quantile_rank__cascade_prerank__base_coef', 'pic_xtr_quantile_rank__cascade_prerank__enable', 'pic_xtr_quantile_rank__cascade_prerank__weights', 
        'prerank_caption_boost_len_max', 'prerank_caption_boost_len_thresh', 'prerank_caption_pic_queue_coef', 'prerank_key_target_hetu_pic_queue_coef', 'prerank_revisited_item_boost_coef', 
        'skip_cascade_prerank_pic_channel_dynamic_shrink', 'cascade_revisited_item_boost_coef', 'cascading_s1_pic_hetu_distribution_colossus_total_count_threshold', 'cascading_s1_pic_hetu_distribution_global_fuse_corr', 
        'cascading_s1_pic_hetu_distribution_hetu_coef_alpha', 'cascading_s1_pic_hetu_distribution_hetu_coef_beta', 'cascading_s1_pic_hetu_distribution_hetu_discount_threshold', 
        'cascading_s1_pic_hetu_distribution_hetu_encourage_threshold', 'cascading_s1_pic_hetu_distribution_max_count', 'enable_cascading_s1_pic_sort_hetu_distribution_adjust', 'enable_pic_action_once_cascade_s1_v2', 
        'enable_pic_cascade_pure_interact_fusion', 'explore_cascade_pic_ctr_pow_weight', 'explore_cascade_pic_xtr_calc_with_ctr', 'explore_pic_enable_mc_oppo_cost_queue', 'explore_pic_mc_adjust_queue_weight_k_cltr', 
        'explore_pic_mc_adjust_queue_weight_k_cmtr', 'explore_pic_mc_adjust_queue_weight_k_ftr', 'explore_pic_mc_adjust_queue_weight_k_ltr', 'explore_pic_mc_adjust_queue_weight_k_wtr', 'explore_pic_mc_adjust_queue_weight_max', 
        'explore_pic_mc_adjust_queue_weight_mode', 'explore_pic_mc_adjust_queue_weight_p', 'explore_pic_mc_oppo_cost_q_pwatch_time_power', 'explore_pic_mc_oppo_cost_q_pwatch_time_power2', 'explore_pic_mc_oppo_cost_q_weights', 
        'pic_cascade_fusion_score_cltr_weight', 'pic_cascade_fusion_score_cmtr_weight', 'pic_cascade_fusion_score_ctr_weight', 'pic_cascade_fusion_score_epstr_weight', 'pic_cascade_fusion_score_evtr_weight', 
        'pic_cascade_fusion_score_fr_score1_ctr_weight', 'pic_cascade_fusion_score_fr_score1_weight', 'pic_cascade_fusion_score_ftr_weight', 'pic_cascade_fusion_score_htr_weight', 'pic_cascade_fusion_score_interact_ctr_weight', 
        'pic_cascade_fusion_score_ltr_weight', 'pic_cascade_fusion_score_lvtr2_ctr_weight', 'pic_cascade_fusion_score_lvtr2_weight', 'pic_cascade_fusion_score_lvtr_ctr_weight', 'pic_cascade_fusion_score_lvtr_weight', 
        'pic_cascade_fusion_score_wtr_weight', 'skip_explore_cascade_pic_xtr_calc', 'cascade_enable_comment_boost', 'cascade_enable_comment_boost__god__coeff_max_w', 'cascade_enable_comment_boost__god__coeff_min_w', 
        'cascade_enable_comment_boost__god__coeff_p', 'cascade_enable_comment_boost__god__coeff_w', 'cascade_enable_comment_boost__hot__coeff_max_w', 'cascade_enable_comment_boost__hot__coeff_min_w', 
        'cascade_enable_comment_boost__hot__coeff_p', 'cascade_enable_comment_boost__hot__coeff_w', 'cascade_final_revisited_item_boost_coef', 'cascade_s2_pic_enable_follow_author_pic_boost', 
        'cascading_final_pic_hetu_distribution_colossus_total_count_threshold', 'cascading_final_pic_hetu_distribution_global_fuse_corr', 'cascading_final_pic_hetu_distribution_hetu_coef_alpha', 
        'cascading_final_pic_hetu_distribution_hetu_coef_beta', 'cascading_final_pic_hetu_distribution_hetu_discount_threshold', 'cascading_final_pic_hetu_distribution_hetu_encourage_threshold', 
        'cascading_final_pic_hetu_distribution_max_count', 'cascading_s2_pic_caption_boost_coef', 'cascading_s2_pic_follow_author_pic_boost_coef', 'enable_cascading_final_pic_sort_hetu_distribution_adjust', 
        'enable_cascading_s2_pic_caption_boost', 'explore_pic_cascade_final_cal_score_type', 'explore_pic_cascade_final_value_seq_fusion_status', 'pic_explore_enable_mc_s2_xhs_target_qualified_photo_boost', 
        'pic_explore_mc_ensemble_pcestr_raw_power_weight', 'pic_explore_mc_ensemble_pcestr_raw_weight', 'pic_explore_mc_ensemble_pcltr_raw_power_weight', 'pic_explore_mc_ensemble_pcltr_raw_weight', 
        'pic_explore_mc_ensemble_pcmtr_raw_power_weight', 'pic_explore_mc_ensemble_pcmtr_raw_weight', 'pic_explore_mc_ensemble_pctr_raw_power_weight', 'pic_explore_mc_ensemble_pctr_raw_weight', 
        'pic_explore_mc_ensemble_pepstr_raw_power_weight', 'pic_explore_mc_ensemble_pepstr_raw_weight', 'pic_explore_mc_ensemble_pftr_raw_power_weight', 'pic_explore_mc_ensemble_pftr_raw_weight', 
        'pic_explore_mc_ensemble_pltr_raw_power_weight', 'pic_explore_mc_ensemble_pltr_raw_weight', 'pic_explore_mc_ensemble_plvtr2_raw_power_weight', 'pic_explore_mc_ensemble_plvtr2_raw_weight', 
        'pic_explore_mc_ensemble_plvtr_raw_power_weight', 'pic_explore_mc_ensemble_plvtr_raw_weight', 'pic_explore_mc_ensemble_pptr_raw_power_weight', 'pic_explore_mc_ensemble_pptr_raw_weight', 
        'pic_explore_mc_ensemble_ppwatch_time_raw_power_weight', 'pic_explore_mc_ensemble_ppwatch_time_raw_weight', 'pic_explore_mc_ensemble_pwtr_raw_power_weight', 'pic_explore_mc_ensemble_pwtr_raw_weight', 
        'pic_explore_mc_ensemble_s2_cascade_score_power_weight', 'pic_explore_mc_ensemble_s2_pcestr_power_weight', 'pic_explore_mc_ensemble_s2_pcltr_power_weight', 'pic_explore_mc_ensemble_s2_pcmtr_power_weight', 
        'pic_explore_mc_ensemble_s2_pctr_power_weight', 'pic_explore_mc_ensemble_s2_pepstr_power_weight', 'pic_explore_mc_ensemble_s2_pftr_power_weight', 'pic_explore_mc_ensemble_s2_phtr_power_weight', 
        'pic_explore_mc_ensemble_s2_pic_stage1_score_power_weight', 'pic_explore_mc_ensemble_s2_pltr_power_weight', 'pic_explore_mc_ensemble_s2_plvtr2_power_weight', 'pic_explore_mc_ensemble_s2_plvtr_power_weight', 
        'pic_explore_mc_ensemble_s2_pptime_power_weight', 'pic_explore_mc_ensemble_s2_produce_cascade_mtcotr_score_power_weight', 'pic_explore_mc_ensemble_s2_produce_cascade_mtcotr_score_weight', 
        'pic_explore_mc_ensemble_s2_produce_cascade_mtctr_score_power_weight', 'pic_explore_mc_ensemble_s2_produce_cascade_mtctr_score_weight', 'pic_explore_mc_ensemble_s2_produce_cascade_mtjtr_score_power_weight', 
        'pic_explore_mc_ensemble_s2_produce_cascade_mtjtr_score_weight', 'pic_explore_mc_ensemble_s2_produce_cascade_sjctr_score_power_weight', 'pic_explore_mc_ensemble_s2_produce_cascade_sjctr_score_weight', 
        'pic_explore_mc_ensemble_s2_produce_cascade_twhtr_score_power_weight', 'pic_explore_mc_ensemble_s2_produce_cascade_twhtr_score_weight', 'pic_explore_mc_ensemble_s2_ptr_power_weight', 
        'pic_explore_mc_ensemble_s2_pwatch_time_power_weight', 'pic_explore_mc_ensemble_s2_pwtr_power_weight', 'pic_explore_mc_s2_xhs_target_qualified_photo_boost_coeff', 'pic_xtr_quantile_rank__cascade_stage2__base_coef', 
        'pic_xtr_quantile_rank__cascade_stage2__enable', 'pic_xtr_quantile_rank__cascade_stage2__weights'
      ]
    )