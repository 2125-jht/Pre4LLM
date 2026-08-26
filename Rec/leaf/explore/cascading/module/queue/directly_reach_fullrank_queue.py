from cascading.module.queue.cascading_channel_sort_queue import ChannelSortQueuePartitioner
from cascading.module.queue.cascading_channel_sort_queue import ChannelSortQueueScorer
from cascading.module.queue.cascade_queues import cluster_variant_sort_queue
from cascading.module.queue.cascade_prerank_queues import prerank_ensemble_sort_queues
from cascading.module.queue.cascade_final_queues import final_channel_sort_queues,cascades2_value_and_rank_score_queues

class DirectlyReachFullrankQueueParitioner(ChannelSortQueuePartitioner):
  def __init__(self, name, flow, config):
    super().__init__(name, flow, config)

  def _calculate_flag(self):
    """
    默认队列，什么也不干，框架最后赋值
    """
    self.flow \
      .copy_attr(
        attrs=[{
          "from_item": "is_directly_reach_fullrank",
          "to_item": self._flag_attr,
        }]
      )


class DirectlyReachFullrankQueuePrerankScorer(ChannelSortQueueScorer):
  def __init__(self, name, flow, config):
    super().__init__(name, flow, config)

  def _caculate_score(self, flag_attr, weight_attr, left_count_attr):
    self._calc_prerank_score(flag_attr, weight_attr)

  def _calc_prerank_score(self, flag_attr, weight_attr):
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
      target_item={flag_attr: 1}
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
      ],
      export_item_attr = [
        {"name": "cascade_prerank_score", "as": self._score_attr}
      ],
      function_name = "CalPreRankScore",
      class_name = "ExploreLightFunctionSetV2",
      target_item={flag_attr: 1}
    ) \
    .copy_attr(
      attrs=[{
        "from_item": self._score_attr,
        "to_item": "cascade_prerank_score"
      }],
      target_item={flag_attr: 1}
    )  # copy_attr 放在 prerank 算分最后

class DirectlyReachFullrankQueueCascadingScorer(ChannelSortQueueScorer):
  def __init__(self, name, flow, config):
    super().__init__(name, flow, config)

  def _caculate_score(self, flag_attr, weight_attr, left_count_attr):
    self._calc_cascading_score(flag_attr, weight_attr, left_count_attr)

  def _calc_cascading_score(self, flag_attr, weight_attr, left_count_attr):
    self.flow \
    .explore_cluster_by_custom_rule(
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
      target_item={flag_attr: 1},
      perf_checkpoint = "cascade",
      enable_shortterm_interest_cluster_opt = "{{enable_shortterm_interest_cluster_opt}}",
      shortterm_hetu_attr = "interest_explore_shortterm_hetu"
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
    )

class DirectlyReachFullrankQueueFinalScorer(ChannelSortQueueScorer):
  def __init__(self, name, flow, config):
    super().__init__(name, flow, config)

  def _cascading_s2_truncate_by_interest_directly_reach_fullrank(self, flag_attr, left_count_attr):
    self.flow \
    .if_("enable_mc_s2_select_photo_by_interest_directly_reach_fullrank == 1") \
      .mc_s2_select_photo_by_interest_directly_reach_fullrank(self._score_attr, flag_attr, left_count_attr) \
    .end_()
    return self

  def _caculate_score(self, flag_attr, weight_attr, left_count_attr):
    self._calc_final_score(flag_attr, weight_attr, left_count_attr)

  def _calc_final_score(self, flag_attr, weight_attr, left_count_attr):
      self.flow \
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
        target_item = {flag_attr : 1}
      )
      self._cascading_s2_truncate_by_interest_directly_reach_fullrank(flag_attr, left_count_attr)
