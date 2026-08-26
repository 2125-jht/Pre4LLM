from cascading.module.queue.cascading_channel_sort_queue import ChannelSortQueuePartitioner
from cascading.module.queue.cascading_channel_sort_queue import ChannelSortQueueScorer
from cascading.module.queue.cascade_queues import *

class U2AQueueParitioner(ChannelSortQueuePartitioner):
  def __init__(self, name, flow, config):
    super().__init__(name, flow, config)

  def _calculate_flag(self):
    self.flow \
      .if_("explore_cascade_channel_sort_u2a_bucket == 1") \
        .set_attr_value(
          target_reason = [10049],
          item_attrs=[{
            "name": self._flag_attr,
            "type": "int",
            "value": 1
          }]
        ) \
      .end_()

class U2AQueuePrerankScorer(ChannelSortQueueScorer):
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
      target_item={ flag_attr: 1 }
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        "cascade_prerank_pctr_weight",
        "cascade_prerank_pltr_weight",
        "cascade_emp_watchtime_score_weight",
        "cascade_prerank_calc_type",
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
      ],
      export_item_attr = [
        {"name": "cascade_prerank_score", "as": self._score_attr}
      ],
      function_name = "CalPreRankScore",
      class_name = "ExploreLightFunctionSetV2",
      target_item={ flag_attr: 1 }
    ) \
    .copy_attr(
      attrs=[{
        "from_item": self._score_attr,
        "to_item": "cascade_prerank_score"
      }]
    )  # copy_attr 放在 prerank 算分最后

class U2AQueueCascadingScorer(ChannelSortQueueScorer):
  def __init__(self, name, flow, config):
    super().__init__(name, flow, config)

  def _caculate_score(self, flag_attr, weight_attr, left_count_attr):
    self._calc_cascading_score(flag_attr, weight_attr, left_count_attr)

  def _calc_cascading_score(self, flag_attr, weight_attr, left_count_attr):
    self.flow \
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
      .case_("photo_duration_quantile") \
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
        .explore_cluster_by_custom_rule(
          skip = 0,
          use_extra_page = "{{explore_use_more_page_photos}}",
          user_info_ptr_attr = "user_info_ptr",
          save_cluster_id_to_attr = "cascade_cluster_id",
          enable_user_profile_top_hetu_level_one_cluster = "{{explore_enable_use_hetu_level1_id}}",
          enable_user_profile_top_hetu_level_two_cluster = "{{explore_enable_use_hetu_level2_id}}",
          enable_user_profile_top_hetu_level_three_cluster = "{{explore_enable_use_hetu_level3_id}}",
          enable_use_photo_age_cluster = "{{explore_enable_use_photo_age_cluster}}",
          enable_hetu_extra_cluster = "{{explore_enable_hetu_extra_cluster}}",
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
          white_author_attr = "is_white_author",
          is_follow_author_attr = "is_follow_author",
          enable_white_author_bucket = "{{explore_interactive_enable_white_author_bucket}}",
          enable_rough_default_cluster = "{{enable_rough_default_cluster}}",
          enable_mc_explore_cluster_mix = "{{enable_mc_explore_cluster_mix}}",
          enable_get_default_hetu_level_one_cluster = "{{enable_get_default_hetu_level_one_cluster}}",
          enable_get_default_hetu_level_two_cluster = "{{enable_get_default_hetu_level_two_cluster}}",
          enable_get_default_hetu_level_three_cluster = "{{enable_get_default_hetu_level_three_cluster}}",
          enable_set_bucket_limit_num_by_ratio = "{{enable_set_bucket_limit_num_by_ratio}}",
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
    .explore_cluster_variant_sort_v2_enrich(
      check_point = "cascade",
      use_superscript_rank = True,
      cluster_attr_name = "cascade_cluster_id",
      hetu_level_one_name = "hetu_tag_level_info__hetu_level_one",
      enable_duration_diversity = "{{enable_explore_cluster_duration_diversity}}",
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
      not_audit_cluster_ratio_adjust = "{{enable_mc_not_audit_cluster_ratio_adjust}}",
      not_audit_cluster_ratio = "{{mc_not_audit_cluster_ratio}}",
      interest_explore_cluster_ratio_adjust = "{{enable_mc_interest_explore_cluster_ratio_adjust}}",
      interest_explore_cluster_ratio = "{{mc_interest_explore_cluster_ratio}}",
      queues = cluster_variant_sort_queue,
      target_item = { # 这个 processor 会对视频做逻辑截断，这里只对打上标签的 item 进行处理
        flag_attr : 1
      }
    ) \

class U2AQueueFinalScorer(ChannelSortQueueScorer):
  def __init__(self, name, flow, config):
    super().__init__(name, flow, config)

  def _caculate_score(self, flag_attr, weight_attr, left_count_attr):
    self._calc_final_score(flag_attr, weight_attr, left_count_attr)

  def _calc_final_score(self, flag_attr, weight_attr, left_count_attr):
      self.flow \
        .explore_calc_ensemble_score(
          use_superscript_rank = True,
          user_power_calc_v2 = "{{explore_mc_ensemble_s2_user_power_calc_v2}}",
          value_seq_fusion_status = "{{explore_mc_ensemble_s2_value_seq_fusion_status}}",
          user_info_ptr_attr = "user_info_ptr",
          action_day = "{{mc_variant_weight_action_day_num_s2}}",
          enable_dynamic_weight_by_user_degree = "{{mc_enable_user_differently_by_degree_s2}}",
          rank_smooth = "{{explore_mc2_rank_smooth}}",
          rank_power_weight = "{{explore_mc2_rank_power_weight}}",
          queues = [
            {
              "name" : "cascade_score",
              "weight" : 1.2,
              "power_weight_attr" : "explore_mc_ensemble_s2_cascade_score_power_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
            },
            {
              "name" : "mc_ensemble_pctr",
              "weight" : 0.1,
              "power_weight_attr" : "explore_mc_ensemble_s2_pctr_power_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
              "avg_xtr" : "explore_mc_ensemble_pctr_avg_xtr",
              "min_ratio" : "explore_mc_ensemble_pctr_min_ratio",
              "max_ratio" : "explore_mc_ensemble_pctr_max_ratio",
              "dynamic_weight" : "explore_mc_ensemble_pctr_dynamic_weight",
              "user_xtr" : "realtime_ctr",
              "rank_cliff_attr": "explore_mc_ensemble_pctr_rank_cliff",
              "rank_cliff_ratio_attr": "explore_mc_ensemble_pctr_rank_cliff_ratio",
              "rank_cliff_min_attr": "explore_mc_ensemble_pctr_rank_cliff_min",
              "raw_weight_attr": "explore_mc_ensemble_pctr_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_pctr_raw_power_weight",
            },
            {
              "name" : "mc_ensemble_pltr",
              "weight" : 0.2,
              "power_weight_attr" : "explore_mc_ensemble_s2_pltr_power_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
              "avg_xtr" : "explore_mc_ensemble_pltr_avg_xtr",
              "min_ratio" : "explore_mc_ensemble_pltr_min_ratio",
              "max_ratio" : "explore_mc_ensemble_pltr_max_ratio",
              "dynamic_weight" : "explore_mc_ensemble_pltr_dynamic_weight",
              "user_xtr" : "realtime_ltr",
              "rank_cliff_attr": "explore_mc_ensemble_pltr_rank_cliff",
              "rank_cliff_ratio_attr": "explore_mc_ensemble_pltr_rank_cliff_ratio",
              "rank_cliff_min_attr": "explore_mc_ensemble_pltr_rank_cliff_min",
              "score_threshold": "user_emp_ltr_cas_threshold",
              "raw_weight_attr": "explore_mc_ensemble_pltr_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_pltr_raw_power_weight",
            },
            {
              "name" : "mc_ensemble_pwtr",
              "weight" : 0.45,
              "power_weight_attr" : "explore_mc_ensemble_s2_pwtr_power_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
              "avg_xtr" : "explore_mc_ensemble_pwtr_avg_xtr",
              "min_ratio" : "explore_mc_ensemble_pwtr_min_ratio",
              "max_ratio" : "explore_mc_ensemble_pwtr_max_ratio",
              "dynamic_weight" : "explore_mc_ensemble_pwtr_dynamic_weight",
              "user_xtr" : "realtime_wtr",
              "rank_cliff_attr": "explore_mc_ensemble_pwtr_rank_cliff",
              "rank_cliff_ratio_attr": "explore_mc_ensemble_pwtr_rank_cliff_ratio",
              "rank_cliff_min_attr": "explore_mc_ensemble_pwtr_rank_cliff_min",
              "score_threshold": "user_emp_wtr_cas_threshold",
              "raw_weight_attr": "explore_mc_ensemble_pwtr_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_pwtr_raw_power_weight",
            },
            {
              "name" : "mc_ensemble_pftr",
              "weight" : 0.05,
              "power_weight_attr" : "explore_mc_ensemble_s2_pftr_power_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
              "avg_xtr" : "explore_mc_ensemble_pftr_avg_xtr",
              "min_ratio" : "explore_mc_ensemble_pftr_min_ratio",
              "max_ratio" : "explore_mc_ensemble_pftr_max_ratio",
              "dynamic_weight" : "explore_mc_ensemble_pftr_dynamic_weight",
              "user_xtr" : "realtime_ftr",
              "rank_cliff_attr": "explore_mc_ensemble_pftr_rank_cliff",
              "rank_cliff_ratio_attr": "explore_mc_ensemble_pftr_rank_cliff_ratio",
              "rank_cliff_min_attr": "explore_mc_ensemble_pftr_rank_cliff_min",
              "score_threshold": "user_emp_ftr_cas_threshold",
              "raw_weight_attr": "explore_mc_ensemble_pftr_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_pftr_raw_power_weight",
            },
            {
              "name" : "mc_ensemble_plvtr",
              "weight" : 0.2,
              "power_weight_attr" : "explore_mc_ensemble_s2_plvtr_power_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
              "raw_weight_attr": "explore_mc_ensemble_plvtr_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_plvtr_raw_power_weight",
            },
            {
              "name" : "mc_ensemble_plvtr2",
              "weight" : 0.12,
              "power_weight_attr" : "explore_mc_ensemble_s2_plvtr2_power_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
              "raw_weight_attr": "explore_mc_ensemble_plvtr2_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_plvtr2_raw_power_weight",
            },
            {
              "name" : "mc_ensemble_psvtr",
              "weight" : -0.1,
              "power_weight_attr" : "explore_mc_ensemble_s2_psvtr_power_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
            },
            {
              "name": "mc_ensemble_ptr",
              "weight": 0.05,
              "power_weight_attr": "explore_mc_ensemble_s2_ptr_power_weight",
              "temperature_attr": "explore_mc_ensemble_s2_temperature",
              "rank_cliff_attr": "explore_mc_ensemble_pptr_rank_cliff",
              "rank_cliff_ratio_attr": "explore_mc_ensemble_pptr_rank_cliff_ratio",
              "rank_cliff_min_attr": "explore_mc_ensemble_pptr_rank_cliff_min",
              "raw_weight_attr": "explore_mc_ensemble_pptr_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_pptr_raw_power_weight",
            },
            {
              "name" : "mc_ensemble_pepstr",
              "weight" : 0.3,
              "power_weight_attr" : "explore_mc_ensemble_s2_pepstr_power_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
              "raw_weight_attr": "explore_mc_ensemble_pepstr_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_pepstr_raw_power_weight",
            },
            {
              "name" : "mc_ensemble_pcestr",
              "weight" : 0.18,
              "power_weight_attr" : "explore_mc_ensemble_s2_pcestr_power_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
              "raw_weight_attr": "explore_mc_ensemble_pcestr_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_pcestr_raw_power_weight",
            },
            {
              "name" : "mc_ensemble_pcmtr",
              "weight" : 0.18,
              "power_weight_attr" : "explore_mc_ensemble_s2_pcmtr_power_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
              "rank_cliff_attr": "explore_mc_ensemble_pcmtr_rank_cliff",
              "rank_cliff_ratio_attr": "explore_mc_ensemble_pcmtr_rank_cliff_ratio",
              "rank_cliff_min_attr": "explore_mc_ensemble_pcmtr_rank_cliff_min",
              "score_threshold": "user_emp_cmtr_cas_threshold",
              "raw_weight_attr": "explore_mc_ensemble_pcmtr_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_pcmtr_raw_power_weight",
            },
            {
              "name" : "mc_ensemble_pptime",
              "weight" : 0.4,
              "power_weight_attr" : "explore_mc_ensemble_s2_pptime_power_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
            },
            {
              "name" : "mc_ensemble_pwatch_time",
              "weight" : 0.45,
              "power_weight_attr" : "explore_mc_ensemble_s2_pwatch_time_power_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
              "raw_weight_attr": "explore_mc_ensemble_ppwatch_time_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_ppwatch_time_raw_power_weight",
            },
            {
              "name": "mc_ensemble_peftr",
              "weight": 0.5,
              "power_weight_attr": "explore_mc_ensemble_s2_eftr_score_power_weight",
              "weight_attr": "mc_eftr_ensemble_sort_weight_s2",
              "temperature_attr": "explore_mc_ensemble_s2_temperature",
              "raw_weight_attr": "explore_mc_ensemble_peftr_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_peftr_raw_power_weight",
            },
            {
              "name": "mc_ensemble_pefctr",
              "weight": 0.5,
              "power_weight_attr": "explore_mc_ensemble_s2_efctr_score_power_weight",
              "weight_attr": "mc_efctr_ensemble_sort_weight_s2",
              "temperature_attr": "explore_mc_ensemble_s2_temperature",
              "raw_weight_attr": "explore_mc_ensemble_pefctr_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_pefctr_raw_power_weight",
            },
            {
              "name" : "mc_ensemble_pcltr",
              "weight" : 0.0,
              "power_weight_attr" : "explore_mc_ensemble_s2_pcltr_power_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
              "variant_weight" : "explore_mc_ensemble_s2_pcltr_variant_weight",
              "avg_xtr" : "avg_cltr",
              "min_ratio" : "explore_mc_ensemble_pcltr_min_ratio",
              "max_ratio" : "explore_mc_ensemble_pcltr_max_ratio",
              "dynamic_weight" : "explore_mc_ensemble_pcltr_dynamic_weight",
              "user_xtr": "realtime_cltr",
              "rank_cliff_attr": "explore_mc_ensemble_pcltr_rank_cliff",
              "rank_cliff_ratio_attr": "explore_mc_ensemble_pcltr_rank_cliff_ratio",
              "rank_cliff_min_attr": "explore_mc_ensemble_pcltr_rank_cliff_min",
              "raw_weight_attr": "explore_mc_ensemble_pcltr_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_pcltr_raw_power_weight",
            },
            {
              "name": "mc_ensemble_opportunity_cost_score",  #粗排机会成本队列
              "weight": 0.8,
              "power_weight_attr": "explore_mc_ensemble_s2_opportunity_cost_power_weight",
              "weight_attr": "explore_mc_ensemble_s2_opportunity_cost_weight",
              "temperature_attr": "explore_mc_ensemble_s2_temperature",
            },
            {
              "name" : "mc_ensemble_pwtd",
              "weight" : 0.0,
              "power_weight_attr" : "explore_mc_ensemble_s2_pwtd_power_weight",
              "weight_attr" : "mc_wtd_ensemble_sort_weight_s2",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
              "rank_cliff_attr": "explore_mc_ensemble_pwtd_rank_cliff",
              "rank_cliff_ratio_attr": "explore_mc_ensemble_pwtd_rank_cliff_ratio",
              "rank_cliff_min_attr": "explore_mc_ensemble_pwtd_rank_cliff_min"
            },
            {
              "name": "mc_ensemble_pwtd_inverse", 
              "weight": 0.0,
              "power_weight_attr": "explore_mc_ensemble_s2_pwtd_inverse_power_weight",
              "weight_attr": "explore_mc_ensemble_s2_pwtd_inverse_weight",
              "temperature_attr": "explore_mc_ensemble_s2_temperature",
              "raw_weight_attr": "explore_mc_ensemble_s2_pwtd_inverse_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_s2_pwtd_inverse_raw_power_weight",
            },
            {
              "name": "mc_ensemble_pfptr",  #粗排播放完成度队列
              "weight": 1.0,
              "power_weight_attr": "explore_mc_ensemble_s2_pfptr_power_weight"    ,
              "weight_attr": "explore_mc_ensemble_s2_pfptr_weight",
              "temperature_attr": "explore_mc_ensemble_s2_temperature",
              "raw_weight_attr": "explore_mc_ensemble_s2_pfptr_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_s2_pfptr_raw_power_weight",
            },
            {
              "name": "mc_ensemble_ordinal_wtd",  
              "weight": 1.0,
              "power_weight_attr": "explore_mc_ensemble_s2_ordinal_wtd_weight",
              "weight_attr": "explore_mc_ensemble_s2_ordinal_wtd_weight",
              "temperature_attr": "explore_mc_ensemble_s2_temperature",
            },
            {
              "name": "mc_ensemble_ordinal_prob",  
              "weight": 1.0,
              "power_weight_attr": "explore_mc_ensemble_s2_ordinal_prob_weight",
              "weight_attr": "explore_mc_ensemble_s2_ordinal_prob_weight",
              "temperature_attr": "explore_mc_ensemble_s2_temperature",
            },
            {
              "name": "mc_ada_xtr_score",  #粗排播放完成度队列
              "weight": 0.0,
              "power_weight_attr": "explore_mc_ensemble_s2_mc_ada_xtr_score_power_weight",
              "weight_attr": "explore_mc_ensemble_s2_mc_ada_xtr_score_power_weight",
              "temperature_attr": "explore_mc_ensemble_s2_temperature",
            },
            {
              "name": "diversity_fr",  #粗排播放完成度队列
              "weight": 0.0,
              "power_weight_attr": "xlife_mc_ensemble_s2_diversity_fr_power_weight",
              "weight_attr": "xlife_mc_ensemble_s2_diversity_fr_power_weight",
              "temperature_attr": "explore_mc_ensemble_s2_temperature",
              "use_new_pow_func": "xlife_diversity_fr_use_new_pow_func",
              "new_pow_func_coeff": "xlife_diversity_fr_new_pow_func_coeff",
              "new_pos_func_bias": "xlife_diversity_fr_new_pos_func_bias"
            },
            {
              "name": "hate_fr",  #粗排topkmgs hate 队列
              "weight": 0.0,
              "power_weight_attr": "explore_mc_ensemble_s2_hate_fr_power_weight",
              "weight_attr": "explore_mc_ensemble_s2_hate_fr_power_weight",
              "temperature_attr": "explore_mc_ensemble_s2_temperature",
              "use_new_pow_func": "explore_hate_fr_use_new_pow_func",
              "new_pow_func_coeff": "explore_hate_fr_new_pow_func_coeff",
              "new_pos_func_bias": "explore_hate_fr_new_pos_func_bias"
            },
            {
              "name": "mc_ensemble_prerank_er",  #粗排 LTR
              "weight": 0.0,
              "power_weight_attr": "explore_mc_ensemble_s2_prerank_er_power_weight",
              "weight_attr": "explore_mc_ensemble_s2_prerank_er_weight",
              "temperature_attr": "explore_mc_ensemble_s2_temperature",
            },
            {
              "name": "cascade_phtr",  #粗排 htr
              "weight": 0.0,
              "reverse_order": True,
              "power_weight_attr": "explore_mc_ensemble_s2_phtr_power_weight",
              "weight_attr": "explore_mc_ensemble_s2_phtr_power_weight",
              "temperature_attr": "explore_mc_ensemble_s2_temperature",
              "rank_cliff_attr": "explore_mc_s2_ensemble_phtr_rank_cliff",
              "rank_cliff_ratio_attr": "explore_mc_s2_ensemble_phtr_rank_cliff_ratio",
              "rank_cliff_min_attr": "explore_mc_s2_ensemble_phtr_rank_cliff_min",
              "rank_height_attr": "explore_mc_s2_ensemble_phtr_rank_height",
            },
            {
              "name": "cascade_fc_s2_pctr",  #粗排播放完成度队列
              "weight": 1.0,
              "power_weight_attr": "explore_mc_ensemble_s2_fc_pctr_power_weight"    ,
              "weight_attr": "explore_mc_ensemble_s2_fc_pctr_weight",
              "temperature_attr": "explore_mc_ensemble_s2_temperature",
              "raw_weight_attr": "explore_mc_ensemble_s2_fc_pctr_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_s2_fc_pctr_raw_power_weight",
            },
            {
              "name": "cascade_fc_s2_plvtr",  #粗排播放完成度队列
              "weight": 1.0,
              "power_weight_attr": "explore_mc_ensemble_s2_fc_plvtr_power_weight"    ,
              "weight_attr": "explore_mc_ensemble_s2_fc_plvtr_weight",
              "temperature_attr": "explore_mc_ensemble_s2_temperature",
              "raw_weight_attr": "explore_mc_ensemble_s2_fc_plvtr_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_s2_fc_plvtr_raw_power_weight",
            },
            {
              "name": "cascade_fc_s2_psvtr",  #粗排播放完成度队列
              "weight": 1.0,
              "power_weight_attr": "explore_mc_ensemble_s2_fc_psvtr_power_weight"    ,
              "weight_attr": "explore_mc_ensemble_s2_fc_psvtr_weight",
              "temperature_attr": "explore_mc_ensemble_s2_temperature",
              "raw_weight_attr": "explore_mc_ensemble_s2_fc_psvtr_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_s2_fc_psvtr_raw_power_weight",
            },
            {
              "name": "cascade_fc_s2_pvtr",  #粗排播放完成度队列
              "weight": 1.0,
              "power_weight_attr": "explore_mc_ensemble_s2_fc_pvtr_power_weight"    ,
              "weight_attr": "explore_mc_ensemble_s2_fc_pvtr_weight",
              "temperature_attr": "explore_mc_ensemble_s2_temperature",
              "raw_weight_attr": "explore_mc_ensemble_s2_fc_pvtr_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_s2_fc_pvtr_raw_power_weight",
            },
            {
              "name": "pdn_rank_score",  #pdn召回rank score队列
              "weight": 0.0,
              "power_weight_attr": "explore_mc_ensemble_s2_pdn_rank_score_power_weight",
              "weight_attr": "explore_mc_ensemble_s2_pdn_rank_score_weight",
              "temperature_attr": "explore_mc_ensemble_s2_temperature",
            },
            {
              "name": "comirec_rank_score",  #comirec召回rank score队列
              "weight": 0.0,
              "power_weight_attr": "explore_mc_ensemble_s2_comirec_rank_score_power_weight",
              "weight_attr": "explore_mc_ensemble_s2_comirec_rank_score_weight",
              "temperature_attr": "explore_mc_ensemble_s2_temperature",
            },
            {
              "name": "colossus_ann_rank_score",  #colossus ann召回rank score队列
              "weight": 0.0,
              "power_weight_attr": "explore_mc_ensemble_s2_colossus_ann_rank_score_power_weight",
              "weight_attr": "explore_mc_ensemble_s2_colossus_ann_rank_score_weight",
              "temperature_attr": "explore_mc_ensemble_s2_temperature",
            },
            {
              "name" : "mc_ensemble_smooth_age_score",
              "weight" : 0.0,
              "power_weight_attr" : "explore_mc_ensemble_s2_smooth_age_score_power_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
            },
            {
              "name" : "mc_ensemble_emp_pop_score",
              "weight" : 0.0,
              "power_weight_attr" : "explore_mc_ensemble_s2_emp_pop_score_power_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
            },
            {
              "name": "mc_htr_cost_score",  #粗排 htr 机会成本
              "weight": 0.0,
              "power_weight_attr": "explore_mc_ensemble_s2_phtr_cost_power_weight",
              "weight_attr": "explore_mc_ensemble_s2_phtr_cost_power_weight",
              "temperature_attr": "explore_mc_ensemble_s2_temperature",
            },
            {
              "name": "mc_interact_fusion_score",
              "weight": 0.0,
              "power_weight_attr" : "explore_mc_ensemble_s2_interact_fusion_score_weight",
              "weight_attr": "explore_mc_ensemble_s2_interact_fusion_score_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
            },
            {
              "name": "mc_watch_time_fusion_score",
              "weight": 0.0,
              "power_weight_attr" : "explore_mc_ensemble_s2_watch_time_fusion_score_weight",
              "weight_attr": "explore_mc_ensemble_s2_watch_time_fusion_score_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
            },
            {
              "name" : "mc_ensemble_psvtr2",
              "reverse_order": True,
              "weight" : 0.0,
              "power_weight_attr" : "explore_mc_ensemble_s2_psvtr2_power_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
            },
            {
              "name": "mc_ctcvr_score",
              "weight": 0.0,
              "power_weight_attr" : "explore_mc_ensemble_s2_ctcvr_score_score_weight",
              "weight_attr": "explore_mc_ensemble_s2_ctcvr_score_score_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
            },
            {
              "name": "mc_ctcvr_gmv_score",
              "weight": 0.0,
              "power_weight_attr" : "explore_mc_ensemble_s2_ctcvr_gmv_score_weight",
              "weight_attr": "explore_mc_ensemble_s2_ctcvr_gmv_score_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
            },
            {
              "name": "mc_elive_ctcvr_score",
              "weight": 0.0,
              "power_weight_attr" : "explore_mc_ensemble_s2_elive_ctcvr_score_score_weight",
              "weight_attr": "explore_mc_ensemble_s2_elive_ctcvr_score_score_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
            },
            {
              "name": "mc_elive_ctcvr_gmv_score",
              "weight": 0.0,
              "power_weight_attr" : "explore_mc_ensemble_s2_elive_ctcvr_gmv_score_weight",
              "weight_attr": "explore_mc_ensemble_s2_elive_ctcvr_gmv_score_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
            },
            {
              "name": "produce_cascade_mtctr",
              "weight": 0.0,
              "power_weight_attr" : "explore_mc_ensemble_s2_produce_cascade_mtctr_score_weight",
              "weight_attr": "explore_mc_ensemble_s2_produce_cascade_mtctr_score_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
              "raw_weight_attr": "explore_mc_ensemble_s2_produce_cascade_mtctr_score_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_s2_produce_cascade_mtctr_score_raw_power_weight",
            },
            {
              "name": "produce_cascade_twhtr",
              "weight": 0.0,
              "power_weight_attr" : "explore_mc_ensemble_s2_produce_cascade_twhtr_score_weight",
              "weight_attr": "explore_mc_ensemble_s2_produce_cascade_twhtr_score_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
              "raw_weight_attr": "explore_mc_ensemble_s2_produce_cascade_twhtr_score_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_s2_produce_cascade_twhtr_score_raw_power_weight",
            },
            {
              "name": "produce_cascade_mtcotr",
              "weight": 0.0,
              "power_weight_attr" : "explore_mc_ensemble_s2_produce_cascade_mtcotr_score_weight",
              "weight_attr": "explore_mc_ensemble_s2_produce_cascade_mtcotr_score_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
              "raw_weight_attr": "explore_mc_ensemble_s2_produce_cascade_mtcotr_score_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_s2_produce_cascade_mtcotr_score_raw_power_weight",
            },
            {
              "name": "produce_cascade_mtjtr",
              "weight": 0.0,
              "power_weight_attr" : "explore_mc_ensemble_s2_produce_cascade_mtjtr_score_weight",
              "weight_attr": "explore_mc_ensemble_s2_produce_cascade_mtjtr_score_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
              "raw_weight_attr": "explore_mc_ensemble_s2_produce_cascade_mtjtr_score_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_s2_produce_cascade_mtjtr_score_raw_power_weight",
            },
            {
              "name": "produce_cascade_kym",
              "weight": 0.0,
              "power_weight_attr" : "explore_mc_ensemble_s2_produce_cascade_kym_score_weight",
              "weight_attr": "explore_mc_ensemble_s2_produce_cascade_kym_score_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
              "raw_weight_attr": "explore_mc_ensemble_s2_produce_cascade_kym_score_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_s2_produce_cascade_kym_score_raw_power_weight",
            },
            {
              "name": "produce_cascade_csti",
              "weight": 0.0,
              "power_weight_attr" : "explore_mc_ensemble_s2_produce_cascade_csti_score_weight",
              "weight_attr": "explore_mc_ensemble_s2_produce_cascade_csti_score_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
              "raw_weight_attr": "explore_mc_ensemble_s2_produce_cascade_csti_score_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_s2_produce_cascade_csti_score_raw_power_weight",
            },
            {
              "name": "produce_cascade_sjctr",
              "weight": 0.0,
              "power_weight_attr" : "explore_mc_ensemble_s2_produce_cascade_sjctr_score_weight",
              "weight_attr": "explore_mc_ensemble_s2_produce_cascade_sjctr_score_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
              "raw_weight_attr": "explore_mc_ensemble_s2_produce_cascade_sjctr_score_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_s2_produce_cascade_sjctr_score_raw_power_weight",
            },
            {
              "name" : "mc_ensemble_prerank_wtd",
              "weight" : 0.0,
              "power_weight_attr" : "explore_mc_ensemble_s2_prerank_wtd_power_weight",
              "weight_attr": "explore_mc_ensemble_s2_prerank_wtd_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
            },
            {
              "name" : "mc_ensemble_prerank_wtd_in_s1",
              "weight" : 0.0,
              "power_weight_attr" : "explore_mc_ensemble_s2_prerank_wtd_in_s1_power_weight",
              "weight_attr": "explore_mc_ensemble_s2_prerank_wtd_in_s1_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
            },
            {
              "name": "cascade_prerank_pctr",
              "weight": 0.0,
              "power_weight_attr" : "explore_mc_ensemble_s2_cascade_prerank_pctr_power_weight",
              "weight_attr": "explore_mc_ensemble_s2_cascade_prerank_pctr_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
              "raw_weight_attr": "explore_mc_ensemble_s2_cascade_prerank_pctr_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_s2_cascade_prerank_pctr_raw_power_weight",
            },
            {
              "name": "cascade_prerank_pltr",
              "weight": 0.0,
              "power_weight_attr" : "explore_mc_ensemble_s2_cascade_prerank_pltr_power_weight",
              "weight_attr": "explore_mc_ensemble_s2_cascade_prerank_pltr_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
              "raw_weight_attr": "explore_mc_ensemble_s2_cascade_prerank_pltr_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_s2_cascade_prerank_pltr_raw_power_weight",
            },
            {
              "name" : "mc_ensemble_psvtr",
              "reverse_order": True,
              "weight" : 0.0,
              "power_weight_attr" : "explore_mc_ensemble_s2_psvtr_v2_power_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
            },
            {
              "name" : "cascade_pwtd",
              "weight" : 0.0,
              "power_weight_attr" : "explore_mc_ensemble_s2_cascade_pwtd_power_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
            },
            {
              "name" : "cascase_life_ctr",
              "weight" : 0.0,
              "power_weight_attr" : "explore_mc_ensemble_s2_lifetab_pctr_power_weight",
              "weight_attr" : "explore_mc_ensemble_s2_lifetab_pctr_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
              "avg_xtr" : "explore_mc_ensemble_lifetab_pctr_avg_xtr",
              "min_ratio" : "explore_mc_ensemble_lifetab_pctr_min_ratio",
              "max_ratio" : "explore_mc_ensemble_lifetab_pctr_max_ratio",
              "dynamic_weight" : "explore_mc_ensemble_lifetab_pctr_dynamic_weight",
              "user_xtr" : "realtime_ctr",
              "rank_cliff_attr": "explore_mc_ensemble_lifetab_pctr_rank_cliff",
              "rank_cliff_ratio_attr": "explore_mc_ensemble_lifetab_pctr_rank_cliff_ratio",
              "rank_cliff_min_attr": "explore_mc_ensemble_lifetab_pctr_rank_cliff_min",
              "raw_weight_attr": "explore_mc_ensemble_lifetab_pctr_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_lifetab_pctr_raw_power_weight",
              "raw_score_normalize_alpha_attr": "explore_mc_ensemble_s2_lifetab_pctr_normalize_alpha"
            },
            {
              "name" : "cascade_pwtd",
              "weight" : 0.0,
              "power_weight_attr" : "explore_mc_ensemble_s2_cascade_pwtd_power_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
            },
            {
              "name" : "cascase_life_ctr",
              "weight" : 0.0,
              "power_weight_attr" : "explore_mc_ensemble_s2_lifetab_pctr_power_weight",
              "weight_attr" : "explore_mc_ensemble_s2_lifetab_pctr_weight",
              "temperature_attr" : "explore_mc_ensemble_s2_temperature",
              "avg_xtr" : "explore_mc_ensemble_lifetab_pctr_avg_xtr",
              "min_ratio" : "explore_mc_ensemble_lifetab_pctr_min_ratio",
              "max_ratio" : "explore_mc_ensemble_lifetab_pctr_max_ratio",
              "dynamic_weight" : "explore_mc_ensemble_lifetab_pctr_dynamic_weight",
              "user_xtr" : "realtime_ctr",
              "rank_cliff_attr": "explore_mc_ensemble_lifetab_pctr_rank_cliff",
              "rank_cliff_ratio_attr": "explore_mc_ensemble_lifetab_pctr_rank_cliff_ratio",
              "rank_cliff_min_attr": "explore_mc_ensemble_lifetab_pctr_rank_cliff_min",
              "raw_weight_attr": "explore_mc_ensemble_lifetab_pctr_raw_weight",
              "raw_power_weight_attr": "explore_mc_ensemble_lifetab_pctr_raw_power_weight",
              "raw_score_normalize_alpha_attr": "explore_mc_ensemble_s2_lifetab_pctr_normalize_alpha"
            },
          ],
          save_score_to_attr = self._score_attr,
          target_item = {
            flag_attr : 1
          }
        )
