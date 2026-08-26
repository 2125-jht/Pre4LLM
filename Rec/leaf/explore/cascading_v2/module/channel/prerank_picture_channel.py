from cascading_v2.module.channel.base_channel import BaseChannelPartitioner
from cascading_v2.module.channel.base_channel import BaseChannelScorer

class PrerankPictureChannelParitioner(BaseChannelPartitioner):
  def __init__(self, name, flow, config):
    super().__init__(name, flow, config)

  def _calculate_flag(self):
    self.flow \
      .copy_attr(
        attrs=[{
          "from_item": "is_picture",
          "to_item": self._flag_attr,
        }],
      )

class PrerankPictureChannelScorer(BaseChannelScorer):
  def __init__(self, name, flow, config):
    super().__init__(name, flow, config)

  def _caculate_score(self, flag_attr, weight_attr, left_count_attr):
    self._calc_prerank_score(flag_attr, weight_attr)

  def _calc_prerank_score(self, flag_attr, weight_attr):
    self.flow \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name":"pic_cascade_prerank_calc_type", "as":"cascade_prerank_calc_type"},
          {"name":"pic_cascade_prerank_pctr_weight", "as":"cascade_prerank_pctr_weight"},
          {"name":"pic_cascade_prerank_pltr_weight", "as":"cascade_prerank_pltr_weight"},
          {"name":"pic_cascade_emp_watchtime_score_weight", "as":"cascade_emp_watchtime_score_weight"},
        ],
        import_item_attr = [
          "cascade_prerank_pctr",
          "cascade_prerank_pltr",
          "cascade_emp_watchtime_score",
        ],
        export_item_attr = [
          {"name": "cascade_prerank_score", "as": self._score_attr}
        ],
        function_name = "CalPreRankScore",
        class_name = "ExploreLightFunctionSetV2",
        target_item = { flag_attr: 1 }
      ) \
      .if_("enable_prerank_key_target_hetu_pic_boost == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "prerank_key_target_hetu_pic_boost_coef", "as": "boost_discount_coeff"}
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
            flag_attr: 1,
            "is_key_target_hetu_pic": 1
          }
        ) \
      .end_() \
      .if_("enable_explore_prerank_pic_search_boost == 1 and (pic_search_boost_user_degree or 0) >= explore_pic_search_boost_user_degree_thresh") \
        .prerank_pic_search_boost(self._score_attr, flag_attr, "is_pic_search_cluster") \
      .end_() \
      .if_("enable_explore_prerank_pic_recent_search_boost == 1 and (pic_search_boost_user_degree or 0) >= explore_pic_search_boost_user_degree_thresh") \
        .prerank_pic_recent_search_boost(self._score_attr, flag_attr, "is_pic_recent_search_cluster") \
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
      .if_("enable_explore_prerank_pic_double_valid_interest_cluster_boost == 1 and (uDoubleOutsideValidPicClusterCnt7dKV or 0) < explore_pic_double_valid_interest_cluster_boost_interest_thresh") \
        .prerank_pic_double_valid_interest_cluster_boost(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_explore_prerank_pic_recent_interest_cluster_boost == 1") \
        .prerank_pic_recent_interest_cluster_boost(self._score_attr, flag_attr) \
      .end_()

    # 动态放弃channel槽位, 放在 _caculate_score 最后
    self.flow \
      .if_("skip_cascade_prerank_pic_channel_dynamic_shrink == 0") \
        .sort(
          score_from_attr = self._score_attr,
          target_item = { flag_attr: 1 }
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
          target_item = { flag_attr: 1 }
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
          target_item = { flag_attr: 1 }
        ) \
      .end_()
