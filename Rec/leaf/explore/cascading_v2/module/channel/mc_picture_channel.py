from cascading_v2.module.channel.prerank_picture_channel import PrerankPictureChannelParitioner
from cascading_v2.module.channel.base_channel import BaseChannelScorer

class McPictureChannelParitioner(PrerankPictureChannelParitioner):
  def __init__(self, name, flow, config):
    super().__init__(name, flow, config)

class McPictureChannelScorer(BaseChannelScorer):
  def __init__(self, name, flow, config):
    super().__init__(name, flow, config)

  def _caculate_score(self, flag_attr, weight_attr, left_count_attr):
    self.__calc_mc_es_score(flag_attr)
    self.flow \
      .pack_item_attr(  # 为了对齐重构之前的有问题的样本（粗排输入 - 原粗排 s1 ，图文 s1 不截断）
        item_source = {
          "reco_results": True,
        },
        mappings = [{
          "from_item_attr": "item_key",
          "to_common_attr": "cascade_output_item_key_list",
          "reset_to_common_attr": False,
        }],
        target_item = {
          flag_attr: 1,
        },
      ) \
      .if_("enable_fr_refactor_pic_mc_same_author == 1") \
        .deduplicate(
          name = "deduplicate_by_author_picture",
          traceback = True,
          on_item_attr = "author__id",
          target_item = {
            flag_attr: 1
          }
        ) \
      .end_() \
      .if_("enable_cascade_s2_pic_quota_v2 == 1") \
        .sort(
          score_from_attr = self._score_attr,
          target_item = { flag_attr: 1 }
        ) \
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
        .split_string( # 打压参数
          input_common_attr = "cascade_s2_quota_prefer_score_weights_str",
          output_common_attr = "cascade_s2_quota_prefer_score_weights",
          delimiters = ",",
          parse_to_double = True
        ) \
        .split_string( # 打压参数
          input_common_attr = "cascade_s2_quota_colossus_score_weights_str",
          output_common_attr = "cascade_s2_quota_colossus_score_weights",
          delimiters = ",",
          parse_to_double = True
        ) \
        .split_string( # 打压参数
          input_common_attr = "cascade_s2_quota_recent_decay_weights_str",
          output_common_attr = "cascade_s2_quota_recent_decay_weights",
          delimiters = ",",
          parse_to_double = True
        ) \
        .split_string( # 打压参数
          input_common_attr = "cascade_s2_quota_pxtr_score_weights_str",
          output_common_attr = "cascade_s2_quota_pxtr_score_weights",
          delimiters = ",",
          parse_to_double = True
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
          ],
          export_item_attr = [
            {"name": "score_attr", "as": self._score_attr},
          ],
          export_common_attr = [
            "pic_final_quota"
          ],
          function_name = "FinalChannelSortPicQueueDynamicShrinkV2",
          class_name = "ExploreLightFunctionSetV2",
          target_item = { flag_attr: 1 }
        ) \
      .end_() \
      .if_("enable_pic_mc_diversity_control == 1") \
        .explore_pic_diversity_control_enricher(
          enable_interest_control = "{{enable_pic_mc_interest_control}}",
          enable_hetu_control = "{{enable_pic_mc_hetu_control}}",
          enable_cluster_control = "{{enable_pic_mc_cluster_control}}",
          enable_actual_hetu_control = "{{enable_pic_mc_actual_hetu_adjust}}",
          keep_size = "pic_final_quota",
          enable_quota_complete = "{{pic_mc_diversity_quota_complete}}",
          quota_complete_adjust_coeff = "{{pic_mc_diversity_quota_complete_adjust}}",
          final_quota_adjust = "{{pic_mc_diversity_final_quota_adjust}}",
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
          # 其他
          es_score_attr = self._score_attr,
          target_item = { flag_attr: 1 }
        ) \
      .end_()

  def __calc_mc_es_score(self, flag_attr):
    self.flow \
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
          target_item = { flag_attr: 1 }
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
            flag_attr: 1
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
            flag_attr: 1
          }
        ) \
      .end_() \
      .if_("enable_explore_pic_mc_real_pctr > 0 and (uDoubleOutsideValidPicClusterCnt7dKV or 0) < explore_pic_mc_real_pctr_interest_thresh") \
        .mc_calc_pic_real_pctr(flag_attr) \
      .end_() \
      .explore_calc_ensemble_score(
        target_item = { flag_attr: 1 },
        use_superscript_rank = False,
        user_power_calc_v2 = 1,
        user_info_ptr_attr = "user_info_ptr",
        queues = self.__get_queue(),
        save_score_to_attr = self._score_attr,
        use_queue_smooth_as_rank_smooth = "{{explore_pic_mc_use_queue_smooth_as_rank_smooth}}",
        value_seq_fusion_status = "{{explore_pic_value_seq_fusion_status}}",
        use_rank_with_absolute_score = "{{explore_pic_mc_use_rank_with_absolute_score}}",
        rank_score_calculate_method = "{{explore_pic_mc_rank_score_calculate_method}}",
        queue_max_raw_score = "{{explore_mc_pic_rerank_queue_max_raw_score}}",
        queue_min_raw_score = "{{explore_mc_pic_rerank_queue_min_raw_score}}",
        enable_normalization_item_score = "{{explore_mc_pic_rerank_enable_normalization_item_score}}",
      ) \
      .if_("cascade_enable_follow_author_pic_mc_boost == 1") \
        .mc_boost_pic_es_by_follow_author(self._score_attr, flag_attr) \
      .end_if_() \
      .if_("enable_cascade_channel_caption_boost == 1") \
         .mc_boost_pic_es_by_caption(self._score_attr, flag_attr) \
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
        .mc_pic_search_boost(self._score_attr, flag_attr, "is_pic_search_cluster") \
      .end_() \
      .if_("enable_explore_mc_pic_recent_search_boost == 1 and (pic_search_boost_user_degree or 0) >= explore_pic_search_boost_user_degree_thresh") \
        .mc_pic_recent_search_boost(self._score_attr, flag_attr, "is_pic_recent_search_cluster") \
      .end_() \
      .if_("enable_explore_mc_pic_double_valid_interest_cluster_boost == 1 and (uDoubleOutsideValidPicClusterCnt7dKV or 0) < explore_pic_double_valid_interest_cluster_boost_interest_thresh") \
        .mc_pic_double_valid_interest_cluster_boost(self._score_attr, flag_attr) \
      .end_() \
      .if_("enable_explore_mc_pic_recent_interest_cluster_boost == 1") \
        .mc_pic_recent_interest_cluster_boost(self._score_attr, flag_attr) \
      .end_()

  def __get_queue(self):
    return [
      {
        "name" : "mc_ensemble_pctr",
        "weight" : 0.1,
        "power_weight_attr" : "explore_mc_ensemble_pic_pctr_power_weight",
        "raw_weight_attr": "explore_mc_ensemble_pic_pctr_raw_weight",
        "raw_power_weight_attr": "explore_mc_ensemble_pic_pctr_raw_power_weight",
        "raw_bias_attr": 'explore_mc_pic_rerank_mc_ensemble_pctr_raw_bias',
        "smooth_attr": "explore_mc_pic_rerank_mc_ensemble_pctr_rank_smooth",
      },
      {
        "name" : "mc_ensemble_pltr",
        "weight" : 0.2,
        "power_weight_attr" : "explore_mc_ensemble_pic_pltr_power_weight",
        "raw_weight_attr": "explore_mc_ensemble_pic_pltr_raw_weight",
        "raw_power_weight_attr": "explore_mc_ensemble_pic_pltr_raw_power_weight",
        "raw_bias_attr": 'explore_mc_pic_rerank_mc_ensemble_pltr_raw_bias',
        "smooth_attr": "explore_mc_pic_rerank_mc_ensemble_pltr_rank_smooth",
      },
      {
        "name" : "mc_ensemble_pwtr",
        "weight" : 0.45,
        "power_weight_attr" : "explore_mc_ensemble_pic_pwtr_power_weight",
        "raw_weight_attr": "explore_mc_ensemble_pic_pwtr_raw_weight",
        "raw_power_weight_attr": "explore_mc_ensemble_pic_pwtr_raw_power_weight",
        "raw_bias_attr": 'explore_mc_pic_rerank_mc_ensemble_pwtr_raw_bias',
        "smooth_attr": "explore_mc_pic_rerank_mc_ensemble_pwtr_rank_smooth",
      },
      {
        "name" : "mc_ensemble_pftr",
        "weight" : 0.05,
        "power_weight_attr" : "explore_mc_ensemble_pic_pftr_power_weight",
        "raw_weight_attr": "explore_mc_ensemble_pic_pftr_raw_weight",
        "raw_power_weight_attr": "explore_mc_ensemble_pic_pftr_raw_power_weight",
        "raw_bias_attr": 'explore_mc_pic_rerank_mc_ensemble_pftr_raw_bias',
        "smooth_attr": "explore_mc_pic_rerank_mc_ensemble_pftr_rank_smooth",
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
      },
      {
        "name" : "mc_ensemble_pcltr",
        "weight" : 0.0,
        "power_weight_attr" : "explore_mc_ensemble_pic_pcltr_power_weight",
        "raw_weight_attr": "explore_mc_ensemble_pic_pcltr_raw_weight",
        "raw_power_weight_attr": "explore_mc_ensemble_pic_pcltr_raw_power_weight",
        "raw_bias_attr": 'explore_mc_pic_rerank_mc_ensemble_pcltr_raw_bias',
        "smooth_attr": "explore_mc_pic_rerank_mc_ensemble_pcltr_rank_smooth",
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
        "name": "mc_ensemble_pic_wtd",
        "weight": 0.0,
        "power_weight_attr": "explore_mc_ensemble_pic_ensemble_pic_wtd_weight",
        "raw_weight_attr": "explore_mc_ensemble_pic_ensemble_pic_wtd_raw_weight",
        "raw_power_weight_attr": "explore_mc_ensemble_pic_ensemble_pic_wtd_raw_power_weight",
        "raw_bias_attr": 'explore_mc_pic_rerank_mc_ensemble_pic_wtd_raw_bias',
        "smooth_attr": "explore_mc_pic_rerank_mc_ensemble_pic_wtd_rank_smooth",
      },
      {
        "name": "mc_ensemble_pic_lvtr",
        "weight": 0.0,
        "power_weight_attr": "explore_mc_ensemble_pic_ensemble_pic_lvtr_weight",
        "raw_weight_attr": "explore_mc_ensemble_pic_ensemble_pic_lvtr_raw_weight",
        "raw_power_weight_attr": "explore_mc_ensemble_pic_ensemble_pic_lvtr_raw_power_weight",
        "raw_bias_attr": 'explore_mc_pic_rerank_mc_ensemble_pic_lvtr_raw_bias',
        "smooth_attr": "explore_mc_pic_rerank_mc_ensemble_pic_lvtr_rank_smooth",
      },
      {
        "name": "mc_ensemble_pic_cpr",
        "weight": 0.0,
        "power_weight_attr": "explore_mc_ensemble_pic_ensemble_pic_cpr_weight",
        "raw_weight_attr": "explore_mc_ensemble_pic_ensemble_pic_cpr_raw_weight",
        "raw_power_weight_attr": "explore_mc_ensemble_pic_ensemble_pic_cpr_raw_power_weight"
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
        "name": "pic_diversity_mgs_score",
        "weight": 0.0,
        "power_weight_attr": "explore_mc_ensemble_pic_diversity_mgs_score_power_weight",
        "raw_weight_attr": "explore_mc_ensemble_pic_diversity_mgs_score_raw_weight",
        "raw_power_weight_attr": "explore_mc_ensemble_pic_diversity_mgs_score_raw_power_weight",
        "raw_bias_attr": 'explore_mc_pic_rerank_pic_diversity_mgs_score_raw_bias',
        "smooth_attr": "explore_mc_pic_rerank_pic_diversity_mgs_score_rank_smooth",
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
        "name": "cascade_phtr",
        "weight": 0.0,
        "power_weight_attr": "explore_mc_ensemble_pic_phtr_order_power_weight",
        "weight_attr": "explore_mc_ensemble_pic_phtr_order_raw_weight",
        "raw_power_weight_attr": "explore_mc_ensemble_pic_phtr_order_raw_power_weight",
        "raw_bias_attr": 'explore_mc_pic_rerank_cascade_phtr_raw_bias',
        "smooth_attr": "explore_mc_pic_rerank_cascade_phtr_rank_smooth",
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
        "name" : "cascade_real_pctr",
        "weight" : 0.0,
        "power_weight_attr" : "explore_pic_mc_real_pctr_pow_weight",
        "raw_weight_attr": "explore_pic_mc_real_pctr_raw_weight",
        "raw_power_weight_attr": "explore_pic_mc_real_pctr_raw_pow_weight",
        "raw_bias_attr": 'explore_pic_mc_real_pctr_raw_bias',
        "smooth_attr": "explore_pic_mc_real_pctr_smooth",
      },
      {
        "name": "pic_search_interest_tagnex_score",
        "weight": 0.0,
        "power_weight_attr": "explore_mc_ensemble_pic_search_interest_tagnex_score_power_weight",
        "raw_weight_attr": "explore_mc_ensemble_pic_search_interest_tagnex_score_raw_weight",
        "raw_power_weight_attr": "explore_mc_ensemble_pic_search_interest_tagnex_score_raw_power_weight",
        "smooth_attr": "explore_mc_ensemble_pic_search_interest_tagnex_score_rank_smooth",
      },
    ]
