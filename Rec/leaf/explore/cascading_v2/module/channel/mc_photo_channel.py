from cascading_v2.module.channel.base_channel import BaseChannelPartitioner
from cascading_v2.module.channel.base_channel import BaseChannelScorer

class McPhotoChannelParitioner(BaseChannelPartitioner):
  def __init__(self, name, flow, config):
    super().__init__(name, flow, config)

  def _calculate_flag(self):
    pass

class McPhotoChannelScorer(BaseChannelScorer):
  ES_QUEUES = [
    {
      "name": "cascade_score",
      "weight": 1.0,
      "power_weight_attr": "explore_mc_s2_es_cascade_score_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_cascade_score_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_cascade_score_raw_power_weight",
      "score_threshold": "explore_mc_s2_es_cascade_score_score_threshold",
      "rank_cliff_attr": "explore_mc_s2_es_cascade_score_rank_threshold",
      "rank_height_attr": "explore_mc_s2_es_cascade_score_rank_height",
    },
    {
      "name" : "mc_ensemble_pwatch_time",
      "weight" : 1.0,
      "power_weight_attr" : "explore_mc_s2_es_pwatch_time_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_pwatch_time_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_pwatch_time_raw_power_weight",
      "score_threshold": "explore_mc_s2_es_pwatch_time_score_threshold",
      "rank_cliff_attr": "explore_mc_s2_es_pwatch_time_rank_threshold",
      "rank_height_attr": "explore_mc_s2_es_pwatch_time_rank_height",
    },
    {
      "name": "mc_ensemble_pwtd_inverse", 
      "weight": 1.0,
      "power_weight_attr": "explore_mc_s2_es_pwtd_inverse_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_pwtd_inverse_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_pwtd_inverse_raw_power_weight",
      "score_threshold": "explore_mc_s2_es_pwtd_inverse_score_threshold",
      "rank_cliff_attr": "explore_mc_s2_es_pwtd_inverse_rank_threshold",
      "rank_height_attr": "explore_mc_s2_es_pwtd_inverse_rank_height",
    },
    {
      "name": "cascade_plvtr",
      "weight": 1.0,
      "power_weight_attr" : "explore_mc_s2_es_cascade_plvtr_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_cascade_plvtr_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_cascade_plvtr_raw_power_weight",
    },
    {
      "name" : "mc_ensemble_plvtr",
      "weight" : 1.0,
      "power_weight_attr" : "explore_mc_s2_es_plvtr_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_plvtr_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_plvtr_raw_power_weight",
    },
    {
      "name" : "mc_ensemble_plvtr2",
      "weight" : 1.0,
      "power_weight_attr" : "explore_mc_s2_es_plvtr2_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_plvtr2_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_plvtr2_raw_power_weight",
    },
    {
      "name" : "mc_ensemble_pctr",
      "weight" : 1.0,
      "power_weight_attr" : "explore_mc_s2_es_pctr_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_pctr_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_pctr_raw_power_weight",
    },
    {
      "name" : "cascade_corr_pctr_psvr",
      "weight" : 1.0,
      "power_weight_attr" : "explore_mc_s2_es_cascade_corr_pctr_psvr_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_cascade_corr_pctr_psvr_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_cascade_corr_pctr_psvr_raw_power_weight",
    },
    {
      "name" : "mc_ensemble_pltr",
      "weight" : 1.0,
      "power_weight_attr" : "explore_mc_s2_es_pltr_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_pltr_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_pltr_raw_power_weight",
      "score_threshold": "explore_mc_s2_es_pltr_rank_cliff_score_threshold",
      "rank_cliff_attr": "explore_mc_s2_es_pltr_rank_cliff_rank_threshold",
      "rank_height_attr": "explore_mc_s2_es_pltr_rank_cliff_rank_height",
    },
    {
      "name" : "mc_ensemble_pwtr",
      "weight" : 1.0,
      "power_weight_attr" : "explore_mc_s2_es_pwtr_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_pwtr_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_pwtr_raw_power_weight",
      "score_threshold": "explore_mc_s2_es_pwtr_rank_cliff_score_threshold",
      "rank_cliff_attr": "explore_mc_s2_es_pwtr_rank_cliff_rank_threshold",
      "rank_height_attr": "explore_mc_s2_es_pwtr_rank_cliff_rank_height",
    },
    {
      "name" : "mc_ensemble_pftr",
      "weight" : 1.0,
      "power_weight_attr" : "explore_mc_s2_es_pftr_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_pftr_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_pftr_raw_power_weight",
    },
    {
      "name" : "mc_ensemble_pepstr",
      "weight" : 1.0,
      "power_weight_attr" : "explore_mc_s2_es_pepstr_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_pepstr_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_pepstr_raw_power_weight",
    },
    {
      "name" : "mc_ensemble_pcmtr",
      "weight" : 1.0,
      "power_weight_attr" : "explore_mc_s2_es_pcmtr_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_pcmtr_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_pcmtr_raw_power_weight",
    },
    {
      "name" : "mc_ensemble_pcltr",
      "weight" : 1.0,
      "power_weight_attr" : "explore_mc_s2_es_pcltr_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_pcltr_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_pcltr_raw_power_weight",
      "score_threshold": "explore_mc_s2_es_pcltr_rank_cliff_score_threshold",
      "rank_cliff_attr": "explore_mc_s2_es_pcltr_rank_cliff_rank_threshold",
      "rank_height_attr": "explore_mc_s2_es_pcltr_rank_cliff_rank_height",
    },
    {
      "name": "mc_ensemble_peftr",
      "weight": 1.0,
      "power_weight_attr": "explore_mc_s2_es_peftr_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_peftr_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_peftr_raw_power_weight",
    },
    {
      "name": "mc_ensemble_pefctr",
      "weight": 1.0,
      "power_weight_attr": "explore_mc_s2_es_pefctr_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_pefctr_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_pefctr_raw_power_weight",
    },
    {
      "name": "mc_ensemble_pcptr",
      "weight": 1.0,
      "power_weight_attr": "explore_mc_s2_es_pcptr_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_pcptr_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_pcptr_raw_power_weight",
    },
    {
      "name": "cascade_phtr",
      "weight": 1.0,
      "reverse_order": True,
      "power_weight_attr": "explore_mc_s2_es_reverse_cascade_phtr_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_reverse_cascade_phtr_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_reverse_cascade_phtr_raw_power_weight",
      "rank_height_attr": "explore_mc_s2_ensemble_phtr_rank_height",
    },
    {
      "name": "cascade_phtr",
      "weight": 1.0,
      "power_weight_attr": "explore_mc_s2_es_cascade_phtr_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_cascade_phtr_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_cascade_phtr_raw_power_weight",
      "score_threshold": "explore_mc_s2_emp_phtr_in_order_rank_cliff_threshold",
      "rank_height_attr": "explore_mc_s2_emp_phtr_in_order_rank_height",
    },
    {
      "name": "mc_ensemble_smooth_age_score",
      "weight": 1.0,
      "power_weight_attr": "explore_mc_s2_es_smooth_age_score_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_smooth_age_score_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_smooth_age_score_raw_power_weight",
    },
    {
      "name": "cascade_prerank_pctr",
      "weight": 1.0,
      "power_weight_attr" : "explore_mc_s2_es_cascade_prerank_pctr_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_cascade_prerank_pctr_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_cascade_prerank_pctr_raw_power_weight",
    },
    {
      "name": "cascade_prerank_pltr",
      "weight": 1.0,
      "power_weight_attr" : "explore_mc_s2_es_cascade_prerank_pltr_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_cascade_prerank_pltr_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_cascade_prerank_pltr_raw_power_weight",
    },
    {
      "name": "cascade_prerank_prstr",
      "weight": 1.0,
      "power_weight_attr" : "explore_mc_s2_es_cascade_prerank_prstr_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_cascade_prerank_prstr_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_cascade_prerank_prstr_raw_power_weight",
    },
    {
      "name": "mc_interact_fusion_score",
      "weight": 1.0,
      "power_weight_attr" : "explore_mc_s2_es_interact_fusion_score_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_interact_fusion_score_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_interact_fusion_score_raw_power_weight",
    },
    {
      "name": "cascade_psvtr",
      "weight": 1.0,
      "power_weight_attr" : "explore_mc_s2_es_cascade_psvtr_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_cascade_psvtr_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_cascade_psvtr_raw_power_weight",
    },
    {
      "name": "mc_ensemble_psvtr",
      "weight": 1.0,
      "power_weight_attr" : "explore_mc_s2_es_psvtr_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_psvtr_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_psvtr_raw_power_weight",
    },
    {
      "name": "mc_ensemble_psvtr",
      "reverse_order": True,
      "weight": 1.0,
      "power_weight_attr" : "explore_mc_s2_es_psvtr_reverse_order_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_psvtr_reverse_order_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_psvtr_reverse_order_raw_power_weight",
    },
    {
      "name": "cascade_hetu_one_xtr_debias_score",
      "weight": 1.0,
      "power_weight_attr" : "explore_mc_s2_es_hetu_one_xtr_debias_score_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_hetu_one_xtr_debias_score_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_hetu_one_xtr_debias_score_raw_power_weight",
    },
    {
      "name" : "cascade_distill_finish",
      "weight" : 1.0,
      "power_weight_attr" : "explore_mc_s2_es_dstill_finish_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_dstill_finish_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_dstill_finish_raw_power_weight",
    },
    {
      "name" : "cascade_distill_play_7s",
      "weight" : 1.0,
      "power_weight_attr" : "explore_mc_s2_es_dstill_play_7s_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_dstill_play_7s_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_dstill_play_7s_raw_power_weight",
    },
    {
      "name" : "topk_new_photo_pctr",
      "weight" : 1.0,
      "power_weight_attr" : "explore_mc_s2_es_topk_new_photo_pctr_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_topk_new_photo_pctr_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_topk_new_photo_pctr_raw_power_weight",
    },
    {
      "name" : "topk_new_cid_photo_pctr",
      "weight" : 1.0,
      "power_weight_attr" : "explore_mc_s2_es_topk_new_cid_photo_pctr_power_weight",
      "raw_weight_attr": "explore_mc_s2_es_topk_new_cid_photo_pctr_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_es_topk_new_cid_photo_pctr_raw_power_weight",
    },
    {
      "name": "sense_view_predict_trans_score",
      "weight": 1.0,
      "power_weight_attr" : "explore_mc_s2_sense_view_predict_trans_score_weight",
      "raw_weight_attr": "explore_mc_s2_sense_view_predict_trans_score_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_sense_view_predict_trans_score_raw_power_weight",
    },
    {
      "name": "cover_view_predict_trans_score",
      "weight": 1.0,
      "power_weight_attr" : "explore_mc_s2_cover_view_predict_trans_score_weight",
      "raw_weight_attr": "explore_mc_s2_cover_view_predict_trans_score_raw_weight",
      "raw_power_weight_attr": "explore_mc_s2_cover_view_predict_trans_score_raw_power_weight",
    },
  ]

  def __init__(self, name, flow, config):
    super().__init__(name, flow, config)

  def _caculate_score(self, flag_attr, weight_attr, left_count_attr):
    self.__calc_cascade_score(flag_attr)
    self.__calc_short_term_score(flag_attr)
    self.__calc_cascade_corr_pctr_psvr(flag_attr)
    self.__calc_cascade_topk_new_photo_pctr(flag_attr)
    self.__calc_interest_tag(flag_attr)
    self.__calc_personalized_power_weight()
    self.__calc_mc_es_score(flag_attr)
    self.__truncate_s1(flag_attr)
    self.__truncate_s2("mc_s2_interest_select_flag")

  def __calc_cascade_score(self, flag_attr):
    self.flow \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "mille_l0_ctr_param",
          "mille_l0_ltr_param",
          "mille_l0_wtr_param",
          "mille_l0_ftr_param",
          "mille_l0_lvtr_param",
          "mille_l0_lvtr2_param",
          "mille_l0_svtr_param",
          "mille_l0_ptr_param",
          "mille_l0_watchtime_param",
          "mille_l0_epstr_param",
          "mille_l0_cestr_param",
          "mille_l0_cmtr_param",
          "mille_l0_livingtr_param",
          "mille_l0_svtr_power_param",
          "mille_l0_cas_xtr_param",
          "mille_l0_cas_final_param",
          "mc_picture_discount_param",
          "mille_l0_wtd_param",
          "hot_mc_cp_ctr_weight",
          "mc_mid_photo_boost_param",
          "mille_l0_phtr_param",
          "mille_l0_phtr_power_param",
          "mille_l0_cltr_param",
          "mille_l0_pwtd_inverse_param",
          "mille_l0_pcptr_param",
        ],
        import_item_attr = [
          {"name": "cascade_corr_pctr", "as": "cascade_pctr"},
          "cascade_pltr",
          "cascade_pwtr",
          "cascade_pftr",
          "cascade_plvtr",
          "cascade_plvtr2",
          "cascade_psvtr",
          "cascade_ptr",
          "cascade_pwatch_time",
          "cascade_pepstr",
          "cascade_pcestr",
          "cascade_pcmtr",
          "cascade_plivingtr",
          "cascade_prerank_pctr",
          "cascade_prerank_pltr",
          "is_picture",
          "cascade_pwtd",
          "duration_ms",
          "cascade_phtr",
          "cascade_pcltr",
          "cascade_pwtd_inverse",
          "cascade_pcptr",
        ],
        export_item_attr = [
          "cascade_score",
        ],
        function_name = "CalMcMergedScore",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          flag_attr: 1
        },
      )

  def __calc_short_term_score(self, flag_attr):
    self.flow \
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
          min_ratio_coeff = "{{explore_short_term_item_min_ratio_coeff}}",
          max_ratio_coeff = "{{explore_short_term_item_max_ratio_coeff}}",
          tagnex_no_click_ratio_threshold = "{{explore_short_term_item_tagnex_no_click_ratio_threshold}}",
          interest_community_tag_no_click_ratio_threshold = "{{explore_short_term_item_interest_community_tag_no_click_ratio_threshold}}",
          cluster_id_no_click_ratio_threshold = "{{explore_short_term_item_cluster_id_no_click_ratio_threshold}}",
          hetu_level2_no_click_ratio_threshold = "{{explore_short_term_item_hetu_level2_no_click_ratio_threshold}}",
          hashtag_no_click_ratio_threshold = "{{explore_short_term_item_hashtag_no_click_ratio_threshold}}",
          hetu_tag_no_click_ratio_threshold = "{{explore_short_term_item_hetu_tag_no_click_ratio_threshold}}",
          enable_tagnex_score = "{{explore_enable_cal_short_term_item_tagnex_score}}",
          enable_interest_community_tag_score = "{{explore_enable_cal_short_term_item_interest_community_tag_score}}",
          enable_cluster_score = "{{explore_enable_cal_short_term_item_cluster_score}}",
          enable_hetu2_score = "{{explore_enable_cal_short_term_item_hetu2_score}}",
          enable_hashtag_score = "{{explore_enable_cal_short_term_item_hashtag_score}}",
          enable_hetu_tag_score = "{{explore_enable_cal_short_term_item_hetu_tag_score}}",
          enable_use_set_tagnex_ratio = "{{explore_enable_short_term_item_use_set_tagnex_ratio}}",
          enable_use_set_interest_community_tag_ratio = "{{explore_enable_short_term_item_use_set_interest_community_tag_ratio}}",
          enable_use_set_cluster_id_ratio = "{{explore_enable_short_term_item_use_set_cluster_id_ratio}}",
          enable_use_set_hetu_level2_ratio = "{{explore_enable_short_term_item_use_set_hetu_level2_ratio}}",
          enable_use_set_hashtag_ratio = "{{explore_enable_short_term_item_use_set_hashtag_ratio}}",
          enable_use_set_hetu_tag_ratio = "{{explore_enable_short_term_item_use_set_hetu_tag_ratio}}",
          enable_tagnex_use_threshold_adjust_score = "{{explore_enable_tagnex_use_threshold_adjust_short_term_item_score}}",
          enable_interest_community_tag_use_threshold_adjust_score = "{{explore_enable_interest_community_tag_use_threshold_adjust_short_term_item_score}}",
          enable_cluster_id_use_threshold_adjust_score = "{{explore_enable_cluster_id_use_threshold_adjust_short_term_item_score}}",
          enable_hetu_level2_use_threshold_adjust_score = "{{explore_enable_hetu_level2_use_threshold_adjust_short_term_item_score}}",
          enable_hashtag_use_threshold_adjust_score = "{{explore_enable_hashtag_use_threshold_adjust_short_term_item_score}}",
          enable_hetu_tag_use_threshold_adjust_score = "{{explore_enable_hetu_tag_use_threshold_adjust_short_term_item_score}}",
          tagnex_attr = "hetu_tag_level_info__hetu_tag",
          interest_community_tag_attr = "interest_community_tag_id",
          cluster_id_attr = "cluster_id_632",
          hetu_level2_attr = "hetu_tag_level_info__hetu_level_two",
          hashtag_attr = "user_hash_tag_id",
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
      .end_()

  def __calc_cascade_corr_pctr_psvr(self, flag_attr):
    self.flow \
      .if_("cal_cascade_cascade_ctr_svr_corr == 1") \
        .mc_replace_cascade_ctr_corr(flag_attr) \
      .end_()

  def __calc_cascade_topk_new_photo_pctr(self, flag_attr):
    self.flow \
      .if_("explore_enable_mc_s2_topk_new_photo_pctr == 1") \
        .mc_topk_new_photo_pctr(flag_attr) \
      .end_()

  def __calc_personalized_power_weight(self) -> None:
    self.flow \
      .if_("explore_mc_sort_weight_adjust_s2 == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "user_emp_ltr",
            "user_emp_wtr",
            "user_emp_ftr",
            "user_emp_cmtr",
            "user_emp_eptr",
            {"name": "explore_weight_adjust_avg_emp_ltr", "as": "all_user_emp_ltr"},
            {"name": "explore_weight_adjust_avg_emp_wtr", "as": "all_user_emp_wtr"},
            {"name": "explore_weight_adjust_avg_emp_ftr", "as": "all_user_emp_ftr"},
            {"name": "explore_weight_adjust_avg_emp_cmtr", "as": "all_user_emp_cmtr"},
            {"name": "explore_weight_adjust_avg_emp_eptr", "as": "all_user_emp_eptr"},
            {"name": "explore_mc_s2_es_pltr_power_weight", "as": "user_ori_ltr_weight"},
            {"name": "explore_mc_s2_es_pwtr_power_weight", "as": "user_ori_wtr_weight"},
            {"name": "explore_mc_s2_es_pftr_power_weight", "as": "user_ori_ftr_weight"},
            {"name": "explore_mc_s2_es_pcmtr_power_weight", "as": "user_ori_cmtr_weight"},
            {"name": "explore_mc_s2_es_pepstr_power_weight", "as": "user_ori_eptr_weight"},
            "explore_weight_adjust_coeff_min",
            "explore_weight_adjust_coeff_max",
          ],
          export_common_attr = [
            {"name": "user_ltr_weight", "as": "explore_mc_s2_es_pltr_power_weight"},
            {"name": "user_wtr_weight", "as": "explore_mc_s2_es_pwtr_power_weight"},
            {"name": "user_ftr_weight", "as": "explore_mc_s2_es_pftr_power_weight"},
            {"name": "user_cmtr_weight", "as": "explore_mc_s2_es_pcmtr_power_weight"},
            {"name": "user_eptr_weight", "as": "explore_mc_s2_es_pepstr_power_weight"},
          ],
          function_name = "UserSortWeightAdjust",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_mc_ensemble_sort_ctr_weight_adjust == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "explore_vv_3d",
            "choose_type",
            {"name": "explore_mc_ensemble_sort_ctr_weight_max", "as": "explore_weight_adjust_coeff_max"},
            {"name": "explore_mc_ensemble_sort_ctr_weight_min", "as": "explore_weight_adjust_coeff_min"}
          ],
          export_common_attr = [
            {"name": "weight_value", "as": "explore_mc_s2_es_pctr_power_weight"},
          ],
          function_name = "DynamicCalculateWeight",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_tnu_user_adjust_cascading_weight == 1 and uIsExploreTnuCrowdUser == 1") \
        .gen_common_attr_by_lua( # 显式判断新回人群逻辑
          attr_map={
            "explore_mc_s2_es_pctr_power_weight": "explore_mc_s2_es_pctr_power_weight * explore_tnu_ctr_cascading_adjust_ratio",
            "explore_mc_s2_es_pltr_power_weight": "explore_mc_ensemble_s2_pltr_power_weight * explore_tnu_ltr_cascading_adjust_ratio",
          }
        ) \
      .end_() \
      .if_("enable_cascading_mc_ensemble_s2_pwtr_weight_new_follow_adjust == 1") \
        .gen_common_attr_by_lua( # 粗排s2涨关摸高
          attr_map={
            "explore_mc_s2_es_pwtr_power_weight": "explore_mc_s2_es_pwtr_power_weight * explore_new_follow_pwtr_cascading_adjust_ratio",
          }
        ) \
      .end_()

  def __calc_interest_tag(self, flag_attr):
    self.flow \
      .explore_colossus_cluster_enricher(
        colossus_v2_attr_name = "colossus_resp_v2",
        user_info_ptr_attr = "user_info_ptr",
        export_colossus_attr_one = "sim_one_tags",
        export_colossus_attr_two = "sim_two_tags",
        export_colossus_attr_three = "sim_three_tags",
        export_colossus_attr_explore = "sim_explore_tags",
        enable_mc_explore_cluster = "{{enable_mc_explore_cluster}}",
        mc_explore_cluster_limit = "{{mc_explore_cluster_limit}}",
        mc_explore_cluster_score_limit = "{{mc_explore_cluster_score_limit}}",
      ) \
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
      .if_("enable_user_history_cids_stat == 1") \
        .user_history_cids_stat_enricher(
          cluster_id_attr = "hetu_sim_cluster_id",
          recent_realshow_items_attr = "explore_realshow_click_common_list",
          recent_realshow_top_ratio = "{{explore_user_history_cids_recent_realshow_top_ratio}}",
          recent_realshow_min_count = "{{explore_user_history_cids_recent_realshow_min_count}}",
          save_recent_realshow_cids_attr = "user_recent_realshow_cids"
        ) \
      .end_() \
      .explore_cluster_by_custom_rule(
        user_info_ptr_attr = "user_info_ptr",
        save_cluster_id_to_attr = "mc_s2_interest_id",
        enable_user_profile_top_hetu_level_one_cluster = "{{explore_enable_use_hetu_level1_id}}",
        user_profile_tag_score_limit = "{{mc_cluster_tag_score_limit}}", # 4
        user_profile_limit_num = "{{mc_cluster_limit_hetulevel1_num}}", # 3
        enable_use_real_show_list = "{{mc_cluster_use_real_show_list}}",
        enable_use_click_list = "{{mc_cluster_use_click_list}}",
        enable_use_like_list = "{{explore_mc_cluster_use_like_list}}",
        enable_use_follow_list = "{{explore_mc_cluster_use_follow_list}}",
        enable_use_forward_list = "{{explore_mc_cluster_use_forward_list}}",
        real_show_weight = "{{explore_mc_cluster_real_show_weight}}", # 1.0
        click_weight = "{{explore_mc_cluster_click_weight}}", #2.0,
        like_weight = "{{explore_mc_cluster_like_weight}}", #3.0,
        follow_weight = "{{explore_mc_cluster_follow_weight}}", # 3.0
        forward_weight = "{{explore_mc_cluster_forward_weight}}", # 3.0
        enable_colossus_cluster = "{{enable_use_colossus_cluster}}",
        enable_mc_explore_cluster = "{{enable_mc_explore_cluster}}",
        input_colossus_attr_one = "sim_one_tags",
        input_colossus_attr_two = "sim_two_tags",
        input_colossus_attr_three = "sim_three_tags",
        input_colossus_attr_explore = "sim_explore_tags",
        hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
        hetu_level_two_attr = "hetu_tag_level_info__hetu_level_two",
        hetu_level_three_attr = "hetu_tag_level_info__hetu_level_three",
        enable_mc_cluster_862_uninterest_cluster = "{{explore_mc_enable_mc_cluster_862_uninterest_cluster}}",
        input_user_interest_cluster_862_attr = "uOldMmuClusterId300ListList",
        enable_mc_cluster_862_uninterest_cluster_by_u2c = "{{explore_mc_enable_mc_cluster_862_uninterest_cluster_by_u2c}}",
        input_user_cluster862_sorted_list_attr = "user_cluster862_sorted_list",
        enable_mc_filter_u2c_score_by_cluster_862_lv1 = "{{explore_mc_enable_mc_filter_u2c_score_by_cluster_862_lv1}}",
        filter_u2c_score_by_cluster_862_lv1_classes_attr = "explore_mc_filter_u2c_score_by_cluster_862_lv1_classes",
        mc_user_uninterest_cluster_862_count = "{{explore_mc_user_uninterest_cluster_862_count}}",
        cluster_862_attr = "mounted_interest_cluster_id",
        audit_b_second_tag_attr = "audit_b_second_tag",
        audit_hot_cover_level_attr = "audit_hot_cover_level",
        enable_mc_cluster_862_uninterest_cluster_impression_filter = "{{explore_mc_enable_cluster_862_uninterest_cluster_impression_filter}}",
        enable_mc_cluster_862_uninterest_cluster_cover_filter = "{{explore_mc_enable_cluster_862_uninterest_cluster_cover_filter}}",
        input_user_recent_realshow_cluster_862_attr = "user_recent_realshow_cids",
        enable_rough_default_cluster = True,
        enable_ignore_profile_candidate_limit_cut = "{{enable_ignore_profile_candidate_limit_cut}}",
        mc_realtime_bucket_limit_num_ratio = "{{mc_realtime_bucket_limit_num_ratio}}",
        perf_checkpoint = "cascade",
        target_item = {
          flag_attr: 1,
        },
      )

  def __calc_mc_es_score(self, flag_attr):
    self.flow \
      .explore_calc_ensemble_score(
        use_superscript_rank = "{{explore_enable_mc_s2_use_superscript_rank}}",
        user_power_calc_v2 = "{{explore_mc_ensemble_s2_user_power_calc_v2}}",
        value_seq_fusion_status = "{{explore_mc_s2_value_seq_fusion_status}}",
        enable_power_weight_norm = "{{explore_enable_mc_s2_power_weight_norm}}",
        power_weight_change_coeff = "{{explore_mc_s2_power_weight_change_coeff}}",
        user_info_ptr_attr = "user_info_ptr",
        rank_smooth = "{{explore_mc2_rank_smooth}}",
        rank_score_calculate_method = "{{explore_mc_s2_rank_score_calculate_method}}",
        queues = self.ES_QUEUES,
        save_score_to_attr = self._score_attr,
        target_item = {
          flag_attr: 1
        }
      )
    self.__boost(flag_attr)

  def __boost(self, flag_attr):
    self.flow \
      .if_("enable_explore_cascading_s2_boost == 1") \
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
              flag_attr: 1
            }
          ) \
        .end_() \
        .if_("mc_enable_high_htr_discount == 1") \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "mc_high_htr_discount_coef", "as": "high_htr_discount_coef"},
              {"name": "mc_high_htr_threshold", "as": "high_htr_threshold"},
              {"name": "mc_high_htr_discount_power", "as": "htr_discount_power"},
            ],
            import_item_attr = [
              {"name": self._score_attr, "as": "es_score"},
              {"name": "cascade_phtr", "as": "htr_score"},
            ],
            export_item_attr = [
              {"name": "es_score", "as": self._score_attr}
            ],
            function_name = "HighHtrMixEsScore",
            class_name = "ExploreLightFunctionSetV2",
            target_item = {
              flag_attr: 1
            }
          ) \
        .end_() \
        .if_("enable_impression_audit_adjust == 1") \
          .mc_impression_audit_adjust(self._score_attr, flag_attr) \
        .end_() \
        .if_("enable_mc_s2_marketing_compensation_discount == 1") \
          .mc_marketing_compensation_discount(self._score_attr, flag_attr) \
        .end_() \
        .if_("enable_mc_s2_interest_cid == 1") \
          .mc_interest_cid_boost(self._score_attr, flag_attr) \
        .end_() \
        .if_("enable_explore_cs_photo_boost_mc_s2 == 1") \
          .mc_cs_boost(self._score_attr, flag_attr, "mc_s2") \
        .end_() \
        .if_("enable_mc_s2_short_uninterest_decay_discount == 1") \
          .mc_short_uninterest_decay_discount(self._score_attr, flag_attr) \
        .end_() \
        .if_("enable_mc_s2_user_uninterest_cluster_862_adjust == 1") \
          .mc_user_uninterest_cluster_862_adjust(self._score_attr, flag_attr) \
        .end_() \
      .end_()

  def __truncate_s1(self, flag_attr):
    self.flow \
      .sort(
        score_from_attr = self._score_attr,
        target_item = {
          flag_attr: 1
        },
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_mc_s2_long_term_interest_keep_ratio", "as": "long_term_interest_keep_ratio"},
          {"name": "explore_mc_s2_short_term_interest_keep_ratio", "as": "short_term_interest_keep_ratio"},
          {"name": "explore_mc_s2_explore_interest_keep_ratio", "as": "explore_interest_keep_ratio"},
          {"name": "explore_mc_s2_cluster_862_uninterest_keep_ratio", "as": "cluster_862_uninterest_keep_ratio"},
          {"name": "explore_mc_s2_default_interest_keep_ratio", "as": "default_interest_keep_ratio"},
          {"name": "is_traceback_request", "as": "need_to_save_interest_detail"},
        ],
        import_item_attr = [
          {"name": "mc_s2_interest_id", "as": "interest_id"},
        ],
        export_common_attr = [
          {"name": "interest_id_list", "as": "mc_s2_interest_id_list"},
          {"name": "interest_count_list", "as": "mc_s2_interest_count_list"},
          {"name": "keep_interest_count_list", "as": "mc_s2_keep_interest_count_list"},
        ],
        export_item_attr = [
          {"name": "interest_select_flag", "as": "mc_s2_interest_select_flag"}
        ],
        function_name = "CalcMcS2InterestSelectFlag",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          flag_attr: 1,
        },
      ) \
      .enrich_attr_by_light_function(
        import_item_attr = [
          {"name": "mc_s2_interest_select_flag", "as": "flag"},
        ],
        export_item_attr = [
          {"name": "score", "as": self._score_attr},
        ],
        function_name = "SetMinimumScoreByFlag",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          flag_attr: 1,
        },
      ) \
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
          "mc_s2_interest_select_flag": 1,
        },
      )

  def __truncate_s2(self, flag_attr):
    self.flow \
      .if_("enable_fr_refactor_mc_same_author == 1") \
        .deduplicate(
          name = "deduplicate_by_author_photo",
          traceback = True,
          on_item_attr = "author__id",
          target_item = {
            flag_attr: 1,
          },
        ) \
      .end_() \
      .copy_item_meta_info(
        save_item_seq_to_attr = "mc_s2_final_index_photo",
        target_item = {
          flag_attr: 1,
        },
      ) \
      .explore_control_hetu_count_enricher(
        hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
        hetu_level_two_attr = "hetu_tag_level_info__hetu_level_two",
        hetu_level_five_attr = "hetu_tag_level_info__hetu_level_five",
        cluster_id_attr = "mounted_interest_cluster_id",
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
        enable_minority_control_diversity = "{{hot_cascade_enable_minority_control_diversity}}",
        is_minority_photo_attr = "is_minority_photo",
        minority_max_size = "{{hot_cascade_control_minority_max_size}}",
        save_is_degraded_common_attr = "mc_s2_hetu_quota_control_is_degraded",
        target_item = {
          flag_attr: 1,
        },
      ) \
      .enrich_attr_by_light_function(
        import_item_attr = [
          {"name": "mc_s2_diversity_select_flag", "as": "flag"},
        ],
        export_item_attr = [
          {"name": "score", "as": self._score_attr},
        ],
        function_name = "SetMinimumScoreByFlag",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          flag_attr: 1,
        },
      )
