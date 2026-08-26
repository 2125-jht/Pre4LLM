from filter import CommonModule

class FilterModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  @property
  def item_attr_map(self) -> dict:
    return {
      "photo_id_attr": "photo_id",
      "audit_hot_high_tag_level_attr": "audit_hot_high_tag_level",
      "duration_ms_attr": "duration_ms",
      "upload_type_attr": "upload_type",
      "show_level_a_attr": "show_level_a",
      "review_pass_level_b_attr": "review_pass_level_b",
      "photo_total_report_count_attr": "explore_stat__report_detail__total_report_count",
      "author_total_report_count_attr": "author__explore_report_thirtyday__total_report_count",
      "title_evil_level_attr": "title_evil_level",
      "ocr_cover_text_evil_level_attr": "ocr_cover_text_evil_level",
      "hetu_level_one_tag_list_attr": "hetu_tag_level_info__hetu_level_one",
      "hetu_level_two_tag_list_attr": "hetu_tag_level_info__hetu_level_two",
      "hetu_level_three_tag_list_attr": "hetu_tag_level_info__hetu_level_three",
      "hetu_level_five_tag_list_attr": "hetu_tag_level_info__hetu_level_five",
      "hetu_face_id_tag_list_attr": "hetu_tag_level_info__hetu_face_id",
      "hetu_tag_list_attr": "hetu_tag_level_info__hetu_tag",
      "hetu_v2_level_one_tag_list_attr": "hetu_tag_level_info_v2__hetu_level_one",
      "hetu_v2_level_two_tag_list_attr": "hetu_tag_level_info_v2__hetu_level_two",
      "hetu_v2_level_three_tag_list_attr": "hetu_tag_level_info_v2__hetu_level_three",
      "audit_hot_cover_level_attr": "audit_hot_cover_level",
      "merchant_item_id_list_attr": "merchant_item_info__item_id_list",
      "merchant_photo_cart_relation_attr": "merchant_photo_cart_relation",
      "caption_length_attr": "caption_length",
      "picture_type_attr": "picture_type",
      "mmu_photo_low_quality_model_attr": "mmu_photo_low_quality_model",
      "risk_man_risk_photo_attr": "risk_level", 
      "need_shuffle_photo_attr": "shuffle_policy",
      "author_fans_count_attr": "author__fans_count",
      "explore_real_show_attr": "explore_stat__real_show_count",
      "nebula_real_show_attr": "nebula_stats__real_show_count",
      "nebula_like_attr": "nebula_stats__like_count",
      "nebula_comment_attr": "nebula_stats__comment_count",
      "nebula_collect_attr": "nebula_stats__collect_count",
      "nebula_forward_attr": "nebula_stats__forward_count",
      "nebula_follow_attr": "nebula_stats__follow_count",
      "nebula_negative_attr": "nebula_stats__negative_count", 
      "thanos_real_show_attr": "thanos_stats__real_show_count",
      "thanos_like_attr": "thanos_stats__like_count",
      "thanos_comment_attr": "thanos_stats__comment_count",
      "thanos_collect_attr": "thanos_stats__collect_count",
      "thanos_forward_attr": "thanos_stats__forward_count",
      "thanos_follow_attr": "thanos_stats__follow_count",
      "thanos_negative_attr": "thanos_stats__negative_count",
      "fountain_real_show_attr": "fountain_stats__real_show_count",
      "fountain_like_attr": "fountain_stats__like_count",
      "fountain_comment_attr": "fountain_stats__comment_count",
      "fountain_collect_attr": "fountain_stats__collect_count",
      "fountain_forward_attr": "fountain_stats__forward_count",
      "fountain_follow_attr": "fountain_stats__follow_count",
      "fountain_negative_attr": "fountain_stats__negative_count",
      "explore_server_show_attr": "explore_stat__show_count",
      "explore_long_play_count_attr": "explore_stat__long_play_count",
      "explore_negative_attr": "explore_stat__negative_count",
      "explore_click_attr": "explore_stat__click_count",
      "explore_like_attr": "explore_stat__like_count",
      "explore_follow_attr": "explore_stat__follow_count",
      "explore_forward_attr": "explore_stat__forward_count",
      "explore_comment_attr": "explore_stat__comment_count",
      "explore_collect_attr": "explore_stat__collect_count",
      "explore_view_length_sum_attr": "explore_stat__view_length_sum",
      "mmu_low_quality_model_score_40_attr": "mmu_low_quality_model_score_40",
      "mmu_low_quality_model_score_42_attr": "mmu_low_quality_model_score_42",
      "mmu_low_quality_model_score_46_attr": "mmu_low_quality_model_score_46",
      "mmu_low_quality_model_score_52_attr": "mmu_low_quality_model_score_52",
      "mmu_low_quality_model_score_63_attr": "mmu_low_quality_model_score_63",
      "mmu_low_quality_model_score_64_attr": "mmu_low_quality_model_score_64",
      "mmu_low_quality_model_score_90_attr": "mmu_low_quality_model_score_90",
      "mmu_low_quality_model_score_104_attr": "mmu_low_quality_model_score_104",
      "mmu_low_quality_model_score_123_attr": "mmu_low_quality_model_score_123",
      "mmu_low_quality_model_score_143_attr": "mmu_low_quality_model_score_143",
      "mmu_low_quality_model_score_145_attr": "mmu_low_quality_model_score_145",
      "mmu_low_quality_model_score_150_attr": "mmu_low_quality_model_score_150",
      "mmu_low_quality_model_score_151_attr": "mmu_low_quality_model_score_151",
      "mmu_low_quality_model_score_163_attr": "mmu_low_quality_model_score_163",
      "mmu_low_quality_model_score_164_attr": "mmu_low_quality_model_score_164",
      "is_sirius_punish_attr": "is_sirius_punish",
      "enable_download_attr": "enable_download",
      "author_id_attr": "author__id",
      "level_hot_online_attr": "content_safety_level_with_namespace__level_hot_online",
      "audit_b_second_tag_attr": "audit_b_second_tag",
      "cold_start_breakout_score_attr": "cold_start_breakout_score",
      "questionaire_info_exposure_count_attr" : "questionnaire_info__exposure_count",
      "questionaire_info_negative_count_attr" : "questionnaire_info__negative_count",
      "questionaire_info_positive_count_attr" : "questionnaire_info__positive_count",
      "questionaire_info_unsure_count_attr" : "questionnaire_info__unsure_count",
      "explore_questionaire_info_exposure_count_attr" : "explore_questionnaire_info__exposure_count",
      "explore_questionaire_info_negative_count_attr" : "explore_questionnaire_info__negative_count",
      "explore_questionaire_info_positive_count_attr" : "explore_questionnaire_info__positive_count",
      "explore_questionaire_info_unsure_count_attr" : "explore_questionnaire_info__unsure_count",
      "upload_time_attr": "upload_time",
      "long_term_photo_attr": "long_term_photo",
      "explore_punish_attr": "explore_punish",
      "explore_punish_city_attr": "explore_punish_city",
      "topk_audit_level_attr": "topk_audit_level",
      "topk_audit_tag_attr": "topk_audit_tag",
      "photo_status_attr": "photo_status",
      "width_attr": "width",
      "height_attr" : "height",
      "high_hot_audit_tag_v2_attr": "high_hot_audit_tag_v2",
      "video_quality_assessment_flag_attr": "video_quality_assessment_flag",
      "audit_user_experiment_level_attr": "audit_user_experiment_level",
      "eyeshot_source_attr": "eyeshot_source",
      "young_inc_tags_attr": "young_inc_tags",
      "final_cross_section_first_class_id_attr": "final_cross_section_first_class_id",
      "light_inc_photo_flag_attr": "light_inc_photo_flag",
      "audit_cold_review_level_attr": "audit_cold_review_level",
      "high_value_pic_flag_attr": "high_value_pic_flag",
      "data_set_tags_attr": "data_set_tags",
      "audit_risk_immd_tag_attr": "audit_risk_immd_tag",
      "photo_dynamic_xtrs_str_attr": "video_cold_start_info__photo_dynamic_xtrs_str",
      "ecom_intent_score_attr": "photo_category_info__ecom_intent_score",
      "hetu_tag_level_info_v2__hetu_tag_attr": "hetu_tag_level_info_v2__hetu_tag",
      "data_set_tags_bit_attr": "data_set_tags_bit",
      "data_set_tags_bit_list_attr": "data_set_tags_bit_list",
      "magic_face_id_attr": "magic_face_id",
      "magic_face_type_attr": "magic_face_type",
      "explore_short_play_attr": "explore_stat__short_play_count",
      "is_repost_photo_attr": "mmu_repost_photo_info__is_repost_photo",
      "sirius_distribution_info__mark_cod_attr": "sirius_distribution_info__mark_cod",
      "live_photo_flag_attr": "live_photo_flag",
      "explore_effective_play_count_attr": "explore_stat__effective_play_count",
      "author_grade_key_attr": "author_grade_key",
      "explore_stats_report_count_attr": "explore_stat__report_count",
      "fountain_stats_report_count_attr": "fountain_stats__report_count",
      "thanos_stats_report_count_attr": "thanos_stats__report_count",
      "nebula_stats_report_count_attr": "nebula_stats__report_count",
      "fountain_stats_short_play_count_attr": "fountain_stats__short_play_count",
      "thanos_stats_short_play_count_attr": "thanos_stats__short_play_count",
      "nebula_stats_short_play_count_attr": "nebula_stats__short_play_count",
      "merchant_hetu_tag_id_photo_attr": "is_merchant_hetu_tag_id",
      "hetu_sim_cluster_id_attr": "hetu_sim_cluster_id",
      "author_age_segment_attr": "author_age_info__age_segment",
      "nebula_stats_view_length_sum_attr": "nebula_stats__view_length_sum",
      "thanos_stats_view_length_sum_attr": "thanos_stats__view_length_sum",
      "fountain_stats_view_length_sum_attr": "fountain_stats__view_length_sum",
      "secure_grading_action_code_attr": "secure_grading_action_code",
      "author_max_item_score_attr": "author_max_item_score",
      "author_shop_score_attr": "author_shop_score",
      "timeliness_flag_attr": "timeliness_flag",
      "coldstart_guarantee_value_attr": "coldstart_guarantee_value",
      "manjiao_markcode_attr": "manjiao_markcode",
      "fangpin_aid_filter_ratio_attr": "fangpin_aid_filter_ratio",
      "plc_business_type_attr": "plc_business_type",
      "author_tail_galaxy_attr": "video_cold_start_info__author_tails__galaxyAuthorExpGroupInteger",
      "author_tail_climb_attr": "video_cold_start_info__author_tails__climb_retrieval_author_tail",
      "author_tail_vcs_attr": "video_cold_start_info__author_tails__vcs_author_tail_161",
      "author_liezhi_pic_count_attr": "author_liezhi_pic_count",
      "author_hash_tag_id_list_attr": "user_hash_tag_id",
      "live_photo_duration_attr": "live_photo_duration",
      "community_survey_markcode_attr": "community_survey_info__survey_title",
      "community_survey_cert_count_attr": "community_survey_info__cert_count",
      "community_survey_not_cert_count_attr": "community_survey_info__not_cert_count",
      "community_survey_uncert_count_attr": "community_survey_info__uncert_count",
      "is_tv_station_bottom_bar_attr": "is_tv_station_bottom_bar",
      "hot_trend_generalized_info_source_attr": "hot_trend_generalized_info__source",
      "author_tail_int_index_map_34_attr": "author_tail_int_index_map_34",
      "author_op_session_class_attr": "author_op_session_class",
      "cover_view_predict_score_attr": "cover_view_predict_score",
      "sense_view_predict_score_attr": "sense_view_predict_score",
      "cover_view_predict_score_v2_attr": "cover_view_predict_score_v2",
      "sense_view_predict_score_v2_attr": "sense_view_predict_score_v2",
      "explore_sid_attr": "explore_sid",
      "is_living_attr": "live_photo_info__is_living",
      "slide_recent_negative_count_attr": "black_industry_filter__slide_recent_negative_count",
      "slide_recent_real_show_count_attr": "black_industry_filter__slide_recent_real_show_count",
      "slide_recent_report_count_attr": "black_industry_filter__slide_recent_report_count"
    }

  @property
  def filters(self) -> list:
    return [
      {
        "name": "not_in_index",
        "enable": True,
      },
      {
        "name": "photo_life",
        "enable": "{{enable_photo_life_filter}}",
        "photo_life_max_hours_attr": "photo_life_max_hours",
        "enable_skip_follow_author_attr": "enable_skip_follow_author",
      },
      {
        "name": "video_filter",
        "enable": "{{enable_explore_video_filter}}",
      },
      {
        "name": "tnu_extend_index",
        "enable": "{{enable_filter_tnu_extend_filter}}",
        "is_tnu_extend_index_photo_attr": "is_tnu_extend_index_photo",
      },
      # topk审劣质
      {
        "name": "topk_audit_bad",
        "enable":"{{enable_filter_topk_audit_bad}}",
        "topk_audit_white_tag_list_attr": "topk_audit_white_tag_list",
        "topk_audit_black_tag_list_attr": "topk_audit_black_tag_list",
        "topk_audit_bad_recall_filter_attr": "explore_topk_audit_bad_recall_filter",
        "topk_audit_bad_recall_filter_use_global_attr": "explore_topk_audit_bad_recall_filter_use_global",
        "topk_audit_bad_recall_filter_credible_ques_cnt_attr": "explore_topk_audit_bad_recall_filter_credible_ques_cnt",
        "topk_audit_bad_recall_filter_pos_threshold_attr": "explore_topk_audit_bad_recall_filter_pos_threshold",
        "topk_audit_bad_recall_filter_mode_attr": "explore_topk_audit_bad_recall_filter_mode",
        "topk_audit_bad_recall_filter_unsure_threshold_attr": "explore_topk_audit_bad_recall_filter_unsure_threshold",
        "topk_audit_bad_recall_filter_neg_threshold_attr": "explore_topk_audit_bad_recall_filter_neg_threshold",
        "topk_audit_bad_recall_filter_hate_threshold_attr": "explore_topk_audit_bad_recall_filter_hate_threshold", 
      },
      # 高热审劣质
      {
        "name": "high_hot_audit_bad",
        "enable": True,
        "high_hot_audit_white_tag_list_attr": "high_hot_audit_white_tag_list",
        "high_hot_audit_black_tag_list_attr": "high_hot_audit_black_tag_list",
        "audit_hot_high_tag_level_attr": "audit_hot_high_tag_level",
        "explore_operation_c_review_level_attr": "explore_operation_c_review_level",
        "user_sexy_interest_score_attr": "uSexyInterestScore",
        "user_sexy_interest_score_ignore_threshold_attr": "high_hot_audit_user_sexy_interest_score_ignore_threshold",
        "user_sexy_interest_exemption_tag_list_attr": "user_sexy_interest_exemption_tag_list",
        "enable_user_sexy_interest_exemption_high_hot_white_tag_attr": "enable_user_sexy_interest_exemption_high_hot_white_tag",
        "enable_user_sexy_interest_exemption_hate_rate_attr": "enable_user_sexy_interest_exemption_hate_rate",
        "user_sexy_interest_exemption_hate_threshold_attr": "user_sexy_interest_exemption_hate_threshold",
        "user_sexy_interest_exemption_hate_rate_threshold_attr": "user_sexy_interest_exemption_hate_rate_threshold",
        "user_sexy_interest_exemption_age_list_attr": "user_sexy_interest_exemption_age_list",
        "user_sexy_interest_exemption_city_level_list_attr": "user_sexy_interest_exemption_city_level_list",
        "user_age_segment_attr": "user_age_segment",
        "user_city_level_attr": "location_city_level_v2",
      },
      # 观感审劣质
      {
        "name": "impression_audit_bad",
        "enable": True,
        "impression_audit_white_tag_list_attr": "impression_audit_white_tag_list",
        "impression_audit_black_tag_list_attr": "impression_audit_black_tag_list",
        "level_hot_online_attr": "content_safety_level_with_namespace__level_hot_online",
        "audit_b_second_tag_attr": "audit_b_second_tag",
        "user_sexy_interest_score_attr": "uSexyInterestScore",
        "user_sexy_interest_score_ignore_threshold_attr": "impression_audit_user_sexy_interest_score_ignore_threshold",
        "user_sexy_interest_extra_filter_tag_list_attr": "user_sexy_interest_extra_filter_tag_list",
        "save_filtered_pid_list_to_attr": "impression_audit_bad_pid_list",
      },
      {
        "name": "zero_impression_level_hot_good",
        "enable": "{{enable_filter_zero_impression}}",
        "explore_zero_play_days_15d_attr": "explore_zero_play_days_15d",
      },
      {
        "name": "high_hot_audit_subdivision_level",
        "enable": True,
        "audit_hot_high_subdivision_level_attr": "audit_hot_high_subdivision_level",
      },
      {
        "name": "high_hot_audit_gray_show",
        "enable": "{{enable_high_hot_audit_gray_show_filter}}",
        "enable_stat_all_page": "explore_high_hot_audit_gray_show_filter_stat_all_page",
        "high_hot_audit_gray_show_threshold": "explore_high_hot_audit_gray_show_filter_threshold",
      },
      {
        "name": "browse_screen_aid",
        "enable": True,
        "browse_screen_aid_list_attr": "browse_screen__aid_list",
        "author_id_attr": "author__id",
      },
      {
        "name": "follow_author",
        "enable": "{{return not close_follow_author_filter or close_follow_author_filter == 0}}",
        "follow_author_filter_timegap_attr": "follow_author_filter_timegap",
        "follow_author_ignore_exptag_list_attr": "follow_author_ignore_exptag_list",
        "author_id_attr": "author__id",
        "upload_time_attr": "upload_time",
      },
      {
        "name": "follow_browse_set",
        "enable": "{{enable_follow_browse_set_filter}}",
      },
      {
        "name": "black_exempt_level_v1_audit",
        "enable": "{{explore_enable_black_exempt_level_v1_audit_filter}}",
        "auto_audit_black_exempt_level_v1_attr": "auto_audit_black_exempt_level_v1",
      },
      {
        "name": "long_term_high_level",
        "enable": "{{enable_filter_some_photo_extend}}",
        "long_term_high_level_photo_attr": "long_term_high_level_photo",
      },
      {
        "name": "photo_status",
        "enable": True,
      },
      {
        "name": "valueable",
        "enable": True,
        "valuable_author_type_list_attr": "valuable_author_type_list",
        "is_cuckoo_photo_attr": "cuckoo_info__is_cuckoo_photo",
        "cuckoo_author_type_attr": "cuckoo_info__author_type",
      },
      {
        "name": "over_180_days",
        "enable": "{{enable_over_days_filter}}",
        "over_days_filter_days_limit_attr": "explore_over_days_filter_days_limit",
        "entertainment_hetu_tags_attr": "explore_entertainment_hetu_tag_str",
        "entertainment_hetu_days_limit_attr": "explore_entertainment_hetu_days_limit_attr",
        "enable_filter_low_like": "enable_explore_over_days_filter_low_like",
        "low_like_limit_attr": "explore_over_days_filter_low_like_limit",
        "low_like_days_limit_attr": "explore_over_days_filter_low_like_days_limit",
        "page_type": "EXPLORE",
        "topn_screen_filter_attr": "explore_over_days_filter_topn_screen_map",
        "enable_filter_by_audit": "enable_explore_over_days_filter_audit",
        "impression_not_audit_hours_limit_attr": "explore_impression_not_audit_hours_limit",
        "impression_audit_gray_hours_limit_attr": "explore_impression_audit_gray_hours_limit",
        "impression_audit_normal_days_limit_attr": "explore_impression_audit_normal_days_limit",
        "impression_audit_high_quality_days_limit_attr": "explore_impression_audit_high_quality_days_limit",
        "high_hot_audit_gray_hours_limit_attr": "explore_high_hot_audit_gray_hours_limit",
        "high_hot_audit_normal_days_limit_attr": "explore_high_hot_audit_normal_days_limit",
        "high_hot_audit_high_quality_days_limit_attr": "explore_high_hot_audit_high_quality_days_limit",
        "enable_impression_audit_timeliness_photo_filter": "explore_enable_impression_audit_timeliness_photo_filter",
        "impression_audit_timeliness_photo_map_attr": "explore_impression_audit_timeliness_photo_map_str",
        "enable_filter_by_pic_days_limit": "explore_over_days_filter_enable_filter_by_pic_days_limit",
        "pic_days_limit_attr": "explore_over_days_filter_pic_days_limit_attr"
      },
      {
        "name": "outdate_news",
        "enable": "{{enable_filter_outdate_news}}",
        "enable_machine_outdate_filter_attr": "enable_machine_outdate_filter",
        "explore_operation_c_review_level_attr": "explore_operation_c_review_level",
        "upload_time_attr": "upload_time",
      },
      {
        "name": "user_self",
        "enable": True,
        "author_id_attr": "author__id",
      },
      {
        "name": "short_duration",
        "enable": "{{explore_enable_short_duration_filter}}",
        "short_duration_limit_attr": "explore_short_duration_filter_limit"
      },
      {
        "name": "upload_type",
        "enable": True,
        "filter_type_list_attr": "filter_upload_type_list",
      },
      {
        "name": "jianguan_risk",
        "enable": True,
        "is_jianguan_risk_photo_attr": "is_jianguan_risk_photo",
      },
      {
        "name": "black_author",
        "enable": True,
        "author_id_attr": "author__id",
      },
      {
        "name": "magic_freq_control",
        "enable": "{{explore_enable_magic_freq_control_filter}}",
        "magic_kk_id_list_attr": "magic_kk_id_list",
        "magic_face_id_attr": "magic_face_id",
        "kuaishan_id_attr": "kuaishan_id",
        "upload_type_attr": "upload_type",
        "outer_material_id_attr": "outer_material_id",
      },
      {
        "name": "report_author",
        "enable": True,
        "author_id_attr": "author__id",
        "enable_report_hetu_short_attr": "enable_report_hetu_short",
        "short_report_hetu_minutes_attr": "short_report_hetu_minutes",
        "long_report_hetu_minutes_attr": "long_report_hetu_minutes",
      },
      {
        "name": "hate_author",
        "enable": True,
        "limit_hate_reason_attr": "explore_limit_hate_reason",
        "enable_short_author_attr": "explore_enable_short_author",
        "short_hate_author_minutes_attr": "explore_short_hate_author_minutes", 
      },
      {
        "name": "emprical_ctr",
        "enable": "{{enable_emp_ctr_filter}}",
        "empctr_filter_threshold_attr": "empctr_filter_threshold",
        "empctr_sample_threshold_str_attr": "explore_empctr_sample_filter_threshold_str",
        "empctr_sample_base_number_attr": "explore_empctr_sample_filter_base_number",
        "empctr_sample_multi_number_attr": "explore_empctr_sample_filter_multi_number",
        "empctr_filter_realshow_threshold_attr": "explore_empctr_filter_realshow_threshold",
      },
      {
        "name": "commerce_extend_index",
        "enable": "{{enable_filter_some_photo_extend}}",
        "is_high_other_photo_attr": "is_high_other_photo",
      },
      {
        "name": "back_fresh_climb",
        "enable": "{{enable_filter_back_fresh_climb}}",
        "show_level_a_attr": "show_level_a",
      },
      {
        "name": "low_porn_report",
        "enable": "{{explore_enable_low_porn_report_filter}}",
        "photo_low_report_count_attr": "explore_stat__report_detail__low_report_count",
        "author_low_report_count_attr": "author__explore_report_thirtyday__low_report_count",
      },
      {
        "name": "total_report",
        "enable": "{{explore_enable_total_report_filter}}",
      },
      {
        "name": "evil_title",
        "enable": "{{enable_filter_evil_title}}",
      },
      {
        "name": "picture",
        "enable": "{{enable_all_pic_filter}}",
        "only_filter_picture_long_and_set_attr": "explore_only_filter_picture_long_and_set",
        "only_filter_high_value_pic_attr": "explore_only_filter_high_value_pic",
      },
      {
        "name": "long_pic",
        "enable": "{{enable_all_long_pic_filter}}",
        "filter_long_pic_picture_type": "long_pic_picture_type",
        "filter_long_pic_upload_type": "long_pic_upload_type",
      },
      {
        "name": "face_90_degree",
        "enable": "{{explore_enable_face_90_degree_filter}}",
        "face_90_degree_pids_data_key": "face_90_degree_pids",
      },
      {
        "name": "short_term_hate",
        "enable": "{{enable_short_term_hate_filter}}",
        "skip_clicked_hate_item_filter_attr": "skip_clicked_hate_item_filter",
        "enable_short_hate_l5_filter_attr": "enable_short_hate_l5_filter",
        "hetu_tag_l5_minutes_cut_attr": "hetu_tag_l5_minutes_cut",
        "hetu_tag_l3_minutes_cut_attr": "hetu_tag_l3_minutes_cut", 
        "enable_hate_author_skip_hetu_filter_attr": "enable_hate_author_skip_hetu_filter",
        "enable_long_hate_filter_attr": "enable_long_hate_filter",
        "hetu_tag_long_term_minutes_cut_attr": "hetu_tag_long_term_minutes_cut",
        "hetu_l2_long_filter_threshold_attr": "hetu_l2_long_filter_threshold",
        "high_hetu_num_threshold_attr": "explore_high_hetu_num_threshold",
        "low_hetu_num_threshold_attr": "explore_low_hetu_num_threshold",
        "hetu_otherl_long_filter_threshold_attr": "hetu_otherl_long_filter_threshold"
      },
      {
        "name": "short_pic_hetu",
        "enable": "{{enable_short_pic_hetu_filter}}",
        "filtered_hetu_tag_list_attr": "xhs_hetu_type"
      },
      {
        "name": "pic_wallpaper",
        "enable": "{{enable_pic_wallpaper_hetu_tag_filter}}",
        "filter_pic_wallpaper_hetu_tag": "pic_wallpaper_hetu_tag",
        "enable_pic_wallpaper_filter_caption_keep_attr": "enable_pic_wallpaper_filter_caption_keep",
        "pic_wallpaper_caption_keep_thresh_attr": "pic_wallpaper_caption_keep_thresh",
      },
      {
        "name": "auto_audit_hot_cover_level_filter",
        "enable": "{{enable_auto_audit_hot_cover_level_filter}}",
        "enable_follow_author_exemption_attr": "enable_auto_audit_follow_author_exemption",
        "enable_impression_good_ignore_attr": "enable_auto_audit_impression_good_ignore",
        "auto_audit_bad_show_limit_attr": "auto_audit_bad_show_limit",
      },
      {
        "name": "audit_hot_cover_level_filter",
        "enable": "{{enable_audit_hot_cover_level_filter}}",
        "save_filtered_pid_list_to_attr": "audit_hot_cover_level_filter_pid_list",
      },
      {
        "name": "audit_gray_cover_level_filter",
        "enable": "{{enable_audit_gray_cover_level_filter}}",
        "page_attr": "page_index",
        "max_page_threshold_attr": "audit_gray_cover_level_max_page_threshold",
        "enable_not_cover_photo_filter_attr": "enable_audit_gray_cover_level_not_cover_filter",
        "enable_tnu_and_reflux_not_cover_photo_filter_attr": "enable_audit_gray_cover_level_tnu_and_reflux_not_cover_filter",
        "enable_first_page_not_cover_photo_filter_attr": "enable_audit_gray_cover_level_first_page_not_cover_photo_filter",
        "hetu_v3_white_tag_fans_threshold_attr": "audit_gray_cover_level_hetu_v3_white_tag_fans_threshold",
        "hetu_v3_level_one_white_tag_list_attr": "audit_gray_cover_hetu_v3_level_one_white_tags",
      },
      {
        "name": "mmu_low_cover_filter",
        "enable": "{{enable_mmu_low_cover_filter}}",
        "lower_cover_mmu_map_strs_attr": "lower_cover_mmu_map_strs",
        "lower_cover_mmu_map_tnu_reflux_strs_attr": "lower_cover_mmu_map_tnu_reflux_strs", #新回、2-14新回配置，命中人群会覆盖lower_cover_mmu_map_strs
        "skip_beauty_photo_filter_attr": "skip_beauty_photo_filter",
        "mmu_enable_follow_author_exemption_attr": "mmu_enable_follow_author_exemption",
        "mmu_enable_impression_good_ignore_attr": "mmu_enable_impression_good_ignore",
        "enable_explore_gender_attr":"enable_explore_gender",
        "user_gender_attr": "basic_info_gender_v2",
        "save_filtered_pid_list_to_attr": "mmu_low_cover_filter_pid_list",
      },
      {
        "name": "new_marketing_sense",
        "enable": "{{enable_new_marketing_sense_filter}}",
        "enable_cart_photo_filter_attr": "enable_cart_photo_filter",
        "enable_hetu_filter_attr": "enable_hetu_filter",
        "enable_audit_tag_filter_attr": "enable_audit_tag_filter",
      },
      {
        "name": "video_quality_assessment_filter",
        "enable": "{{enable_video_quality_assessment_filter}}",
        "skip_video_quality_assessment_follow_author_attr": "skip_video_quality_assessment_follow_author"
      },
      {
        "name": "not_audit_level_b",
        "enable": "{{enable_not_audit_photo_filter}}",
        "cold_start_breakout_score_threshold_attr": "cold_start_breakout_score_threshold",
        "high_fans_threshold_attr": "not_audit_filter_high_fans_threshold",
        "ctr_threshold_attr": "not_audit_filter_ctr_threshold",
        "higher_action_threshold_attr": "not_audit_filter_higher_action_threshold",
        "need_high_quality_mmu_score_attr": "not_audit_need_high_quality_mmu_score",
        "high_quality_mmu_map_strs_attr": "not_audit_high_quality_mmu_map_strs",
        "skip_not_audit_zero_value_attr": "skip_not_audit_zero_value",
        "skip_not_audit_follow_author_attr": "skip_not_audit_follow_author"
      },
      {
        "name": "quetionaire_info_filter",
        "enable": "{{enable_questionnaire_info_filter}}",
        "questionaire_info_negtive_rate_threhold_attr": "questionnaire_info_filter_max_negative_rate",
        "questionaire_info_negtive_rate_high_threhold_attr": "questionnaire_info_filter_max_negative_high_rate",
        "questionaire_info_positive_rate_threhold_attr": "questionnaire_info_filter_min_positive_rate",
        "questionaire_info_unsure_rate_threhold_attr": "questionnaire_info_filter_max_undefined_rate",
        "questionaire_info_credible_total_count_attr": "credible_questionnaire_total_count",
        "questionaire_thompson_filter_attr": "questionaire_thompson_filter",
        "questionaire_use_global_data_attr": "questionaire_use_global_data",
        "questionaire_filter_neg_weight_attr": "questionaire_filter_neg_weight",
        "questionaire_filter_pos_weight_attr": "questionaire_filter_pos_weight",
        "questionaire_filter_unsure_weight_attr": "questionaire_filter_unsure_weight",
        "questionaire_filter_click_weight_attr": "questionaire_filter_click_weight",
        "questionaire_filter_unclick_weight_attr": "questionaire_filter_unclick_weight",
        "questionaire_info_replace_topk_result": "explore_questionaire_info_replace_topk_result",
        "questionaire_info_topk_level_threshold_attr": "explore_ques_info_topk_level_threshold",
        "questionaire_info_audit_level_threshold_attr": "explore_ques_info_audit_level_threshold", 
      },
      {
        "name": "risk_man_risk_photo", # 监管需求，对高等级用户只出高等级视频
        "enable": "{{enable_filter_risk_man_risk_photo}}",
        "tmp_be_risk_user_attr": "is_tmp_risk_user",
        "explore_user_risk_min_attr": "explore_user_risk_min",
        "black_white_change_risk": "enable_black_white_change_risk"
      },
      {
        "name": "need_shuffle_photo", # 监管需求，对特定视频进行打散，过滤仅为了防止打散做不好
        "enable": "{{enable_filter_need_shuffle_photo}}",
        "tmp_be_shuffle_user_attr": "is_tmp_risk_user",
        "black_white_change_shuffle": "enable_black_white_change_risk"
      },
      {
        "name": "low_real_show",
        "enable": "{{enable_filter_low_real_show}}",
        "low_real_show_threshold": "explore_num_of_sum_real_show",
        "black_hetu_set_low_real_show_attr": "black_hetu_set"
      },
      {
        "name": "low_fans",
        "enable": "{{enable_filter_low_fans}}",
        "low_fans_threshold": "explore_num_of_fans_need_filter",
        "black_hetu_set_low_fans_attr": "black_hetu_set"
      },
      {
        "name": "high_explore_show",
        "enable": "{{enable_filter_high_explore_show}}",
        "rate_of_high_explore_show": "explore_rate_of_show_need_filter",
        "min_show_of_high_explore_show": "explore_min_show_of_show",
        "max_show_of_high_explore_show": "explore_max_show_of_show",
        "black_hetu_set_high_explore_show_attr": "black_hetu_set",
        "enable_ratio_dynamic_adjust_attr": "enable_explore_adjust_rate_of_show_need_filter",
        "page_index_attr": "page_index",
        "refresh_times_attr": "refreshTimes",
        "age_segment_attr": "basic_info_age_segment_v2",
        "ratio_coef_for_first_screen_attr": "explore_rate_of_show_need_filter_adjust_coef_for_first_screen",
        "ratio_coef_for_top3_screen_attr": "explore_rate_of_show_need_filter_adjust_coef_for_top3_screen",
        "ratio_coef_for_below_30_user_attr": "explore_rate_of_show_need_filter_adjust_coef_for_below_30_user"
      },
      {
        "name": "pic_filter_before_admin",
        "enable": "{{enable_pic_filter_before_admin}}",
        "pic_mmu_low_quality_type_map": "pic_mmu_low_quality_type_map",
        "explore_server_show_threshold": "explore_server_show_threshold",
        "explore_ctr_threshold": "explore_ctr_threshold",
      },
      {
        "name": "is_sirius_punish",
        "enable": "{{enable_filter_sirius_punish_photo}}",
      },
      {
        "name": "download_disabled_pic",
        "enable": "{{enable_filter_download_disabled_pic}}",
      },
      {
        "name": "black_photos",
        "enable": "{{enable_filter_black_photo_results}}",
        "black_photos_attr": "black_photos_set"
      },
      {
        "name": "impression_audit_gray_show",
        "enable": "{{enable_impression_audit_gray_show_filter}}",
        "impression_audit_gray_tag_list_attr" : "impression_audit_gray_tag_list",
        "impression_audit_gray_show_limit_attr": "impression_audit_gray_show_limit",
      },
      {
        "name": "high_emp_phtr_filter",
        "enable": "{{enable_emp_phtr_filter}}",
        "emphtr_exemption_for_above11_attr": "emphtr_exemption_for_above11",
        "emp_realshow_show_threshold_attr": "emp_realshow_show_threshold",
        "emphtr_filter_threshold_attr": "emphtr_filter_threshold",
        "enable_hate_cost_attr": "enable_hate_cost",
        "emphtr_filter_ctr_weight_attr": "emphtr_filter_ctr_weight",
        "emphtr_filter_ltr_weight_attr": "emphtr_filter_ltr_weight",
        "emphtr_filter_wtr_weight_attr": "emphtr_filter_wtr_weight",
        "emphtr_filter_ftr_weight_attr": "emphtr_filter_ftr_weight",
        "emphtr_filter_cmtr_weight_attr": "emphtr_filter_cmtr_weight",
        "emphtr_filter_time_weight_attr": "emphtr_filter_time_weight",
        "emphtr_filter_report_weight_attr": "emphtr_filter_report_weight",
        "emphtr_filter_normal_time_weight_attr": "emphtr_filter_normal_time_weight",
        "enable_adpt_threshold_attr": "enable_adpt_threshold",
        "emphtr_filter_threshold_list_attr": "emphtr_filter_threshold_list",
        "enable_hate_count_filter_attr": "enable_hate_count_filter",
        "emp_realshow_show_high_threshold_attr": "emp_realshow_show_high_threshold",
        "emp_hate_cnt_filter_threshold_attr": "emp_hate_cnt_filter_threshold",
        "enable_adpt_threshold_by_realshow_attr": "enable_adpt_threshold_by_realshow",
        "emphtr_filter_adpt_threshold_coeff_max_attr": "emphtr_filter_adpt_threshold_coeff_max",
        "emphtr_filter_adpt_threshold_coeff_min_attr": "emphtr_filter_adpt_threshold_coeff_min",
        "emphtr_filter_adpt_threshold_alpha_attr": "emphtr_filter_adpt_threshold_alpha",
        "emphtr_filter_adpt_threshold_beta_attr": "emphtr_filter_adpt_threshold_beta",
        "emphtr_filter_adpt_threshold_omega_attr": "emphtr_filter_adpt_threshold_omega",
        "emphtr_filter_adpt_threshold_exp_upper_attr": "emphtr_filter_adpt_threshold_exp_upper",
      },
      { # 20 大
        "name": "explore_punish_filter",
        "enable": "{{explore_enable_explore_punish_filter}}"
      },
      { # 20 大
        "name": "explore_punish_city_filter",
        "enable": "{{explore_enable_explore_punish_city_filter}}"
      },
      {
        "name": "negative_thompson_filter",
        "enable": "{{enable_negative_thompson_filter}}",
        "thompson_filter_threshold_attr": "thompson_filter_threshold",
        "enable_interaction_base_attr": "enable_interaction_base",
        "thompson_filter_realshow_divisor_attr": "thompson_filter_realshow_divisor",
        "thompson_filter_ctr_weight_attr": "thompson_filter_ctr_weight",
        "thompson_filter_ltr_weight_attr": "thompson_filter_ltr_weight",
        "thompson_filter_wtr_weight_attr": "thompson_filter_wtr_weight",
        "thompson_filter_ftr_weight_attr": "thompson_filter_ftr_weight",
        "thompson_filter_cmtr_weight_attr": "thompson_filter_cmtr_weight",
        "thompson_filter_time_weight_attr": "thompson_filter_time_weight",
        "thompson_filter_report_weight_attr": "thompson_filter_report_weight",
        "thompson_filter_normal_time_weight_attr": "thompson_filter_normal_time_weight",
        "thompson_filter_enable_skip_low_emphtr_attr": "thompson_filter_enable_skip_low_emphtr",
        "thompson_filter_no_click_weight_attr": "thompson_filter_no_click_weight",
        "thompson_filter_lvtr_weight_attr": "thompson_filter_lvtr_weight",
        "thompson_filter_low_emphtr_threshold_attr": "thompson_filter_low_emphtr_threshold"
      },
      {
        "name": "explore_boost_photo_filter",
        "enable": "{{explore_enable_boost_photo_filter}}",
        "boost_photo_reason_list_attr": "boost_photo_reason_list",
      },
      {
        "name": "duration_random_filter",
        "enable": "{{explore_enable_duration_random_filter}}",
        "ignore_reason_attr": "duration_random_ignore_reasons",
        "default_cut_off_ratio_attr": "explore_duration_random_default_cut_off_ratio",
        "adjust_cut_off_ratio_attr": "explore_duration_random_adjust_cut_off_ratio",
        "enable_random_cut_off_attr": "explore_duration_random_enable_random_cut_off",
        "lt_longview_ratio_threshold_attr": "explore_duration_random_lt_longview_ratio_threshold",
        "sharp_change_confidence_threshold_attr": "explore_duration_random_sharp_change_confidence_threshold",
      },
      {
        "name": "duration_emp_watchtime_sample_filter",
        "enable": "{{explore_enable_duration_emp_watchtime_sample_filter}}",
        "duration_sample_threshold_attr": "explore_duration_sample_filter_threshold_str",
        "duration_sample_base_number_attr": "explore_duration_sample_filter_base_number",
        "duration_sample_multi_number_attr": "explore_duration_sample_filter_multi_number",
      },
      {
        "name": "xtab_life_index_filter",
        "enable": "{{explore_enable_xtab_life_index_filter}}",
        "key_hetu_category_list_attr": "key_hetu_category_list",
        "key_hetu_category_l2_list_attr": "xhs_hetu_type",
        "key_hetu_blacklist_category_list_attr": "key_hetu_blacklist_category_list",
        "key_hetu_blacklist_category_l2_list_attr": "key_hetu_blacklist_category_l2_list",
        "enable_key_hetu_category_filter_attr": "explore_enable_key_hetu_category_filter",
        "enable_key_hetu_category_l2_filter_attr": "explore_enable_key_hetu_category_l2_filter",
        "enable_key_hetu_blacklist_category_filter_attr": "explore_enable_key_hetu_blacklist_category_filter",
        "enable_key_hetu_blacklist_category_l2_filter_attr": "explore_enable_key_hetu_blacklist_category_l2_filter",
      },
      {
        "name": "lifecate_pic_filter",
        "enable": "{{explore_enable_lifecate_pic_filter}}",
        "explore_lifecate_hetu1_list_attr": "lifecate_hetu1_set",
      },
      {
        "name": "pic_exptag_filter",
        "enable": "{{explore_enable_pic_exptag_filter}}",
        "pic_exptag_filter_str_attr": "explore_pic_exptag_filter_map_str",
      },
      {
        "name": "fresh_request_filter", # 生活tab首刷策略
        "enable": "{{explore_enable_fresh_request_filter}}",
        "is_fresh_request_attr": "is_fresh_request",
        "show_threshold_attr": "explore_fresh_request_show_threshold",
      },
      {
        "name": "multi_audit_gray_filter",
        "enable": "{{enable_explore_multi_audit_gray_filter}}",
        "audit_gray_count_threshold_attr": "explore_multi_audit_gray_filter_count_threshold",
        "multi_audit_gray_days_limit_attr": "explore_multi_audit_gray_filter_days_limit",
      },
      {
        "name": "audit_rule_adjust_filter",
        "enable": "{{explore_enable_audit_rule_adjust_filter}}",
        "audit_rule_adjust_tags_attr": "explore_audit_rule_adjust_tags",
      },
      {
        "name": "merchant_holdout_filter",
        "enable": "{{explore_enable_merchant_holdout_filter}}",
        "merchant_author_list_ptr_attr": "merchant_author_list_ptr",
        "enable_filter_living_merchant_photo": "explore_enable_filter_living_merchant_photo",
        "enable_filter_living_merchant_author": "explore_enable_filter_living_merchant_author",
        "enable_high_negative_feedback_rate_filter": "explore_enable_high_negative_feedback_rate_filter",
        "photo_real_show_count_thres": "explore_photo_real_show_count_thres",
        "photo_hate_like_rate_thres": "explore_photo_hate_like_rate_thres",
      },
      { 
        "name": "specified_group_gray_audit_filter",
        "enable": "{{enable_specified_group_explore_all_gray_filter}}",
        "audit_impression_limit_list_attr": "audit_impression_limit_list",
        "audit_high_hot_limit_list_attr": "audit_high_hot_limit_list",
        "audit_topk_limit_list_attr": "audit_topk_limit_list",
        "is_satisfy_user": "is_la_correct_user"
      },
      {
        "name": "audit_user_experiment_level_filter",
        "enable": "{{explore_enable_audit_user_experiment_level_filter}}",
        "audit_user_experiment_level_map_attr": "explore_audit_user_experiment_level_map_str"
      },
      {
        "name": "personified_author_filter",
        "enable": "{{explore_enable_personified_author_filter}}",
        "personified_author_filter_flag": "explore_personified_author_filter_flag"
      },
      # 影视价值验证holdout
      {
        "name": "movie_copyright_holdout_filter",
        "enable": "{{explore_enable_movie_copyright_holdout_filter}}",
        "filter_bits_list_attr": "movie_copyright_filter_bits_list"
      },
      # 明星运营验证holdout
      {
        "name": "star_holdout_filter",
        "enable": "{{explore_enable_star_holdout_filter}}",
        "filter_bits_list_attr": "star_holdout_filter_bits_list"
      },
      # 年轻人垂类验证
      {
        "name": "young_inc_tags_holdout_filter",
        "enable": "{{explore_enable_young_inc_tags_holdout_filter}}",
        "young_inc_category_list_attr": "young_inc_category_list",
        "young_inc_category_hetu_list_attr": "young_inc_category_hetu_list",
        "filter_flag_attr": "explore_young_inc_tags_filter_flag",
        "filter_ratio_attr": "explore_young_inc_tags_filter_ratio",
        "filter_prime_attr": "explore_young_inc_tags_filter_prime",
        "upload_time_limit_attr": "explore_young_inc_tags_filter_upload_time_limit"
      },
      {
        "name": "be_black_author_filter",
        "enable": "{{explore_enable_be_black_author_filter}}",
        "be_black_list_attr": "be_black_list"
      },
      # 光合验证
      {
        "name": "light_inc_holdout_filter",
        "enable": "{{explore_enable_light_inc_holdout_filter}}",
      },
      # 粉段验证
      {
        "name": "fans_count_random_holdout_filter",
        "enable": "{{explore_enable_fans_count_random_holdout_filter}}",
        "filter_ratio_attr": "explore_fans_count_random_holdout_filter_ratio",
        "filter_prime_attr": "explore_fans_count_random_holdout_filter_prime",
        "fans_bucket_list_attr": "explore_fans_count_random_holdout_filter_fans_bucket_list"
      },
      # 生活 tab 首刷优化
      {
        "name": "low_comment_cnt_filter",
        "enable": "{{explore_enable_low_comment_cnt_filter}}",
        "low_comment_cnt_threshold_attr": "low_comment_cnt_filter_threshold",
        "is_first_page_attr" : "page"
      },
      # 诱导互动作品过滤
      {
        "name": "audit_hack_photo_filter",
        "enable": "{{explore_enable_audit_hack_photo_filter}}",
        "audit_hack_tag_set_attr": "audit_hack_tags_str",
        "min_show_attr": "audit_hack_photo_filter_min_show",
        "max_ltr_attr": "audit_hack_photo_filter_max_ltr",
        "max_wtr_attr": "audit_hack_photo_filter_max_wtr",
        "max_cmtr_attr": "audit_hack_photo_filter_max_cmtr"
      },
      {
        "name": "audit_cold_review_level_filter",
        "enable": "{{enable_audit_cold_review_level_filter}}",
        "audit_cold_review_level_black_tag_set_attr": "audit_cold_review_level_black_tag_set_str",
        "audit_cold_review_level_top_list_inferior_tag_set_attr": "audit_cold_review_level_top_list_inferior_tag_set_str",
        "audit_cold_review_level_top_list_inferior_vv_limit_attr": "audit_cold_review_level_top_list_inferior_vv_limit",
        "explore_enable_audit_cold_review_level_for_all_user_attr": "explore_enable_audit_cold_review_level_for_all_user",
        "explore_enable_not_audit_cold_review_level_filter_attr": "explore_enable_not_audit_cold_review_level_filter",
        "page_attr": "page_index",
        "not_audit_cold_review_max_page_threshold_attr": "explore_not_audit_cold_review_max_page_threshold",
      },
      {
        "name": "user_reco_neg_photo_filter",
        "enable": "{{explore_enable_user_reco_neg_photo_filter}}",
        "reco_neg_photo_list_attr": "reco_neg_photo_id_filter_list"
      },
      {
        "name": "data_set_tags_filter",
        "enable": "{{explore_enable_data_set_tags_filter}}",
        "filter_tags_list_attr": "data_set_tags_filter_tags_list"
      },
      {
        "name": "hetu_tag_filter",
        "enable": "{{explore_enable_hetu_tag_filter}}",
        "hetu_v2_whitelist_categories_l1_list_attr": "hetu_v2_whitelist_categories_l1_list",
        "hetu_v2_whitelist_categories_l2_list_attr": "hetu_v2_whitelist_categories_l2_list",
        "hetu_v2_blacklist_categories_l1_list_attr": "hetu_v2_blacklist_categories_l1_list",
        "hetu_v2_blacklist_categories_l2_list_attr": "hetu_v2_blacklist_categories_l2_list",
        "enable_low_vv_filter_attr": "explore_enable_low_vv_filter",
        "explore_vv_3d_threshold_attr": "explore_vv_3d_threshold",
        "explore_vv_3d_attr": "explore_vv_3d",
        "is_zero_play_user_attr": "is_zero_play_user",
        "enable_only_zero_play_filter_attr": "explore_enable_only_zero_play_filter"
      },
      # 显式判断新回人群逻辑删除 to_be_delete = 2024-09-20
      {
        "name": "hetu_sim_cluster_id_filter",
        "enable": "{{return explore_enable_hetu_sim_cluster_id_filter == 1 and uIsExploreTnuCrowdUser == 1}}",
        "hetu_sim_cluster_id_blacklist_attr": "hetu_sim_cluster_id_blacklist"
      },
      {
        "name": "quality_audit_filter",
        "enable": "{{explore_enable_quality_audit_filter_final}}",
        "filter_tags_list_attr": "quality_audit_filter_tags_list"
      },
      {
        "name": "quality_control_filter",
        "enable": "{{explore_enable_quality_control_filter}}",
        "is_first_page_attr": "page",
        "explore_quality_control_threshold_attr": "explore_quality_control_threshold",
        "explore_audit_gray_weight_attr": "explore_audit_gray_weight",
        "explore_mmu_score_gray_weight_attr": "explore_mmu_score_gray_weight",
        "impression_audit_gray_tag_list_attr": "impression_audit_gray_tag_list"
      },
      {
        "name": "empirical_xtr",
        "enable": "{{enable_explore_empirical_xtr_filter}}",
        "explore_realshow_threshold_attr": "explore_realshow_threshold",
        "explore_upload_date_threshold_attr": "explore_upload_date_threshold",
        "explore_emp_ctr_dropout_rate_attr": "explore_emp_ctr_dropout_rate",
        "explore_emp_playtime_dropout_rate_attr": "explore_emp_playtime_dropout_rate",
        "explore_emp_cross_dropout_rate_attr": "explore_emp_cross_dropout_rate",
        "emp_ctr_threshold_str_attr": "explore_emp_ctr_threshold_str",
        "emp_playtime_threshold_str_attr": "explore_emp_playtime_threshold_str",
        "emp_cross_threshold_str_attr": "explore_emp_cross_threshold_str",
      },
      {
        "name": "dynamic_xtr_filter",
        "enable": "{{enable_explore_dynamic_xtr_filter}}",
        "dynamic_xtrs_threshold_list_attr": "dynamic_xtrs_threshold_list",
        "dynamic_filter_old_photo_days_attr": "explore_filter_old_photo_days",
        "dynamic_filter_save_follow_author_attr": "enable_explore_save_follow_author"
      },
      {
        "name": "ecom_intent_score_filter",
        "enable": "{{explore_enable_ecom_intent_score_filter}}",
        "ecom_intent_score_threshold_attr": "ecom_intent_score_threshold",
        "explore_enable_ecom_intent_score_for_all_user_attr": "explore_enable_ecom_intent_score_for_all_user"
      },
      {
        "name": "hetu_author_category_holdout_filter",
        "enable": "{{explore_enable_hetu_author_category_holdout_filter}}",
        "fans_count_limit_attr": "explore_hetu_author_category_holdout_filter_fans_count_limit",
        "hetu_author_category_list_attr": "explore_hetu_author_category_holdout_filter_list"
      },
      {
        "name": "pic_low_quality_filter",
        "enable": "{{explore_enable_pic_low_quality_filter}}",
        "pic_low_quality_filter_thresh_attr": "pic_low_quality_filter_thresh_list",
        "explore_pic_low_quality_tag_list_attr": "explore_pic_low_quality_tag_list",
      },
      {
        "name": "pic_low_cost_filter",
        "enable": "{{explore_enable_pic_low_cost_filter}}",
        "explore_low_cost_pic_max_cnt_attr": "explore_low_cost_pic_max_cnt",
        "explore_low_cost_pic_cnt_mode_attr": "explore_low_cost_pic_cnt_mode",
      },
      {
        "name": "pic_hack_act_filter",
        "enable": "{{explore_enable_pic_hack_act_filter}}",
        "explore_hack_act_pic_tags_attr": "explore_hack_act_pic_tags_str",
        "explore_hack_act_pic_types_attr": "explore_hack_act_pic_types_str",
        "explore_hack_act_pic_max_cnt_attr": "explore_hack_act_pic_max_cnt",
        "explore_hack_act_pic_cnt_mode_attr": "explore_hack_act_pic_cnt_mode",
      },
      {
        "name": "pic_low_act_filter",
        "enable": "{{explore_enable_pic_low_act_filter}}",
        "explore_pic_low_act_vv_thres_attr": "explore_pic_low_act_vv_thres",
        "explore_pic_low_act_rate_thres_attr": "explore_pic_low_act_rate_thres"
      },
      {
        "name": "pic_sexy_filter",
        "enable": "{{explore_enable_pic_sexy_filter}}",
        "sexy_pic_max_cnt_attr": "explore_sexy_pic_max_cnt",
        "sexy_pic_cnt_mode_attr": "explore_sexy_pic_cnt_mode",
      },
      {
        "name": "data_set_tags_bit_filter",
        "enable": "{{explore_enable_data_set_tags_bit_filter}}",
        "filter_bits_list_attr": "data_set_tags_bit_filter_bits_list"
      },
      {
        "name": "product_block_filter",  # 产品需求，只在单列分发，在双列屏蔽，老板拍板直接推全
        "enable": "{{enable_product_block_filter}}",
        "list_index_attr": "product_block_filter_list_index",
        "bit_index_attr": "product_block_filter_bit_index",
      },
      {
        "name": "merchant_cart_holdout_filter", # 挂车视频过滤
        "enable": "{{explore_enable_merchant_cart_holdout_filter}}",
        "enable_first_fresh_merchant_filter_attr": "enable_first_fresh_merchant_cart_filter_attr",
        "page_index_attr": "page_index",
      },
      {
        "name": "high_photo_count_author_filter",
        "enable": "{{explore_enable_high_photo_count_author_filter}}",
        "high_photo_count_author_map_ptr_attr": "high_upload_photo_author_map_ptr",
        "realshow_threshold_attr": "explore_high_photo_count_author_photo_realshow_threshold",
        "pos_neg_ratio_coeff_attr": "explore_high_photo_count_author_pos_neg_ratio_coeff",
        "fans_count_limit_attr": "explore_high_photo_count_author_fans_count_limit"
      },
      {
        "name": "douyin_author_holdout_filter",
        "enable": "{{explore_enable_douyin_author_holdout_filter}}",
        "filter_flag_attr": "explore_douyin_author_holdout_filter_flag",
        "fans_count_limit_attr": "explore_douyin_author_holdout_filter_fans_count_limit",
        "hetu_author_category_list_attr": "explore_douyin_author_holdout_filter_list",
        "douyin_10w_author_set_ptr_attr": "douyin_10w_author_set_ptr",
        "douyin_100w_author_set_ptr_attr": "douyin_100w_author_set_ptr"
      },
      {
        "name": "short_play_filter",
        "enable": "{{explore_enable_short_play_filter}}",
        "explore_short_play_threshold_attr": "explore_short_play_threshold",
        "explore_short_play_smooth_attr": "explore_short_play_smooth"
      },
      {
        "name": "continuous_hitting_filter",
        "enable": "{{explore_enable_continuous_hitting_filter}}",
        "explore_hitting_threshold_attr": "explore_continuous_hitting_threshold",
        "enable_hetu_five_white_list": "explore_continuous_hitting_filter_hetu_five_white_list",
        "hetu_five_whitelist_attr": "explore_continuous_hitting_hetu_level_five_whitelist_str",
        "realshow_unclick_item_cnt_attr": "continuous_hitting_filter_hetu_cnt_common_attr",
        "realshow_unclick_item_id_attr": "continuous_hitting_filter_hetu_id_common_attr",
      },
      # 中长视频holdout
      {
        "name": "mid_long_video_holdout_filter",
        "enable": "{{explore_enable_mid_long_video_holdout_filter}}",
        "duration_lowerbound_attr": "explore_mid_long_video_holdout_filter_duration_lowerbound",
        "duration_upperbound_attr": "explore_mid_long_video_holdout_filter_duration_upperbound",
        "filter_tags_list_attr": "explore_mid_long_video_holdout_filter_tags_list"
      },
      # 生产类别过滤
      {
        "name": "produce_type_filter",
        "enable": "{{explore_enable_produce_type_filter}}",
        "produce_magic_type_filter_flag_attr": "explore_produce_magic_type_filter_flag",
        "produce_need_filter_magic_type_list_attr": "explore_produce_need_filter_magic_type_list"
      },
      {
        "name": "high_global_emphtr_filter",
        "enable": "{{explore_enable_high_global_emphtr_filter}}",
        "global_emp_realshow_show_threshold_attr": "global_emp_realshow_show_threshold",
        "global_emphtr_filter_threshold_attr": "global_emphtr_filter_threshold"
      },
      {
        "name": "first_slide_impression_audit_filter",
        "enable": "{{explore_enable_first_slide_impression_audit_filter}}",
        "page_attr": "page_index",
        "enable_only_first_slide_filter_attr": "enable_only_first_slide_filter",
        "impression_audit_whitelist_categories_list_attr": "impression_audit_whitelist_categories_list",
        "need_filter_photo_type_list_attr": "need_filter_photo_type_list"
      },
      # 搬运视频holdout
      {
        "name": "repost_photo_filter",
        "enable": "{{explore_enable_repost_photo_filter}}",
      },
      # 首刷退场过过滤
      {
        "name": "first_refresh_filter",
        "enable": "{{explore_enable_first_refresh_filter}}",
        "filter_list_size_attr": "explore_first_refresh_filter_list_size",
        "page_index_attr": "page_index",
        "refresh_times_attr": "refreshTimes",
        "enable_global_data_attr": "explore_first_refresh_filter_enable_global_data"
      },
      # mmu营销感视频过滤
      {
        "name": "mmu_merchant_photo_filter",
        "enable": "{{explore_enable_mmu_merchant_photo_filter}}",
        "mmu_merchant_filter_black_list_attr": "mmu_merchant_photo_blacklist_str",
        "page_index_attr": "page_index",
      },
      # 首刷观感审营销感视频过滤
      {
        "name": "first_fresh_ad_impression_audit_filter",
        "enable": "{{explore_enable_fresh_impression_audit_filter}}",
        "fresh_ad_impression_audit_blacklist_list_attr": "fresh_ad_impression_audit_blacklist_str",
        "page_attr": "page_index",
      },
      # 营销感过滤
      {
        "name": "sirius_distribution_photo_filter",
        "enable": "{{explore_enable_sirius_distribution_photo_filter}}",
        "filter_tags_list_attr": "explore_sirius_distribution_photo_tags_list",
        "hack_author_young_mutual_filter_ratio": "explore_hack_author_young_mutual_filter_ratio",
        "hack_author_young_mutual_tolerate_score": "hack_author_young_mutual_tolerate_score",
        "hack_author_young_mutual_tolerate_score_min": "explore_hack_author_young_mutual_tolerate_score_min",
        "hack_author_young_mutual_tolerate_score_max": "explore_hack_author_young_mutual_tolerate_score_max",
        "hack_author_induce_interaction_filter_ratio": "explore_hack_author_induce_interaction_filter_ratio",
        "hack_author_induce_interaction_tolerate_score": "hack_author_induce_interaction_tolerate_score",
        "hack_author_induce_interaction_tolerate_score_min": "explore_hack_author_induce_interaction_tolerate_score_min",
        "hack_author_induce_interaction_tolerate_score_max": "explore_hack_author_induce_interaction_tolerate_score_max",
      },
      # 15天内高热非优质过滤
      {
        "name": "proximate_audit_hot_high_bad_filter",
        "enable": "{{enable_proximate_audit_hot_high_bad_filter}}",
        "page_index_attr": "page_index",
        "audit_hot_high_tag_level_attr": "audit_hot_high_tag_level",
        "proximate_hot_high_page_threshold_attr": "proximate_audit_hot_high_page_threshold",
      },
      # 封面未进审过滤
      {
        "name": "no_cover_audit_filter",
        "enable": "{{explore_enable_no_cover_audit_photo_filter}}",
        "default_thres_attr": "explore_no_cover_audit_photo_filter_default_thres",
        "filter_map_str_attr": "explore_no_cover_audit_photo_filter_map_str",
        "realshow_thres_attr": "explore_no_cover_audit_photo_filter_realshow_thres",
        "enable_dynamic_realshow_thres_attr": "explore_no_cover_audit_photo_filter_enable_dynamic_realshow_thres",
        "realshow_per_day_attr": "explore_no_cover_audit_photo_filter_realshow_per_day",
        "enable_protogenetic_advertise_filter_attr": "explore_no_cover_audit_photo_filter_enable_advertise_filter",
        "filter_protogenetic_advertise_list_attr": "explore_no_cover_audit_photo_filter_advertise_list",
      },
      # reason3125 过滤
      {
        "name": "reason_3125_filter",
        "enable": "{{explore_enable_reason_3125_filter}}",
        "hetu_filter_map_str_attr": "explore_reason_3125_hetu_filter_map_str",
        "hetu_default_thres_attr": "explore_reason_3125_hetu_default_thres",
        "cover_view_predict_score_thres_attr": "explore_reason_3125_cover_view_predict_score_thres",
        "marketing_filter_tags_list_attr": "explore_reason_3125_marketing_filter_tags_list",
        "marketing_default_thres_attr": "explore_reason_3125_marketing_default_thres",
        "cover_sense_view_score_version_attr": "explore_cover_sense_view_score_version",
      },
      # 封面审、观感审预估分数过滤
      {
        "name": "cover_sense_view_predict_score_filter",
        "enable": "{{explore_enable_cover_sense_view_score_filter}}",
        "cover_view_predict_score_thres_attr": "explore_filter_cover_view_predict_score_thres",
        "sense_view_predict_score_thres_attr": "explore_filter_sense_view_predict_score_thres",
        "enable_filter_cover_view_missing_score_attr": "enable_explore_filter_cover_view_missing_score",
        "enable_filter_sense_view_missing_score_attr": "enable_explore_filter_sense_view_missing_score",
        "cover_sense_view_score_version_attr": "explore_cover_sense_view_score_version",
        "enable_filter_audit_cover_view_score_attr": "enable_explore_filter_audit_cover_view_score",
        "enable_filter_audit_sense_view_score_attr": "enable_explore_filter_audit_sense_view_score",
      },
      # 原生广告过滤
      {
        "name": "protogenetic_advertise_tags_filter",
        "enable": "{{explore_enable_protogenetic_advertise_tags_filter}}",
        "filter_advertise_list_attr": "protogenetic_advertise_tags_blacklist",
        "filter_advertise_cover_view_score_thres_attr": "explore_filter_advertise_cover_view_score_thres",
        "filter_advertise_audit_cover_level_attr": "explore_filter_advertise_audit_cover_level",
      },
      # live_photo_flag 过滤
      {
        "name": "live_photo_flag_filter",
        "enable": "{{explore_enable_live_photo_flag_filter}}",
        "live_photo_flag_thres_attr": "explore_live_photo_flag_default_thres"
      },
      # 非优质画风 过滤
      {
        "name": "terrible_quality_style_filter",
        "enable": "{{explore_enable_terrible_quality_style_filter}}",
      },
      # 视频生命周期过滤
      {
        "name": "emp_xtr_decrease_filter",
        "enable": "{{explore_enable_emp_xtr_decrease_filter}}",
        "emp_xtr_decrease_photo_set_ptr_attr": "explore_emp_xtr_decrease_photo_set_ptr",
        "enable_random_attr": "explore_emp_xtr_decrease_photo_filter_enable_random",
        "filter_ratio_attr": "explore_emp_xtr_decrease_photo_filter_filter_ratio",
      },
      #生命周期汤普森过滤
      {
        "name": "emp_xtr_decrease_tonpson_filter",
        "enable": "{{explore_enable_emp_xtr_tonpson_decrease_filter}}",
        "emp_xtr_decrease_tonpson_photo_map_ptr_attr": "explore_emp_topson_decrease_down_photo_map_ptr",
        "filter_ratio_attr": "explore_emp_xtr_tonpson_decrease_photo_filter_ratio",
      },
      # 新回人群审核基线过滤
      {
        "name": "tnu_impression_audit_bad_filter",
        "enable": "{{return explore_enable_tnu_impression_audit_bad_filter == 1 and uIsExploreTnuCrowdUser == 1}}",
        "tnu_impression_audit_whitelist_attr": "tnu_impression_audit_whitelist"
      },
      # 社区负向作者过滤 @liuhao07
      {
        "name": "negative_aid_filter",
        "enable": "{{explore_enable_negative_aid_filter}}",
        "negative_aid_set_ptr_attr": "negative_aid_set_ptr"
      },
      {
        "name": "valid_play_ratio_filter",
        "enable": "{{explore_enable_valid_play_ratio_filter}}",
        "explore_click_threshold_attr": "explore_click_threshold",
        "explore_upload_days_threshold_attr": "explore_upload_days_threshold",
        "explore_valid_play_ratio_threshold_attr": "explore_valid_play_ratio_threshold",
        "explore_valid_play_ratio_dropout_rate_attr": "explore_valid_play_ratio_dropout_rate"
      },
      # emotions_pic 过滤
      {
        "name": "emotions_pic_filter",
        "enable": "{{explore_enable_emotions_pic_filter}}",
        "emotions_pic_filter_ratio_attr": "explore_emotions_pic_filter_ratio",
        "emotions_pic_show_count_threshold_attr": "explore_emotions_pic_show_count_threshold"
      },
      # 图文负向作者过滤
      {
        "name": "pic_author_filter",
        "enable": "{{explore_enable_pic_author_filter}}",
        "author_grade_thresh_attr": "explore_pic_author_grade_thresh",
        "author_punish_cnt_mode_attr": "explore_pic_author_punish_cnt_mode",
        "author_filter_markcode_attr": "explore_pic_author_filter_markcode",
        "author_punish_markcode_attr": "explore_pic_author_punish_markcode"
      },
      {
        "name": "over_distribute_filter",
        "enable": "{{explore_enable_over_distribute_filter}}",
        "explore_show_limit_attr": "explore_over_distribute_filter_show_limit_attr",
        "explore_ctr_limit_attr": "explore_over_distribute_filter_ctr_limit_attr"
      },
      # 连续封面合集过滤 @liuhao07
      {
        "name": "serial_cover_photo_collection_filter",
        "enable": "{{explore_enable_serial_cover_photo_collection_filter}}",
        "photo_collection_pids_set_ptr_attr": "photo_collection_pids_set_ptr"
      },
      # mmu 营销感视频过滤 @liuhao07
      {
        "name": "merchant_hetu_tag_photo_filter",
        "enable": "{{explore_enable_merchant_hetu_tag_photo_filter}}",
        "show_count_limit_attr": "explore_merchant_hetu_tag_photo_filter_show_limit_count",
        "is_random_filter_attr": "explore_merchant_hetu_tag_photo_filter_is_random",
        "random_filter_percent_attr": "explore_merchant_hetu_tag_photo_filter_random_percent"
      },
      # 图文生态负向特征过滤：高举报
      {
        "name": "pic_ecology_high_report_filter",
        "enable": "{{enable_explore_pic_ecology_high_report_filter}}",
        "explore_pic_ecology_high_report_rate_threshold_attr": "explore_pic_ecology_high_report_rate_threshold",
        "explore_pic_ecology_high_report_count_threshold_attr": "explore_pic_ecology_high_report_count_threshold",
        "pic_ecology_high_report_fans_count_threshold_attr": "explore_pic_ecology_high_report_fans_count_threshold"
      },
      # 图文生态负向特征过滤：高负正反馈率
      {
        "name": "pic_ecology_high_neg_pos_rate_filter",
        "enable": "{{enable_explore_pic_ecology_high_neg_pos_rate_filter}}",
        "explore_pic_ecology_high_neg_pos_rate_threshold_attr": "explore_pic_ecology_high_neg_pos_rate_threshold"
      },
      # 图文生态负向特征过滤：高短播
      {
        "name": "pic_ecology_high_short_play_rate_filter",
        "enable": "{{enable_explore_pic_ecology_high_short_play_rate_filter}}",
        "explore_pic_ecology_high_short_play_rate_threshold_attr": "explore_pic_ecology_high_short_play_rate_threshold",
        "explore_pic_ecology_neg_rate_threshold_attr": "explore_pic_ecology_neg_rate_threshold"
      },
      # 图文生态负向特征过滤：bad avg view ( 高短播且高次均 )
      {
        "name": "pic_ecology_bad_avg_time_filter",
        "enable": "{{enable_explore_pic_ecology_bad_avg_time_filter}}",
        "explore_pic_ecology_bad_view_time_for_short_play_rate_threshold_attr": "explore_pic_ecology_bad_view_time_for_short_play_rate_threshold",
        "explore_pic_ecology_bad_view_time_threshold_attr": "explore_pic_ecology_bad_view_time_threshold"
      },
      # 图文生态负向特征过滤：综合互动
      {
        "name": "pic_ecology_mix_interact_rate_filter",
        "enable": "{{enable_explore_pic_ecology_mix_interact_rate_filter}}",
        "pic_ecology_interact_rate_threshold_attr": "explore_pic_ecology_interact_rate_threshold",
        "pic_ecology_interact_avg_view_time_threshold_attr": "explore_pic_ecology_interact_avg_view_time_threshold",
        "pic_ecology_interact_vv_threshold_attr": "explore_pic_ecology_interact_vv_threshold"
      },
      {
        "name": "minority_photo_filter",
        "enable": "{{explore_enable_minority_photo_filter}}",
        "filter_bits_list_attr": "minority_photo_filter_bits_list",
        "pass_num_limit_attr": "explore_minority_photo_filter_pass_num_limit"
      },
      {
        "name": "teenager_author_filter",
        "enable": "{{explore_enable_teenager_author_filter}}",
        "show_count_limit_attr": "explore_teenager_author_filter_show_limit_count"
      },
      # 高发布或低质作者过滤
      {
        "name": "pic_ecology_high_release_author_filter",
        "enable": "{{explore_enable_pic_ecology_high_release_author_filter}}",
        "filter_bits_list_attr": "explore_pic_ecology_high_release_author_bits_list"
      },
      # 高删文作者过滤
      {
        "name": "pic_ecology_high_delete_author_filter",
        "enable": "{{explore_enable_pic_ecology_high_delete_author_filter}}",
        "filter_bits_list_attr": "explore_pic_ecology_high_delete_author_bits_list"
      },
      # 图文 mmu hetu tag 过滤
      {
        "name": "pic_mmu_hetu_tag_filter",
        "enable": "{{explore_enable_pic_mmu_hetu_tag_filter}}",
        "mmu_tag_prob_str_attr": "explore_pic_filter_mmu_tag_prob_str",
        "mmu_tag_skip_hv_str_attr": "explore_pic_filter_mmu_tag_skip_hv_str",
        "mmu_tag_vv_thr_str_attr": "explore_pic_filter_mmu_tag_vv_thr_str",
      },
      # 作者黑名单过滤
      {
        "name": "induced_author_black_list_filter",
        "enable": "{{enable_induced_author_black_list_filter}}",
        "enable_filter_induced_curiosity_author_list_attr": "explore_enable_filter_induced_curiosity_author",
        "induced_curiosity_authorid_blacklist_attr": "induced_curiosity_authorid_blacklist",
        "enable_filter_bad_audit_list_attr": "explore_enable_filter_bad_audit_author",
        "bad_audit_authorid_blacklist_attr": "bad_audit_authorid_blacklist"
      },
      # xtr 动作过滤
      {
        "name": "lower_emp_xtr_act_filter",
        "enable": "{{enable_explore_lower_emp_xtr_act_filter}}",
        "enbale_filter_low_like_act_attr": "enbale_explore_filter_low_like_act",
        "enbale_filter_low_follow_act_attr": "enbale_explore_filter_low_follow_act",
        "low_real_show_count_threshold_attr": "explore_low_real_show_count_threshold",
        "low_empltr_act_filter_threshold_attr": "explore_low_empltr_act_filter_threshold",
        "low_empftr_act_filter_threshold_attr": "explore_low_empftr_act_filter_threshold",
        "limit_for_like_empctr_threshold_attr": "explore_limit_for_like_empctr_threshold",
        "limit_for_follow_empctr_threshold_attr": "explore_limit_for_follow_empctr_threshold",
      },
      # 大模型过滤 liuhao07
      {
        "name": "llm_negative_photos_filter",
        "enable": "{{explore_enable_llm_negative_photos_filter}}",
        "teenager_filter_tag_map_str_attr": "explore_llm_negative_photos_filter_teenager_tag_map_str",
        "filter_tag_map_str_attr": "explore_llm_negative_photos_filter_tag_map_str",
        "is_teenager_attr": "is_teenager",
        "show_count_limit_map_str_attr": "explore_llm_negative_photos_filter_show_count_limit_map_str",
        "enable_filter_no_impression_audit_result_attr": "explore_llm_negative_photos_filter_impression_audit_result",
        "filter_impression_audit_level_attr": "explore_llm_negative_photos_filter_impression_audit_level",
        "report_count_coeff_attr": "explore_llm_negative_photos_filter_report_count_coeff",
        "report_ratio_coeff_attr": "explore_llm_negative_photos_filter_report_ratio_coeff"
      },
      # 图文 data_set_tags_bit 过滤
      {
        "name": "pic_data_set_tags_bit_filter",
        "enable": "{{explore_enable_pic_data_set_tags_bit_filter}}",
        "pic_filter_bits_str_attr": "explore_pic_filter_data_set_tags_bits_str",
        "pic_punish_bits_str_attr": "explore_pic_punish_data_set_tags_bits_str",
        "skip_filter_mark_cod_str_attr": "explore_pic_skip_filter_mark_cod_str",
        "punish_vv_thresh_attr": "explore_pic_punish_data_set_tags_bit_vv_thresh",
        "punish_filter_prob_attr": "explore_pic_punish_data_set_tags_bit_filter_prob"
      },
      # 热点退场过滤
      {
        "name": "hot_point_pid_filter",
        "enable": "{{explore_enable_photo_hot_point_filter}}",
        "hot_point_pid_filter_default_days_attr": "explore_hot_point_pid_filter_default_days",
        "hot_point_pid_filter_bits_list_attr":"explore_hot_point_pid_filter_bits_gen_list",
        "hot_point_pid_filter_hetu_str_attr":"explore_hot_point_pid_filter_hetu_str",
        "hot_point_pid_filter_up_days_attr":"explore_hot_point_pid_filter_up_days",
        "hot_point_pid_filter_collect_rate_limit_attr":"explore_hot_point_pid_filter_collect_rate_limit",
        "hot_point_pid_filter_follow_rate_limit_attr":"explore_hot_point_pid_filter_follow_rate_limit",
      },
      # 图文安全审过滤
      {
        "name": "pic_secure_grade_filter",
        "enable": "{{explore_enable_pic_secure_grade_filter}}",
        "secure_grade_filter_code_attr": "explore_pic_secure_grade_filter_code_str",
        "secure_grade_punish_code_attr": "explore_pic_secure_grade_punish_code_str",
        "skip_audit_b_second_tags_attr": "explore_pic_secure_grade_filter_skip_audit_b_second_tags_str"
      },
      #  外流店铺分过滤
      {
        "name": "author_shop_score_filter",
        "enable": "{{explore_enable_author_shop_score_filter}}",
        "author_shop_score_limit_attr": "explore_author_shop_score_filter_limit_count",
        "author_shop_zero_protect_attr": "explore_enable_author_shop_zero_protect"
      },
      #  外流商品分过滤
      {
        "name": "author_goods_score_filter",
        "enable": "{{explore_enable_author_goods_score_filter}}",
        "author_goods_score_limit_attr": "explore_author_goods_score_filter_limit_count",
        "author_goods_zero_protect_attr": "explore_enable_author_goods_zero_protect"
      },
      # 机审过时退场
      {
        "name": "audit_overtime_filter",
        "enable": "{{explore_enable_audit_overtime_filter}}",
        "upload_days_limit_attr": "explore_overtime_filter_upload_days_limit"
      },  
      # 营销号单图低综合互动率过滤
      {
        "name": "pic_mix_interact_rate_filter",
        "enable": "{{explore_enable_pic_mix_interact_rate_filter}}",
        "base_vv_threshold_attr": "explore_pic_mix_interact_rate_filter_base_vv_threshold",
        "author_filter_mark_cod_str_attr": "explore_pic_mix_interact_rate_filter_author_mark_cod_str",
        "interact_rate_thresholds_str_attr": "explore_pic_mix_interact_rate_thresholds_str",
        "vv_thresholds_str_attr": "explore_pic_mix_interact_rate_filter_vv_thresholds_str",
        "filter_probs_str_attr": "explore_pic_mix_interact_rate_filter_probs_str",
      },
      # 视频负反馈比例过滤
      {
        "name": "emp_neg_feedback_filter",
        "enable": "{{explore_enable_emp_neg_feedback_filter}}",
        "emp_neg_feedback_photo_set_ptr_attr": "explore_emp_neg_feedback_photo_set_ptr",
        "filter_ratio_attr": "explore_emp_neg_feedback_filter_ratio",
      },
      {
        "name": "high_report_photo_filter",
        "enable": "{{explore_enable_high_report_photo_filter}}",
        "realshow_threshold_attr": "explore_high_report_photo_filter_realshow_threshold",
        "repoprt_ratio_limit_attr": "explore_high_report_photo_filter_report_ratio_limit",
      },
      # 营销号泛单图过滤
      {
        "name": "marketing_static_video_filter",
        "enable": "{{explore_enable_marketing_static_video_filter}}",
        "static_video_tag_id_attr": "explore_static_video_hetu_tag_id",
        "static_video_tag_prob_thd_attr": "explore_static_video_hetu_tag_prob_thd",
        "marketing_mark_cod_str_attr": "explore_marketing_static_video_filter_mark_cod_str",
        "base_vv_threshold_attr": "explore_marketing_static_video_filter_base_vv_threshold",
        "interact_rate_thresholds_str_attr": "explore_marketing_static_video_filter_interact_rate_thresholds_str",
        "vv_thresholds_str_attr": "explore_marketing_static_video_filter_vv_thresholds_str",
        "skip_market_mark_attr": "explore_static_video_skip_market_mark",
        "filter_probs_str_attr": "explore_marketing_static_video_filter_probs_str"
      },
      {
        "name": "coldstart_holdout_filter",
        "enable": "{{explore_enable_coldstart_holdout_filter}}",
        "nebula_thanos_realshow_limit_attr": "explore_coldstart_holdout_filter_nebula_thanos_realshow_limit",
        "guarantee_rank_limit_attr": "explore_coldstart_holdout_filter_guarantee_rank_limit",
        "enable_filter_if_double_shield_attr": "explore_coldstart_holdout_filter_enable_filter_if_double_shield"
      },
      {
        "name": "sexy_induce_author_filter",
        "enable": "{{return explore_enable_sexy_induce_author_filter == 1 and uIsExploreTnuCrowdUser == 1}}",
        "sexy_induce_photo_set_ptr_attr": "sexy_induce_photo_set_ptr"
      },
      {
        "name": "poor_quality_author_filter",
        "enable": "{{explore_enable_poor_quality_author_filter}}",
        "enable_filter_by_gaofen_signs_uids": "explore_enable_filter_by_gaofen_signs_uids",
        "gaofen_signs_uids_set_ptr_attr": "gaofen_signs_uids_set_ptr",
        "enable_filter_by_hierarchy_label_uids": "explore_enable_filter_by_hierarchy_label_uids",
        "hierarchy_label_uids_map_ptr_attr": "hierarchy_label_uids_map_ptr",
        "hierarchy_label_uids_filter_ratio": "explore_hierarchy_label_uids_filter_ratio",
        "enable_filter_by_hack_author_uids": "explore_enable_filter_by_hack_author_uids",
        "hack_author_uids_map_ptr_attr": "hack_author_uid_map_ptr",
        "hack_author_filter_types_and_ratios_str": "explore_hack_author_filter_types_and_ratios_str",
        "hack_author_high_p_tolerate_score": "hack_author_high_p_tolerate_score",
        "hack_author_high_p_tolerate_score_min": "explore_hack_author_high_p_tolerate_score_min",
        "hack_author_high_p_tolerate_score_max": "explore_hack_author_high_p_tolerate_score_max",
      },
      # 发现页前几屏强控
      {
        "name": "topn_screen_filter",
        "enable": "{{explore_enable_topn_screen_filter}}",
        "impression_audit_level_limit_attr": "explore_topn_screen_filter_impression_audit_level_limit",
        "high_hot_audit_level_limit_attr": "explore_topn_screen_filter_high_hot_audit_level_limit",
        "emp_ntpr_limit_attr": "explore_topn_screen_filter_emp_ntpr_limit",
        "explore_today_vv_attr": "explore_today_vv",
        "active_days_avg_vv_attr": "active_days_avg_vv",
        "filter_ratio_attr": "explore_topn_screen_filter_filter_ratio",
        "avg_vv_coeff_attr": "explore_topn_screen_filter_avg_vv_coeff",
        "page_index_attr": "page_index",
        "hetu_l1_blacklist_list_attr": "hetu_l1_blacklist_list",
        "hetu_l2_blacklist_list_attr": "hetu_l2_blacklist_list",
        "hetu_l3_blacklist_list_attr": "hetu_l3_blacklist_list",
        "enable_topn_merchant_cart_filter_ratio_attr": "explore_enable_topn_merchant_cart_filter_ratio",
        "manjiao_markcode_blacklist_list_attr": "manjiao_markcode_blacklist_list",
        "enable_first_index_control_attr":"explore_topn_screen_filter_enable_first_index_control_ratio"
      },
      #  房聘需求
      {
        "name": "fangpin_aid_filter",
        "enable": "{{explore_enable_fangpin_aid_filter}}"
      },
      # 图文低成本营销号过滤
      {
        "name": "pic_low_cost_marketing_filter",
        "enable": "{{explore_enable_pic_low_cost_marketing_filter}}",
        "low_cost_markcode_attr": "explore_low_cost_markcode_str",
        "yanghao_markcode_attr": "explore_yanghao_markcode_str"
      },
      # 小程序 holdout 过滤
      {
        "name": "plc_business_type_filter",
        "enable": "{{explore_enable_plc_business_type_filter}}",
        "filter_tags_list_attr": "explore_plc_business_type_filter_tags_list"
      },
      #  冷启价值验证屏蔽
      {
        "name": "valuable_photo_open_filter",
        "enable": "{{explore_enable_valuable_photo_open_filter}}",
        "valuable_rules_kconf_key_attr": "explore_valuable_open_filter_kconf_key"
      },
      # 图文劣质作者过滤
      {
        "name": "pic_liezhi_author_filter",
        "enable": "{{explore_enable_pic_liezhi_author_filter}}",
        "author_liezhi_pic_count_thresh_attr": "explore_author_liezhi_pic_count_thresh",
        "author_fans_count_thresh_attr": "explore_pic_liezhi_author_filter_fans_count_thresh"
      },
      # 图文新星作者过滤
      {
        "name": "pic_xinxing_author_filter",
        "enable": "{{explore_enable_pic_xinxing_author_filter}}"
      },
      # 运营话题hashtag作品过滤
      {
        "name": "pic_operation_hash_tag_filter",
        "enable": "{{explore_enable_pic_operation_hash_tag_filter}}",
        "pic_operation_filter_hash_tagid_attr" : "pic_operation_filter_hash_tagid_ptr"
      },
      # 图文封面审劣质过滤
      {
        "name": "pic_bad_cover_filter",
        "enable": "{{explore_enable_pic_bad_cover_filter}}",
        "pic_bad_cover_tags_attr": "explore_pic_bad_cover_tags_str"
      },
      # 图文新回审劣质过滤
      {
        "name": "pic_audit_cold_review_level_filter",
        "enable": "{{explore_enable_pic_audit_cold_review_level_filter}}",
        "filter_audit_cold_review_level_str_attr": "explore_pic_filter_audit_cold_review_level_str"
      },
      # 长实况图过滤
      {
        "name": "pic_long_live_photo_filter",
        "enable": "{{explore_enable_pic_long_live_photo_filter}}",
        "pic_long_live_photo_vv_thresh_attr": "explore_pic_long_live_photo_vv_thresh",
        "pic_long_live_photo_duration_thresh_attr": "explore_pic_long_live_photo_duration_thresh"
      },
      # 热点holdout
      {
        "name": "hot_spot_holdout_filter",
        "enable": "{{explore_enable_hot_spot_holdout_filter}}",
        "filter_level_list_attr": "hot_spot_filter_level_list",
        "filter_source_list_attr": "hot_spot_filter_source_list"
      },
      # 封面二维码内容过滤 使用 hetu 判断
      {
        "name": "cover_qr_code_filter",
        "enable": "{{explore_enable_cover_qr_code_filter}}",
      },
      # 封面敏感词过滤 使用 mem data 判断
      {
        "name": "cover_sensitive_word_filter",
        "enable": "{{explore_enable_cover_sensitive_word_filter}}",
        "cover_sensitive_word_ptr_attr": "illegal_word_pids_ptr"
      },
       #  作者冷启价值验证屏蔽
      {
        "name": "valuable_author_photo_open_filter",
        "enable": "{{explore_enable_valuable_author_photo_open_filter}}",
        "valuable_author_photo_rules_name_attr": "explore_valuable_author_photo_rules_name"
      },
      # 外流封面内容不一致过滤 使用hetu_tag 判断
      {
        "name": "explore_cover_video_not_correlation_filter",
        "enable": "{{explore_enable_cover_video_not_correlation_filter}}",
      },
      # 根据sid过滤负反馈视频
      {
        "name": "explore_user_hate_sid_video_filter",
        "enable": "{{explore_enable_user_hate_sid_video_filter}}",
        "recent_hate_sid_list_attr": "recent_hate_sid_list",
        "is_use_first_level_sid_attr": "explore_is_use_first_level_sid"
      },
      # 社区问卷过滤
      {
        "name": "community_survey_filter",
        "enable": "{{explore_enable_community_survey_filter}}",
        "survey_filter_markcode_str_attr": "explore_filter_community_survey_markcode_str",
        "survey_markcode_2_cert_ratio_threshold_str_attr": "explore_filter_survey_markcode_2_cert_ratio_threshold_str",
        "survey_markcode_2_cert_cnt_threshold_str_attr": "explore_filter_survey_markcode_2_cert_cnt_threshold_str",
      },
      {
        "name": "author_living_whitelist_filter",
        "enable": True,
        "author_whitelist_attr": "author_living_whitelist",
        "author_blacklist_attr": "author_living_blacklist",
      },
      # 1222舆情相关tagnex过滤
      {
        "name": "public_opinion_tagnex_filter",
        "enable": "{{return explore_enable_public_opinion_tagnex_filter == 1 or (explore_enable_public_opinion_tagnex_filter_reflux == 1 and uIsRefluxCrowdUser == 1)}}",
        "public_opinion_tagnex_str_attr": "explore_filter_public_opinion_tagnex_str",
      },
      # 性暗示作品过滤
      {
        "name": "sexually_photo_filter",
        "enable": "{{return (explore_enable_new_reflux_sexually_photo_filter == 1 and (is_new_device == 1 or is_growth_reflux == 1))}}",
        "sexually_manjiao_markcode_str_attr": "sexually_manjiao_markcode_str",
      },
      # 舆情灰产兜底策略，常规状态无需开启
      {
        "name": "high_hate_report_filter",
        "enable": "{{explore_enable_high_hate_report_filter}}",
        "recent_real_show_thres_attr": "explore_filter_recent_real_show_thres_attr",
        "recent_hate_ratio_thres_attr": "explore_filter_recent_hate_ratio_thres_attr",
        "recent_report_ratio_thres_attr": "explore_filter_recent_report_ratio_thres_attr",
        "recent_hate_count_thres_attr": "explore_filter_recent_hate_count_thres_attr",
        "recent_report_count_thres_attr": "explore_filter_recent_report_count_thres_attr",
        "real_show_thres_attr": "explore_filter_real_show_thres_attr",
        "hate_ratio_thres_attr": "explore_filter_hate_ratio_thres_attr",
        "report_ratio_thres_attr": "explore_filter_report_ratio_thres_attr",
        "hate_count_thres_attr": "explore_filter_hate_count_thres_attr",
        "report_count_thres_attr": "explore_filter_report_count_thres_attr"
      },
      {
        "name": "topk_audit_not_pass_filter",   # 兜底策略，只保留topk_audit_level为3 审核通过视频
        "enable": "{{enable_retr_filter_downgrade}}",
      },
      # 未成年年龄概率过滤
      {
        "name": "teenager_age_prob_filter",
        "enable": "{{explore_enable_teenager_age_prob_filter}}",
        "age_is_teenager_attr": "is_teenager",
        "teenager_age_prob_map_str_attr": "explore_teenager_age_prob_filter_map_str",
        "teenager_age_prob_weight_attr": "explore_teenager_age_prob_filter_weight",
      },
      {
        "name": "content_dup",  # 必须放在最后一个
        "enable": "{{enable_explore_content_dup_filter}}",
        "filter_content_type_list_attr": "filter_content_type_list",
        "dup_cluster_id_attr": "dup_cluster_id",
        "pic_and_selfdup_id_attr": "pic_and_selfdup_id",
        "sim_remove_dup_id_attr": "sim_remove_dup_id",
        "skip_high_xtr_dup_filter_attr": "skip_high_xtr_dup_filter",
        "skip_high_hot_quality_pic_attr": "skip_high_hot_quality_pic",
        "explore_skip_high_hot_quality_attr": "explore_skip_high_hot_quality",
        "skip_dup_realshow_threshold_attr": "skip_dup_realshow_threshold",
        "skip_dup_watchtime_threshold_attr": "skip_dup_watchtime_threshold",
        "skip_dup_fvtr_threshold_attr": "skip_dup_fvtr_threshold",
        "skip_dup_ctr_threshold_attr": "skip_dup_ctr_threshold",
        "filter_content_type_list_for_pic_attr": "filter_content_type_list_for_pic"
      },
   ]

  @property
  def truncation_map(self) -> dict:
    return {
      "196": 360,
      "6300": 350,
      "674": 140,
      "3099": 700,
      "88": 420,
      "191": 137,
      "4499": 700,
      "221": 350,
      "921": 720,
      "941": 1050,
      "942": 1050,
      "943": 1050,
      "945": 1050,
      "946": 700,
      "3061": 1260,
      "3064": 560,
      "3070": 840,
      "3073": 840,
      "6301": 280,
      "6315": 350,
      "676": 295,
      "677": 280,
      "932": 350,
      "3077": 420,
      "1836": 345,
      "3052": 255,
      "1899": 420,
      "3060": 456,
      "649": 350,
      "675": 360,
      "3069": 700,
      "3068": 500,
      "6311": 140,
      "678": 350,
      "3084": 420,
      "10001": 350,
      "10023": 350,
      "10022": 350,
      "10024": 350,
      "3106": 500,
      "212": 200,
      "219": 300,
      "10037": 1500,
      "10033": 1500,
      "10043": 1500, # 生活tab召回扩量, 放宽截断量
      "13020": 1500,
      "13022": 1500, # 生活tab召回扩量, 放宽截断量
      "13007": 1500, # 生活tab召回扩量, 放宽截断量
      "4050": 1500,  # 放大截断限制进行收益摸底
      "13027": 1500, # 放大截断限制进行收益摸底
      "9999": 4000, # 兜底策略召回reason的filter上线数量，新增truncation_map注意保持一致
    }

  def process(self):
    self.flow \
      .split_string(
        input_common_attr = "topk_audit_second_level_white_tags",
        output_common_attr = "topk_audit_white_tag_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True,
      ) \
      .split_string(
        input_common_attr = "topk_audit_second_level_black_tags",
        output_common_attr = "topk_audit_black_tag_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True,
      ) \
      .split_string(
        input_common_attr = "high_hot_audit_user_sexy_interest_exemption_tags",
        output_common_attr = "user_sexy_interest_exemption_tag_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True,
      ) \
      .split_string(
        input_common_attr = "high_hot_audit_user_sexy_interest_exemption_ages",
        output_common_attr = "user_sexy_interest_exemption_age_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True,
      ) \
      .split_string(
        input_common_attr = "high_hot_audit_user_sexy_interest_exemption_city_levels",
        output_common_attr = "user_sexy_interest_exemption_city_level_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = False,
      ) \
      .split_string(
        input_common_attr = "impression_audit_user_sexy_interest_extra_filter_tags",
        output_common_attr = "user_sexy_interest_extra_filter_tag_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True,
      ) \
      .split_string(
        input_common_attr = "high_hot_audit_second_level_white_tags",
        output_common_attr = "high_hot_audit_white_tag_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True,
      ) \
      .split_string(
        input_common_attr = "impression_audit_gray_tags",
        output_common_attr = "impression_audit_gray_tag_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True,
      ) \
      .split_string(
        input_common_attr = "high_hot_audit_second_level_black_tags",
        output_common_attr = "high_hot_audit_black_tag_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True,
      ) \
      .split_string(
        input_common_attr = "follow_author_ignore_exptags",
        output_common_attr = "follow_author_ignore_exptag_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True,
      ) \
      .split_string(
        input_common_attr = "impression_audit_second_level_white_tags",
        output_common_attr = "impression_audit_white_tag_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True,
      ) \
      .split_string(
        input_common_attr = "impression_audit_second_level_black_tags",
        output_common_attr = "impression_audit_black_tag_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True,
      ) \
      .split_string(
        input_common_attr = "explore_dynamic_xtr_filter_threshold_str",
        output_common_attr = "dynamic_xtrs_threshold_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_double = True,
      ) \
      .split_string(
        input_common_attr = "valuable_author_types",
        output_common_attr = "valuable_author_type_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
      ) \
      .split_string(
        input_common_attr = "filter_content_types",
        output_common_attr = "filter_content_type_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True,
      ) \
      .split_string(
        input_common_attr="filter_content_types_for_pic",
        output_common_attr="filter_content_type_list_for_pic",
        delimiters=",",
        trim_spaces=True,
        skip_empty_tokens=True,
        parse_to_int=True,
      ) \
      .split_string(
        input_common_attr = "boost_photo_reasons",
        output_common_attr = "boost_photo_reason_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True,
      ) \
      .split_string(
        input_common_attr = "explore_duration_random_ignore_reasons",
        output_common_attr = "duration_random_ignore_reasons",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True, 
      ) \
      .split_string(
        input_common_attr = "audit_impression_limit_str",
        output_common_attr = "audit_impression_limit_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True,
      ) \
      .split_string(
        input_common_attr = "audit_high_hot_limit_str",
        output_common_attr = "audit_high_hot_limit_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True,
      ) \
      .split_string(
        input_common_attr = "audit_topk_limit_str",
        output_common_attr = "audit_topk_limit_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True,
      ) \
      .split_string(
        input_common_attr = "explore_key_hetu_categories",
        output_common_attr = "key_hetu_category_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True,
      ) \
      .split_string(
        input_common_attr = "explore_key_hetu_blacklist_categories",
        output_common_attr = "key_hetu_blacklist_category_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True,
      ) \
      .split_string(
        input_common_attr = "explore_key_hetu_blacklist_categories_l2",
        output_common_attr = "key_hetu_blacklist_category_l2_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True,
      ) \
      .split_string(
        input_common_attr = "explore_audit_rule_adjust_tags_str",
        output_common_attr = "explore_audit_rule_adjust_tags",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True,
      ) \
      .split_string(
        input_common_attr = "emphtr_filter_threshold_list_str",
        output_common_attr = "emphtr_filter_threshold_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_double = True,
      ) \
      .split_string(
        input_common_attr = "explore_produce_need_filter_magic_type_str",
        output_common_attr = "explore_produce_need_filter_magic_type_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True,
      ) \
      .split_string(
        input_common_attr = "explore_no_cover_audit_photo_filter_advertise_str_v2",
        output_common_attr = "explore_no_cover_audit_photo_filter_advertise_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True,
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "_USER_ID_", "as": "user_id"},
          "explore_risk_user_set"
        ],
        export_common_attr = [
          "is_tmp_risk_user"
        ],
        function_name = "CanBeRiskUser",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .explore_memory_data_enrich(
        data_key = "{{explore_face_90_degree_pids}}",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "face_90_degree_pids"
      ) \
      .set_attr_value(
        common_attrs = [
          {
            "name": "filter_upload_type_list",
            "type": "int_list",
            "value": [27],
          },
        ],
      ) \
      .split_string(
        input_common_attr = "explore_young_inc_category_list_str",
        output_common_attr = "young_inc_category_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .split_string(
        input_common_attr = "explore_young_inc_category_hetu_list_str",
        output_common_attr = "young_inc_category_hetu_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .split_string(
        input_common_attr = "explore_fans_count_random_holdout_filter_fans_bucket_str",
        output_common_attr = "explore_fans_count_random_holdout_filter_fans_bucket_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .split_string(
        input_common_attr = "explore_data_set_tags_filter_tags_list_str",
        output_common_attr = "data_set_tags_filter_tags_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .split_string(
        input_common_attr = "explore_data_set_tags_bit_filter_bits_list_str",
        output_common_attr = "data_set_tags_bit_filter_bits_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .if_("explore_enable_movie_copyright_holdout_filter == 1") \
        .split_string(
          input_common_attr = "explore_movie_copyright_filter_bits_list_str",
          output_common_attr = "movie_copyright_filter_bits_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
      .end_() \
      .if_("explore_enable_star_holdout_filter == 1") \
        .split_string(
          input_common_attr = "explore_star_holdout_filter_bits_list_str",
          output_common_attr = "star_holdout_filter_bits_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
      .end_() \
      .split_string(
        input_common_attr = "explore_pic_ecology_high_release_author_bits_list_str",
        output_common_attr = "explore_pic_ecology_high_release_author_bits_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .split_string(
        input_common_attr = "explore_pic_ecology_high_delete_author_bits_list_str",
        output_common_attr = "explore_pic_ecology_high_delete_author_bits_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .split_string(
        input_common_attr = "explore_hetu_v2_whitelist_categories_l1",
        output_common_attr = "hetu_v2_whitelist_categories_l1_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .split_string(
        input_common_attr = "explore_hetu_v2_whitelist_categories_l2",
        output_common_attr = "hetu_v2_whitelist_categories_l2_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .split_string(
        input_common_attr = "explore_hetu_v2_blacklist_categories_l1",
        output_common_attr = "hetu_v2_blacklist_categories_l1_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .split_string(
        input_common_attr = "explore_hetu_v2_blacklist_categories_l2",
        output_common_attr = "hetu_v2_blacklist_categories_l2_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .split_string(
        input_common_attr = "explore_tnu_impression_audit_whitelist",
        output_common_attr = "tnu_impression_audit_whitelist",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .split_string(
        input_common_attr = "explore_hetu_sim_cluster_id_blacklist",
        output_common_attr = "hetu_sim_cluster_id_blacklist",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .split_string(
        input_common_attr = "explore_quality_audit_filter_tags_list_str_final",
        output_common_attr = "quality_audit_filter_tags_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .split_string(
        input_common_attr = "explore_hetu_author_category_holdout_filter_list_str",
        output_common_attr = "explore_hetu_author_category_holdout_filter_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .split_string(
        input_common_attr = "explore_pic_low_quality_tag_str",
        output_common_attr = "explore_pic_low_quality_tag_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .split_string(
        input_common_attr = "pic_low_quality_filter_thresh_list_str",
        output_common_attr = "pic_low_quality_filter_thresh_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_double = True
      ) \
      .split_string(
        input_common_attr = "explore_douyin_author_holdout_filter_list_str",
        output_common_attr = "explore_douyin_author_holdout_filter_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .split_string(
        input_common_attr = "explore_mid_long_video_holdout_filter_tags_list_str",
        output_common_attr = "explore_mid_long_video_holdout_filter_tags_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .split_string(
        input_common_attr = "explore_hot_point_pid_filter_bits_list",
        output_common_attr = "explore_hot_point_pid_filter_bits_gen_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .split_string(
        input_common_attr = "explore_impression_audit_whitelist_categories",
        output_common_attr = "impression_audit_whitelist_categories_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .split_string(
        input_common_attr = "explore_need_filter_photo_type_categories",
        output_common_attr = "need_filter_photo_type_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .split_string(
        input_common_attr = "explore_topn_hetu_l1_blacklist_list",
        output_common_attr = "hetu_l1_blacklist_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .split_string(
        input_common_attr = "explore_topn_hetu_l2_blacklist_list",
        output_common_attr = "hetu_l2_blacklist_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .split_string(
        input_common_attr = "explore_topn_hetu_l3_blacklist_list",
        output_common_attr = "hetu_l3_blacklist_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .split_string(
        input_common_attr = "explore_topn_manjiao_markcode_blacklist_list",
        output_common_attr = "manjiao_markcode_blacklist_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .if_("explore_enable_minority_photo_filter == 1") \
        .split_string(
          input_common_attr = "explore_minority_photo_filter_bits_list",
          output_common_attr = "minority_photo_filter_bits_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
      .end_() \
      .if_("enable_explore_mmu_merchant_hetu_tag_id == 1") \
        .split_string(
          input_common_attr = "mmu_merchant_tag_black_tags_str",
          output_common_attr = "mmu_merchant_tag_black_tags_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True,
        ) \
        .enrich_attr_by_light_function(
          import_item_attr = [
            {"name": "hetu_tag_level_info__hetu_tag", "as": "attrs"},
          ],
          import_common_attr = [
            {"name": "mmu_merchant_tag_black_tags_list", "as": "attr_list"},
          ],
          export_item_attr = [
            {"name": "is_in_set", "as": "is_merchant_hetu_tag_id"},
          ],
          function_name = "AttrListIsInSet",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .enrich_attr_by_light_function(
          item_list_from_attr = "explore_recent_play_list",
          import_item_attr = [
            {"name": "hetu_tag_level_info__hetu_tag", "as": "attrs"},
          ],
          import_common_attr = [
            {"name": "mmu_merchant_tag_black_tags_list", "as": "attr_list"},
          ],
          export_item_attr = [
            {"name": "is_in_set", "as": "is_merchant_hetu_tag_id"},
          ],
          function_name = "AttrListIsInSet",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .if_("explore_ranking_diversity_enable_standard_explore_realshow_pid_list == 1") \
          .enrich_attr_by_light_function(
            item_list_from_attr = "standard_explore_realshow_pid_list",
            import_item_attr = [
              {"name": "hetu_tag_level_info__hetu_tag", "as": "attrs"},
            ],
            import_common_attr = [
              {"name": "mmu_merchant_tag_black_tags_list", "as": "attr_list"},
            ],
            export_item_attr = [
              {"name": "is_in_set", "as": "is_merchant_hetu_tag_id"},
            ],
            function_name = "AttrListIsInSet",
            class_name = "ExploreLightFunctionSetV2",
          ) \
        .end_() \
      .end_() \
      .if_("enable_explore_gen_fangpin_aid == 1") \
        .cast_attr_type(
          attr_type_cast_configs=[
            {
              "to_type": "string",
              "from_item_attr": "author__id",
              "to_item_attr": "aid_to_string"
            }
          ]
        ) \
        .lookup_kconf (
          kconf_configs = [{
            "kconf_key": "reco.live2.recruitAuthorFilterMap",
            "value_type": "map_string_double",
            "lookup_attr": "aid_to_string",
            "output_attr": "fangpin_aid_filter_ratio",
            "is_common_attr": False,
          }]
        )\
      .end_() \
      .if_("explore_enable_reason_3125_filter == 1") \
        .split_string(
          input_common_attr = "explore_reason_3125_marketing_filter_tags_list_str",
          output_common_attr = "explore_reason_3125_marketing_filter_tags_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
      .end_() \
      .if_("explore_enable_plc_business_type_filter == 1") \
        .split_string(
          input_common_attr = "explore_plc_business_type_filter_tags_str",
          output_common_attr = "explore_plc_business_type_filter_tags_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
      .end_() \
      .if_("explore_enable_fetch_rank_neg_photo == 1") \
        .split_string(
          input_common_attr = "rank_neg_photo_id_list_str",
          output_common_attr = "rank_neg_photo_id_filter_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
      .end_() \
      .if_("explore_enable_fetch_rerank_neg_photo == 1", to_be_delete = "date=2024-05-29;committer=liuhao07") \
        .split_string(
          input_common_attr = "rerank_neg_photo_id_list_str",
          output_common_attr = "rerank_neg_photo_id_filter_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
      .end_() \
      .if_("explore_enable_sirius_distribution_photo_filter == 1") \
        .split_string(
          input_common_attr = "explore_sirius_distribution_photo_tags_list_str",
          output_common_attr = "explore_sirius_distribution_photo_tags_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
      .end_() \
      .if_("enable_explore_hack_author_high_p_tolerate_score == 1") \
        .select_list_values(
          index_attr = "explore_hack_author_high_p_tolerate_score_index",
          list_values = [ 
            {"from": "uHackHighpToleranceScoreListList", "to": "hack_author_high_p_tolerate_score"},
          ],  
          is_common_attr=True
        ) \
      .end_() \
      .if_("enable_explore_hack_author_young_mutual_tolerate_score == 1") \
        .select_list_values(
          index_attr = "explore_hack_author_young_mutual_tolerate_score_index",
          list_values = [ 
            {"from": "uYoungMutualLikeToleranceScoreListList", "to": "hack_author_young_mutual_tolerate_score"},
          ],  
          is_common_attr=True
        ) \
      .end_() \
      .if_("enable_explore_hack_author_induce_interaction_tolerate_score == 1") \
        .select_list_values(
          index_attr = "explore_hack_author_induce_interaction_tolerate_score_index",
          list_values = [ 
            {"from": "uInduceInteractionToleranceScoreListList", "to": "hack_author_induce_interaction_tolerate_score"},
          ],  
          is_common_attr=True
        ) \
      .end_() \
      .if_("explore_enable_protogenetic_advertise_tags_filter == 1") \
        .if_("enable_explore_filter_low_active_customization_advertise_and_impression_audit == 1 and is_explore_new_la_user == 1") \
          .copy_attr(
            attrs = [{
              "from_common": "explore_protogenetic_advertise_tags_blacklist_str_low_active",
              "to_common": "explore_protogenetic_advertise_tags_blacklist_str"
            }, {
              "from_common": "explore_enable_fresh_impression_audit_filter_low_active",
              "to_common": "explore_enable_fresh_impression_audit_filter"
            }]
          ) \
        .end_() \
        .split_string(
          input_common_attr = "explore_protogenetic_advertise_tags_blacklist_str",
          output_common_attr = "protogenetic_advertise_tags_blacklist",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
      .end_() \
      .if_("explore_enable_hot_spot_holdout_filter == 1") \
        .split_string(
          input_common_attr = "explore_hot_spot_filter_level_list_str",
          output_common_attr = "hot_spot_filter_level_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
        .split_string(
          input_common_attr = "explore_hot_spot_filter_source_list_str",
          output_common_attr = "hot_spot_filter_source_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
      .end_() \
      .pack_common_attr(
        input_common_attrs = [
          "rank_neg_photo_id_filter_list",
          "rerank_neg_photo_id_filter_list"
        ],
        output_common_attr = "reco_neg_photo_id_filter_list",
        deduplicate = True,
      ) \
      .if_("explore_enable_continuous_hitting_filter == 1") \
        .enrich_attr_by_light_function(
          item_list_from_attr = "explore_realshow_click_common_list",
          import_common_attr = [
            "explore_realshow_click_timestamp_common_list",
            "explore_click_common_list",
            "explore_realshow_hetu_five_common_list",
            {"name": "explore_continuous_hitting_window_size", "as": "timestamp_window_thred"},
            {"name": "explore_continuous_hitting_realshow_num_limit", "as": "realshow_num_limit"},
          ],
          export_common_attr = [
            "continuous_hitting_filter_hetu_id_common_attr",
            "continuous_hitting_filter_hetu_cnt_common_attr",
          ],
          function_name = "CalculateRealshowUnclickCnt",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_pic_author_filter_enable_specific_age_seg_diff == 1 and basic_info_age_segment_v2 ~= nil") \
        .split_string(
          input_common_attr = "explore_pic_author_filter_specific_age_seg_str",
          output_common_attr = "explore_pic_author_filter_specific_age_seg_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
        .find_value(
          input = "{{explore_pic_author_filter_specific_age_seg_list}}",
          value = "{{basic_info_age_segment_v2}}",
          result = "is_pic_author_filter_specific_age_seg"
        ) \
        .if_("is_pic_author_filter_specific_age_seg == 1") \
          .copy_attr(
            attrs = [
              {
                "from_common": "explore_pic_author_filter_markcode_for_specific_age_seg",
                "to_common": "explore_pic_author_filter_markcode"
              },
              {
                "from_common": "explore_pic_author_punish_markcode_for_specific_age_seg",
                "to_common": "explore_pic_author_punish_markcode"
              },
            ]
          ) \
        .end_() \
      .end_() \
      .if_("explore_enable_pic_liezhi_author_filter_elder == 1 and (basic_info_age_segment_v2 or 0) >= explore_pic_liezhi_author_filter_elder_age_thresh") \
        .copy_attr(
          attrs = [
            {
              "from_common": "explore_enable_pic_liezhi_author_filter_elder",
              "to_common": "explore_enable_pic_liezhi_author_filter"
            },
            {
              "from_common": "explore_author_liezhi_pic_count_thresh_elder",
              "to_common": "explore_author_liezhi_pic_count_thresh"
            },
          ]
        ) \
      .end_() \
      .if_("enable_author_living_whitelist_filter == 1") \
        .pack_common_attr(
          input_common_attrs = [
            "author_living_whitelist",
            "author_living_whitelist2",
            "author_living_whitelist3",
            "author_living_whitelist4",
            "author_living_whitelist5",
          ],
          output_common_attr = "author_living_whitelist",
        ) \
      .else_() \
        .set_attr_value(
          common_attrs = [
            {
              "name": "author_living_whitelist",
              "type": "int_list",
              "value": [],
            }
          ]
        ) \
      .end_() \
      .if_("enable_author_living_blacklist_filter == 1") \
        .pack_common_attr(
          input_common_attrs = [
            "author_living_blacklist",
            "author_living_blacklist2",
            "author_living_blacklist3",
            "author_living_blacklist4",
            "author_living_blacklist5",
          ],
          output_common_attr = "author_living_blacklist",
        ) \
      .else_() \
        .set_attr_value(
          common_attrs = [
            {
              "name": "author_living_blacklist",
              "type": "int_list",
              "value": [],
            }
          ]
        ) \
      .end_() \
      .get_kconf_params(
        kconf_configs = [
          {
            "kconf_key": "reco.explore.product_block_filter",
            "json_path": "enable",
            "default_value": True,
            "export_common_attr": "enable_product_block_filter",
          },
          {
            "kconf_key": "reco.explore.product_block_filter",
            "json_path": "list_index",
            "default_value": 0,
            "export_common_attr": "product_block_filter_list_index",
          },
          {
            "kconf_key": "reco.explore.product_block_filter",
            "json_path": "bit_index",
            "default_value": 27,
            "export_common_attr": "product_block_filter_bit_index",
          },
        ],
      ) \
      .pack_item_attr(
        item_source = {
          "reco_results": False,
          "common_attr": ["recent_hate_pid_list"]
        },
        mappings = [{
          "aggregator": "concat",
          "from_item_attr": "explore_sid",
          "to_common_attr": "recent_hate_sid_list"
        }]
      ) \
      .switch_("explore_filter_mode") \
        .case_(1) \
          .explore_retrieval_filter(
            name = "explore_retr_filter_skip",
            traceback = True,
            user_info_ptr_attr = "user_info_ptr",
            item_attr_map = self.item_attr_map,
            filters = self.filters,
            truncation_map = {
              "10040": 2000,
              "9999": 4000
            }
          ) \
        .case_(2) \
          .explore_retrieval_filter(
            name = "explore_retr_filter_no_limit",
            traceback = True,
            user_info_ptr_attr = "user_info_ptr",
            item_attr_map = self.item_attr_map,
            filters = self.filters,
            truncation_map = {
              "default": 2000,
              "9999": 4000,
            }
          ) \
        .case_(3) \
          .explore_retrieval_filter(
            name = "explore_retr_filter_limit_5k",
            traceback = True,
            user_info_ptr_attr = "user_info_ptr",
            item_attr_map = self.item_attr_map,
            filters = self.filters,
            truncation_map = {
              "default": 5000
            }
          ) \
        .case_(4) \
          .explore_retrieval_filter(
            name = "explore_retr_filter_part_limit_2k",
            traceback = True,
            user_info_ptr_attr = "user_info_ptr",
            item_attr_map = self.item_attr_map,
            filters = self.filters,
            truncation_map = {
              "3125": 2000,
              "3097": 2000,
              "3099": 2000,
              "3131": 2000,
              "9999": 4000,
            }
          ) \
        .default_() \
          .explore_retrieval_filter(
            name = "explore_retr_filter",
            traceback = True,
            user_info_ptr_attr = "user_info_ptr",
            item_attr_map = self.item_attr_map,
            filters = self.filters,
            truncation_map = self.truncation_map,
          ) \
      .end_() \
      .if_("enable_temp_upload_time_filter == 1") \
        .filter_by_kconf_list(
          enable_white = True,
          enable_black = False,
          kconf_key = "reco.grpr.antiHackLiveStreamConfig",
          white_list_name = "photo_aid_white_list_kconf_name_list",
          filter_item_attr = "author__id",
          select_item = {
            "join": "and",
            "filters": [{
              "attr_name": "upload_time",
              "select_if": ">",
              "compare_to": "{{temp_upload_time_filter_start}}",
            }, {
              "attr_name": "upload_time",
              "select_if": "<=",
              "compare_to": "{{temp_upload_time_filter_end}}",
            }]
          }
        ) \
      .end_() \
      .if_("explore_enable_gen_filter_neg_list == 1") \
        .pack_common_attr(
          input_common_attrs = [
            "audit_hot_cover_level_filter_pid_list",
            "mmu_low_cover_filter_pid_list",
          ],
          output_common_attr = "send_other_pid_list",
        ) \
        .enrich_attr_by_light_function(  
          item_list_from_attr = "send_other_pid_list",
          export_item_attr = [
            "upload_time",
          ],
          function_name = "EmptyFunction",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .enrich_attr_by_light_function(
          item_list_from_attr = "send_other_pid_list",
          import_item_attr = [
            "upload_time",
          ],
          export_item_attr = [
            "photo_age_hour",
          ],
          function_name = "CalcAgeHour",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .filter_by_attr(
          item_list_from_attr = "send_other_pid_list",
          attr_name = "photo_age_hour",
          remove_if = ">",
          compare_to = "{{fresh_photo_threshold}}",
          remove_if_attr_missing = True,
        ) \
        .pack_common_attr(
          input_common_attrs = [
            "impression_audit_bad_pid_list",
            "send_other_pid_list",
          ],
          output_common_attr = "send_neg_pid_list",
        ) \
        .shuffle_list_attr(
          common_attr = "send_neg_pid_list",
        ) \
        .pack_common_attr(
          input_common_attrs = [
            "send_neg_pid_list",
          ],
          output_common_attr = "send_final_pid_list",
          limit_num = "{{send_final_pid_list_num}}",
        ) \
        .enrich_attr_by_light_function(  
          item_list_from_attr = "send_final_pid_list",
          export_item_attr = [
            "author__id",
          ],
          function_name = "EmptyFunction",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .pack_item_attr(
          item_source = {
            "reco_results": False,
            "common_attr": ["send_final_pid_list"],
          },
          mappings = [{
            "from_item_attr": "author__id",
            "to_common_attr": "send_final_aid_list",
            "default_val": 0,
          }]
        ) \
      .end_() \



  def post_process(self) -> None:
    self.flow \
    .pack_item_attr(  # 保存filter结束后（进入粗排）的结果集
      item_source = {
        "reco_results": True
      },
      mappings = [{
        "aggregator": "concat",
        "from_item_attr": "photo_id",
        "to_common_attr": "filter_output_item_key_list"
      }],
    )
