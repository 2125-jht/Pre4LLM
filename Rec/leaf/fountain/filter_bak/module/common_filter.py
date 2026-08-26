ITEM_ATTR_MAP = {
  "photo_id_attr": "photo_id",
  "author_id_attr": "author__id",
  "photo_status_attr": "photo_status",
  "upload_time_attr": "upload_time",
  "upload_type_attr": "upload_type",
  "topk_audit_level_attr": "topk_audit_level",
  "topk_audit_tag_attr": "topk_audit_tag",
  "is_mid_video_photo_attr": "is_mid_video_photo",
  "questionaire_info_exposure_count_attr" : "questionnaire_info__exposure_count",
  "questionaire_info_negative_count_attr" : "questionnaire_info__negative_count",
  "questionaire_info_positive_count_attr" : "questionnaire_info__positive_count",
  "questionaire_info_unsure_count_attr" : "questionnaire_info__unsure_count",
  "explore_questionaire_info_exposure_count_attr" : "explore_questionnaire_info__exposure_count",
  "explore_questionaire_info_negative_count_attr" : "explore_questionnaire_info__negative_count",
  "explore_questionaire_info_positive_count_attr" : "explore_questionnaire_info__positive_count",
  "explore_questionaire_info_unsure_count_attr" : "explore_questionnaire_info__unsure_count",
  "level_hot_online_attr": "content_safety_level_with_namespace__level_hot_online",
  "audit_b_second_tag_attr": "audit_b_second_tag",
  "audit_hot_high_tag_level_attr": "audit_hot_high_tag_level",
  "risk_man_risk_photo_attr": "risk_level",
  "duration_ms_attr": "duration_ms",
  "long_term_photo_attr": "long_term_photo",
  "explore_punish_city_attr": "explore_punish_city",
  "explore_real_show_attr": "explore_stat__real_show_count",
  "explore_negative_attr": "explore_stat__negative_count",
  "explore_like_attr": "explore_stat__like_count",
  "explore_click_attr":"explore_stat__click_count",
  "photo_total_report_count_attr": "explore_stat__report_detail__total_report_count",
  "explore_follow_attr": "explore_stat__follow_count",
  "explore_forward_attr": "explore_stat__forward_count",
  "explore_comment_attr": "explore_stat__comment_count",
  "explore_view_length_sum_attr": "explore_stat__view_length_sum",
  "fountain_real_show_attr": "fountain_stats__real_show_count",
  "fountain_negative_attr": "fountain_stats__negative_count",
  "fountain_like_attr": "fountain_stats__like_count",
  "thanos_real_show_attr": "thanos_stats__real_show_count",
  "thanos_negative_attr": "thanos_stats__negative_count",
  "thanos_like_attr": "thanos_stats__like_count",
  "nebula_real_show_attr": "nebula_stats__real_show_count",
  "nebula_negative_attr": "nebula_stats__negative_count",
  "nebula_like_attr": "nebula_stats__like_count",
  "picture_type_attr": "picture_type",
  "hetu_v2_level_one_tag_list_attr": "hetu_tag_level_info_v2__hetu_level_one",
  "hetu_v3_level_one_tag_list_attr": "hetu_tag_level_info_v2__hetu_level_one",
  "hetu_level_one_tag_list_attr": "hetu_tag_level_info__hetu_level_one",
  "width_attr": "width",
  "height_attr" : "height",
  "high_hot_audit_tag_v2_attr": "high_hot_audit_tag_v2",
  "merchant_item_id_list_attr": "merchant_item_info__item_id_list",
  "audit_user_experiment_level_attr": "audit_user_experiment_level",
  "eyeshot_source_attr": "eyeshot_source",
  "hetu_level_two_tag_list_attr": "hetu_tag_level_info__hetu_level_two",
  "hetu_level_three_tag_list_attr": "hetu_tag_level_info__hetu_level_three",
  "hetu_level_five_tag_list_attr": "hetu_tag_level_info__hetu_level_five",
  "hetu_tag_list_attr": "hetu_tag_level_info__hetu_tag",
  "hetu_face_id_tag_list_attr": "hetu_tag_level_info__hetu_face_id",
  "young_inc_tags_attr": "young_inc_tags",
  "final_cross_section_first_class_id_attr": "final_cross_section_first_class_id",
  "light_inc_photo_flag_attr": "light_inc_photo_flag",
  "author_fans_count_attr": "author__fans_count",
  "high_value_pic_flag_attr": "high_value_pic_flag",
  "data_set_tags_attr": "data_set_tags",
  "photo_dynamic_xtrs_str_attr": "video_cold_start_info__photo_dynamic_xtrs_str",
  "audit_risk_immd_tag_attr": "audit_risk_immd_tag",
  "data_set_tags_bit_attr": "data_set_tags_bit",
  "magic_face_type_attr": "magic_face_type",
  "magic_face_id_attr": "magic_face_id",
}

FILTERS = [
  {
    "name": "not_in_index",
    "enable": True,
  },
  {
    "name": "server_show_aid",
    "enable": True,
    "server_show_aid_list_attr": "browsedAuthorIds",
  },
  {
    "name": "photo_status",
    "enable": True,
  },
  {
    "name": "video_filter",
    "enable": "{{enable_fountain_video_filter}}",
  },
  {
    "name": "over_180_days",
    "enable": "{{enable_fountain_over_days_filter}}",
    "over_days_filter_days_limit_attr": "fountain_over_days_filter_days_limit",
    "entertainment_hetu_tags_attr": "fountain_entertainment_hetu_tag_str",
    "entertainment_hetu_days_limit_attr": "fountain_entertainment_hetu_days_limit_attr",
    "enable_filter_low_like": "enable_fountain_over_days_filter_low_like",
    "low_like_limit_attr": "fountain_over_days_filter_low_like_limit",
    "low_like_days_limit_attr": "fountain_over_days_filter_low_like_days_limit",
    "page_type": "FOUNTAIN",
    "topn_screen_filter_attr": "fountain_over_days_filter_topn_screen_map",
    "enable_filter_by_audit": "enable_fountain_over_days_filter_audit",
    "impression_audit_gray_hours_limit_attr": "fountain_impression_audit_gray_hours_limit",
    "impression_audit_normal_days_limit_attr": "fountain_impression_audit_normal_days_limit",
    "impression_audit_high_quality_days_limit_attr": "fountain_impression_audit_high_quality_days_limit",
    "high_hot_audit_gray_hours_limit_attr": "fountain_high_hot_audit_gray_hours_limit",
    "high_hot_audit_normal_days_limit_attr": "fountain_high_hot_audit_normal_days_limit",
    "high_hot_audit_high_quality_days_limit_attr": "fountain_high_hot_audit_high_quality_days_limit",
  },
  {
    "name": "impression_audit_bad",
    "enable": True,
    "impression_audit_white_tag_list_attr": "impression_audit_white_tags",
    "impression_audit_black_tag_list_attr": "impression_audit_black_tags",
    "level_hot_online_attr": "content_safety_level_with_namespace__level_hot_online",
    "audit_b_second_tag_attr": "audit_b_second_tag",
  },
  {
    "name": "high_hot_audit_bad",
    "enable": True,
    "high_hot_audit_white_tag_list_attr": "high_hot_audit_white_tags",
    "high_hot_audit_black_tag_list_attr": "high_hot_audit_black_tags",
    "audit_hot_high_tag_level_attr": "audit_hot_high_tag_level",
    "explore_operation_c_review_level_attr": "explore_operation_c_review_level",
  },
  {
    "name": "upload_type",
    "enable": True,
    "filter_type_list_attr": "filter_upload_type_lsit",
    "enable_skip_high_value_pic": "fountain_enable_skip_high_value_pic",
    "enable_skip_climbing_high_value_pic": "fountain_enable_skip_climbing_high_value_pic",
  },
  {
    "name": "picture_type",
    "enable": True,
    "filter_type_list_attr": "filter_picture_type_lsit",
    "enable_skip_high_value_pic": "fountain_enable_skip_high_value_pic",
    "enable_skip_climbing_high_value_pic": "fountain_enable_skip_climbing_high_value_pic",
  },
  {
    "name": "zero_duration",
    "enable": "{{fountain_enable_zero_duration_filter}}",
  },
  {
    "name": "short_duration",
    "enable": "{{fountain_enable_short_duration_filter}}",
    "short_duration_limit_attr": "fountain_short_duration_filter_limit"
  },
  {
    "name": "topk_audit_bad",
    "enable": "{{return skip_fountain_topk_audit_filter_new == 0}}",
    "topk_audit_white_tag_list_attr": "topk_audit_white_tags",
    "topk_audit_black_tag_list_attr": "topk_audit_black_tags",
    "topk_audit_bad_recall_filter_attr": "fountain_topk_audit_bad_recall_filter",
    "topk_audit_bad_recall_filter_use_global_attr": "fountain_topk_audit_bad_recall_filter_use_global",
    "topk_audit_bad_recall_filter_credible_ques_cnt_attr": "fountain_topk_audit_bad_recall_filter_credible_ques_cnt",
    "topk_audit_bad_recall_filter_pos_threshold_attr": "fountain_topk_audit_bad_recall_filter_pos_threshold",
    "topk_audit_bad_recall_filter_mode_attr": "fountain_topk_audit_bad_recall_filter_mode",
    "topk_audit_bad_recall_filter_unsure_threshold_attr": "fountain_topk_audit_bad_recall_filter_unsure_threshold",
    "topk_audit_bad_recall_filter_neg_threshold_attr": "fountain_topk_audit_bad_recall_filter_neg_threshold",
    "topk_audit_bad_recall_filter_hate_threshold_attr": "fountain_topk_audit_bad_recall_filter_hate_threshold",
  },
  {
    "name": "mid_video",
    "enable": "{{return fountain_skip_mid_video_photo_filter == 0}}",
  },
  {
    "name": "commerce_extend_index",
    "enable": True,
    "is_high_other_photo_attr": "is_high_other_photo",
  },
  {
    "name": "picture",
    "enable": "{{return skip_fountain_filter_all_picture == 0}}",
  },
  {
    "name": "risk_man_risk_photo",
    "enable": "{{fountain_enable_risk_man_risk_photo_filter}}",
    "explore_user_risk_min_attr": "user_risk_min",
  },
  {
    "name": "impression_audit_gray_show",
    "enable": "{{return fountain_skip_impression_audit_gray_filter == 0}}",
    "impression_audit_gray_show_limit_attr": "fountain_impression_audit_gray_show_limit",
  },
  {
    "name": "photo_life",
    "enable": True,
    "photo_life_max_hours_attr": "photo_life_max_hours",
  },
  { # 20 大
    "name": "explore_punish_city_filter",
    "enable": "{{fountain_enable_explore_punish_city_filter}}"
  },
  {
    "name": "negative_thompson_filter",
    "enable": "{{fountain_enable_negative_thompson_filter}}",
    "thompson_filter_enable_fountain_cnt":"enable_fountain_thompson_filter_use_fountain",
    "thompson_filter_enable_thanos_cnt":"enable_fountain_thompson_filter_use_thanos",
    "thompson_filter_enable_nebula_cnt":"enable_fountain_thompson_filter_use_nebula",
    "thompson_filter_enable_explore_cnt":"enable_fountain_thompson_filter_use_explore",
    "thompson_filter_threshold_attr": "fountain_thompson_filter_threshold",
    "thompson_filter_ctr_weight_attr" : "fountain_thompson_filter_ctr_weight",
    "thompson_filter_ltr_weight_attr" : "fountain_thompson_filter_ltr_weight",
    "thompson_filter_wtr_weight_attr" : "fountain_thompson_filter_wtr_weight",
    "thompson_filter_ftr_weight_attr" : "fountain_thompson_filter_ftr_weight",
    "thompson_filter_cmtr_weight_attr" : "fountain_thompson_filter_cmtr_weight",
    "thompson_filter_time_weight_attr" : "fountain_thompson_filter_time_weight",
    "thompson_filter_normal_time_weight_attr" : "fountain_thompson_filter_normal_time_weight",
    "thompson_filter_report_weight_attr" : "fountain_thompson_filter_report_weight",
    "thompson_filter_realshow_weight_attr" : "fountain_thompson_filter_realshow_weight",
    "enable_interaction_base_attr": "fountain_enable_interaction_base",
    "thompson_filter_realshow_divisor_attr": "fountain_thompson_filter_realshow_divisor",
  },
  {
    "name": "quetionaire_info_filter",
    "enable": "{{enable_fountain_questionnaire_info_filter_v2}}",
    "questionaire_info_replace_topk_result": "fountain_questionaire_info_replace_topk_result",
    "questionaire_info_topk_level_threshold_attr": "fountain_ques_info_topk_level_threshold",
    "questionaire_info_audit_level_threshold_attr": "fountain_ques_info_audit_level_threshold",
    "questionaire_info_negtive_rate_threhold_attr": "fountain_questionnaire_filter_max_negative_rate",
    "questionaire_info_negtive_rate_high_threhold_attr": "fountain_questionnaire_filter_max_negative_high_rate",
    "questionaire_info_positive_rate_threhold_attr": "fountain_questionnaire_filter_min_positive_rate",
    "questionaire_info_unsure_rate_threhold_attr": "fountain_questionnaire_filter_max_unsure_rate",
    "questionaire_info_credible_total_count_attr": "fountain_questionnaire_filter_min_total_count",
    "questionaire_thompson_filter_attr": "fountain_questionaire_thompson_filter",
    "questionaire_filter_neg_weight_attr": "fountain_questionaire_filter_neg_weight",
    "questionaire_filter_pos_weight_attr": "fountain_questionaire_filter_pos_weight",
    "questionaire_filter_unsure_weight_attr": "fountain_questionaire_filter_unsure_weight",
    "questionaire_filter_click_weight_attr": "fountain_questionaire_filter_click_weight",
    "questionaire_filter_unclick_weight_attr": "fountain_questionaire_filter_unclick_weight",
    "questionaire_use_global_data_attr": "fountain_questionaire_use_global_data", 
  },
  {
    "name": "black_author",
    "enable": True,
    "author_id_attr": "author__id",
  },
  {
    "name": "hate_author",
    "enable": True,
    "limit_hate_reason_attr": "fountain_limit_hate_reason",
  },
  {
    "name": "long_term",
    "enable": "{{filter_not_high_quality_out_of_date}}",
    "days_threshold_attr": "filter_not_high_quality_days",
  },
  {
    "name": "duration_random_filter",
    "enable": "{{fountain_enable_duration_random_filter}}",
    "ignore_reason_attr": "duration_random_ignore_reasons",
    "default_cut_off_ratio_attr": "fountain_duration_random_default_cut_off_ratio",
    "adjust_cut_off_ratio_attr": "fountain_duration_random_adjust_cut_off_ratio",
    "enable_random_cut_off_attr": "fountain_duration_random_enable_random_cut_off",
    "lt_longview_ratio_threshold_attr": "fountain_duration_random_lt_longview_ratio_threshold",
    "sharp_change_confidence_threshold_attr": "fountain_duration_random_sharp_change_confidence_threshold",
    "page_type_attr": "fountain"
  },
  {
    "name": "duration_emp_watchtime_sample_filter",
    "enable": "{{fountain_enable_duration_emp_watchtime_sample_filter}}",
    "duration_sample_threshold_attr": "fountain_duration_sample_filter_threshold_str",
    "duration_sample_base_number_attr": "fountain_duration_sample_filter_base_number",
    "duration_sample_multi_number_attr": "fountain_duration_sample_filter_multi_number",
  },
  {
    "name": "multi_audit_gray_filter",
    "enable": "{{enable_fountain_multi_audit_gray_filter}}",
    "audit_gray_count_threshold_attr": "fountain_multi_audit_gray_filter_count_threshold",
    "multi_audit_gray_days_limit_attr": "fountain_multi_audit_gray_filter_days_limit",
  },
  {
    "name": "high_hot_audit_gray_show",
    "enable": "{{enable_fountain_high_hot_audit_gray_show_filter}}",
    "enable_stat_all_page": "fountain_high_hot_audit_gray_show_filter_stat_all_page",
    "high_hot_audit_gray_show_threshold": "fountain_high_hot_audit_gray_show_filter_threshold",
  },
  {
    "name": "xtab_life_index_filter",
    "enable": "{{fountain_enable_xtab_life_index_filter}}",
    "key_hetu_category_list_attr": "key_hetu_category_list",
  },
  {
    "name": "high_emp_phtr_filter",
    "enable": "{{enable_fountain_emp_phtr_filter}}",
    "emp_realshow_show_threshold_attr": "fountain_emphtr_realshow_show_threshold",
    "emphtr_filter_threshold_attr": "fountain_emphtr_filter_threshold",
    "enable_hate_cost_attr": "fountain_enable_hate_cost",
    "emphtr_filter_ctr_weight_attr": "fountain_emphtr_filter_ctr_weight",
    "emphtr_filter_ltr_weight_attr": "fountain_emphtr_filter_ltr_weight",
    "emphtr_filter_wtr_weight_attr": "fountain_emphtr_filter_wtr_weight",
    "emphtr_filter_ftr_weight_attr": "fountain_emphtr_filter_ftr_weight",
    "emphtr_filter_cmtr_weight_attr": "fountain_emphtr_filter_cmtr_weight",
    "emphtr_filter_time_weight_attr": "fountain_emphtr_filter_time_weight",
    "emphtr_filter_normal_time_weight_attr": "fountain_emphtr_filter_normal_time_weight",
    "enable_adpt_threshold_attr": "fountain_enable_adpt_threshold",
    "emphtr_filter_threshold_list_attr": "fountain_emphtr_filter_threshold_list"
  },
  {
    "name": "audit_rule_adjust_filter",
    "enable": "{{fountain_enable_audit_rule_adjust_filter}}",
    "audit_rule_adjust_tags_attr": "fountain_audit_rule_adjust_tags",
  },
  {
    "name": "merchant_holdout_filter",
    "enable": "{{fountain_enable_merchant_holdout_filter}}",
    "merchant_author_list_ptr_attr": "merchant_author_list_ptr",
    "enable_filter_living_merchant_photo": "fountain_enable_filter_living_merchant_photo",
    "enable_filter_living_merchant_author": "fountain_enable_filter_living_merchant_author",
  },
  {
    "name": "audit_user_experiment_level_filter",
    "enable": "{{fountain_enable_audit_user_experiment_level_filter}}",
    "audit_user_experiment_level_map_attr": "fountain_audit_user_experiment_level_map_str"
  },
  {
    "name": "personified_author_filter",
    "enable": "{{fountain_enable_personified_author_filter}}",
    "personified_author_filter_flag": "fountain_personified_author_filter_flag"
  },
  # 影视价值验证holdout
  {
    "name": "movie_copyright_holdout_filter",
    "enable": "{{fountain_enable_movie_copyright_holdout_filter}}",
    "photo_ip_info_map_ptr_attr": "photo_ip_info_map_ptr",
    "target_filter_flag_attr": "fountain_movie_copyright_holdout_filter_target_flag"
  },
  # 年轻人垂类验证
  {
    "name": "young_inc_tags_holdout_filter",
    "enable": "{{fountain_enable_young_inc_tags_holdout_filter}}",
    "young_inc_category_list_attr": "young_inc_category_list",
    "young_inc_category_hetu_list_attr": "young_inc_category_hetu_list",
    "filter_flag_attr": "fountain_young_inc_tags_filter_flag",
    "filter_ratio_attr": "fountain_young_inc_tags_filter_ratio",
    "filter_prime_attr": "fountain_young_inc_tags_filter_prime",
    "upload_time_limit_attr": "fountain_young_inc_tags_filter_upload_time_limit"
  },
  # 光合验证
  {
    "name": "light_inc_holdout_filter",
    "enable": "{{fountain_enable_light_inc_holdout_filter}}",
  },
  # 粉段验证
  {
    "name": "fans_count_random_holdout_filter",
    "enable": "{{fountain_enable_fans_count_random_holdout_filter}}",
    "filter_ratio_attr": "fountain_fans_count_random_holdout_filter_ratio",
    "filter_prime_attr": "fountain_fans_count_random_holdout_filter_prime",
    "fans_bucket_list_attr": "fountain_fans_count_random_holdout_filter_fans_bucket_list",
  },
  # 诱导互动作品过滤
  {
    "name": "audit_hack_photo_filter",
    "enable": "{{fountain_enable_audit_hack_photo_filter}}",
    "audit_hack_tag_set_attr": "audit_hack_tags_str",
    "min_show_attr": "audit_hack_photo_filter_min_show",
    "max_ltr_attr": "audit_hack_photo_filter_max_ltr",
    "max_wtr_attr": "audit_hack_photo_filter_max_wtr",
    "max_cmtr_attr": "audit_hack_photo_filter_max_cmtr",
  },
  {
    "name": "user_reco_neg_photo_filter",
    "enable": "{{fountain_enable_user_reco_neg_photo_filter}}",
    "reco_neg_photo_list_attr": "reco_neg_photo_id_filter_list",
    "candidate_count_attr": "filter_candidate_count",
    "candidate_count_limit": "fountain_user_reco_neg_filter_candidate_count_limit"
  },
  {
    "name": "short_term_negative_filter",
    "enable": "{{enable_fountain_short_term_negative_filter}}",
    "short_minutes_cut_attr": "fountain_negative_filter_short_minutes_cut",
    "long_minutes_cut_attr": "fountain_negative_filter_long_minutes_cut"
  },
  {
    "name": "data_set_tags_filter",
    "enable": "{{fountain_enable_data_set_tags_filter}}",
    "filter_tags_list_attr": "data_set_tags_filter_tags_list"
  },
  {
    "name": "quality_audit_filter",
    "enable": "{{fountain_enable_quality_audit_filter_final}}",
    "filter_tags_list_attr": "quality_audit_filter_tags_list"
  },
  {
    "name": "dynamic_xtr_filter",
    "enable": "{{enable_fountain_dynamic_xtr_filter}}",
    "dynamic_xtrs_threshold_list_attr": "dynamic_xtrs_threshold_list",
    "dynamic_filter_old_photo_days_attr": "fountain_filter_old_photo_days",
    "dynamic_filter_save_follow_author_attr": "enable_fountain_save_follow_author"
  },
  {
    "name": "hetu_author_category_holdout_filter",
    "enable": "{{fountain_enable_hetu_author_category_holdout_filter}}",
    "fans_count_limit_attr": "fountain_hetu_author_category_holdout_filter_fans_count_limit",
    "hetu_author_category_list_attr": "fountain_hetu_author_category_holdout_filter_list"
  },
  {
    "name": "data_set_tags_bit_filter",
    "enable": "{{fountain_enable_data_set_tags_bit_filter}}",
    "filter_bits_list_attr": "data_set_tags_bit_filter_bits_list"
  },
  {
    "name": "merchant_cart_holdout_filter", # 挂车视频过滤
    "enable": "{{fountain_enable_merchant_cart_holdout_filter}}",
  },
  {
    "name": "high_photo_count_author_filter",
    "enable": "{{fountain_enable_high_photo_count_author_filter}}",
    "high_photo_count_author_map_ptr_attr": "high_photo_count_author_map_ptr",
    "realshow_threshold_attr": "fountain_high_photo_count_author_photo_realshow_threshold",
    "post_num_base_attr": "fountain_high_photo_count_author_post_num_base"
  },
  {
    "name": "douyin_author_holdout_filter",
    "enable": "{{fountain_enable_douyin_author_holdout_filter}}",
    "filter_flag_attr": "fountain_douyin_author_holdout_filter_flag",
    "fans_count_limit_attr": "fountain_douyin_author_holdout_filter_fans_count_limit",
    "hetu_author_category_list_attr": "fountain_douyin_author_holdout_filter_list",
    "douyin_10w_author_set_ptr_attr": "douyin_10w_author_set_ptr",
    "douyin_100w_author_set_ptr_attr": "douyin_100w_author_set_ptr"
  },
  # 中长视频holdout
  {
    "name": "mid_long_video_holdout_filter",
    "enable": "{{fountain_enable_mid_long_video_holdout_filter}}",
    "duration_lowerbound_attr": "fountain_mid_long_video_holdout_filter_duration_lowerbound",
    "duration_upperbound_attr": "fountain_mid_long_video_holdout_filter_duration_upperbound",
    "filter_tags_list_attr": "fountain_mid_long_video_holdout_filter_tags_list"
  },
  # 生产类别过滤
  {
    "name": "produce_type_filter",
    "enable": "{{fountain_enable_produce_type_filter}}",
    "produce_magic_type_filter_flag_attr": "fountain_produce_magic_type_filter_flag",
    "produce_need_filter_magic_type_list_attr": "fountain_produce_need_filter_magic_type_list"
  }
]

def append_prepare_processors(flow):
  flow \
    .split_string(
      input_common_attr = "fountain_impression_audit_second_level_white_tags",
      output_common_attr = "impression_audit_white_tags",
      delimiters=",",
      parse_to_int = True,
    ) \
    .split_string(
      input_common_attr = "fountain_impression_audit_second_level_black_tags",
      output_common_attr = "impression_audit_black_tags",
      delimiters=",",
      parse_to_int = True,
    ) \
    .split_string(
      input_common_attr = "fountain_high_hot_audit_second_level_white_tags",
      output_common_attr = "high_hot_audit_white_tags",
      delimiters=",",
      parse_to_int = True,
    ) \
    .split_string(
      input_common_attr = "fountain_high_hot_audit_second_level_black_tags",
      output_common_attr = "high_hot_audit_black_tags",
      delimiters=",",
      parse_to_int = True,
    ) \
    .split_string(
      input_common_attr = "fountain_audit_rule_adjust_tags_str",
      output_common_attr = "fountain_audit_rule_adjust_tags",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True,
    ) \
    .split_string(
      input_common_attr = "fountain_produce_need_filter_magic_type_str",
      output_common_attr = "fountain_produce_need_filter_magic_type_list",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True,
    ) \
    .set_attr_value(
      common_attrs = [
        {
          "name": "single_picture_upload_type_list",
          "type": "int_list",
          "value": [7, 70],
        },
        {
          "name": "single_picture_picture_type_list",
          "type": "int_list",
          "value": [1],
        },
      ],
      skip = "{{fountain_skip_filter_by_picture_single_variant_attr}}",
    ) \
    .set_attr_value(
      common_attrs = [
        {
          "name": "variant_picture_upload_type_list",
          "type": "int_list",
          "value": [10, 12],
        },
        {
          "name": "variant_picture_picture_type_list",
          "type": "int_list",
          "value": [3],
        },
      ],
      skip = "{{fountain_skip_filter_by_picture_variant_attr}}",
    ) \
    .set_attr_value(
      common_attrs = [
        {
          "name": "variant_picture_set_upload_type_list",
          "type": "int_list",
          "value": [11],
        },
        {
          "name": "variant_picture_set_picture_type_list",
          "type": "int_list",
          "value": [2],
        },
      ],
      skip = "{{fountain_skip_filter_by_picture_set_variant_attr}}",
    ) \
    .set_attr_value(
      common_attrs = [
        {
          "name": "long_article_upload_type_list",
          "type": "int_list",
          "value": [27],
        },
      ],
    ) \
    .pack_common_attr(
      input_common_attrs = [
        "single_picture_upload_type_list",
        "variant_picture_upload_type_list",
        "long_article_upload_type_list",
        "variant_picture_set_upload_type_list"
      ],
      output_common_attr = "filter_upload_type_lsit",
      deduplicate = False,
    ) \
    .pack_common_attr(
      input_common_attrs = [
        "single_picture_picture_type_list",
        "variant_picture_picture_type_list",
        "variant_picture_set_picture_type_list"
      ],
      output_common_attr = "filter_picture_type_lsit",
      deduplicate = False,
    ) \
    .set_attr_value(
      common_attrs = [
        {
          "name": "photo_life_max_hours",
          "type": "int",
          "value": 168,
        },
      ],
    ) \
    .split_string(
      input_common_attr = "fountain_topk_audit_second_level_white_tags",
      output_common_attr = "topk_audit_white_tags",
      delimiters=",",
      parse_to_int = True,
    ) \
    .split_string(
      input_common_attr = "fountain_topk_audit_second_level_black_tags",
      output_common_attr = "topk_audit_black_tags",
      delimiters=",",
      parse_to_int = True,
    ) \
    .split_string(
      input_common_attr = "fountain_duration_random_ignore_reasons",
      output_common_attr = "duration_random_ignore_reasons",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True,
    ) \
    .split_string(
      input_common_attr = "fountain_key_hetu_categories",
      output_common_attr = "key_hetu_category_list",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True,
    ) \
    .split_string(
      input_common_attr = "fountain_young_inc_category_list_str",
      output_common_attr = "young_inc_category_list",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True
    ) \
    .split_string(
      input_common_attr = "fountain_young_inc_category_hetu_list_str",
      output_common_attr = "young_inc_category_hetu_list",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True
    ) \
    .split_string(
      input_common_attr = "fountain_fans_count_random_holdout_filter_fans_bucket_str",
      output_common_attr = "fountain_fans_count_random_holdout_filter_fans_bucket_list",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True
    ) \
    .split_string(
      input_common_attr = "fountain_data_set_tags_filter_tags_list_str",
      output_common_attr = "data_set_tags_filter_tags_list",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True
    ) \
    .split_string(
      input_common_attr = "fountain_data_set_tags_bit_filter_bits_list_str",
      output_common_attr = "data_set_tags_bit_filter_bits_list",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True
    ) \
    .split_string(
      input_common_attr = "fountain_dynamic_xtr_filter_threshold_str",
      output_common_attr = "dynamic_xtrs_threshold_list",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_double = True,
    ) \
    .split_string(
      input_common_attr = "fountain_quality_audit_filter_tags_list_str_final",
      output_common_attr = "quality_audit_filter_tags_list",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True
    ) \
    .split_string(
      input_common_attr = "fountain_hetu_author_category_holdout_filter_list_str",
      output_common_attr = "fountain_hetu_author_category_holdout_filter_list",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True
    ) \
    .split_string(
      input_common_attr = "fountain_douyin_author_holdout_filter_list_str",
      output_common_attr = "fountain_douyin_author_holdout_filter_list",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True
    ) \
    .split_string(
      input_common_attr = "fountain_mid_long_video_holdout_filter_tags_list_str",
      output_common_attr = "fountain_mid_long_video_holdout_filter_tags_list",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True
    ) \
    .get_kconf_params(
      kconf_configs = [{
        "kconf_key": "reco.hot.recoExploreUserRiskMin",
        "export_common_attr": "user_risk_min",
        "value_type": "int64"
      }]
    ) \
    .explore_memory_data_enrich(
      data_key = "livestream_merchant_author",
      data_type = "uint64_set",
      save_data_ptr_to_attr = "merchant_author_list_ptr"
    ) \
    .explore_memory_data_enrich(
      data_key = "photo_ip_info_map",
      data_type = "uint64_string_map",
      save_data_ptr_to_attr = "photo_ip_info_map_ptr"
    ) \
    .explore_memory_data_enrich(
      data_key = "high_photo_count_author_map",
      data_type = "uint64_uint64_map",
      save_data_ptr_to_attr = "high_photo_count_author_map_ptr"
    ) \
    .explore_memory_data_enrich(
      data_key = "douyin_10w_author",
      data_type = "uint64_set",
      save_data_ptr_to_attr = "douyin_10w_author_set_ptr"
    ) \
    .explore_memory_data_enrich(
      data_key = "douyin_100w_author",
      data_type = "uint64_set",
      save_data_ptr_to_attr = "douyin_100w_author_set_ptr"
    ) \
    .if_("fountain_enable_fetch_rank_neg_photo == 1") \
      .split_string(
        input_common_attr = "rank_neg_photo_id_list_str",
        output_common_attr = "rank_neg_photo_id_filter_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
    .end_() \
    .if_("fountain_enable_fetch_rerank_neg_photo == 1") \
      .split_string(
        input_common_attr = "rerank_neg_photo_id_list_str",
        output_common_attr = "rerank_neg_photo_id_filter_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
    .end_() \
    .if_("fountain_enable_fetch_mc_neg_photo == 1") \
      .split_string(
        input_common_attr = "mc_neg_photo_id_list_str",
        output_common_attr = "mc_neg_photo_id_filter_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
    .end_() \
    .pack_common_attr(
      input_common_attrs = [
        "rank_neg_photo_id_filter_list",
        "rerank_neg_photo_id_filter_list",
        "mc_neg_photo_id_filter_list"
      ],
      output_common_attr = "reco_neg_photo_id_filter_list",
      deduplicate = True,
    ) \
    .count_reco_result(
      save_count_to = "filter_candidate_count"
    )
