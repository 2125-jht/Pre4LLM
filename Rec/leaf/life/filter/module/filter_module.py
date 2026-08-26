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
      "hetu_v3_level_one_tag_list_attr": "hetu_tag_level_info_v3__hetu_level_one",
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
      "nebula_forward_attr": "nebula_stats__forward_count",
      "nebula_follow_attr": "nebula_stats__follow_count",
      "nebula_collect_attr": "nebula_stats__collect_count",
      "nebula_negative_attr": "nebula_stats__negative_count", 
      "thanos_real_show_attr": "thanos_stats__real_show_count",
      "thanos_like_attr": "thanos_stats__like_count",
      "thanos_comment_attr": "thanos_stats__comment_count",
      "thanos_forward_attr": "thanos_stats__forward_count",
      "thanos_follow_attr": "thanos_stats__follow_count",
      "thanos_collect_attr": "thanos_stats__collect_count",
      "thanos_negative_attr": "thanos_stats__negative_count",
      "fountain_real_show_attr": "fountain_stats__real_show_count",
      "fountain_like_attr": "fountain_stats__like_count",
      "fountain_comment_attr": "fountain_stats__comment_count",
      "fountain_forward_attr": "fountain_stats__forward_count",
      "fountain_follow_attr": "fountain_stats__follow_count",
      "fountain_collect_attr": "fountain_stats__collect_count",
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
      "is_picture": "is_picture",
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
      "sirius_distribution_info__mark_cod_attr": "sirius_distribution_info__mark_cod",
      "author_grade_key_attr": "author_grade_key",
      "author_shop_score_attr":"author_shop_score",
      "author_max_item_score_attr":"author_max_item_score",
      "secure_grading_action_code_attr": "secure_grading_action_code",
      "explore_stats_report_count_attr": "explore_stat__report_count",
      "fountain_stats_report_count_attr": "fountain_stats__report_count",
      "thanos_stats_report_count_attr": "thanos_stats__report_count",
      "nebula_stats_report_count_attr": "nebula_stats__report_count",
      "explore_short_play_attr": "explore_stat__short_play_count",
      "fountain_stats_short_play_count_attr": "fountain_stats__short_play_count",
      "thanos_stats_short_play_count_attr": "thanos_stats__short_play_count",
      "nebula_stats_short_play_count_attr": "nebula_stats__short_play_count",
      "nebula_stats_view_length_sum_attr": "nebula_stats__view_length_sum",
      "thanos_stats_view_length_sum_attr": "thanos_stats__view_length_sum",
      "fountain_stats_view_length_sum_attr": "fountain_stats__view_length_sum",
    }

  @property
  def filters(self) -> list:
    return [
      {
        "name": "not_in_index",
        "enable": True,
      },
      {
        "name": "passby_user_low_vv_filter",
        "enable": "{{enable_xlife_passby_user_low_vv_photo_filter}}",
        "low_vv_thresh_attr": "xlife_passby_user_photo_low_vv_thresh",
        "high_vv_thresh_attr": "xlife_passby_user_photo_high_vv_thresh",
        "act_rate_thresh_attr": "xlife_passby_user_photo_act_rate_thresh",
      },
      { # 生活页过路/低消用户首刷过滤
        "name": "passby_low_active_user_first_page_filter",
        "enable": "{{life_enable_passby_low_active_user_first_page_filter}}"
      },
      { # 产品需求，部分aid在生活tab屏蔽，直接推全
        "name": "server_show_aid",
        "enable": "{{enable_operation_video_filter_life}}",
        "server_show_aid_list_attr": "life_operation_video_filter_aid_list"
      },
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
      },
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
        "name": "black_exempt_level_v1_audit",
        "enable": True,
        "auto_audit_black_exempt_level_v1_attr": "auto_audit_black_exempt_level_v1",
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
      },
      {
        "name": "outdate_news",
        "enable": "{{enable_filter_outdate_news}}",
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
        "enable": True,
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
      },
      {
        "name": "hate_author",
        "enable": True,
        "limit_hate_reason_attr": "explore_limit_hate_reason",
      },
      {
        "name": "back_fresh_climb",
        "enable": "{{enable_filter_back_fresh_climb}}",
        "show_level_a_attr": "show_level_a",
      },
      {
        "name": "low_porn_report",
        "enable": True,
        "photo_low_report_count_attr": "explore_stat__report_detail__low_report_count",
        "author_low_report_count_attr": "author__explore_report_thirtyday__low_report_count",
      },
      {
        "name": "total_report",
        "enable": True,
      },
      {
        "name": "evil_title",
        "enable": "{{enable_filter_evil_title}}",
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
        "enable_hate_content_reason_filter_attr": "enable_hate_content_reason_filter",
        "hate_content_reason_minutes_cut_attr": "hate_content_reason_minutes_cut", 
        "hetu_tag_l3_minutes_cut_attr": "hetu_tag_l3_minutes_cut", 
        "enable_hate_author_skip_hetu_filter_attr": "enable_hate_author_skip_hetu_filter",
        "enable_long_hate_filter_attr": "enable_long_hate_filter",
        "hetu_tag_long_term_minutes_cut_attr": "hetu_tag_long_term_minutes_cut",
        "hetu_l2_long_filter_threshold_attr": "hetu_l2_long_filter_threshold",
        "hetu_otherl_long_filter_threshold_attr": "hetu_otherl_long_filter_threshold"
      },
      {
        "name": "audit_hot_cover_level_filter",
        "enable": "{{enable_audit_hot_cover_level_filter}}",
      },
      {
        "name": "audit_gray_cover_level_filter", # todo: @suweiwei03 enable_audit_gray_cover_level_part_filter和enable_audit_gray_cover_level_part_filter_series_switch是一个串联开关，需要23年年度comb结束以后合并为一个开关
        "enable": "{{enable_audit_gray_cover_level_filter}}",
        "enable_audit_gray_cover_level_filter_escape_attr": "enable_audit_gray_cover_level_filter_escape",
        "enable_audit_gray_cover_level_part_filter_attr": "enable_audit_gray_cover_level_part_filter",
        "enable_audit_gray_cover_level_part_filter_series_switch_attr": "enable_audit_gray_cover_level_part_filter_series_switch",
        "infer_uv_ctr_attr": "infer_uv_ctr",
        "infer_uv_ctr_threshold_max_attr": "infer_uv_ctr_threshold_max",
        "infer_uv_ctr_threshold_min_attr": "infer_uv_ctr_threshold_min",
        "refresh_times_attr": "refreshTimes",
        "refresh_times_threshold_max_attr": "refresh_times_threshold_max",
        "refresh_times_threshold_min_attr": "refresh_times_threshold_min"
      },
      {
        "name": "mmu_low_cover_filter",
        "enable": "{{enable_mmu_low_cover_filter}}",
        "lower_cover_mmu_map_strs_attr": "lower_cover_mmu_map_strs",
        "lower_cover_mmu_map_tnu_reflux_strs_attr": "lower_cover_mmu_map_tnu_reflux_strs", #新回、2-14新回配置，命中人群会覆盖lower_cover_mmu_map_strs
        "skip_beauty_photo_filter_attr": "skip_beauty_photo_filter",
        "mmu_enable_follow_author_exemption_attr": "mmu_enable_follow_author_exemption",
        "mmu_enable_impression_good_ignore_attr": "mmu_enable_impression_good_ignore",
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
        "name": "is_sirius_punish",
        "enable": "{{enable_filter_sirius_punish_photo}}",
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
        "emphtr_filter_threshold_list_attr": "emphtr_filter_threshold_list"
      },
      {
        "name":"clickbait_filter",
        "enable": "{{life_enabel_clickbait_filter}}",
        "voyage_emp_real_show_threshold_attr": "life_emp_real_show_threshold",
        "voyage_emp_low_like_filter_threshold_attr":"life_emp_low_like_filter_threshold",
        "voyage_emp_high_comment_filter_threshold_attr":"life_emp_high_comment_filter_threshold",
        "voyage_enable_pltr_over_pctr_filter_attr":"life_enable_pltr_over_pctr_filter",
        "voyage_enable_comment_over_like_filter_attr":"life_enable_comment_over_like_filter"
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
        "name": "lifecate_pic_filter",
        "enable": "{{explore_enable_lifecate_pic_filter}}",
        "explore_lifecate_hetu1_list_attr": "lifecate_hetu1_set",
      },
      {
        "name": "audit_rule_adjust_filter",
        "enable": "{{explore_enable_audit_rule_adjust_filter}}",
        "audit_rule_adjust_tags_attr": "explore_audit_rule_adjust_tags",
      },
      {
        "name": "personified_author_filter",
        "enable": "{{explore_enable_personified_author_filter}}",
        "personified_author_filter_flag": "explore_personified_author_filter_flag"
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
        "name": "quality_audit_filter",
        "enable": "{{explore_enable_quality_audit_filter_final}}",
        "filter_tags_list_attr": "quality_audit_filter_tags_list"
      },
      { # 封面机审灰劣过滤
        "name": "auto_audit_hot_cover_level_filter",
        "enable": "{{enable_auto_audit_hot_cover_level_filter}}",
        "enable_follow_author_exemption_attr": "enable_auto_audit_follow_author_exemption",
        "enable_impression_good_ignore_attr": "enable_auto_audit_impression_good_ignore",
        "auto_audit_bad_show_limit_attr": "auto_audit_bad_show_limit",
      },
      {
        "name": "quality_control_filter",
        "enable": "{{explore_enable_quality_control_filter}}",
        "is_first_page_attr": "page",
        "skip_first_page_control_attr": "xlife_quality_control_skip_first_page",
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
        "name": "hetu_author_category_holdout_filter",
        "enable": "{{explore_enable_hetu_author_category_holdout_filter}}",
        "fans_count_limit_attr": "explore_hetu_author_category_holdout_filter_fans_count_limit",
        "hetu_author_category_list_attr": "explore_hetu_author_category_holdout_filter_list"
      },
      {
        "name": "pic_low_quality_filter",
        "enable": "{{explore_enable_pic_low_quality_filter}}",
        "pic_low_quality_filter_thresh_attr": "pic_low_quality_filter_thresh",
        "explore_pic_low_quality_tag_list_attr": "explore_pic_low_quality_tag_list",
      },
      {
        "name": "data_set_tags_bit_filter",
        "enable": "{{explore_enable_data_set_tags_bit_filter}}",
        "filter_bits_list_attr": "data_set_tags_bit_filter_bits_list"
      },
      {
        "name": "xlife_index_filter",
        "enable": "{{enable_xlife_index_filter}}",
        "xlife_low_quality_filter_thresh_attr": "xlife_index_low_quality_filter_thresh",
        "xlife_low_quality_tag_list_attr": "xlife_index_low_quality_tag"
      },
      {
        "name": "xlife_content_control_filter",
        "enable": "{{enable_xlife_content_control_filter}}",
        "pic_vv_threshold_attr": "xlife_pic_low_cost_threshold",
        "mmu_40_filter_score_attr": "xlife_mmu_40_filter_threshold"
      },
      {
        "name": "merchant_cart_holdout_filter", # 挂车视频过滤
        "enable": "{{enable_xlife_merchant_cart_holdout_filter_final}}",
      },
      {
        "name": "audit_b_second_tag_filter",
        "enable": "{{enable_xlife_audit_b_second_tag_filter_final}}",
        "filter_audit_b_second_tag_str_attr": "xlife_filter_audit_b_second_tag_str"
      },
      { # 原生广告过滤
        "name": "protogenetic_advertise_tags_filter",
        "enable": "{{life_enable_protogenetic_advertise_tags_filter}}",
        "filter_advertise_list_attr": "protogenetic_advertise_tags_blacklist"
      },
      { # 新回用户过滤
        "name": "tnu_content_filter",
        "enable": "{{life_enable_tnu_content_filter}}",
        "enable_cover_filter": "life_tnu_enable_cover_filter",
        "enable_impression_filter": "life_tnu_enable_impression_filter",
        "enable_hate_filter": "life_tnu_enable_hate_filter",
        "hate_cnt_thresh_attr": "life_tnu_hate_cnt_thresh",
        "hate_rate_thresh_attr": "life_tnu_hate_rate_thresh"
      },
      { # 负向作者过滤
        "name": "life_author_filter",
        "enable": "{{enable_life_author_filter}}",
        "author_grade_thresh_attr": "life_author_grade_thresh",
        "author_punish_cnt_mode_attr": "life_author_punish_cnt_mode",
        "author_filter_markcode_attr": "life_author_filter_markcode",
        "author_punish_markcode_attr": "life_author_punish_markcode"
      },
      #  店铺分过滤
      {
        "name": "author_shop_score_filter",
        "enable": "{{life_enable_author_shop_score_filter}}",
        "author_shop_score_limit_attr": "life_author_shop_score_filter_limit_count",
        "author_shop_zero_protect_attr": "life_enable_author_shop_zero_protect"
      },
      #  举报内容退场
      {
        "name": "life_report_hetu_filter",
        "enable": "{{life_enable_report_hetu_filter}}",
        "short_report_hetu_minutes_attr": "life_short_report_hetu_minutes",
        "long_report_hetu_minutes_attr": "life_long_report_hetu_minutes",
      },
      #  商品分过滤
      {
        "name": "author_goods_score_filter",
        "enable": "{{life_enable_author_goods_score_filter}}",
        "author_goods_score_limit_attr": "life_author_goods_score_filter_limit_count",
        "author_goods_zero_protect_attr": "life_enable_author_goods_zero_protect"
      },
      # 低成本图文过滤
      {
        "name": "pic_low_cost_filter",
        "enable": "{{life_enable_pic_low_cost_filter}}",
        "explore_low_cost_pic_max_cnt_attr": "life_low_cost_pic_max_cnt",
        "explore_low_cost_pic_cnt_mode_attr": "life_low_cost_pic_cnt_mode",
      },
      # 性感内容图文过滤
      {
        "name": "pic_sexy_filter",
        "enable": "{{life_enable_pic_sexy_filter}}",
        "sexy_pic_max_cnt_attr": "life_sexy_pic_max_cnt",
        "sexy_pic_cnt_mode_attr": "life_sexy_pic_cnt_mode",
      },
      # 图文生态负向特征过滤：高举报
      {
        "name": "pic_ecology_high_report_filter",
        "enable": "{{enable_life_pic_ecology_high_report_filter}}",
        "explore_pic_ecology_high_report_rate_threshold_attr": "life_pic_ecology_high_report_rate_threshold",
        "explore_pic_ecology_high_report_count_threshold_attr": "life_pic_ecology_high_report_count_threshold",
        "pic_ecology_high_report_fans_count_threshold_attr": "life_pic_ecology_high_report_fans_count_threshold"
      },
      # 图文生态负向特征过滤：高负正反馈率
      {
        "name": "pic_ecology_high_neg_pos_rate_filter",
        "enable": "{{enable_life_pic_ecology_high_neg_pos_rate_filter}}",
        "explore_pic_ecology_high_neg_pos_rate_threshold_attr": "life_pic_ecology_high_neg_pos_rate_threshold"
      },
      # 图文生态负向特征过滤：高短播
      {
        "name": "pic_ecology_high_short_play_rate_filter",
        "enable": "{{enable_life_pic_ecology_high_short_play_rate_filter}}",
        "explore_pic_ecology_high_short_play_rate_threshold_attr": "life_pic_ecology_high_short_play_rate_threshold",
        "explore_pic_ecology_neg_rate_threshold_attr": "life_pic_ecology_neg_rate_threshold"
      },
      # 图文生态负向特征过滤：综合互动
      {
        "name": "pic_ecology_mix_interact_rate_filter",
        "enable": "{{enable_life_pic_ecology_mix_interact_rate_filter}}",
        "pic_ecology_interact_rate_threshold_attr": "life_pic_ecology_interact_rate_threshold",
        "pic_ecology_interact_avg_view_time_threshold_attr": "life_pic_ecology_interact_avg_view_time_threshold",
        "pic_ecology_interact_vv_threshold_attr": "life_pic_ecology_interact_vv_threshold"
      },
      # 图文 mmu hetu tag 过滤
      {
        "name": "pic_mmu_hetu_tag_filter",
        "enable": "{{life_enable_pic_mmu_hetu_tag_filter}}",
        "mmu_tag_prob_str_attr": "life_pic_filter_mmu_tag_prob_str",
        "mmu_tag_skip_hv_str_attr": "life_pic_filter_mmu_tag_skip_hv_str",
        "mmu_tag_vv_thr_str_attr": "life_pic_filter_mmu_tag_vv_thr_str",
      },
      # 图文负向作者过滤
      {
        "name": "pic_author_filter",
        "enable": "{{life_enable_pic_author_filter}}",
        "author_grade_thresh_attr": "life_pic_author_grade_thresh",
        "author_punish_cnt_mode_attr": "life_pic_author_punish_cnt_mode",
        "author_filter_markcode_attr": "life_pic_author_filter_markcode",
        "author_punish_markcode_attr": "life_pic_author_punish_markcode"
      },
      # 图文 data_set_tags_bit 过滤
      {
        "name": "pic_data_set_tags_bit_filter",
        "enable": "{{life_enable_pic_data_set_tags_bit_filter}}",
        "pic_filter_bits_str_attr": "life_pic_filter_data_set_tags_bits_str",
        "pic_punish_bits_str_attr": "life_pic_punish_data_set_tags_bits_str",
        "skip_filter_mark_cod_str_attr": "life_pic_skip_filter_mark_cod_str",
        "punish_vv_thresh_attr": "life_pic_punish_data_set_tags_bit_vv_thresh",
        "punish_filter_prob_attr": "life_pic_punish_data_set_tags_bit_filter_prob"
      },
      # 图文安全审过滤
      {
        "name": "pic_secure_grade_filter",
        "enable": "{{life_enable_pic_secure_grade_filter}}",
        "secure_grade_filter_code_attr": "life_pic_secure_grade_filter_code_str"
      },
      # 营销号单图低综合互动率过滤
      {
        "name": "pic_mix_interact_rate_filter",
        "enable": "{{life_enable_pic_mix_interact_rate_filter}}",
        "base_vv_threshold_attr": "life_pic_mix_interact_rate_filter_base_vv_threshold",
        "author_filter_mark_cod_str_attr": "life_pic_mix_interact_rate_filter_author_mark_cod_str",
        "interact_rate_thresholds_str_attr": "life_pic_mix_interact_rate_thresholds_str",
        "vv_thresholds_str_attr": "life_pic_mix_interact_rate_filter_vv_thresholds_str",
        "filter_probs_str_attr": "life_pic_mix_interact_rate_filter_probs_str",
      },
      # 营销号泛单图过滤
      {
        "name": "marketing_static_video_filter",
        "enable": "{{life_enable_marketing_static_video_filter}}",
        "static_video_tag_id_attr": "life_static_video_hetu_tag_id",
        "static_video_tag_prob_thd_attr": "life_static_video_hetu_tag_prob_thd",
        "marketing_mark_cod_str_attr": "life_marketing_static_video_filter_mark_cod_str",
        "base_vv_threshold_attr": "life_marketing_static_video_filter_base_vv_threshold",
        "interact_rate_thresholds_str_attr": "life_marketing_static_video_filter_interact_rate_thresholds_str",
        "vv_thresholds_str_attr": "life_marketing_static_video_filter_vv_thresholds_str",
        "filter_probs_str_attr": "life_marketing_static_video_filter_probs_str"
      },
      { # 极速新双列低俗治理
        "name": "life_vulgar_content_filter",
        "enable": "{{life_enable_vulgar_content_filter}}",
        "enable_llm_tag_filter": "life_enable_vulgar_llm_tag_filter",
        "enable_mmu_tag_filter": "life_enable_vulgar_mmu_tag_filter",
        "enable_low_score_filter": "life_enable_vulgar_low_score_filter",
        "enable_safety_review_filter": "life_enable_safety_review_filter",
        "audit_b_second_tag_attr": "audit_b_second_tag",
        "is_picture": "is_picture",
      },
      { # 浅消费首刷保护
        "name": "life_first_page_content_filter",
        "enable": "{{life_enable_first_page_content_filter_final}}",
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
        "skip_dup_realshow_threshold_attr": "skip_dup_realshow_threshold",
        "skip_dup_watchtime_threshold_attr": "skip_dup_watchtime_threshold",
        "skip_dup_fvtr_threshold_attr": "skip_dup_fvtr_threshold",
        "skip_dup_ctr_threshold_attr": "skip_dup_ctr_threshold",
        "filter_content_type_list_for_pic_attr": "filter_content_type_list_for_pic"
      }
    ]

  @property
  def sec_tab_truncation_map(self) -> dict:
    return {
      "10041": 7000 # 二级顶导 reason 限制
    }

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
        input_common_attr = "xlife_index_low_quality_filter_thresh_list_attr",
        output_common_attr = "xlife_index_low_quality_filter_thresh",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_double = True
      ) \
      .split_string(
        input_common_attr = "xlife_index_low_quality_tag_list_attr",
        output_common_attr = "xlife_index_low_quality_tag",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
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
      .if_("explore_enable_fetch_rerank_neg_photo == 1") \
        .split_string(
          input_common_attr = "rerank_neg_photo_id_list_str",
          output_common_attr = "rerank_neg_photo_id_filter_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
      .end_() \
      .if_("life_enable_protogenetic_advertise_tags_filter == 1") \
        .if_("enable_life_advertise_tags_blacklist_str_low_active == 1 and uIsLifeHighActive ~= 1") \
          .split_string(
            input_common_attr = "life_protogenetic_advertise_tags_blacklist_str_low_active",
            output_common_attr = "protogenetic_advertise_tags_blacklist",
            delimiters = ",",
            trim_spaces = True,
            skip_empty_tokens = True,
            parse_to_int = True
          ) \
        .else_() \
          .split_string(
            input_common_attr = "life_protogenetic_advertise_tags_blacklist_str",
            output_common_attr = "protogenetic_advertise_tags_blacklist",
            delimiters = ",",
            trim_spaces = True,
            skip_empty_tokens = True,
            parse_to_int = True
          ) \
        .end_() \
      .end_() \
      .pack_common_attr(
        input_common_attrs = [
          "rank_neg_photo_id_filter_list",
          "rerank_neg_photo_id_filter_list"
        ],
        output_common_attr = "reco_neg_photo_id_filter_list",
        deduplicate = True,
      ) \
      .enrich_attr_by_lua(
        import_common_attr = [
          "uNebulaXlifeVisitDays30dKV",
          "uNebulaDoubleFindVisitDays30dKV",
          "page",
          "refreshTimes",
          "life_enable_first_page_content_filter_page_thred",
          "life_enable_first_page_content_filter_refresh_thred",
        ],
        export_common_attr = [
          "is_low_active_user_and_first_page",
        ],
        function_for_common = "filter_switch_cal",
        lua_script = """
          function filter_switch_cal()
            local uNebulaXlifeVisitDays30dKV = uNebulaXlifeVisitDays30dKV or 0
            local uNebulaDoubleFindVisitDays30dKV = uNebulaDoubleFindVisitDays30dKV or 0
            local page = page or 0
            local refreshTimes = refreshTimes or 0
            if (uNebulaXlifeVisitDays30dKV + uNebulaDoubleFindVisitDays30dKV) <= 1 and (page <= life_enable_first_page_content_filter_page_thred and refreshTimes <= life_enable_first_page_content_filter_refresh_thred) then
              return 1
            end
            return 0
          end
        """,
      ) \
      .gen_common_attr_by_lua(
        attr_map={
          "enable_xlife_merchant_cart_holdout_filter_final": "enable_xlife_merchant_cart_holdout_filter == 1 or (enable_xlife_merchant_cart_holdout_filter_first_page == 1 and page == 1)",
          "enable_xlife_audit_b_second_tag_filter_final": "enable_xlife_audit_b_second_tag_filter == 1 or (enable_xlife_audit_b_second_tag_filter_first_page == 1 and page == 1) or (enable_xlife_audit_b_second_tag_filter_first_page_low_active == 1 and page == 1 and uIsLifeHighActive ~= 1)",
          "enable_xlife_passby_user_low_vv_photo_filter": "enable_xlife_passby_user_low_vv_photo_filter == 1 and uIsNotLifePassBy ~= 1",
          "life_enable_tnu_content_filter": "life_enable_tnu_content_filter == 1 and uIsTnuCrowdUser == 1",
          "life_enable_passby_low_active_user_first_page_filter": "life_enable_passby_low_active_user_first_page_filter == 1 and page == 1 and (uIsNotLifePassBy ~= 1 or uIsLifeHighActive ~= 1)",
          "life_enable_first_page_content_filter_final": "life_enable_first_page_content_filter == 1 and is_low_active_user_and_first_page == 1",
        }
      ) \
      .if_("enable_second_tab > 0") \
        .explore_life_retrieval_filter(
          name = "explore_sec_tab_filter",
          traceback = True,
          user_info_ptr_attr = "user_info_ptr",
          item_attr_map = self.item_attr_map,
          filters = self.filters,
          truncation_map = self.sec_tab_truncation_map,
        ) \
      .else_() \
        .switch_("explore_filter_mode") \
          .case_(1) \
            .explore_life_retrieval_filter(
              name = "explore_retr_filter_skip",
              traceback = True,
              user_info_ptr_attr = "user_info_ptr",
              item_attr_map = self.item_attr_map,
              filters = self.filters,
            ) \
          .case_(2) \
            .if_("life_enable_first_page_content_filter_final == 1") \
              .explore_life_retrieval_filter(
                name = "explore_retr_filter_no_limit_tnu",
                traceback = True,
                user_info_ptr_attr = "user_info_ptr",
                item_attr_map = self.item_attr_map,
                filters = self.filters,
                truncation_map = {
                  "default": 10000
                }
              ) \
            .else_() \
              .explore_life_retrieval_filter(
                name = "explore_retr_filter_no_limit",
                traceback = True,
                user_info_ptr_attr = "user_info_ptr",
                item_attr_map = self.item_attr_map,
                filters = self.filters,
                truncation_map = {
                  "default": 2000
                }
              ) \
            .end_() \
          .default_() \
            .explore_life_retrieval_filter(
              name = "explore_retr_filter",
              traceback = True,
              user_info_ptr_attr = "user_info_ptr",
              item_attr_map = self.item_attr_map,
              filters = self.filters,
              truncation_map = self.truncation_map,
            ) \
        .end_() \
      .end_if_() \

  def post_process(self) -> None:
    self.flow \
      .if_("explore_nearline_user_update_flag == 1") \
        .pack_item_attr(
          item_source = {
            "reco_results": True,
          },
          mappings = [{
            "from_item_attr": "photo_id",
            "to_common_attr": "explore_nearline_candidates_after_filter",
            "aggregator": "concat"
          }]
        ) \
        .export_attr_to_kafka(
          kafka_topic = "explore_nearline_leaf_message",
          common_attrs = ["userInfo", "explore_nearline_candidates_after_filter"],
        ) \
      .end_() \
