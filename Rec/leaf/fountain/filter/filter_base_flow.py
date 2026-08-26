#!/usr/bin/env python3
# coding=utf-8


from dragonfly.common_leaf_dsl import LeafFlow
from dragonfly.ext.subdivision.subdivision_api_mixin import subdivisionApiMixin
from util import enrich_ab_param


class FilterBaseFlow(LeafFlow, subdivisionApiMixin):
  _ITEM_ATTR_MAP = {
    "photo_id_attr": "photo_id",
    "author_id_attr": "author__id",
    "is_living_attr": "live_photo_info__is_living",
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
    "data_set_tags_bit_list_attr": "data_set_tags_bit_list",
    "magic_face_type_attr": "magic_face_type",
    "magic_face_id_attr": "magic_face_id",
    "audit_hot_cover_level_attr": "audit_hot_cover_level",
    "hetu_tag_level_info_v2__hetu_tag_attr": "hetu_tag_level_info_v2__hetu_tag",
    "is_repost_photo_attr": "mmu_repost_photo_info__is_repost_photo",
    "sirius_distribution_info__mark_cod_attr": "sirius_distribution_info__mark_cod",
    "live_photo_flag_attr": "live_photo_flag",
    "merchant_hetu_tag_id_photo_attr": "is_merchant_hetu_tag_id",
    "audit_cold_review_level_attr": "audit_cold_review_level",
    "author_age_segment_attr": "author_age_info__age_segment",
    "explore_stats_report_count_attr": "explore_stat__report_count",
    "fountain_stats_report_count_attr": "fountain_stats__report_count",
    "thanos_stats_report_count_attr": "thanos_stats__report_count",
    "nebula_stats_report_count_attr": "nebula_stats__report_count",
    "explore_short_play_attr": "explore_stat__short_play_count",  #
    "fountain_stats_short_play_count_attr": "fountain_stats__short_play_count",
    "thanos_stats_short_play_count_attr": "thanos_stats__short_play_count",
    "nebula_stats_short_play_count_attr": "nebula_stats__short_play_count",
    "nebula_stats_view_length_sum_attr": "nebula_stats__view_length_sum",
    "thanos_stats_view_length_sum_attr": "thanos_stats__view_length_sum",
    "fountain_stats_view_length_sum_attr": "fountain_stats__view_length_sum",
    "explore_collect_attr": "explore_stat__collect_count",
    "fountain_collect_attr": "fountain_stats__collect_count",
    "thanos_collect_attr": "thanos_stats__collect_count",
    "nebula_collect_attr": "nebula_stats__collect_count",
    "author_max_item_score_attr": "author_max_item_score",
    "author_shop_score_attr": "author_shop_score",
    "secure_grading_action_code_attr": "secure_grading_action_code",
    "coldstart_guarantee_value_attr": "coldstart_guarantee_value",
    "fangpin_aid_filter_ratio_attr": "fangpin_aid_filter_ratio",
    "plc_business_type_attr": "plc_business_type",
    "author_tail_galaxy_attr": "video_cold_start_info__author_tails__galaxyAuthorExpGroupInteger",
    "author_tail_climb_attr": "video_cold_start_info__author_tails__climb_retrieval_author_tail",
    "author_tail_vcs_attr": "video_cold_start_info__author_tails__vcs_author_tail_161",
    "author_liezhi_pic_count_attr": "author_liezhi_pic_count",
    "author_hash_tag_id_list_attr": "user_hash_tag_id",
    "live_photo_duration_attr": "live_photo_duration",
    "is_tv_station_bottom_bar_attr": "is_tv_station_bottom_bar",
    "hot_trend_generalized_info_source_attr": "hot_trend_generalized_info__source",
    "author_tail_int_index_map_34_attr": "author_tail_int_index_map_34",
    "author_op_session_class_attr": "author_op_session_class",
    "community_survey_markcode_attr": "community_survey_info__survey_title",
    "community_survey_cert_count_attr": "community_survey_info__cert_count",
    "community_survey_not_cert_count_attr": "community_survey_info__not_cert_count",
    "community_survey_uncert_count_attr": "community_survey_info__uncert_count",
    "cover_view_predict_score_attr": "cover_view_predict_score",
    "sense_view_predict_score_attr": "sense_view_predict_score",
    "slide_recent_negative_count_attr": "black_industry_filter__slide_recent_negative_count",
    "slide_recent_real_show_count_attr": "black_industry_filter__slide_recent_real_show_count",
    "slide_recent_report_count_attr": "black_industry_filter__slide_recent_report_count"
  }

  _FILTERS = [
    {
      "name": "not_in_index",
      "enable": True,
    },
    {
      "name": "server_show_aid",
      "enable": "{{enable_fountain_server_show_aid_filter}}",
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
      "enable_impression_audit_timeliness_photo_filter": "fountain_enable_impression_audit_timeliness_photo_filter",
      "impression_audit_timeliness_photo_map_attr": "fountain_impression_audit_timeliness_photo_map_str"
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
      "enable": "{{fountain_enable_upload_type_filter}}",
      "filter_type_list_attr": "filter_upload_type_lsit",
      "enable_skip_high_value_pic": "fountain_enable_skip_high_value_pic",
      "enable_skip_climbing_high_value_pic": "fountain_enable_skip_climbing_high_value_pic",
      "enable_skip_useful_high_value_pic": "fountain_enable_skip_useful_high_value_pic",
    },
    {
      "name": "picture_type",
      "enable": "{{fountain_enable_picture_type_filter}}",
      "filter_type_list_attr": "filter_picture_type_lsit",
      "enable_skip_high_value_pic": "fountain_enable_skip_high_value_pic",
      "enable_skip_climbing_high_value_pic": "fountain_enable_skip_climbing_high_value_pic",
      "enable_skip_useful_high_value_pic": "fountain_enable_skip_useful_high_value_pic",
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
      "only_filter_high_value_pic_attr": "fountain_only_filter_high_value_pic",
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
      "filter_bits_list_attr": "movie_copyright_filter_bits_list"
    },
    # 明星运营验证holdout
    {
      "name": "star_holdout_filter",
      "enable": "{{fountain_enable_star_holdout_filter}}",
      "filter_bits_list_attr": "star_holdout_filter_bits_list"
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
      "fans_bucket_list_attr": "fountain_fans_count_random_holdout_filter_fans_bucket_list"
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
      "name": "negative_retr_filter",
      "enable": "{{enable_fountain_negative_filter}}",
      "negative_retr_list_attr": "fountain_negative_i2i_retr_results"
    },
    {
      "name": "data_set_tags_filter",
      "enable": "{{fountain_enable_data_set_tags_filter}}",
      "filter_tags_list_attr": "data_set_tags_filter_tags_list"
    },
    {
      "name": "short_term_negative_filter",
      "enable": "{{enable_fountain_short_term_negative_filter}}",
      "short_minutes_cut_attr": "fountain_negative_filter_short_minutes_cut",
      "long_minutes_cut_attr": "fountain_negative_filter_long_minutes_cut"
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
      "name": "product_block_filter",  # 产品需求，只在单列分发，在双列屏蔽，老板拍板直接推全
      "enable": "{{enable_product_block_filter}}",
      "list_index_attr": "product_block_filter_list_index",
      "bit_index_attr": "product_block_filter_bit_index",
    },
    {
      "name": "merchant_cart_holdout_filter", # 挂车视频过滤
      "enable": "{{fountain_enable_merchant_cart_holdout_filter}}",
    },
    {
      "name": "high_photo_count_author_filter",
      "enable": "{{fountain_enable_high_photo_count_author_filter}}",
      "high_photo_count_author_map_ptr_attr": "high_upload_photo_author_map_ptr",
      "realshow_threshold_attr": "fountain_high_photo_count_author_photo_realshow_threshold",
      "pos_neg_ratio_coeff_attr": "fountain_high_photo_count_author_pos_neg_ratio_coeff",
      "fans_count_limit_attr": "fountain_high_photo_count_author_fans_count_limit"
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
    },
    # 魔表过滤
    {
      "name": "magic_id_filter",
      "enable": "{{fountain_enable_magic_id_filter}}",
      "produce_need_filter_magic_id_set_attr": "fountain_produce_need_filter_magic_id_set"
    },
    {
      "name": "pic_sexy_filter",
      "enable": "{{fountain_enable_pic_sexy_filter}}",
      "sexy_pic_max_cnt_attr": "fountain_sexy_pic_max_cnt",
      "sexy_pic_cnt_mode_attr": "fountain_sexy_pic_cnt_mode",
    },
    {
      "name": "pic_bad_cover_filter",
      "enable": "{{fountain_enable_pic_bad_cover_filter}}",
      "pic_bad_cover_tags_attr": "fountain_pic_bad_cover_tags_str"
    },
    {
      "name": "pic_low_quality_filter",
      "enable": "{{fountain_enable_pic_low_quality_filter}}",
      "pic_low_quality_filter_thresh_attr": "fountain_pic_low_quality_filter_thresh_list",
      "explore_pic_low_quality_tag_list_attr": "fountain_pic_low_quality_tag_list",
    },
    {
      "name": "pic_low_cost_filter",
      "enable": "{{fountain_enable_pic_low_cost_filter}}",
      "explore_low_cost_pic_max_cnt_attr": "fountain_low_cost_pic_max_cnt",
      "explore_low_cost_pic_cnt_mode_attr": "fountain_low_cost_pic_cnt_mode",
    },
    {
      "name": "pic_hack_act_filter",
      "enable": "{{fountain_enable_pic_hack_act_filter}}",
      "explore_hack_act_pic_tags_attr": "fountain_hack_act_pic_tags_str",
      "explore_hack_act_pic_types_attr": "fountain_hack_act_pic_types_str",
      "explore_hack_act_pic_max_cnt_attr": "fountain_hack_act_pic_max_cnt",
      "explore_hack_act_pic_cnt_mode_attr": "fountain_hack_act_pic_cnt_mode",
    },
    # 搬运视频holdout
    {
      "name": "repost_photo_filter",
      "enable": "{{fountain_enable_repost_photo_filter}}",
    },
    # 营销感过滤
    {
      "name": "sirius_distribution_photo_filter",
      "enable": "{{fountain_enable_sirius_distribution_photo_filter}}",
      "filter_tags_list_attr": "fountain_sirius_distribution_photo_tags_list",
    },
    # live_photo_flag 过滤
    {
      "name": "live_photo_flag_filter",
      "enable": "{{fountain_enable_live_photo_flag_filter}}",
      "live_photo_flag_thres_attr": "fountain_live_photo_flag_default_thres"
    },
    # 社区负向作者过滤 @liuhao07
    {
      "name": "negative_aid_filter",
      "enable": "{{fountain_enable_negative_aid_filter}}",
      "negative_aid_set_ptr_attr": "negative_aid_set_ptr"
    },
    # emotions_pic 过滤
    {
      "name": "emotions_pic_filter",
      "enable": "{{fountain_enable_emotions_pic_filter}}",
      "emotions_pic_filter_ratio_attr": "fountain_emotions_pic_filter_ratio",
      "emotions_pic_show_count_threshold_attr": "fountain_emotions_pic_show_count_threshold"
    },
    # 图文负向作者过滤
    {
      "name": "pic_author_filter",
      "enable": "{{fountain_enable_pic_author_filter}}",
      "author_grade_thresh_attr": "fountain_pic_author_grade_thresh",
      "author_punish_cnt_mode_attr": "fountain_pic_author_punish_cnt_mode",
      "author_filter_markcode_attr": "fountain_pic_author_filter_markcode",
      "author_punish_markcode_attr": "fountain_pic_author_punish_markcode"
    },
    # 新回作者内容控制 @liuhao07
    # 显式判断新回人群逻辑删除 to_be_delete = 2024-09-20
    {
      "name": "tnu_content_control_filter",
      "enable": "{{return fountain_enable_tnu_content_control_filter == 1 and uIsExploreTnuCrowdUser == 1}}",
      "enable_filter_audit_cold_review_attr": "fountain_enable_tnu_filter_audit_cold_review",
      "enable_filter_hetu_tags_attr": "fountain_enable_tnu_filter_hetu_tags",
      "enable_filter_mmu_merchant_hetu_photo_attr": "fountain_enable_tnu_filter_mmu_merchant_hetu_photo",
      "filter_ratio_attr": "fountain_enable_tnu_filter_mmu_merchant_hetu_photo_ratio",
      "hetu_tags_list_attr": "fountain_tnu_content_control_filter_hetu_tags",
      "cold_review_audit_tags_list_attr": "fountain_tnu_content_control_filter_audit_cold_review_tags",
    },
    {
      "name": "teenager_author_filter",
      "enable": "{{fountain_enable_teenager_author_filter}}",
      "show_count_limit_attr": "fountain_teenager_author_filter_show_limit_count"
    },
    # 青少年视频日龄概率过滤
    {
      "name": "teenager_age_prob_filter",
      "enable": "{{fountain_enable_teenager_age_prob_filter}}",
      "age_is_teenager_attr": "is_teenager",
      "teenager_age_prob_map_str_attr": "fountain_teenager_age_prob_filter_map_str",
      "teenager_age_prob_weight_attr": "fountain_teenager_age_prob_filter_weight",
    },
    # 图文生态负向特征过滤：高举报
    {
      "name": "pic_ecology_high_report_filter",
      "enable": "{{enable_fountain_pic_ecology_high_report_filter}}",
      "explore_pic_ecology_high_report_rate_threshold_attr": "fountain_pic_ecology_high_report_rate_threshold",
      "explore_pic_ecology_high_report_count_threshold_attr": "fountain_pic_ecology_high_report_count_threshold",
      "pic_ecology_high_report_fans_count_threshold_attr": "fountain_pic_ecology_high_report_fans_count_threshold"
    },
    # 图文生态负向特征过滤：高负正反馈率
    {
      "name": "pic_ecology_high_neg_pos_rate_filter",
      "enable": "{{enable_fountain_pic_ecology_high_neg_pos_rate_filter}}",
      "explore_pic_ecology_high_neg_pos_rate_threshold_attr": "fountain_pic_ecology_high_neg_pos_rate_threshold"
    },
    # 图文生态负向特征过滤：高短播
    {
      "name": "pic_ecology_high_short_play_rate_filter",
      "enable": "{{enable_fountain_pic_ecology_high_short_play_rate_filter}}",
      "explore_pic_ecology_high_short_play_rate_threshold_attr": "fountain_pic_ecology_high_short_play_rate_threshold",
      "explore_pic_ecology_neg_rate_threshold_attr": "fountain_pic_ecology_neg_rate_threshold"
    },
    # 图文生态负向特征过滤：bad avg view ( 高短播且高次均 )
    {
      "name": "pic_ecology_bad_avg_time_filter",
      "enable": "{{enable_fountain_pic_ecology_bad_avg_time_filter}}",
      "explore_pic_ecology_bad_view_time_for_short_play_rate_threshold_attr": "fountain_pic_ecology_bad_view_time_for_short_play_rate_threshold",
      "explore_pic_ecology_bad_view_time_threshold_attr": "fountain_pic_ecology_bad_view_time_threshold"
    },
    # 图文生态负向特征过滤：综合互动相关
    {
      "name": "pic_ecology_mix_interact_rate_filter",
      "enable": "{{enable_fountain_pic_ecology_mix_interact_rate_filter}}",
      "pic_ecology_interact_rate_threshold_attr": "fountain_pic_ecology_interact_rate_threshold",
      "pic_ecology_interact_avg_view_time_threshold_attr": "fountain_pic_ecology_interact_avg_view_time_threshold",
      "pic_ecology_interact_vv_threshold_attr": "fountain_pic_ecology_interact_vv_threshold"
    },
    # 高负正反馈比视频过滤
    {
      "name": "high_emp_ntpr_filter",
      "enable": "{{fountain_enable_hig_emp_ntpr_filter}}",
      "emp_ntpr_realshow_show_high_threshold_attr": "fountain_emp_ntpr_realshow_show_high_threshold",
      "emp_ntpr_filter_threshold_attr": "fountain_emp_ntpr_filter_threshold",
      "emp_ntpr_filter_report_weight_attr": "fountain_emp_ntpr_filter_report_weight",
      "emp_ntpr_enable_adpt_threshold_by_realshow_attr": "fountain_emp_ntpr_enable_adpt_threshold_by_realshow",
      "emp_ntpr_adpt_threshold_coeff_max_attr": "fountain_emp_ntpr_adpt_threshold_coeff_max",
      "emp_ntpr_adpt_threshold_coeff_min_attr": "fountain_emp_ntpr_adpt_threshold_coeff_min",
      "emp_ntpr_adpt_threshold_alpha_attr": "fountain_emp_ntpr_adpt_threshold_alpha",
      "emp_ntpr_adpt_threshold_beta_attr": "fountain_emp_ntpr_adpt_threshold_beta",
      "emp_ntpr_adpt_threshold_omega_attr": "fountain_emp_ntpr_adpt_threshold_omega",
      "emp_ntpr_adpt_threshold_exp_upper_attr": "fountain_emp_ntpr_adpt_threshold_exp_upper",
      "emp_ntpr_enable_adpt_fountain_consume_data_attr": "fountain_emp_ntpr_enable_adpt_fountain_consume_data",
    },
    # 举报河图退场
    {
      "name": "short_term_report_filter",
      "enable": "{{enable_fountain_short_term_report_filter}}",
      "short_minutes_cut_attr": "fountain_report_filter_short_minutes_cut",
      "long_minutes_cut_attr": "fountain_report_filter_long_minutes_cut"
    },
    # 高发布或低质作者过滤
    {
      "name": "pic_ecology_high_release_author_filter",
      "enable": "{{fountain_enable_pic_ecology_high_release_author_filter}}",
      "filter_bits_list_attr": "fountain_pic_ecology_high_release_author_bits_list"
    },
    # 高删文作者过滤
    {
      "name": "pic_ecology_high_delete_author_filter",
      "enable": "{{fountain_enable_pic_ecology_high_delete_author_filter}}",
      "filter_bits_list_attr": "fountain_pic_ecology_high_delete_author_bits_list"
    },
    # 图文 mmu hetu tag 过滤
    {
      "name": "pic_mmu_hetu_tag_filter",
      "enable": "{{fountain_enable_pic_mmu_hetu_tag_filter}}",
      "mmu_tag_prob_str_attr": "fountain_pic_filter_mmu_tag_prob_str",
      "mmu_tag_skip_hv_str_attr": "fountain_pic_filter_mmu_tag_skip_hv_str",
      "mmu_tag_vv_thr_str_attr": "fountain_pic_filter_mmu_tag_vv_thr_str",
    },
    # 大模型过滤 liuhao07
    {
      "name": "llm_negative_photos_filter",
      "enable": "{{fountain_enable_llm_negative_photos_filter}}",
      "teenager_filter_tag_map_str_attr": "fountain_llm_negative_photos_filter_teenager_tag_map_str",
      "filter_tag_map_str_attr": "fountain_llm_negative_photos_filter_tag_map_str",
      "is_teenager_attr": "is_teenager",
      "show_count_limit_map_str_attr": "fountain_llm_negative_photos_filter_show_count_limit_map_str",
      "enable_filter_no_impression_audit_result_attr": "fountain_llm_negative_photos_filter_impression_audit_result",
      "filter_impression_audit_level_attr": "fountain_llm_negative_photos_filter_impression_audit_level",
      "report_count_coeff_attr": "fountain_llm_negative_photos_filter_report_count_coeff",
      "report_ratio_coeff_attr": "fountain_llm_negative_photos_filter_report_ratio_coeff"
    },
    # 图文 data_set_tags_bit 过滤
    {
      "name": "pic_data_set_tags_bit_filter",
      "enable": "{{fountain_enable_pic_data_set_tags_bit_filter}}",
      "pic_filter_bits_str_attr": "fountain_pic_filter_data_set_tags_bits_str",
      "pic_punish_bits_str_attr": "fountain_pic_punish_data_set_tags_bits_str",
      "skip_filter_mark_cod_str_attr": "fountain_pic_skip_filter_mark_cod_str",
      "punish_vv_thresh_attr": "fountain_pic_punish_data_set_tags_bit_vv_thresh",
      "punish_filter_prob_attr": "fountain_pic_punish_data_set_tags_bit_filter_prob"
    },
    # 内流商铺分过滤
    {
      "name": "author_shop_score_filter",
      "enable": "{{fountain_enable_author_shop_score_filter}}",
      "author_shop_score_limit_attr": "fountain_author_shop_score_filter_limit_count",
      "author_shop_zero_protect_attr": "fountain_enable_author_shop_zero_protect"
    },
    {
      "name": "author_goods_score_filter",
      "enable": "{{fountain_enable_author_goods_score_filter}}",
      "author_goods_score_limit_attr": "fountain_author_goods_score_filter_limit_count",
      "author_goods_zero_protect_attr": "fountain_enable_author_goods_zero_protect"
    },
    # 图文安全审过滤
    {
      "name": "pic_secure_grade_filter",
      "enable": "{{fountain_enable_pic_secure_grade_filter}}",
      "secure_grade_filter_code_attr": "fountain_pic_secure_grade_filter_code_str",
      "secure_grade_punish_code_attr": "fountain_pic_secure_grade_punish_code_str",
      "skip_audit_b_second_tags_attr": "foutain_pic_secure_grade_filter_skip_audit_b_second_tags_str"
    },
    # 营销号单图低综合互动率过滤
    {
      "name": "pic_mix_interact_rate_filter",
      "enable": "{{fountain_enable_pic_mix_interact_rate_filter}}",
      "base_vv_threshold_attr": "fountain_pic_mix_interact_rate_filter_base_vv_threshold",
      "author_filter_mark_cod_str_attr": "fountain_pic_mix_interact_rate_filter_author_mark_cod_str",
      "interact_rate_thresholds_str_attr": "fountain_pic_mix_interact_rate_thresholds_str",
      "vv_thresholds_str_attr": "fountain_pic_mix_interact_rate_filter_vv_thresholds_str",
      "filter_probs_str_attr": "fountain_pic_mix_interact_rate_filter_probs_str",
    },
    {
      "name": "high_report_photo_filter",
      "enable": "{{fountain_enable_high_report_photo_filter}}",
      "realshow_threshold_attr": "fountain_high_report_photo_filter_realshow_threshold",
      "repoprt_ratio_limit_attr": "fountain_high_report_photo_filter_report_ratio_limit",
    },
    # 营销号泛单图过滤
    {
      "name": "marketing_static_video_filter",
      "enable": "{{fountain_enable_marketing_static_video_filter}}",
      "static_video_tag_id_attr": "fountain_static_video_hetu_tag_id",
      "static_video_tag_prob_thd_attr": "fountain_static_video_hetu_tag_prob_thd",
      "marketing_mark_cod_str_attr": "fountain_marketing_static_video_filter_mark_cod_str",
      "base_vv_threshold_attr": "fountain_marketing_static_video_filter_base_vv_threshold",
      "interact_rate_thresholds_str_attr": "fountain_marketing_static_video_filter_interact_rate_thresholds_str",
      "vv_thresholds_str_attr": "fountain_marketing_static_video_filter_vv_thresholds_str",
      "filter_probs_str_attr": "fountain_marketing_static_video_filter_probs_str"
    },
    {
      "name": "coldstart_holdout_filter",
      "enable": "{{fountain_enable_coldstart_holdout_filter}}",
      "nebula_thanos_realshow_limit_attr": "fountain_coldstart_holdout_filter_nebula_thanos_realshow_limit",
      "guarantee_rank_limit_attr": "fountain_coldstart_holdout_filter_guarantee_rank_limit",
      "enable_filter_if_double_shield_attr": "fountain_coldstart_holdout_filter_enable_filter_if_double_shield"
    },
    {
      "name": "sexy_induce_author_filter",
      "enable": "{{return fountain_enable_sexy_induce_author_filter == 1 and uIsExploreTnuCrowdUser == 1}}",
      "sexy_induce_photo_set_ptr_attr": "sexy_induce_photo_set_ptr"
    },
    {
      "name": "poor_quality_author_filter",
      "enable": "{{fountain_enable_poor_quality_author_filter}}",
      "enable_filter_by_gaofen_signs_uids": "fountain_enable_filter_by_gaofen_signs_uids",
      "gaofen_signs_uids_set_ptr_attr": "gaofen_signs_uids_set_ptr",
      "enable_filter_by_hierarchy_label_uids": "fountain_enable_filter_by_hierarchy_label_uids",
      "hierarchy_label_uids_map_ptr_attr": "hierarchy_label_uids_map_ptr",
      "hierarchy_label_uids_filter_ratio": "fountain_hierarchy_label_uids_filter_ratio"
    },
    {
      "name": "fangpin_aid_filter",
      "enable": "{{fountain_enable_fangpin_aid_filter}}"
    },
    # 图文低成本营销号过滤
    {
      "name": "pic_low_cost_marketing_filter",
      "enable": "{{fountain_enable_pic_low_cost_marketing_filter}}",
      "low_cost_markcode_attr": "fountain_low_cost_markcode_str",
      "yanghao_markcode_attr": "fountain_yanghao_markcode_str"
    },
    # 小程序 holdout 过滤
    {
      "name": "plc_business_type_filter",
      "enable": "{{fountain_enable_plc_business_type_filter}}",
      "filter_tags_list_attr": "fountain_plc_business_type_filter_tags_list"
    },
    #  冷启价值验证屏蔽
    {
      "name": "valuable_photo_open_filter",
      "enable": "{{fountain_enable_valuable_photo_open_filter}}",
      "valuable_rules_kconf_key_attr": "fountain_valuable_open_filter_kconf_key"
    },
    # 图文劣质作者过滤
    {
      "name": "pic_liezhi_author_filter",
      "enable": "{{fountain_enable_pic_liezhi_author_filter}}",
      "author_liezhi_pic_count_thresh_attr": "fountain_author_liezhi_pic_count_thresh",
      "author_fans_count_thresh_attr": "fountain_pic_liezhi_author_filter_fans_count_thresh"
    },
    # 图文新星计划作者过滤
    {
      "name": "pic_xinxing_author_filter",
      "enable": "{{fountain_enable_pic_xinxing_author_filter}}",
    },
    # 图文新回审劣质过滤
    {
      "name": "pic_audit_cold_review_level_filter",
      "enable": "{{fountain_enable_pic_audit_cold_review_level_filter}}",
      "filter_audit_cold_review_level_str_attr": "fountain_pic_filter_audit_cold_review_level_str"
    },
    # 长实况图过滤
    {
      "name": "pic_long_live_photo_filter",
      "enable": "{{fountain_enable_pic_long_live_photo_filter}}",
      "pic_long_live_photo_vv_thresh_attr": "fountain_pic_long_live_photo_vv_thresh",
      "pic_long_live_photo_duration_thresh_attr": "fountain_pic_long_live_photo_duration_thresh"
    },
    # 热点holdout
    {
      "name": "hot_spot_holdout_filter",
      "enable": "{{fountain_enable_hot_spot_holdout_filter}}",
      "filter_level_list_attr": "hot_spot_filter_level_list",
      "filter_source_list_attr": "hot_spot_filter_source_list"
    },
    # 封面二维码内容过滤 使用 hetu 判断
    {
      "name": "cover_qr_code_filter",
      "enable": "{{fountain_enable_cover_qr_code_filter}}",
    },
    # 封面敏感词过滤 使用 mem data 判断
    {
      "name": "cover_sensitive_word_filter",
      "enable": "{{fountain_enable_cover_sensitive_word_filter}}",
      "cover_sensitive_word_ptr_attr": "illegal_word_pids_ptr"
    },
    #  作者冷启价值验证屏蔽
    {
      "name": "valuable_author_photo_open_filter",
      "enable": "{{fountain_enable_valuable_author_photo_open_filter}}",
      "valuable_author_photo_rules_name_attr": "fountain_valuable_author_photo_rules_name"
    },
    # 社区问卷过滤
    {
      "name": "community_survey_filter",
      "enable": "{{fountain_enable_community_survey_filter}}",
      "survey_filter_markcode_str_attr": "fountain_filter_community_survey_markcode_str",
      "survey_markcode_2_cert_ratio_threshold_str_attr": "fountain_filter_survey_markcode_2_cert_ratio_threshold_str",
      "survey_markcode_2_cert_cnt_threshold_str_attr": "fountain_filter_survey_markcode_2_cert_cnt_threshold_str",
    },
    # 观感审分数过滤
    {
      "name": "cover_sense_view_predict_score_filter",
      "enable": "{{fountain_enable_cover_sense_view_predict_score_filter}}",
      "cover_view_predict_score_thres_attr": "fountain_filter_cover_view_predict_score_thres",
      "sense_view_predict_score_thres_attr": "fountain_filter_sense_view_predict_score_thres",
      "enable_filter_cover_view_missing_score_attr": "fountain_enable_filter_cover_view_missing_score",
      "enable_filter_sense_view_missing_score_attr": "fountain_enable_filter_sense_view_missing_score"
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
      "enable": "{{fountain_enable_public_opinion_tagnex_filter}}",
      "public_opinion_tagnex_str_attr": "fountain_filter_public_opinion_tagnex_str",
    },
    # 舆情灰产兜底策略，常规状态无需开启
    {
      "name": "high_hate_report_filter",
      "enable": "{{fountain_enable_high_hate_report_filter}}",
      "recent_real_show_thres_attr": "fountain_filter_recent_real_show_thres_attr",
      "recent_hate_ratio_thres_attr": "fountain_filter_recent_hate_ratio_thres_attr",
      "recent_report_ratio_thres_attr": "fountain_filter_recent_report_ratio_thres_attr",
      "recent_hate_count_thres_attr": "fountain_filter_recent_hate_count_thres_attr",
      "recent_report_count_thres_attr": "fountain_filter_recent_report_count_thres_attr",
      "real_show_thres_attr": "fountain_filter_real_show_thres_attr",
      "hate_ratio_thres_attr": "fountain_filter_hate_ratio_thres_attr",
      "report_ratio_thres_attr": "fountain_filter_report_ratio_thres_attr",
      "hate_count_thres_attr": "fountain_filter_hate_count_thres_attr",
      "report_count_thres_attr": "fountain_filter_report_count_thres_attr"
    },
    # 兜底过滤
    {
      "name": "topk_audit_not_pass_filter",
      "enable": "{{enable_retr_filter_downgrade}}"
    }
  ]

  def _get_commmo_abtest_params(self):
    """
    填充首屏和非首屏共用 ab 参数
    """
    self \
      .get_abtest_params(
        biz_name = "RECO_RPC",
        ab_params = enrich_ab_param([
          ("fountain_impression_audit_second_level_white_tags", "2008292,2008293"),
          ("fountain_high_hot_audit_second_level_white_tags", "2008226"),
          ("fountain_topk_audit_second_level_white_tags", "2008234,2008235,2008236"),
          ("fountain_topk_audit_second_level_black_tags", ""),
          ("fountain_high_hot_audit_second_level_black_tags", ""),
          ("fountain_impression_audit_second_level_black_tags", ""),
          ("skip_fountain_topk_audit_filter_new", 1),
          ("fountain_topk_audit_bad_recall_filter", False),
          ("fountain_topk_audit_bad_recall_filter_use_global", False),
          ("fountain_topk_audit_bad_recall_filter_credible_ques_cnt", 30),
          ("fountain_topk_audit_bad_recall_filter_pos_threshold", 1.0),
          ("fountain_topk_audit_bad_recall_filter_mode", 0),
          ("fountain_topk_audit_bad_recall_filter_unsure_threshold", 1.0),
          ("fountain_topk_audit_bad_recall_filter_neg_threshold", 1.0),
          ("fountain_topk_audit_bad_recall_filter_hate_threshold", 1.0),
          ("fountain_skip_mid_video_photo_filter", 1),
          ("fountain_questionnaire_filter_min_total_count", 100),
          ("enable_fountain_questionnaire_info_filter_v2",False),
          ("fountain_questionaire_info_replace_topk_result",0),
          ("fountain_ques_info_topk_level_threshold", 3),
          ("fountain_ques_info_audit_level_threshold", 3),
          ("fountain_questionnaire_filter_min_positive_rate", 0.5),
          ("fountain_questionnaire_filter_max_negative_rate", 0.5),
          ("fountain_questionnaire_filter_max_negative_high_rate", 0.5),
          ("fountain_questionnaire_filter_max_unsure_rate", 0.5),
          ("fountain_questionaire_thompson_filter",False),
          ("fountain_questionaire_filter_neg_weight", 1.0),
          ("fountain_questionaire_filter_pos_weight", 1.0),
          ("fountain_questionaire_filter_unsure_weight", 1.0),
          ("fountain_questionaire_filter_click_weight", 1.0),
          ("fountain_questionaire_filter_unclick_weight", 1.0),
          ("fountain_questionaire_use_global_data", False), 
          ("fountain_skip_impression_audit_gray_filter", 1),
          ("fountain_impression_audit_gray_show_limit", 50000),
          ("fountain_enable_explore_punish_city_filter", 0),
          ("fountain_thompson_filter_threshold",0.85),
          ("enable_fountain_thompson_filter_use_fountain",0),
          ("enable_fountain_thompson_filter_use_thanos",0),
          ("enable_fountain_thompson_filter_use_nebula",0),
          ("enable_fountain_thompson_filter_use_explore", 1),
          ("fountain_thompson_filter_ctr_weight",0.0),
          ("fountain_thompson_filter_ltr_weight",1.0),
          ("fountain_thompson_filter_wtr_weight",0.0),
          ("fountain_thompson_filter_ftr_weight",0.0),
          ("fountain_thompson_filter_cmtr_weight",0.0),
          ("fountain_thompson_filter_time_weight",0.0),
          ("fountain_thompson_filter_normal_time_weight",0.0),
          ("fountain_thompson_filter_realshow_weight",0.0),
          ("fountain_enable_interaction_base",0),
          ("fountain_thompson_filter_realshow_divisor",10000.0),
          ("fountain_enable_zero_duration_filter", False),
          ("fountain_limit_hate_reason", False),
          ("enable_fountain_emp_phtr_filter",False),
          ("fountain_enable_hate_cost", False),
          ("fountain_emphtr_filter_threshold",0.01),
          ("fountain_emphtr_filter_ctr_weight", 0.01),
          ("fountain_emphtr_filter_ltr_weight", 0.01),
          ("fountain_emphtr_filter_wtr_weight", 0.01),
          ("fountain_emphtr_filter_ftr_weight", 0.01),
          ("fountain_emphtr_filter_cmtr_weight", 0.01),
          ("fountain_emphtr_filter_time_weight", 0.0),
          ("fountain_emphtr_filter_normal_time_weight", 0.01),
          ("fountain_emphtr_realshow_show_threshold", 100),
          ("fountain_emphtr_realshow_show_threshold", 100),
          ("fountain_enable_adpt_threshold", False),
          ("fountain_emphtr_filter_threshold_list", ""),
          ("enable_fountain_negative_filter", False),
          ("enable_fountain_short_term_negative_filter", False),
          ("fountain_negative_filter_short_minutes_cut", 5),
          ("fountain_negative_filter_long_minutes_cut", 30),
          ("enable_fountain_short_term_report_filter", False),
          ("fountain_report_filter_short_minutes_cut", 5),
          ("fountain_report_filter_long_minutes_cut", 30),
          ("enable_fountain_over_days_filter", True),
          ("fountain_over_days_filter_days_limit", 180),
          ("fountain_entertainment_hetu_tag_str", ""),
          ("fountain_entertainment_hetu_days_limit_attr", 90),
          ("enable_fountain_over_days_filter_low_like", False),
          ("fountain_over_days_filter_low_like_limit", 50),
          ("fountain_over_days_filter_low_like_days_limit", 7),
          ("fountain_over_days_filter_topn_screen_map", ""),
          ("fountain_enable_duration_random_filter", False),
          ("fountain_duration_random_ignore_reasons", ""),
          ("fountain_duration_random_default_cut_off_ratio", "0:0.5;1:0.2;2:0.6"),
          ("fountain_duration_random_adjust_cut_off_ratio", "0:0.5;1:0.2;2:0.6"),
          ("fountain_duration_random_enable_random_cut_off", True),
          ("fountain_duration_random_lt_longview_ratio_threshold", 0.05),
          ("fountain_duration_random_sharp_change_confidence_threshold", 0.8),
          ("fountain_enable_duration_emp_watchtime_sample_filter", False),
          ("fountain_duration_sample_filter_threshold_str", ""),
          ("fountain_duration_sample_filter_base_number", 2.0),
          ("fountain_duration_sample_filter_multi_number", 10.0),
          ("fountain_enable_short_duration_filter", True),
          ("fountain_short_duration_filter_limit", 3),
          ("fountain_follow_author_filter_timegap", 0),
          ("enable_fountain_multi_audit_gray_filter", False),
          ("fountain_multi_audit_gray_filter_count_threshold", 2),
          ("fountain_multi_audit_gray_filter_days_limit", 0),
          ("enable_fountain_high_hot_audit_gray_show_filter", True),
          ("fountain_high_hot_audit_gray_show_filter_stat_all_page", False),
          ("fountain_high_hot_audit_gray_show_filter_threshold", 300000),
          ("fountain_enable_audit_rule_adjust_filter", False),
          ("fountain_audit_rule_adjust_tags_str", ""),
          ("fountain_enable_merchant_holdout_filter", False),
          ("fountain_enable_filter_living_merchant_photo", False),
          ("fountain_enable_filter_living_merchant_author", False),
          ("fountain_enable_risk_man_risk_photo_filter", True),
          ("fountain_enable_audit_user_experiment_level_filter", False),
          ("fountain_audit_user_experiment_level_map_str", "500:0"),
          ("fountain_personified_author_filter_flag", 1),
          ("fountain_enable_movie_copyright_holdout_filter", False),
          ("fountain_movie_copyright_filter_bits_list_str", ""),
          ("fountain_enable_star_holdout_filter", False),
          ("fountain_star_holdout_filter_bits_list_str", ""),
          ("fountain_young_inc_category_list_str", ""),
          ("fountain_young_inc_category_hetu_list_str", ""),
          ("fountain_young_inc_tags_filter_flag", 0),
          ("fountain_young_inc_tags_filter_ratio", 1.0),
          ("fountain_young_inc_tags_filter_prime", 503),
          ("fountain_young_inc_tags_filter_upload_time_limit", 0),
          ("fountain_fans_count_random_holdout_filter_ratio", 1.0),
          ("fountain_fans_count_random_holdout_filter_prime", 423),
          ("fountain_fans_count_random_holdout_filter_fans_bucket_str", ""),
          ("fountain_enable_skip_high_value_pic", 0),
          ("fountain_enable_skip_climbing_high_value_pic", 0),
          ("fountain_enable_skip_useful_high_value_pic", 0),
          ("enable_fountain_over_days_filter_audit", False),
          ("fountain_impression_audit_gray_hours_limit", 48),
          ("fountain_impression_audit_normal_days_limit", 30),
          ("fountain_impression_audit_high_quality_days_limit", 30),
          ("fountain_high_hot_audit_gray_hours_limit", 48),
          ("fountain_high_hot_audit_normal_days_limit", 90),
          ("fountain_high_hot_audit_high_quality_days_limit", 180),
          ("fountain_enable_audit_hack_photo_filter", False),
          ("audit_hack_tags_str", "2037808,2037809,2037810,2037811,2037812,2037813"),
          ("audit_hack_photo_filter_min_show", 10000),
          ("audit_hack_photo_filter_max_ltr", 0.08),
          ("audit_hack_photo_filter_max_wtr", 0.01),
          ("audit_hack_photo_filter_max_cmtr", 0.02),
          ("fountain_enable_data_set_tags_filter", False),
          ("fountain_data_set_tags_filter_tags_list_str", ""),
          ("fountain_enable_data_set_tags_bit_filter", False),
          ("fountain_data_set_tags_bit_filter_bits_list_str", ""),
          ("fountain_user_reco_neg_filter_candidate_count_limit", 1500),
          ("fountain_enable_quality_audit_filter_final", True),
          ("fountain_quality_audit_filter_tags_list_str_final", "2147250,2147252,2147253,2147254"),
          ("fountain_enable_hetu_author_category_holdout_filter", False),
          ("fountain_hetu_author_category_holdout_filter_fans_count_limit", 100000),
          ("fountain_hetu_author_category_holdout_filter_list_str", ""),
          ("fountain_enable_merchant_cart_holdout_filter", False),
          ("fountain_enable_high_photo_count_author_filter", False),
          ("fountain_high_photo_count_author_photo_realshow_threshold", 500000),
          ("fountain_high_photo_count_author_pos_neg_ratio_coeff", 1.0),
          ("fountain_high_photo_count_author_fans_count_limit", 100000),
          ("fountain_enable_douyin_author_holdout_filter", False),
          ("fountain_douyin_author_holdout_filter_flag", 0),
          ("fountain_douyin_author_holdout_filter_fans_count_limit", 100000),
          ("fountain_douyin_author_holdout_filter_list_str", ""),
          ("fountain_enable_mid_long_video_holdout_filter", False),
          ("fountain_mid_long_video_holdout_filter_duration_lowerbound", 0),
          ("fountain_mid_long_video_holdout_filter_duration_upperbound", 0),
          ("fountain_mid_long_video_holdout_filter_tags_list_str", ""),
          ("fountain_enable_produce_type_filter", False),
          ("fountain_produce_magic_type_filter_flag", 0),
          ("fountain_produce_need_filter_magic_type_str", ""),
          ("fountain_enable_magic_id_filter", False),
          ("fountain_enable_pic_sexy_filter", False),
          ("fountain_sexy_pic_max_cnt", 100000000),
          ("fountain_sexy_pic_cnt_mode", 0),
          ("fountain_enable_pic_bad_cover_filter", False),
          ("fountain_pic_bad_cover_tags_str", "2023746,2231037"),
          ("fountain_enable_pic_low_quality_filter", False),
          ("fountain_enable_gen_fangpin_aid", False),
          ("fountain_pic_low_quality_tag_str", "4009000,4009001,4009002,4009003,4009004,4009006,4009007"),
          ("fountain_pic_low_quality_filter_thresh_list_str", "-1,-1,-1,-1,-1,-1,-1"),
          ("fountain_enable_pic_low_cost_filter", False),
          ("fountain_low_cost_pic_max_cnt", 100000000),
          ("fountain_low_cost_pic_cnt_mode", 0),
          ("fountain_enable_pic_hack_act_filter", False),
          ("fountain_hack_act_pic_tags_str", "2037808,2037809,2037810,2037811,2037812,2037813"),
          ("fountain_hack_act_pic_types_str", "1"),
          ("fountain_hack_act_pic_max_cnt", 100000000),
          ("fountain_hack_act_pic_cnt_mode", 0),
          ("fountain_enable_repost_photo_filter", False),
          ("fountain_enable_sirius_distribution_photo_filter", False),
          ("fountain_sirius_distribution_photo_tags_list_str", ""),
          ("fountain_enable_live_photo_flag_filter", False),
          ("fountain_live_photo_flag_default_thres", 0),
          ("fountain_enable_negative_aid_filter", False),
          ("fountain_enable_emotions_pic_filter", False),
          ("fountain_emotions_pic_filter_ratio", 1.0),
          ("fountain_emotions_pic_show_count_threshold", 50000),
          ("fountain_enable_pic_author_filter", False),
          ("fountain_pic_author_grade_thresh", 0),
          ("fountain_pic_author_punish_cnt_mode", 0),
          ("fountain_pic_author_filter_markcode", ""),
          ("fountain_pic_author_punish_markcode", ""),
          ("fountain_pic_author_filter_enable_specific_age_seg_diff", False),
          ("fountain_pic_author_filter_specific_age_seg_str", "5,6,7"),
          ("fountain_pic_author_filter_markcode_for_specific_age_seg", ""),
          ("fountain_pic_author_punish_markcode_for_specific_age_seg", ""),
          ("fountain_enable_tnu_content_control_filter", False),
          ("fountain_enable_tnu_filter_audit_cold_review", False),
          ("fountain_enable_tnu_filter_hetu_tags", False),
          ("fountain_enable_tnu_filter_mmu_merchant_hetu_photo", False),
          ("fountain_enable_tnu_filter_mmu_merchant_hetu_photo_ratio", 0.0),
          ("fountain_tnu_content_control_filter_hetu_tags_str", ""),
          ("fountain_tnu_content_control_filter_audit_cold_review_tags_str", ""),
          ("fountain_mmu_merchant_tag_black_tags_str", ""),
          ("fountain_enable_teenager_author_filter", False),
          ("fountain_teenager_author_filter_show_limit_count", 2000),
          ("fountain_enable_teenager_age_prob_filter", False),
          ("fountain_teenager_age_prob_filter_map_str", ""),
          ("fountain_teenager_age_prob_filter_weight", 0.0),
          ("fountain_enable_hig_emp_ntpr_filter", False),
          ("fountain_emp_ntpr_realshow_show_high_threshold", 12000),
          ("fountain_emp_ntpr_filter_threshold", 0.5),
          ("fountain_emp_ntpr_filter_report_weight", 0.5),
          ("fountain_emp_ntpr_enable_adpt_threshold_by_realshow", False),
          ("fountain_emp_ntpr_adpt_threshold_coeff_max", 1.0),
          ("fountain_emp_ntpr_adpt_threshold_coeff_min", -1.0),
          ("fountain_emp_ntpr_adpt_threshold_alpha", 1.0),
          ("fountain_emp_ntpr_adpt_threshold_beta", 2.0),
          ("fountain_emp_ntpr_adpt_threshold_omega", 5000000.0),
          ("fountain_emp_ntpr_adpt_threshold_exp_upper", 10.0),
          ("fountain_emp_ntpr_enable_adpt_fountain_consume_data", 0),
          ("fountain_enable_author_shop_score_filter", False),
          ("fountain_enable_author_goods_score_filter", False),
          ("fountain_author_shop_score_filter_limit_count", 4.4),
          ("fountain_author_goods_score_filter_limit_count", 4.4),
          ("fountain_enable_author_shop_zero_protect", 1),
          ("fountain_enable_author_goods_zero_protect", 1),
          {
            "attr_name": "enable_fountain_pic_ecology_high_report_filter",
            "param_name": "enable_fountain_pic_ecology_high_report_filter",
            "default_value": False
          },
          {
            "attr_name": "enable_fountain_pic_ecology_high_neg_pos_rate_filter",
            "param_name": "enable_fountain_pic_ecology_high_neg_pos_rate_filter",
            "default_value": False
          },
          {
            "attr_name": "enable_fountain_pic_ecology_bad_avg_time_filter",
            "param_name": "enable_fountain_pic_ecology_bad_avg_time_filter",
            "default_value": False
          },
          {
            "attr_name": "enable_fountain_pic_ecology_high_short_play_rate_filter",
            "param_name": "enable_fountain_pic_ecology_high_short_play_rate_filter",
            "default_value": False
          },
          {
            "attr_name": "fountain_pic_ecology_high_report_rate_threshold",
            "param_name": "fountain_pic_ecology_high_report_rate_threshold",
            "default_value": 0.000254
          },
          {
            "attr_name": "fountain_pic_ecology_high_report_count_threshold",
            "param_name": "fountain_pic_ecology_high_report_count_threshold",
            "default_value": 10
          },
          {
            "attr_name": "fountain_pic_ecology_high_neg_pos_rate_threshold",
            "param_name": "fountain_pic_ecology_high_neg_pos_rate_threshold",
            "default_value": 0.226
          },
          {
            "attr_name": "fountain_pic_ecology_high_short_play_rate_threshold",
            "param_name": "fountain_pic_ecology_high_short_play_rate_threshold",
            "default_value": 0.5165
          },
          {
            "attr_name": "fountain_pic_ecology_neg_rate_threshold",
            "param_name": "fountain_pic_ecology_neg_rate_threshold",
            "default_value": 0.00136
          },
          {
            "attr_name": "fountain_pic_ecology_bad_view_time_for_short_play_rate_threshold",
            "param_name": "fountain_pic_ecology_bad_view_time_for_short_play_rate_threshold",
            "default_value": 0.5165
          },
          {
            "attr_name": "fountain_pic_ecology_bad_view_time_threshold",
            "param_name": "fountain_pic_ecology_bad_view_time_threshold",
            "default_value": 17.13
          },
          ("fountain_pic_ecology_high_report_fans_count_threshold", 1000000),
          ("fountain_enable_pic_ecology_high_release_author_filter", False),
          ("fountain_enable_pic_ecology_high_delete_author_filter", False),
          ("fountain_pic_ecology_high_release_author_bits_list_str", ""),
          ("fountain_pic_ecology_high_delete_author_bits_list_str", ""),
          ("enable_fountain_pic_ecology_mix_interact_rate_filter", False),
          ("fountain_pic_ecology_interact_rate_threshold", 0.0143),
          ("fountain_pic_ecology_interact_avg_view_time_threshold", 17.13),
          ("fountain_pic_ecology_interact_vv_threshold", 10000),
          ("fountain_enable_pic_mmu_hetu_tag_filter", False),
          ("fountain_pic_filter_mmu_tag_prob_str", ""),
          ("fountain_pic_filter_mmu_tag_skip_hv_str", ""),
          ("fountain_pic_filter_mmu_tag_vv_thr_str", ""),
          ("fountain_enable_impression_audit_timeliness_photo_filter", False),
          ("fountain_impression_audit_timeliness_photo_map_str", ""),
          ("fountain_enable_llm_negative_photos_filter", False),
          ("fountain_llm_negative_photos_filter_teenager_tag_map_str", ""),
          ("fountain_llm_negative_photos_filter_tag_map_str", ""),
          ("fountain_llm_negative_photos_filter_impression_audit_result", False),
          ("fountain_llm_negative_photos_filter_impression_audit_level", 0),
          ("fountain_llm_negative_photos_filter_report_count_coeff", 0.0),
          ("fountain_llm_negative_photos_filter_report_ratio_coeff", 0.0),
          ("fountain_enable_pic_data_set_tags_bit_filter", False),
          ("fountain_pic_filter_data_set_tags_bits_str", ""),
          ("fountain_pic_punish_data_set_tags_bits_str", ""),
          ("fountain_pic_skip_filter_mark_cod_str", ""),
          ("fountain_pic_punish_data_set_tags_bit_vv_thresh", 0),
          ("fountain_pic_punish_data_set_tags_bit_filter_prob", 0.0),
          ("fountain_llm_negative_photos_filter_show_count_limit_map_str", ""),
          ("fountain_enable_pic_secure_grade_filter", False),
          ("fountain_pic_secure_grade_filter_code_str", ""),
          ("fountain_pic_secure_grade_punish_code_str", ""),
          ("foutain_pic_secure_grade_filter_skip_audit_b_second_tags_str", ""),
          ("fountain_enable_pic_mix_interact_rate_filter", False),
          ("fountain_pic_mix_interact_rate_filter_base_vv_threshold", 1000),
          ("fountain_pic_mix_interact_rate_filter_author_mark_cod_str", ""),
          ("fountain_pic_mix_interact_rate_thresholds_str", ""),
          ("fountain_pic_mix_interact_rate_filter_vv_thresholds_str", ""),
          ("fountain_pic_mix_interact_rate_filter_probs_str", ""),
          ("fountain_enable_high_report_photo_filter", False),
          ("fountain_high_report_photo_filter_realshow_threshold", 10000),
          ("fountain_high_report_photo_filter_report_ratio_limit", 1.0),
          ("fountain_enable_marketing_static_video_filter", False),
          ("fountain_static_video_hetu_tag_id", 4009921),
          ("fountain_static_video_hetu_tag_prob_thd", 0.99),
          ("fountain_marketing_static_video_filter_mark_cod_str", ""),
          ("fountain_marketing_static_video_filter_base_vv_threshold", 1000),
          ("fountain_marketing_static_video_filter_interact_rate_thresholds_str", "0.0091,0.0292,1.0"),
          ("fountain_marketing_static_video_filter_vv_thresholds_str", "1000,5000,10000"),
          ("fountain_marketing_static_video_filter_probs_str", "1.0,0.7,0.5"),
          ("fountain_enable_coldstart_holdout_filter", False),
          ("fountain_coldstart_holdout_filter_nebula_thanos_realshow_limit", 50000),
          ("fountain_coldstart_holdout_filter_guarantee_rank_limit", 5),
          ("fountain_coldstart_holdout_filter_enable_filter_if_double_shield", False),
          ("fountain_enable_sexy_induce_author_filter", False),
          ("fountain_enable_poor_quality_author_filter", False),
          ("fountain_enable_filter_by_gaofen_signs_uids", 1),
          ("fountain_enable_filter_by_hierarchy_label_uids", 0),
          ("fountain_hierarchy_label_uids_filter_ratio", 0.0),
          ("fountain_enable_fangpin_aid_filter", False),
          ("fountain_enable_pic_low_cost_marketing_filter", False),
          ("fountain_low_cost_markcode_str", ""),
          ("fountain_yanghao_markcode_str", ""),
          ("fountain_enable_plc_business_type_filter", False),
          ("fountain_plc_business_type_filter_tags_str", ""),
          ("fountain_enable_valuable_photo_open_filter", False),
          ("fountain_valuable_open_filter_kconf_key", "reco.author.vcsShieldPhotoRuleExp4"),
          ("fountain_enable_pic_liezhi_author_filter", False),
          ("fountain_author_liezhi_pic_count_thresh", 3),
          ("fountain_pic_liezhi_author_filter_fans_count_thresh", 1000000),
          ("fountain_enable_pic_liezhi_author_filter_elder", False),
          ("fountain_pic_liezhi_author_filter_elder_age_thresh", 5),
          ("fountain_author_liezhi_pic_count_thresh_elder", 3),
          ("fountain_enable_pic_xinxing_author_filter", False),
          ("fountain_enable_pic_audit_cold_review_level_filter", False),
          ("fountain_pic_filter_audit_cold_review_level_str", ""),
          ("fountain_enable_pic_long_live_photo_filter", False),
          ("fountain_pic_long_live_photo_vv_thresh", 10000),
          ("fountain_pic_long_live_photo_duration_thresh", 10.0),
          ("fountain_enable_hot_spot_holdout_filter", False),
          ("fountain_hot_spot_filter_level_list_str", ""),
          ("fountain_hot_spot_filter_source_list_str", ""),
          ("fountain_enable_cover_qr_code_filter", False),
          ("fountain_enable_cover_sensitive_word_filter", False),
          ("fountain_enable_valuable_author_photo_open_filter", False),
          ("fountain_valuable_author_photo_rules_name", "ProduceShieldExpAuthor"),
          ("fountain_enable_community_survey_filter", False),
          ("fountain_filter_community_survey_markcode_str", ""),
          ("fountain_filter_survey_markcode_2_cert_ratio_threshold_str", ""),
          ("fountain_filter_survey_markcode_2_cert_cnt_threshold_str", ""),
          ("fountain_enable_public_opinion_tagnex_filter", False),
          ("fountain_filter_public_opinion_tagnex_str", ""),
          ("fountain_enable_cover_sense_view_predict_score_filter", False),
          ("fountain_filter_cover_view_predict_score_thres", 1.0),
          ("fountain_filter_sense_view_predict_score_thres", 1.0),
          ("fountain_enable_filter_cover_view_missing_score", 0),
          ("fountain_enable_filter_sense_view_missing_score", 0)
        ]),
        prioritized_suffix = "{{_ABTEST_SUFFIX_LIST_}}",
      ) \
      .get_abtest_params(
        biz_name = "RECO_RPC",
        ab_params = [
          {
            "attr_name": "fountain_skip_filter_by_picture_single_variant_attr",
            "default_value": 0,
            "param_name": "fountain_skip_filter_by_picture_single_variant_attr",
            "param_type": "int",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "fountain_enable_xtab_life_index_filter",
            "default_value": False,
            "param_name": "fountain_enable_xtab_life_index_filter",
            "param_type": "bool",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "fountain_key_hetu_categories",
            "default_value": "4,5,7,10,11,12,16,17,18,25,26,27,36",
            "param_name": "fountain_key_hetu_categories",
            "param_type": "string",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "enable_fountain_server_show_aid_filter",
            "default_value": True,
            "param_name": "enable_fountain_server_show_aid_filter",
            "param_type": "bool",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "enable_fountain_video_filter",
            "default_value": False,
            "param_name": "enable_fountain_video_filter",
            "param_type": "bool",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "fountain_skip_filter_by_picture_variant_attr",
            "default_value": 1,
            "param_name": "fountain_skip_filter_by_picture_variant_attr",
            "param_type": "int",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "fountain_skip_filter_by_picture_set_variant_attr",
            "default_value": 0,
            "param_name": "fountain_skip_filter_by_picture_set_variant_attr",
            "param_type": "int",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "skip_fountain_filter_all_picture",
            "default_value": 1,
            "param_name": "skip_fountain_filter_all_picture",
            "param_type": "int",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "fountain_only_filter_high_value_pic",
            "default_value": 0,
            "param_name": "fountain_only_filter_high_value_pic",
            "param_type": "int",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "fountain_enable_personified_author_filter",
            "default_value": False,
            "param_name": "fountain_enable_personified_author_filter",
            "param_type": "bool",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "fountain_enable_fans_count_random_holdout_filter",
            "default_value": False,
            "param_name": "fountain_enable_fans_count_random_holdout_filter",
            "param_type": "bool",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "fountain_enable_negative_thompson_filter",
            "default_value": False,
            "param_name": "fountain_enable_negative_thompson_filter",
            "param_type": "bool",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "fountain_enable_young_inc_tags_holdout_filter",
            "default_value": False,
            "param_name": "fountain_enable_young_inc_tags_holdout_filter",
            "param_type": "bool",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "fountain_enable_upload_type_filter",
            "default_value": True,
            "param_name": "fountain_enable_upload_type_filter",
            "param_type": "bool",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "fountain_enable_picture_type_filter",
            "default_value": True,
            "param_name": "fountain_enable_picture_type_filter",
            "param_type": "bool",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "fountain_enable_light_inc_holdout_filter",
            "default_value": False,
            "param_name": "fountain_enable_light_inc_holdout_filter",
            "param_type": "bool",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
        ],
        prioritized_suffix = "{{_ABTEST_SUFFIX_LIST_}}"
      )

    return self

  def _filter_by_attr_with_perf(self, **kwargs):
    attr_name = "default_attr"
    if_skip = 0
    for key, value in kwargs.items():
      if key == "attr_name":
        attr_name= value
      elif key == "skip":
        if_skip = value

    self.filter_by_attr(
      **kwargs
    ) \
    .perflog_reason_count(
      check_point = "filter_by_" + attr_name,
      skip = if_skip
    )

    return self

  def _common_filter(self):
    """
    首屏和非首屏共用的过滤
    """
    self \
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
      .get_kconf_params(
        kconf_configs = [{
          "kconf_key": "poster.magicFace.hotMagicFaceHoldOutIdSet",
          "export_common_attr": "fountain_produce_need_filter_magic_id_set",
          "value_type": "set_int64"
        }]
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
      .if_("fountain_enable_movie_copyright_holdout_filter == 1") \
        .split_string(
          input_common_attr = "fountain_movie_copyright_filter_bits_list_str",
          output_common_attr = "movie_copyright_filter_bits_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
      .end_() \
      .if_("fountain_enable_star_holdout_filter == 1") \
        .split_string(
          input_common_attr = "fountain_star_holdout_filter_bits_list_str",
          output_common_attr = "star_holdout_filter_bits_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
      .end_() \
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
      .split_string(
        input_common_attr = "fountain_tnu_content_control_filter_audit_cold_review_tags_str",
        output_common_attr = "fountain_tnu_content_control_filter_audit_cold_review_tags",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .split_string(
        input_common_attr = "fountain_tnu_content_control_filter_hetu_tags_str",
        output_common_attr = "fountain_tnu_content_control_filter_hetu_tags",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .split_string(  # 生成 mmu 营销感标记
        input_common_attr = "fountain_mmu_merchant_tag_black_tags_str",
        output_common_attr = "mmu_merchant_tag_black_tags_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True,
      ) \
      .if_("fountain_enable_plc_business_type_filter == 1") \
        .split_string(
          input_common_attr = "fountain_plc_business_type_filter_tags_str",
          output_common_attr = "fountain_plc_business_type_filter_tags_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
      .end_() \
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
      .if_("fountain_enable_gen_fangpin_aid == 1") \
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
      .get_kconf_params(
        kconf_configs = [{
          "kconf_key": "reco.hot.recoExploreUserRiskMin",
          "export_common_attr": "user_risk_min",
          "value_type": "int64"
        }]
      ) \
      .get_kconf_params(
        kconf_configs = [
          {
            "kconf_key": "reco.index.enableWhiteListAuthor",
            "value_type": "bool",
            "default_value": False,
            "export_common_attr": "enable_author_living_whitelist_filter",
          },
          {
            "kconf_key": "reco.grprLiveAuthorWhiteList.whiteList",
            "value_type": "set_int64",
            "default_value": [],
            "export_common_attr": "author_living_whitelist",
          },
          {
            "kconf_key": "reco.grprLiveAuthorWhiteList2.whiteList",
            "value_type": "set_int64",
            "default_value": [],
            "export_common_attr": "author_living_whitelist2",
          },
          {
            "kconf_key": "reco.grprLiveAuthorWhiteList3.whiteList",
            "value_type": "set_int64",
            "default_value": [],
            "export_common_attr": "author_living_whitelist3",
          },
          {
            "kconf_key": "reco.grprLiveAuthorWhiteList4.whiteList",
            "value_type": "set_int64",
            "default_value": [],
            "export_common_attr": "author_living_whitelist4",
          },
          {
            "kconf_key": "reco.grprLiveAuthorWhiteList5.whiteList",
            "value_type": "set_int64",
            "default_value": [],
            "export_common_attr": "author_living_whitelist5",
          },
          {
            "kconf_key": "reco.index.enableBlackListAuthor",
            "value_type": "bool",
            "default_value": False,
            "export_common_attr": "enable_author_living_blacklist_filter",
          },
          {
            "kconf_key": "reco.grprLiveAuthorBlackList.heiChanBlackList",
            "value_type": "set_int64",
            "default_value": [],
            "export_common_attr": "author_living_blacklist",
          },
          {
            "kconf_key": "reco.grprLiveAuthorBlackList2.heiChanBlackList",
            "value_type": "set_int64",
            "default_value": [],
            "export_common_attr": "author_living_blacklist2",
          },
          {
            "kconf_key": "reco.grprLiveAuthorBlackList3.heiChanBlackList",
            "value_type": "set_int64",
            "default_value": [],
            "export_common_attr": "author_living_blacklist3",
          },
          {
            "kconf_key": "reco.grprLiveAuthorBlackList4.heiChanBlackList",
            "value_type": "set_int64",
            "default_value": [],
            "export_common_attr": "author_living_blacklist4",
          },
          {
            "kconf_key": "reco.grprLiveAuthorBlackList5.heiChanBlackList",
            "value_type": "set_int64",
            "default_value": [],
            "export_common_attr": "author_living_blacklist5",
          },
        ],
      ) \
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
      .explore_memory_data_enrich(
        data_key = "livestream_merchant_author",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "merchant_author_list_ptr"
      ) \
      .explore_memory_data_enrich( #待废弃
        data_key = "high_photo_count_author_map",
        data_type = "uint64_uint64_map",
        save_data_ptr_to_attr = "high_photo_count_author_map_ptr"
      ) \
      .explore_memory_data_enrich(
        data_key = "high_upload_photo_author_map",
        data_type = "uint64_double_map",
        save_data_ptr_to_attr = "high_upload_photo_author_map_ptr"
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
      .explore_memory_data_enrich(
        data_key = "negative_aid",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "negative_aid_set_ptr",
      ) \
      .explore_memory_data_enrich(
        data_key = "yanghao_disu_uids",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "sexy_induce_photo_set_ptr"
      ) \
      .explore_memory_data_enrich(
        data_key = "gaofen_signs_uids",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "gaofen_signs_uids_set_ptr"
      ) \
      .explore_memory_data_enrich(
        data_key = "hierarchy_label_uids",
        data_type = "uint64_uint64_map",
        save_data_ptr_to_attr = "hierarchy_label_uids_map_ptr"
      ) \
      .explore_memory_data_enrich(
        data_key = "illegal_word_pids",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "illegal_word_pids_ptr"
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
      .if_("fountain_enable_fetch_rerank_neg_photo == 1", to_be_delete = "date=2024-05-29;committer=denghong") \
        .split_string(
          input_common_attr = "rerank_neg_photo_id_list_str",
          output_common_attr = "rerank_neg_photo_id_filter_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
      .end_() \
      .if_("fountain_enable_fetch_mc_neg_photo == 1", to_be_delete = "date=2024-05-29;committer=denghong") \
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
      .if_("fountain_enable_pic_low_quality_filter == 1") \
        .split_string(
          input_common_attr = "fountain_pic_low_quality_tag_str",
          output_common_attr = "fountain_pic_low_quality_tag_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
        .split_string(
          input_common_attr = "fountain_pic_low_quality_filter_thresh_list_str",
          output_common_attr = "fountain_pic_low_quality_filter_thresh_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_double = True
        ) \
      .end_() \
      .if_("fountain_enable_sirius_distribution_photo_filter == 1") \
        .split_string(
          input_common_attr = "fountain_sirius_distribution_photo_tags_list_str",
          output_common_attr = "fountain_sirius_distribution_photo_tags_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
      .end_() \
      .split_string(
        input_common_attr = "fountain_pic_ecology_high_release_author_bits_list_str",
        output_common_attr = "fountain_pic_ecology_high_release_author_bits_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .split_string(
        input_common_attr = "fountain_pic_ecology_high_delete_author_bits_list_str",
        output_common_attr = "fountain_pic_ecology_high_delete_author_bits_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .if_("fountain_enable_hot_spot_holdout_filter == 1") \
        .split_string(
          input_common_attr = "fountain_hot_spot_filter_level_list_str",
          output_common_attr = "hot_spot_filter_level_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
        .split_string(
          input_common_attr = "fountain_hot_spot_filter_source_list_str",
          output_common_attr = "hot_spot_filter_source_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
      .end_() \
      .if_("fountain_pic_author_filter_enable_specific_age_seg_diff == 1 and basic_info_age_segment_v2 ~= nil") \
        .split_string(
          input_common_attr = "fountain_pic_author_filter_specific_age_seg_str",
          output_common_attr = "fountain_pic_author_filter_specific_age_seg_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
        .find_value(
          input = "{{fountain_pic_author_filter_specific_age_seg_list}}",
          value = "{{basic_info_age_segment_v2}}",
          result = "is_pic_author_filter_specific_age_seg"
        ) \
        .if_("is_pic_author_filter_specific_age_seg == 1") \
          .copy_attr(
            attrs = [
              {
                "from_common": "fountain_pic_author_filter_markcode_for_specific_age_seg",
                "to_common": "fountain_pic_author_filter_markcode"
              },
              {
                "from_common": "fountain_pic_author_punish_markcode_for_specific_age_seg",
                "to_common": "fountain_pic_author_punish_markcode"
              },
            ]
          ) \
        .end_() \
      .end_() \
      .if_("fountain_enable_pic_liezhi_author_filter_elder == 1 and (basic_info_age_segment_v2 or 0) >= fountain_pic_liezhi_author_filter_elder_age_thresh") \
        .copy_attr(
          attrs = [
            {
              "from_common": "fountain_enable_pic_liezhi_author_filter_elder",
              "to_common": "fountain_enable_pic_liezhi_author_filter"
            },
            {
              "from_common": "fountain_author_liezhi_pic_count_thresh_elder",
              "to_common": "fountain_author_liezhi_pic_count_thresh"
            },
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
      .count_reco_result(
        save_count_to = "filter_candidate_count"
      )
    return self

  def _timestamp_begin(self, name: str):
    return self \
      .gen_common_attr_by_lua(
        attr_map = {
          name + "_begin_ts": "util.GetTimestamp()",
        },
      )

  def _timestamp_end(self, name: str):
    return self \
      .gen_common_attr_by_lua(
        attr_map = {
          name + "_ts": "util.GetTimestamp() - " + name + "_begin_ts",
        },
      )

  def _count_stage_cpu_cost(self, name: str):
    return self \
      .copy_user_meta_info(
        save_flow_cpu_cost_to_attr = name + "_cpu_cost_ts",
      )

  def _perf_local_life_info(self, ckp):
    self \
    .perflog_attr_value(
      check_point = ckp,
      aggregator = "count",
      item_attrs=["is_local_life_photo", "is_same_city_local_life"]
    ) \
    .perflog_attr_value(
      check_point=ckp,
      aggregator = "avg",
      item_attrs=["is_local_life_photo", "is_same_city_local_life"]
    )
    return self

  def _count_photo_type_distribution(self, stage):
    self \
      .count_reco_result(
        save_count_to = "%s_single_picture_count" % stage,
        target_item = {"picture_type": 1}
      ) \
      .count_reco_result(
        save_count_to = "%s_long_picture_count" % stage,
        target_item = {"picture_type": 2}
      ) \
      .count_reco_result(
        save_count_to = "%s_cluster_picture_count" % stage,
        target_item = {"picture_type": 3}
      ) \
    
    return self
