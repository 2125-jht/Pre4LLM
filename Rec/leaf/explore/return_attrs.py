DEFAULT_COMMON_ATTRS = [
  "retr_pic_cluster_distr_str", # 先知使用
  "prerank_pic_cluster_distr_str", # 先知使用
  "mc_s1_pic_cluster_distr_str", # 先知使用
  "mc_s2_pic_cluster_distr_str", # 先知使用
  "rank_pic_cluster_distr_str", # 先知使用
  "rerank_pic_cluster_distr_str", # 先知使用
  "user_pic_interest_hetu_distr_str",  # 先知使用，之后会删掉
  "mc_s1_cluster_id", # 先知使用
  "mc_s1_cluster_cnt", # 先知使用
  "mc_s1_cluster_after_truncation_cnt", # 先知使用
  "mc_s2_hetu_quota_control_is_degraded", # 先知使用
  "mc_s2_directly_reach_fullrank_hetu_quota_control_is_degraded", #先知使用
  "prerank_hetu_quota_control_is_degraded", # 先知使用
  "gamora_interest_hetu2_list", # 先知使用
  "fountain_interest_hetu2_list", # 先知使用
  "short_uninterest_cid_stat", # 先知使用
  "short_uninterest_hetu5_stat", # 先知使用
  "explore_realshow_max_timestamp", # 先知使用
  "ssd_div_score", #先知使用
  "interest_migration_is_degraded", #先知使用
  "pic_insert_flag",  # 先知使用
  "fr_carm_pid_list",  # 精排主模型 carm 特征回流
  "mc_s2_interest_count_list", # 先知使用
  "mc_s2_interest_id_list", # 先知使用
  "mc_s2_keep_interest_count_list", # 先知使用
]

DEFAULT_ITEM_ATTRS = [
  "author__id",
  "upload_time",
  "click_count",
  "like_count",
  "follow_count",
  "forward_count",
  "comment_count",
  "width",
  "height",
  "color",
  "duration_ms",
  "mod",
  "chn",
  "filter",
  "music",
  "explore_stat",
  "hot_trend_generalized_info",
  "hetu_tag_level_info",
  "hetu_tag_level_info_v2",
  "nearby_feeling",
  "empirical_ctr",
  "empirical_ltr",
  "empirical_wtr",
  "empirical_ftr",
  "empirical_lvtr",
  "empirical_svtr",
  "empirical_ptr",
  "empirical_htr",
  "empirical_cmtr",
  "cascade_pctr",
  "cascade_pltr",
  "cascade_pwtr",
  "cascade_plvtr",
  "cascade_plvtr2",
  "cascade_psvtr",
  "cascade_ptr",
  "cascade_pwatch_time",
  "cascade_pepstr",
  "cascade_pcestr",
  "cascade_pcmtr",
  "cascade_plivingtr",
  "cascade_pcltr",
  "cascade_pdtr",
  "cascade_pcptr",
  "pctr",
  "pltr",
  "pwtr",
  "pcmtr",
  "pptr",
  "pcmef",
  "phtr",
  "pevtr",
  "fr_score1",
  "fr_score2",
  "pepstr",
  "pdtr",
  "pcltr",
  "fetr",
  "pftr",
  "psvr",
  "retr_rank",
  "awesome_wtd",
  "original_explore_fr_ensemble_score",
  "explore_fr_pxtr_absolute_score",
  "prerank_ctr",
  "prerank_ltr",
  "prerank_wtd",
  "pctr_index",
  "plvtr_index",
  "pvtr_index",
  "pltr_index",
  "pftr_index",
  "pwtr_index",
  "pesptr_index",
  "psvr_index",
  "consume_time_ctr", # 先知使用
  "photo_source_type", #透传混排标
  "pic_interest_cluster", # 先知使用
  "is_recommend_by_friend", # leaf 排序及混排使用
  "recommend_friend_list", # 客户端展示使用
  "is_new_interest_explore", # 客户端内部展示工具“白盒”使用
  "is_valid_interest_explore", # 客户端内部展示工具“白盒”使用
  "cluster_id_632" # 客户端内部展示工具“白盒”使用
]

BACKUP_COMMON_ATTRS = [
  "filter_output_item_key_list",
  "cascade_final_output_item_key_list",
  "cascade_output_item_key_list",
  "cascade_input_item_key_list",
  "ranking_pos_sample_list",
  "rerank_cltr_adjust_ratio_attr",
  "rerank_cmef_adjust_ratio_attr",
  "rerank_cmtr_adjust_ratio_attr",
  "rerank_ctr_adjust_ratio_attr",
  "rerank_duration_adjust_ratio_attr",
  "rerank_ensemble_score_adjust_ratio_attr",
  "rerank_epstr_adjust_ratio_attr",
  "rerank_fetr_adjust_ratio_attr",
  "rerank_fountain_eff_adjust_ratio_attr",
  "rerank_fr_score1_adjust_ratio_attr",
  "rerank_fr_score2_adjust_ratio_attr",
  "rerank_ftr_adjust_ratio_attr",
  "rerank_l2r_score_adjust_ratio_attr",
  "rerank_ltr_adjust_ratio_attr",
  "rerank_ptr_adjust_ratio_attr",
  "rerank_wtr_adjust_ratio_attr",
  "retr_pic_cluster_distr_str", # 先知使用
  "prerank_pic_cluster_distr_str", # 先知使用
  "mc_s1_pic_cluster_distr_str", # 先知使用
  "mc_s2_pic_cluster_distr_str", # 先知使用
  "rank_pic_cluster_distr_str", # 先知使用
  "rerank_pic_cluster_distr_str", # 先知使用
  "user_pic_interest_hetu_distr_str",  # 先知使用，之后会删掉
  "continuous_hitting_filter_hetu_id_common_attr",  # 先知使用
  "continuous_hitting_filter_hetu_cnt_common_attr",   # 先知使用
  "mc_s1_cluster_id", # 先知使用
  "mc_s1_cluster_cnt", # 先知使用
  "mc_s1_cluster_after_truncation_cnt", # 先知使用
  "mc_s2_hetu_quota_control_is_degraded", # 先知使用
  "mc_s2_directly_reach_fullrank_hetu_quota_control_is_degraded", #先知使用
  "prerank_hetu_quota_control_is_degraded", # 先知使用
  "gamora_interest_hetu2_list", # 先知使用
  "fountain_interest_hetu2_list", # 先知使用
  "short_uninterest_cid_stat", # 先知使用
  "short_uninterest_hetu5_stat", # 先知使用
  "explore_realshow_max_timestamp", # 先知使用
  "interest_migration_is_degraded" #先知使用
]

BACKUP_ITEM_ATTRS = [
  "pliving_ctr",
  "pliving_wtr",
  "author",
  "author__gender",
  "author_age_info",
  "chn",
  "click_count",
  "color",
  "comment_count",
  "content_safety_level_with_namespace",
  "explore_stat",
  "filter",
  "forward_count",
  "height",
  "hetu_tag_level_info",
  "hetu_tag_level_info_v2",
  "hot_trend_generalized_info",
  "lda_topic",
  "like_count",
  "location",
  "mmu_cluster_music_id",
  "mmu_content_id",
  "mmu_face_age",
  "mmu_face_gender",
  "mod",
  "music_info__music_combo_id",
  "nearby_feeling",
  "ocr_cover_text_word_count",
  "report_count",
  "show_level_b",
  "tag",
  "view_length_sum",
  "width",
  "retr_rank",
  "original_explore_fr_ensemble_score",
  "explore_fr_pxtr_absolute_score",
  "pctr_index",
  "plvtr_index",
  "pvtr_index",
  "pltr_index",
  "pftr_index",
  "pwtr_index",
  "pesptr_index",
  "psvr_index",
  "consume_time_ctr", # 先知使用
  "pic_interest_cluster", # 先知使用
]

TRACEBACK_ITEM_ATTRS = [
  "i2i_retr__trigger_pid"
]
