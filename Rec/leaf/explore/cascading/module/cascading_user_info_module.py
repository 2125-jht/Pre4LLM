from cascading import CommonModule

# 主要用于集中生成后续粗排策略中需要用到的 userInfo / colossus 的信息

class CascadingUserInfoModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
    self.pic_colossus_stat_attrs = ["pic_stat_video_play_cnt", "pic_stat_pic_play_cnt", "pic_stat_pic_like_cnt",
            "pic_stat_pic_follow_cnt", "pic_stat_pic_forward_cnt", "pic_stat_pic_comment_cnt",
            "pic_stat_video_like_cnt", "pic_stat_video_follow_cnt", "pic_stat_video_forward_cnt",
            "pic_stat_video_comment_cnt", "pic_stat_pic_recent_play_cnt"]

  def process(self) -> None:
    self.flow \
      .if_("explore_mc_sort_weight_adjust == 1") \
        .explore_user_emp_xtr_enricher(
          colossus_resp_attr = "colossus_resp_v2",
          user_info_ptr_attr = "user_info_ptr",
          enable_colossus_item_limit = "{{enable_colossus_item_limit}}",
          max_colossus_item_num = "{{max_colossus_item_num}}",
          user_colossus_min_sec_ago = "{{user_colossus_min_sec_ago}}",
          user_colossus_max_sec_ago = "{{user_colossus_max_sec_ago}}",
          save_user_click_count = "user_colossus_click_count",
          save_user_emp_ctr = "user_emp_ctr",
          save_user_emp_ltr = "user_emp_ltr",
          save_user_emp_wtr = "user_emp_wtr",
          save_user_emp_ftr = "user_emp_ftr",
          save_user_emp_htr = "user_emp_htr",
          save_user_emp_cmtr = "user_emp_cmtr",
          save_user_emp_eptr = "user_emp_eptr",
          save_user_emp_svtr = "user_emp_svtr",
          save_user_emp_evtr = "user_emp_evtr",
          save_user_emp_lvtr = "user_emp_lvtr",
          save_user_emp_fintr = "user_emp_fintr",
          save_user_emp_finish_rate = "user_emp_finish_rate",
          save_user_emp_watch_time = "user_emp_watchtime",
          save_user_emp_fountain_time_ratio = "user_emp_fountain_time_ratio"
        ) \
      .end_() \
      .if_("explore_pic_interest_explore__enable == 1") \
        .split_string(
          input_common_attr="user_level_for_pic_explore_str",
          output_common_attr="user_level_for_pic_explore",
          delimiters=",",
          trim_spaces=True,
          skip_empty_tokens=True,
          parse_to_int=True,
        ) \
        .split_string(
          input_common_attr="user_gender_for_pic_explore_str",
          output_common_attr="user_gender_for_pic_explore",
          delimiters=",",
          trim_spaces=True,
          skip_empty_tokens=True,
          parse_to_int=True,
        ) \
        .split_string(
          input_common_attr="user_age_for_pic_explore_str",
          output_common_attr="user_age_for_pic_explore",
          delimiters=",",
          trim_spaces=True,
          skip_empty_tokens=True,
          parse_to_int=True,
        ) \
        .if_("explore_pic_valid_cluster_enable == 0 or ((uDoubleOutsideValidPicClusterCnt7dKV or 0) < 1 and find_user_active_degree >= explore_pic_cluster_active_degree_start and find_user_active_degree <= explore_pic_cluster_active_degree_end)")\
          .enrich_attr_by_light_function(
            import_common_attr=[
              "user_info_ptr",
              "user_level_for_pic_explore",
              "user_gender_for_pic_explore",
              "user_age_for_pic_explore",
              "uStandardRealShowPicAllIdList",
              "explore_pic_interval_hour",
              "uDoubleOutsideValidPicClusterCnt7dKV",
              "explore_pic_valid_cluster_enable",
              "explore_calc_pic_insert_flag_by_ddp_feature",
              "uStandardExploreRealshowPhotoIdList",
              "uStandardExploreRealshowTimestampList",
            ],
            import_item_attr=[],
            export_common_attr=[
              "enable_pic_explore_flag",
            ],
            function_name="CalcEnableExplorePicForUser",
            class_name="ExploreLightFunctionSetV2",
          ) \
        .end_()\
      .end_() \
      .if_("explore_pic_enable_mc_adjust_weights_by_emp_xtr == 1 or explore_pic_tower_model_infer_v3_skip == 0 " +
           "or explore_pic_prerank_pic_colossus_skip == 0 or skip_cascade_enrich_pic_play_cnt == 0 or explore_pic_s1_cluster_sort__enable == 1 ") \
        .explore_pic_colossus_stat(
          colossus_attr_name = "colossus_resp_v2",
          user_info_ptr_attr = "user_info_ptr",
          save_pic_like_cnt = "pic_stat_pic_like_cnt",
          save_pic_follow_cnt = "pic_stat_pic_follow_cnt",
          save_pic_forward_cnt = "pic_stat_pic_forward_cnt",
          save_pic_comment_cnt = "pic_stat_pic_comment_cnt",
          save_pic_play_cnt = "pic_stat_pic_play_cnt",
          save_pic_recent_play_cnt = "pic_stat_pic_recent_play_cnt",
          save_pic_eff_play_cnt = "pic_stat_pic_eff_play_cnt",
          save_eff_item_size = "pic_stat_eff_item_size",
          save_video_like_cnt = "pic_stat_video_like_cnt",
          save_video_follow_cnt = "pic_stat_video_follow_cnt",
          save_video_forward_cnt = "pic_stat_video_forward_cnt",
          save_video_comment_cnt = "pic_stat_video_comment_cnt",
          save_video_play_cnt = "pic_stat_video_play_cnt",
          save_video_eff_play_cnt = "pic_stat_video_eff_play_cnt",
          save_pic_play_list = "pic_play_list",
          save_pic_like_list = "pic_like_list",
          save_pic_follow_list = "pic_follow_list",
          save_pic_comment_list = "pic_comment_list",
          save_pic_comment_aid_list = "pic_comment_aid_list",
          save_pic_hetu_l1_cnt = "pic_hetu_l1_cnt",
          save_pic_hetu_l1_cnt2 = "pic_hetu_l1_cnt2",
          save_user_pic_interest_hetu_distr = "user_pic_interest_hetu_distr",
          save_user_pic_interest_hetu_distr_str = "user_pic_interest_hetu_distr_str",
          save_short_term_pic_cnt = "short_term_pic_cnt",
          save_short_term_video_cnt = "short_term_video_cnt",
          effect_playtime_thresh_s = "user_effect_playtime_thresh_s",
          enable_user_pic_hetu_distr_attr = "{{explore_enable_user_pic_hetu_distr}}",
          interest_distr_use_hetu_l1_backup = "{{explore_interest_distr_use_hetu_l1_backup}}",
          recent_pic_play_thresh_m = "{{recent_pic_play_thresh_m}}",
        ) \
        .if_("enable_picture_interest_explore == 1") \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "user_pic_interest_hetu_distr", "as": "user_pic_interest_hetu_map"},
              {"name": "pic_interest_explore_time_gap_min", "as": "recent_time_gap_min"},
              {"name": "uStandardRealShowPicAllIdList", "as": "pic_realshow_pids"},
              "user_info_ptr",
              "pic_interest_limit_explore_page",
              "pic_interest_tail_trunc_ratio",
              "pic_interest_trunc_size_thresh",
              "enable_user_pic_unbiased_interest",
              "user_pic_unbiased_interest_map_ptr"
            ],
            export_common_attr = [
              "pic_interest_explore_hetu_list",
            ],
            function_name = "GenHetuListForPictureInterestExplore",
            class_name = "ExploreLightFunctionSetV2",
          ) \
        .end_() \
        .if_("explore_pic_enable_write_user_cnts_to_redis == 1") \
          .set_attr_value(
            no_overwrite=True,
            common_attrs=self._get_pic_user_cnts_default_values()
          ) \
          .str_format(
            format_string=",".join(["%d"] * len(self.pic_colossus_stat_attrs)),
            input_attrs=self.pic_colossus_stat_attrs,
            output_attr="user_pic_colossus_stats",
          ) \
          .copy_user_meta_info(
            save_request_id_to_attr = "request_id",
          ) \
          .write_to_redis(
            kcc_cluster = "explorerCache",
            key_prefix = "{{explore_user_pic_stat_redis_key_prefix}}",
            key = "{{request_id}}",
            value = "{{user_pic_colossus_stats}}",
            timeout_ms = 10,
            expire_second = "{{explore_user_pic_stat_redis_expire_sec}}"
          ) \
        .end_() \
      .end_() \
      .if_("enable_cal_explore_intrest_adjust_weight == 1 and (enable_explore_intrest_adjust_location_filter == 0 or featureUserRequestCityId == 16842752)") \
        .explore_intrest_adjust_enricher(
          gamora_hetu_adjust_history_list_attr = "gamora_hetu_adjust_history_list",
          opt_card_like_list_attr = "opt_card_like_list",
          opt_card_dis_like_list_attr = "opt_card_dis_like_list",
          interest_adjust_decay_attr = "{{interest_adjust_decay}}",
          interest_adjust_immediate_adjust_thres_attr = "{{interest_adjust_immediate_adjust_thres}}",
          interest_adjust_immediate_adjust_weight_attr = "{{interest_adjust_immediate_adjust_weight}}",
          output_intrest_key_list_attr = "output_intrest_key_list",
          output_intrest_value_list_attr = "output_intrest_value_list"
        ) \
        .if_("output_intrest_key_list and output_intrest_value_list") \
          .get_abtest_params(
            biz_name = "MOBILE",
            ab_params = [{
              "param_name": "enable_interest_adjust_report",
              "param_type": "bool",
              "default_value": False,
              "report_ab_hit": True
            }],
          ) \
        .end_() \
        .if_("explore_enable_user_need_break_cocoon == 1") \
          .find_value(
            input = "{{output_intrest_key_list}}",
            value = 0,
            result = "user_need_break_cocoon_flag"
          ) \
        .end_() \
      .end_() \
      .if_("enable_fr_hetu_distribution_adjust == 1 or enable_mc_hetu_distribution_adjust == 1 or enable_kl_fusion_hetu_distribution == 1") \
        .explore_photo_distribution_colossus_stat_enricher(
          enable_only_hot_stat = "{{fr_hetu_distribution_only_hot}}",
          colossus_resp_attr = "colossus_resp_v2",
          save_total_count = "colossus_hetu_distribution_total_count",
          save_user_hetu_stat_attr = "colossus_hetu_distribution_hetu_stat",
        ) \
      .end_() \
      .if_( # TODO(fenglei03) 推全或不用后清理冗余ab参数
        "enable_enrich_pic_feasury_action_list == 1 or explore_pic_quota_enable_recent_realshow_decay == 1") \
        .explore_merchant_global_data_enricher(
          kuiba_user_attr = "kuibaUserAttrStr",
          export_common_attr = [
            "uStandardRealShowPicAllIdList",
            "uStandardClickPicAllIdList",
            "uStandardLongviewPicAllIdList",
            "uIsPicDeep",
          ]
        ) \
      .end_() \
      .if_("explore_pic_quota_enable_recent_realshow_decay == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "expl_pic_recent_realshow_time_gap_min", "as": "recent_time_gap_min"},
            {"name": "uStandardRealShowPicAllIdList", "as": "pic_realshow_pids"},
            {"name": "uStandardClickPicAllIdList", "as": "pic_click_pids"},
            "user_info_ptr",
          ],
          export_common_attr = [
            "pic_recent_realshow_not_click_cnt",
          ],
          function_name = "ProccessPicActionList",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "expl_pic_recent_realshow_not_click_max", "as": "realshow_not_click_max"},
            {"name": "expl_pic_recent_realshow_not_click_min", "as": "realshow_not_click_min"},
            {"name": "expl_pic_recent_realshow_ctr_base", "as": "ctr_base"},
            {"name": "pic_recent_realshow_not_click_cnt", "as": "realshow_not_click_cnt"},
            "pic_da_user_pref_ptr",
            "basic_info_age_segment_v2",
            "uIsPicDeep",
          ],
          export_common_attr = [
            "user_pic_recent_ctr_score",
          ],
          function_name = "PicCtrByRealshow",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .switch_("enable_explore_get_user_group_emp_xtr") \
        .case_(1) \
          .get_user_group_emp_xtr() \
        .case_(2) \
          .get_user_new_group_emp_xtr() \
      .end_() \
      .if_("explore_enable_user_mau_emp_xtr == 1") \
        .get_user_mau_emp_xtr() \
      .end_() \
      .if_("enable_explore_gemini_refresh_scene == 1", to_be_delete = "date=2024-05-29;committer=caoying03") \
        .get_user_gemini_refresh_scene() \
      .end_() \
      .switch_("explore_cal_user_personalized_score_switch_method") \
        .case_(1) \
          .explore_photo_distribution_colossus_stat_enricher(
            colossus_resp_attr = "colossus_resp_v2",
            save_user_hetu_entropy_attr = "explore_colossus_hetu_personalized_score",
            enable_user_hetu_entropy = "{{explore_enable_user_hetu_entropy}}"
          ) \
        .case_(2) \
          .enrich_attr_by_light_function(
            import_common_attr = [
              "uOldMmuClusterId300ListList",
              {"name": "fountain_rerank_dpp_cid_upper_bound", "as": "cluster_id_cid_rate_upper_bound"},
              {"name": "fountain_rerank_dpp_cid_lower_bound", "as": "cluster_id_cid_rate_lower_bound"},
              {"name": "fountain_rerank_dpp_cid_avg_bound", "as": "cluster_id_cid_rate_avg_bound"},
            ],
            export_common_attr = [
              {"name": "cid_rate", "as": "explore_colossus_hetu_personalized_score"}, #取值范围为0.03，使用的时候可以加权
            ],
            function_name = "DynamicClusterIdRate",
            class_name = "ExploreLightFunctionSetV2"
          ) \
      .end_() \
      .if_("enable_explore_get_pic_interest == 1") \
        .switch_("enable_explore_only_pic_interest") \
          .case_(1) \
            .explore_pic_interest_stat_enricher(
              colossus_resp_attr = "colossus_resp_v2",
              interest_time_interval_day = "{{explore_pic_interest_time_interval_day}}",
              short_term_weight = "{{explore_pic_interest_short_term_weight}}",
              long_term_weight = "{{explore_pic_interest_long_term_weight}}",
              interest_distr_trans_pic_pic_weight = "{{explore_pic_distr_trans_pic_weight}}",
              reward_normal_min_score = "{{explore_pic_interest_normal_min_score}}",
              reward_normal_max_score = "{{explore_pic_interest_normal_max_score}}",
              explore_interest_count = "{{explore_pic_explore_interest_list_count}}",
              short_weight_adjust_vv_thres = "{{explore_pic_short_term_weight_adjust_vv_thres}}",
              short_term_adjust_coeff = "{{explore_pic_short_term_weight_adjust_coeff}}",
              save_pic_interest_stat_attr = "colossus_actual_reward_hetu_stat",
              save_short_term_interest_attr = "explore_pic_short_interest_list",
              save_long_term_interest_attr = "explore_pic_long_interest_list",
              save_explore_interest_attr = "explore_pic_explore_interest_list"
            ) \
          .case_(2) \
            .explore_pic_interest_stat_enricher_v2(
              # 输入输出 attr
              colossus_resp_attr = "colossus_resp_v2",
              save_pic_interest_stat_attr = "colossus_actual_reward_hetu_stat",
              save_short_term_interest_attr = "explore_pic_short_interest_list",
              save_long_term_interest_attr = "explore_pic_long_interest_list",
              save_explore_interest_attr = "explore_pic_explore_interest_list",
              # ab param
              explore_page_weight = "{{explore_pic_interest_v2_explore_page_weight}}",
              other_page_weight = "{{explore_pic_interest_v2_other_page_weight}}",
              pic_weight = "{{explore_pic_interest_v2_pic_weight}}",
              video_weight = "{{explore_pic_interest_v2_video_weight}}",
              stat_limit_day = "{{explore_pic_interest_v2_stat_limit_day}}",
              calc_method = "{{explore_pic_interest_v2_calc_method}}",
              short_interval_thresh_day = "{{explore_pic_interest_v2_short_interval_thresh_day}}",
              index_decay_pow_base = "{{explore_pic_interest_v2_index_decay_pow_base}}",
              index_decay_min = "{{explore_pic_interest_v2_index_decay_min}}",
              index_bucket_h = "{{explore_pic_interest_v2_index_bucket_h}}",
              interest_min_thresh = "{{explore_pic_interest_v2_interest_min_thresh}}",
              long_interest_weight = "{{explore_pic_interest_v2_long_interest_weight}}",
              short_interest_weight = "{{explore_pic_interest_v2_short_interest_weight}}",
              explore_interest_count = "{{explore_pic_interest_v2_explore_interest_count}}",
              explore_interest_weight = "{{explore_pic_interest_v2_explore_interest_weight}}",
              smooth_or_sharp_power = "{{explore_pic_interest_v2_smooth_or_sharp_power}}",
            ) \
          .default_() \
            .explore_photo_distribution_colossus_stat_enricher(
              enable_only_positive_stat = "{{explore_hetu_distribution_stat_only_positive}}",
              colossus_resp_attr = "colossus_resp_v2",
              save_total_count = "colossus_hetu_distribution_total_count",
              save_user_hetu_stat_attr = "colossus_hetu_distribution_hetu_stat",
              interest_stat_use_reward = "{{explore_interest_stat_use_reward}}",
              interest_stat_vv_weight = "{{explore_interest_stat_vv_weight}}",
              interest_stat_reward_weight = "{{explore_interest_stat_reward_weight}}",
              interest_stat_avg_reward_smooth = "{{explore_interest_stat_avg_reward_smooth}}",
              enable_interest_stat_avg_reward = "{{explore_enable_interest_stat_avg_reward}}",
              minus_hate_stat_coeff = "{{explore_interest_stat_minus_hate_coeff}}",
              minus_sv_stat_coeff = "{{explore_interest_stat_minus_sv_coeff}}",
              enable_use_actual_reward = "{{explore_colossus_enable_use_actual_reward}}",
              max_history_size = "{{explore_colossus_actual_reward_max_history_size}}",
              save_actual_hetu_stat_attr = "colossus_actual_reward_hetu_stat",
              enable_interest_stat_use_true_feedback = "{{enable_explore_mc_interest_stat_use_true_feedback}}",
            ) \
        .end_() \
        .if_("enable_explore_pic_interest_decay == 1") \
          .explore_pic_recent_interest_adjust_enricher(
            actual_hetu_stat_attr = "colossus_actual_reward_hetu_stat",
            user_info_attr = "user_info_ptr",
            pic_real_show_attr = "uStandardRealShowPicAllIdList",
            pic_click_attr = "uStandardClickPicAllIdList",
            interest_decay_time_gap = "{{pic_interest_decay_time_gap_min}}",
            interest_decay_coeff = "{{pic_interest_decay_coeff}}",
            interest_boost_coeff = "{{pic_interest_boost_coeff}}",
            consec_realshow_time_gap = "{{pic_interest_consec_realshow_time_gap_min}}",
            consec_show_not_click_decay = "{{pic_interest_consec_show_not_click_decay}}",
            consec_show_max = "{{pic_interest_consec_show_max}}",
            consec_hetu_level = "{{enable_pic_rerank_realshow_decay_hetu_level}}",
            save_consec_show_not_click_decay_attr = "colossus_pic_consec_show_decay",
          ) \
        .end_() \
      .end_() \
      .if_("explore_external_prefer_user_explore__enable == 1") \
        .enrich_with_protobuf(
          from_extra_var = "user_info_ptr",
          attrs = [
            dict(name="click_page_type_list", path="user_profile_v1.click_list.page_type"),
            dict(name="click_time_ms_list", path="user_profile_v1.click_list.time_ms"),
            dict(name="fountain_click_time_ms_list", path="fountain_reco_user_profile.click_list.time_ms"),
          ]
        ) \
        .enrich_attr_by_light_function(
          import_common_attr=[
            "click_page_type_list",
            "click_time_ms_list",
            "fountain_click_time_ms_list",
            {"name": "external_prefer_user_fountain_vv_bound", "as": "fountain_vv_bound"},
            {"name": "external_prefer_user_external_vv_bound", "as": "external_vv_bound"},
          ],
          export_common_attr=[
            "external_prefer_user_flag",
          ],
          function_name="FlagExternalPreferUser",
          class_name="ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_enable_write_filter_neg_list_to_redis == 1") \
        .write_to_redis(
          kcc_cluster = "recoHotLauRank",
          timeout = 10,
          expire_second = "{{explore_neg_pid_list_redis_expire_seconds}}",
          key_prefix = "{{explore_neg_pid_list_key_prefix}}",
          key = "{{_REQ_ID_}}",
          value = "{{send_final_pid_list}}"
        ) \
        .write_to_redis(
          kcc_cluster = "recoHotLauRank",
          timeout = 10,
          expire_second = "{{explore_neg_aid_list_redis_expire_seconds}}",
          key_prefix = "{{explore_neg_aid_list_key_prefix}}",
          key = "{{_REQ_ID_}}",
          value = "{{send_final_aid_list}}"
        ) \
      .end_() \
      .if_("explore_enable_user_cocoon_flag == 1") \
        .explore_user_interest_cocoon_enricher(
          colossus_v2_attr_name = "colossus_resp_v2",
          user_info_ptr_name = "user_info_ptr",
          user_valid_avg_vv_name = "active_days_avg_vv",
          user_cocoon_code_name = "uCocoonCodeKV",
          output_user_vv_type = "user_vv_flag",
          output_user_cocoon_type = "user_cocoon_flag",
          day_upper = "{{explore_user_cocoon_day_upper}}", 
          colossus_num_limit = "{{explore_user_cocoon_colossus_num_limit}}", 
          realshow_action_limit = "{{explore_user_cocoon_realshow_action_limit}}", 
          click_action_limit = "{{explore_user_cocoon_click_action_limit}}", 
          vv_in_interest_explore_freq = "{{explore_user_cocoon_vv_in_interest_explore_freq}}", 
          real_show_rate_threshold = "{{explore_user_cocoon_real_show_rate_threshold}}", 
          click_rate_threshold = "{{explore_user_cocoon_click_rate_threshold}}", 
          user_vv_calculate_type = "{{explore_user_cocoon_user_vv_calculate_type}}", 
          user_cocoon_calculate_type = "{{explore_user_cocoon_user_cocoon_calculate_type}}", 
          consume_concentration_threshold = "{{explore_user_cocoon_consume_concentration_threshold}}", 
          show_concentration_threshold = "{{explore_user_cocoon_show_concentration_threshold}}", 
          cocoon_active_day_threshold = "{{explore_user_cocoon_cocoon_active_day_threshold}}", 
          cocoon_consider_show_concentration = "{{explore_user_cocoon_cocoon_consider_show_concentration}}", 
        ) \
      .end_() \
      .if_("explore_enable_user_all_page_interest_migration_flag == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "user_page_prefer_score", "as": "score"},
            {"name": "explore_user_all_page_interest_migration_threshold", "as": "threshold"}
          ],
          export_common_attr = [
            {"name": "final_flag", "as": "all_page_interest_user_migration_flag"}
          ],
          function_name = "IsBelowThreshold",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_enable_gen_is_diversity_degraded == 1") \
        .gen_is_diversity_degraded() \
      .end_() \
      .if_("enable_explore_calc_pic_search_boost_user_degree == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "search_click_list",
            "search_click_list_timestamps",
            {"name": "uStandardClickPicAllIdList", "as": "pic_click_list"},
            {"name": "explore_search_click_pic_time_gap_min", "as": "time_gap_min"},
            {"name": "uDoubleOutsideValidPicClusterCnt7dKV", "as": "user_cluster_cnt"},
            {"name": "explore_pic_search_boost_user_cluster_thresh", "as": "user_cluster_thresh"},
          ],
          export_common_attr = [
            {"name": "search_degree", "as": "pic_search_boost_user_degree"},
          ],
          function_name = "CalcPicSearchBoostUserDegree",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_enable_mc_cal_ef_view_weight == 1") \
        .cal_explore_fountain_view_weight() \
      .end_() \

  def post_process(self) -> None:
      self.flow \
        .if_("explore_mc_sort_weight_adjust == 1") \
          .perflog_attr_value(
            check_point="default.user_emp_xtr",
            common_attrs=["user_colossus_click_count",
              "user_emp_ltr","user_emp_wtr",
              "user_emp_ftr","user_emp_htr",
              "user_emp_cmtr","user_emp_eptr",
              "user_emp_watchtime", "user_emp_fountain_time_ratio",
              "user_emp_ctr", "enable_pic_explore_flag"]
          ) \
        .end_() \
        .log_debug_info(
          common_attrs = ["user_effect_playtime_thresh_s", "short_term_pic_cnt", "short_term_video_cnt", "uIsPicDeep", "user_pic_recent_ctr_score"],
          for_debug_request_only = True,
          target_item = { "is_picture" : 1 }
        )

  def _get_pic_user_cnts_default_values(self) -> list:
    return [{
              "name": attr,
              "type": "int",
              "value": -1
            } for attr in self.pic_colossus_stat_attrs]
