from cascading import CommonModule

# 主要用于集中生成后续粗排策略中需要用到的 userInfo / colossus 的信息

class CascadingUserInfoModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
    self.pic_colossus_stat_attrs = ["pic_stat_video_play_cnt", "pic_stat_pic_play_cnt", "pic_stat_pic_like_cnt",
            "pic_stat_pic_follow_cnt", "pic_stat_pic_forward_cnt", "pic_stat_pic_comment_cnt",
            "pic_stat_video_like_cnt", "pic_stat_video_follow_cnt", "pic_stat_video_forward_cnt",
            "pic_stat_video_comment_cnt"]

  def process(self) -> None:
    self.flow \
      .if_("explore_mc_sort_weight_adjust == 1") \
        .if_("enable_life_user_emp_xtr_method == 1") \
          .if_("uIsLifeHighActive ~= 1") \
            .copy_attr(
              attrs=[
                {
                  "from_common": "life_user_emp_ctr_init_alpha_low_active",
                  "to_common": "life_user_emp_ctr_init_alpha"
                },
                {
                  "from_common": "life_user_emp_ctr_init_beta_low_active",
                  "to_common": "life_user_emp_ctr_init_beta"
                },
              ]
            ) \
          .end_() \
          .explore_life_user_emp_xtr_enricher(
            colossus_resp_attr = "colossus_resp_v2",
            user_info_ptr_attr = "user_info_ptr",
            enable_colossus_item_limit = "{{enable_colossus_item_limit}}",
            max_colossus_item_num = "{{max_colossus_item_num}}",
            user_colossus_min_sec_ago = "{{user_colossus_min_sec_ago}}",
            user_colossus_max_sec_ago = "{{user_colossus_max_sec_ago}}",
            init_ctr_alpha = "{{life_user_emp_ctr_init_alpha}}",
            init_ctr_beta = "{{life_user_emp_ctr_init_beta}}",
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
        .else_() \
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
      .end_() \
      .if_("enable_explore_weight_adjust_v2 == 1") \
        .explore_memory_data_enrich(
          data_key = "{{explore_colossus_user_emp_xtr_map}}",
          data_type = "string_double_vector_map",
          save_data_ptr_to_attr = "explore_colossus_user_emp_xtr_map_ptr",
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
        .enrich_attr_by_light_function(
          import_common_attr=[
            "user_info_ptr",
            "user_level_for_pic_explore",
            "user_gender_for_pic_explore",
            "user_age_for_pic_explore",
            "uStandardRealShowPicAllIdList",
            "explore_pic_interval_hour"
          ],
          import_item_attr=[],
          export_common_attr=[
            "enable_pic_explore_flag",
          ],
          function_name="CalcEnableExplorePicForUser",
          class_name="ExploreLightFunctionSetV2",
        ) \
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
        ) \
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
      .if_("enable_cal_explore_opt_card_weight == 1") \
        .explore_intrest_adjust_enricher(
          gamora_hetu_adjust_history_list_attr = "gamora_hetu_adjust_history_list",
          opt_card_like_list_attr = "opt_card_like_list",
          opt_card_dis_like_list_attr = "opt_card_dis_like_list",
          adjust_mode = "{{adjust_mode}}",
          opt_card_adjust_smooth = "{{opt_card_adjust_smooth}}",
          opt_card_adjust_weight = "{{opt_card_adjust_weight}}",
          opt_card_min_score = "{{opt_card_min_score}}",
          opt_card_max_score = "{{opt_card_max_score}}",
          output_intrest_key_list_attr = "output_opt_card_key_list",
          output_intrest_value_list_attr = "output_opt_card_value_list"
        ) \
        .if_("output_opt_card_key_list and output_opt_card_value_list") \
          .get_abtest_params(
            biz_name = "MOBILE",
            ab_params = [{
              "param_name": "enable_opt_card_adjust_report",
              "param_type": "bool",
              "default_value": False,
              "report_ab_hit": True
            }],
          ) \
          .if_("enable_opt_card_adjust_report == 1") \
            .set_attr_value(
                common_attrs = [
                  {
                    "name": "output_opt_card_key_list",
                    "type": "int_list",
                    "value": []
                  },
                  {
                    "name": "output_opt_card_value_list",
                    "type": "double_list",
                    "value": []
                  }
                ]
              ) \
          .end_() \
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
      .if_("enable_explore_get_pic_interest == 1") \
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
          enable_only_hot_stat = "{{life_hetu_distribution_stat_only_hot}}",
          enable_interest_stat_use_true_feedback = "{{enable_explore_mc_interest_stat_use_true_feedback}}",
        ) \
      .end_() \
      .if_("enable_life_calc_user_positive_hetu == 1 and page == 1 and (life_user_pos_hetu_boost_limit_low_active ~= 1 or uIsLifeHighActive ~= 1)") \
        .explore_life_user_positive_hetu_enricher(
          user_info_ptr_attr = "user_info_ptr",
          save_user_positive_hetu2_to_attr = "user_positive_hetu2_list",
          action_days_thresh = "{{life_user_positive_hetu_action_days_thresh}}",
          action_minutes_thresh = "{{life_user_positive_hetu_action_minutes_thresh}}",
          action_time_mode = "{{life_user_positive_hetu_action_time_mode}}",
          save_hetu_count = "{{life_user_positive_hetu_save_hetu_count}}",
          enable_use_click_list = "{{life_user_positive_hetu_enable_use_click_list}}",
          enable_use_like_list = "{{life_user_positive_hetu_enable_use_like_list}}",
          enable_use_follow_list = "{{life_user_positive_hetu_enable_use_follow_list}}",
          enable_use_forward_list = "{{life_user_positive_hetu_enable_use_forward_list}}",
          enable_use_comment_list = "{{life_user_positive_hetu_enable_use_comment_list}}",
          enable_use_collect_list = "{{life_user_positive_hetu_enable_use_collect_list}}",
          enable_user_video_play_stats = "{{life_user_positive_hetu_enable_user_video_play_stats}}",
          click_weight = "{{life_user_positive_hetu_click_weight}}",
          like_weight = "{{life_user_positive_hetu_like_weight}}",
          follow_weight = "{{life_user_positive_hetu_follow_weight}}",
          forward_weight = "{{life_user_positive_hetu_forward_weight}}",
          collect_weight = "{{life_user_positive_hetu_collect_weight}}",
          comment_weight = "{{life_user_positive_hetu_comment_weight}}",
          video_play_weight = "{{life_user_positive_hetu_video_play_weight}}",
          min_effective_play_length = "{{life_user_positive_hetu_min_effective_play_length}}"
        ) \
      .end_() \
      .if_("enable_life_calc_user_recent_uninterest_rate == 1") \
        .enrich_attr_by_light_function(
          import_common_attr=[
            "user_info_ptr",
            {"name": "life_user_recent_uninterest_play_cnt_thr", "as": "play_cnt_thr"},
            {"name": "life_user_recent_uninterest_real_show_cnt_thr", "as": "real_show_cnt_thr"},
            {"name": "life_user_recent_uninterest_time_gap_min", "as": "time_gap_min"},
            {"name": "life_user_recent_uninterest_short_play_thresh", "as": "short_play_thresh"}
          ],
          export_common_attr=[
            "recent_short_play_rate",
            "recent_unclick_rate"
          ],
          function_name="CalcUserRecentUninterestRate",
          class_name="ExploreLifeLightFunctionSet",
        ) \
      .end_() \
      .if_("enable_life_get_user_group_emp_xtr == 1") \
        .get_user_group_emp_xtr() \
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
          common_attrs = ["user_effect_playtime_thresh_s", "short_term_pic_cnt", "short_term_video_cnt", "uIsPicDeep"],
          for_debug_request_only = True,
          target_item = { "is_picture" : 1 }
        )

  def _get_pic_user_cnts_default_values(self) -> list:
    return [{
              "name": attr,
              "type": "int",
              "value": -1
            } for attr in self.pic_colossus_stat_attrs]