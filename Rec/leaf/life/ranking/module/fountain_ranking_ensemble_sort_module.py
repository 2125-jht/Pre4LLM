from ranking import CommonModule
from ranking.fountain_ranking_queues import fullrank_ensemble_queues

class FountainRankingEnsembleSortModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def lost_fractile_scores(self):
    return ["fullrank_sim_click_score_fractile_score", "fullrank_sim_pcltr_fractile_score", 
            "fullrank_sim_pepstr_fractile_score", "fullrank_sim_pevtr_fractile_score", 
            "fullrank_sim_pl2r_fractile_score", "fullrank_sim_plvtr_fractile_score", 
            "fullrank_sim_psvr_fractile_score", "fullrank_sim_pwatchtime_no_bias_fractile_score",
            "fountain_splash_slide", "questionnaire_score"]

  def process(self) -> None:
    self.flow \
      .enrich_attr_by_lua(
        import_common_attr = [
          "fountain_fullrank_duration_discount_weight",
          "fountain_fullrank_lvtr_sigmoid_bias",
          "fountain_vtr_max_value",
          "fountain_vtr_sigmoid_decay_rate",
          "fountain_vtr_smooth_rate",
          "fountain_vtr_sigmoid_bias",
          "enable_fountain_pwatch_time_sigmoid_bias_new",
          "fountian_vtr_big_duration_discount_bias",
          "fountian_vtr_big_duration_discount_slope",
          "fountain_vtr_score_discount_fix",
          "enable_fountain_longview_score_remove_click_coef",
          "fountain_fullrank_sim_pevtr_coef_weight",
          "fountain_fullrank_sim_pvtr_coef_weight",
          "fullrank_pvtr_trans_score_threshold",
          "skip_fountain_act_vtr_norm",
          "fountain_fullrank_act_l2r_max",
          "fountain_fullrank_act_l2r_merge_weight",
          "fountain_fullrank_act_vtr_merge_weight",
          "skip_fountain_act_vtr_merge",
          "skip_fountain_act_l2r_replace",
          "fountain_fullrank_distill_score_evtr_v2_weight"
        ],
        import_item_attr = [
          "fullrank_detail_new_pevtr_v2",
          "fullrank_sim_pwtd_v2_playtime",
          "duration_ms",
          "fullrank_sim_pvtr",
          "explore_stat__view_length_sum",
          "explore_stat__click_count",
          "fullrank_sim_click_score",
          "fullrank_sim_plvtr",
          "picture_variant_attr",
          "fullrank_sim_pwtr",
          "fullrank_sim_pptr",
          "fountain_act_vtr_max",
          "fullrank_act_wtd",
          "fullrank_ltr_score",
          "fullrank_distill_rerank_score",
        ],
        export_item_attr = [
          "fullrank_sim_pwatchtime_no_bias",
          "fullrank_sim_longview_score_no_bias",
          "fullrank_sim_pvtr_multi_pwtr",
          "fullrank_sim_pvtr_multi_pptr",
          "fullrank_sim_evtr_v2_multi_pfintr",
          "fullrank_trans_pvtr_score",
          "fullrank_act_wtd",
          "fullrank_ltr_score",
          "fullrank_evtr_distill_score",
        ],
        skip = "{{skip_fullrank_calc_fullrank_score_lua_v1}}",
        function_for_item = "calc_fullrank_score",
        lua_script_file = "life/ranking/lua/module/fountain_ranking_score__calc_fullrank_score.lua",
      ) \
      .if_("skip_fullrank_gen_min_act_rank_reci ==0") \
        .gen_min_act_rank_reci() \
      .end_if_() \
      .enrich_attr_by_lua( # 计算 ensemble score
        skip = "{{skip_fullrank_user_adaptive_weight_cal}}",
        import_common_attr = [
          "fountain_fullrank_user_ada_pxtr_avg_weight",
          "fountain_fullrank_skip_ada_wt_user_emp_xtr",
          "fountain_fullrank_skip_ada_vv_user_emp_xtr",
          "fountain_fullrank_skip_ada_act_user_emp_xtr",
          "fountain_fullrank_skip_ada_user_emp_xtr",
          "enable_filter_fountain_ada_weight_lower_one",
          "enable_filter_fountain_ada_weight_over_one",
          "user_colossus_click_count_ada",
          "user_emp_ltr_ada",
          "user_emp_wtr_ada",
          "user_emp_ftr_ada",
          "user_emp_cmtr_ada",
          "user_emp_eptr_ada",
          "user_emp_svtr_ada",
          "user_emp_lvtr_ada",
          "fullrank_splash_pre_filter_keep_photo_size",
          "xlife_fountain_ensemble_power_weight_fullrank_like_score",
          "xlife_fountain_ensemble_power_weight_fullrank_follow_score",
          "fountain_ensemble_power_weight_fullrank_pcmtr_score",
          "fountain_ensemble_power_weight_fullrank_pptr_score",
          "fountain_ensemble_power_weight_fullrank_pepstr_score",
          "fountain_ensemble_weight_forward_score",
          "fountain_ensemble_power_weight_fullrank_pvtr_multi_pwtr",
          "fountain_ensemble_power_weight_fullrank_pvtr_multi_pptr",
          "fountain_ensemble_weight_fullrank_pthanos_svr",
          "fountain_ensemble_power_weight_fullrank_svr_in_order_score",
          "fountain_ensemble_weight_fullrank_sim_longview_score_no_bias_debias",
          "fountain_ensemble_power_weight_adjust_ratio_min",
          "fountain_ensemble_power_weight_adjust_ratio_max",
          "fountain_ensemble_power_weight_fullrank_like_emp",
          "fountain_ensemble_power_weight_fullrank_follow_emp",
          "fountain_ensemble_power_weight_fullrank_pcmtr_emp",
          "fountain_ensemble_power_weight_fullrank_pptr_emp",
          "fountain_ensemble_power_weight_fullrank_psvtr_emp",
          "fountain_ensemble_power_weight_fullrank_plvtr_emp",
          "fountain_ensemble_power_weight_fullrank_forward_emp",
          "fountain_fullrank_ensemble_use_absolute_score_queue_power_weight",
          "xlife_fountain_fullrank_ensemble_like_raw_pow_weight_attr",
          "xlife_fountain_fullrank_ensemble_follow_raw_pow_weight_attr",
          "fountain_fullrank_ensemble_comment_raw_pow_weight_attr",
          "fountain_fullrank_ensemble_pepstr_raw_pow_weight_attr",
          "fountain_fullrank_ensemble_pftr_raw_pow_weight_attr",
          "userExpLtr",
          "userExpWtr",
          "userExpCmtr",
          "userExpPtr",
          "userExpSvtr",
          "userExpLvtr",
          "userExpFtr",
          "psvr_avg",
          "pltr_avg",
          "pwtr_avg",
          "pftr_avg",
          "pcmtr_avg",
          "pptr_avg",
          "plvtr_avg",
        ],
        export_common_attr = [
          "xlife_fountain_ensemble_power_weight_fullrank_like_score",
          "xlife_fountain_ensemble_power_weight_fullrank_follow_score",
          "fountain_ensemble_power_weight_fullrank_pcmtr_score",
          "fountain_ensemble_power_weight_fullrank_pptr_score",
          "fountain_ensemble_power_weight_fullrank_pepstr_score",
          "fountain_ensemble_power_weight_fullrank_pvtr_multi_pwtr",
          "fountain_ensemble_power_weight_fullrank_pvtr_multi_pptr",
          "fountain_ensemble_weight_fullrank_pthanos_svr",
          "fountain_ensemble_power_weight_fullrank_svr_in_order_score",
          "fountain_ensemble_weight_fullrank_sim_longview_score_no_bias_debias",
          "fountain_ensemble_weight_forward_score",
          "xlife_fountain_fullrank_ensemble_like_raw_pow_weight_attr",
          "xlife_fountain_fullrank_ensemble_follow_raw_pow_weight_attr",
          "fountain_fullrank_ensemble_comment_raw_pow_weight_attr",
          "fountain_fullrank_ensemble_pepstr_raw_pow_weight_attr",
          "fountain_fullrank_ensemble_pftr_raw_pow_weight_attr",
        ],
        function_for_common = "cal_fullrank_adaptive_weights_v2",
        lua_script_file = "life/ranking/lua/module/fountain_ranking_score__cal_adaptive_weight.lua",
      ) \
      .set_attr_value(
        no_overwrite=True,
        item_attrs=[
          {
            "name": score,
            "type": "double",
            "value": 0
          }
          for score in self.lost_fractile_scores()
        ]
      ) \
      .fountain_calc_ensemble_score(
        save_score_to_attr = "fullrank_ensemble_score",
        save_ori_ensemble_score_to_attr = "fullrank_ensemble_ori_score",
        save_absolute_score_to_attr = "fullrank_ensemble_absolute_score",
        save_fractile_score_to_attr = "fullrank_ensemble_fractile_score",
        use_dist_calc = "{{fountain_fullrank_ensemble_use_dist_calc}}",
        dis_factor = "{{fountain_fullrank_ensemble_dis_factor}}",
        user_new_proportion = "{{fullrank_ensemble_score_user_new_proportion}}",
        user_power_calc = "{{fountain_fullrank_variant_enable_power_calc}}",
        user_power_calc_v2 = "{{fountain_fullrank_variant_enable_power_calc_v2}}",
        rank_smooth = "{{fountain_fullrank_rank_smooth}}",
        fractile_smooth = "{{fountain_fullrank_fractile_smooth}}",
        use_queue_smooth_as_rank_smooth = "{{fountain_fullrank_ensemble_use_queue_smooth_as_rank_smooth}}",
        use_reciprocal = "{{fountain_fullrank_use_reciprocal}}",
        duration_min = "{{fountain_fullrank_duration_min}}",
        duration_max = "{{fountain_fullrank_duration_max}}",
        user_info_ptr_attr = "user_info_ptr",
        action_day = "{{fountain_fullrank_variant_weight_action_day_num}}",
        min_rank_weight = "{{fountain_fr_fullrank_min_rank_weight}}",
        queue_head_boost_index = "{{fountain_fullrank_ensemble_queue_head_boost_index}}",
        queue_tail_discount_index = "{{fountain_fullrank_ensemble_queue_tail_discount_index}}",
        queues = fullrank_ensemble_queues,
        use_absolute_score_queue_power_weight = "{{fountain_fullrank_ensemble_use_absolute_score_queue_power_weight}}",
        queue_head_boost_threshold = "{{fountain_fullrank_ensemble_queue_head_boost_threshold}}",
        queue_tail_discount_threshold = "{{fountain_fullrank_ensemble_queue_tail_discount_threshold}}",
        ensemble_score_head_coef = "{{fountain_fullrank_ensemble_ensemble_score_head_coef}}",
        ensemble_score_tail_coef = "{{fountain_fullrank_ensemble_ensemble_score_tail_coef}}",
        use_rank_with_absolute_score = "{{fountain_fullrank_ensemble_use_rank_with_absolute_score}}"
      ) \
      ._dump_attr_to_kafka( # ES 排序之后, 将全部item的重要 item attr 落盘
        stage_name = "fr_s2_score",
        dump_item_attr_list = [
          # 推全排序队列
          "fullrank_sim_click_score",
          "fullrank_sim_like_score",
          "fullrank_sim_pvtr_multi_pwtr",
          "fullrank_sim_pcmtr",
          "fullrank_sim_pvtr_multi_pptr",
          "fullrank_sim_pcmef",
          "fullrank_sim_pcltr",
          "fullrank_sim_plvtr",
          "fullrank_sim_pwatchtime_no_bias",
          "fullrank_sim_pcpr",
          "fullrank_sim_pepstr",
          "fullrank_sim_phtr",
          "fullrank_sim_pfintr",
          "fullrank_hate_similary_score",
          # "fullrank_action_once_watchtime_score",
          "fullrank_ltr_score",
          "fullrank_ltr_v4_fountain_finish_rate",
          # "fullrank_opportunity_cost_score",
          "fullrank_sim_pftr",
          "fullrank_min_act_rank_reci",
          "fullrank_sim_longview_score_no_bias",
          "fullrank_sim_out_pctr",
          "fullrank_sim_lstr",
          "fullrank_sim_psvr",
          "fullrank_ltr_v4_fountain_next",
          "fullrank_detail_new_pevtr_v2",
          "fullrank_ori_pswptr",
          "comment_ltr",
          # "fullrank_sim_pwatchtime_no_bias_debias",
          # "xgb_ltr",
          # "fullrank_pre_filter_score",
          # "fullrank_ada_xtr_score",
          # 排序分
          "fullrank_ensemble_score"
        ]
      ) \
      .if_("request_type == 'fountain_splash_life' or request_type == 'fountain_splash_life_pic_inside'") \
        .enrich_attr_by_lua(
          import_common_attr = [
            "enable_fountain_movie_ip_boost",
            "fountain_movie_ip_boost_ratio",
            "long_duration_boost",
            "long_duration_boost_min_plvtr",
            "fullrank_enable_questionnaire_boost",
            "fullrank_questionnaire_boost_ratio",
            "fullrank_questionnaire_boost_threshold",
          ],
          import_item_attr = [
            "fullrank_ensemble_score",
            "source_related_score",
            "duration_ms",
            "reason",
            "fullrank_sim_plvtr",
            "questionnaire_score"
          ],
          export_item_attr = [
            "fullrank_ensemble_score_after_adjust",
          ],
          function_for_item = "fullrank_score_adjust_splash",
          lua_script_file = "life/ranking/lua/module/fountain_ranking_score__fullrank_score_adjust.lua",
        ) \
      .else_() \
        .enrich_attr_by_lua(
          import_item_attr = [
            "hetu_tag_level_info__hetu_level_one",
            "hetu_tag_level_info__hetu_level_two",
            "hetu_tag_level_info__hetu_level_five",
          ],
          export_item_attr = [
            "hetu_level_one",
            "hetu_level_two",
            "hetu_level_five",
          ],
          function_for_item = "calculate",
          lua_script_file = "life/ranking/lua/module/fountain_ranking_score__trans_item_attr.lua",
        ) \
        .if_("skip_fullrank_negative_feedback_discount == 0") \
          .fountain_negative_feedback_discount(
            user_info_attr = "userInfo",
            save_score_to_attr = "fullrank_discount_ratio",
            discount_score = "{{fullrank_negative_feedback_discount_score}}",
            min_neg_feedback = "{{fullrank_negative_feedback_min_count}}",
            time_limit_second = "{{fullrank_negative_feedback_time_limit_sec}}"
          ) \
        .end_if_() \
        .if_("skip_fullrank_negative_feedback_discount_v2 == 0") \
          .fountain_negative_feedback_discount_v2(
            user_info_attr = "user_info_ptr",
            save_score_to_attr = "fullrank_discount_ratio",
            enable_fountain_user_profile = "{{fountain_nfd_v2_enable_fountain_profile}}",
            enable_hot_user_profile = "{{fountain_nfd_v2_enable_hot_profile}}",
            enable_not_click_list = "{{fountain_nfd_v2_enable_not_click_list}}",
            enable_play_stat_list = "{{fountain_nfd_v2_enable_play_stat_list}}",
            enable_hate_list = "{{fountain_nfd_v2_enable_hate_list}}",
            discount_score = "{{fountain_nfd_v2_discount_score}}",
            neg_feedback_threshold = "{{fountain_nfd_v2_neg_feedback_threshold}}",
            period_decay_factor = "{{fountain_nfd_v2_period_decay_factor}}",
            no_click_factor = "{{fountain_nfd_v2_not_click_discount_factor}}",
            video_play_stat_factor = "{{fountain_nfd_v2_play_stat_discount_factor}}",
            hate_list_factor = "{{fountain_nfd_v2_hate_list_discount_factor}}",
            play_time_thresold_0 = "{{fountain_nfd_v2_play_time_thresold_0}}",
            play_time_thresold_1 = "{{fountain_nfd_v2_play_time_thresold_1}}",
            time_limit_second = "{{fountain_nfd_v2_time_limit_second}}",
            attr_keys = ["hetu_level_one", "hetu_level_two", "photo_dnn_cluster_id", "mmu_img_cluster_v3", "tag"],
          ) \
        .end_if_() \
        .enrich_attr_by_lua(
          import_common_attr = [
            "skip_fullrank_negative_feedback_discount",
            "skip_fullrank_negative_feedback_discount_v2",
            "enable_community_discount",
            "community_discount_ratio",
            "long_duration_boost",
            "long_duration_boost_min_plvtr",
            "fullrank_enable_questionnaire_boost",
            "fullrank_questionnaire_boost_ratio",
            "fullrank_questionnaire_boost_threshold",
          ],
          import_item_attr = [
            "fullrank_ensemble_score",
            "fullrank_discount_ratio",
            "explore_operation_c_review_level",
            "duration_ms",
            "fullrank_sim_plvtr",
            "questionnaire_score",
          ],
          export_item_attr = [
            "fullrank_ensemble_score_after_adjust",
          ],
          function_for_item = "fullrank_score_adjust_fast",
          lua_script_file = "life/ranking/lua/module/fountain_ranking_score__fullrank_score_adjust.lua",
        ) \
      .end_if_() \
      .if_("life_fountain_enable_rank_marketing_compensation_adjust == 1") \
        .rank_marketing_compensation_adjust() \
      .end_() \
      .sort(
        score_from_attr = "fullrank_ensemble_score_after_adjust",
      )

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = [
          "fountain_ensemble_weight_fullrank_sim_longview_score_no_bias_debias",
          "fountain_fullrank_ensemble_hate_cliff_score_bias_attr",
          "fountain_fullrank_ensemble_hate_rank_cliff_attr",
          "fountain_fullrank_ensemble_hate_rank_cliff_ratio_attr",
          "fountain_fullrank_ensemble_hate_rank_height_attr", 
          "fountain_splash_slide", 
          "fullrank_ltr_v4_fountain_finish_rate", 
          "fullrank_ltr_v4_fountain_next", 
          "fullrank_ltr_v4_fountain_reward"
        ],
        item_attrs = [
          "fullrank_ensemble_score",
          "fullrank_ensemble_score_after_adjust",
          "fullrank_ensemble_absolute_score",
          "fullrank_ensemble_fractile_score",
          "fullrank_ensemble_ori_score",
          "hetu_level_five"
        ],
        for_debug_request_only = True,
        item_num_limit = 10,
      )
