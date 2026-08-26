from ranking import CommonModule

class RankingScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
    .enrich_attr_by_lua(
      import_item_attr = [
        "fr_score2",
      ],
      export_item_attr = [
        "fr_score2",
      ],
      function_for_item = "fr_score2_change",
      lua_script_file = "explore/ranking/lua/module/ranking_score__fr_score2.lua",
      skip = "{{explore_skip_fr_score2_change}}"
    ) \
    .gen_common_attr_by_lua(
      attr_map={
        "pctr_upper_bound": "(user_emp_ctr or 0.0) + explore_pctr_upper_bound_bias"
      }
    ) \
    .enrich_attr_by_lua(
      import_common_attr = [
        "enable_ctr_socre_ensemble",
        "ctr_socre_ensemble_coef",
        "enable_follow_author_pwtr_corr",
        "ranking_follow_author_pwtr_corr_coef",
        "explore_fullrank_calibration_ctr_param",
        "pctr_upper_bound",
        "enable_explore_pctr_upper_bound_limit"
      ],
      import_item_attr = [
        "pctr",
        "explore_diversity_ctr_score",
        "pwtr",
        "is_follow_author",
      ],
      export_item_attr = [
        "corr_pctr",
        "corr_pwtr"
      ],
      function_for_item = "pxtr_change",
      lua_script_file = "explore/ranking/lua/module/ranking_score__pxtr_change.lua",
      skip = "{{explore_skip_pxtr_change}}"
    ) \
    .if_("explore_enable_pxtr_calibration == 1") \
      .explore_pxtr_calibration() \
      .if_("explore_enable_rank_short_window_ctr_cali == 1") \
        .fr_short_window_ctr_cali() \
      .end_() \
      .if_("enable_corr_pctr_short_uninterest_photo_discount == 1") \
        .short_uninterest_photo_discount("corr_pctr", "corr_pctr") \
      .end_() \
      .if_("enable_corr_pctr_realshow_emctr_unbias == 1") \
        .explore_realshow_emctr_unbias(
          prev_item_from_attr = "explore_realshow_click_common_list",
          prev_click_item_from_attr = "explore_click_common_list",
          cluster_id_attr = "hetu_sim_cluster_id",
          input_pctr_attr = "corr_pctr",
          output_pctr_attr = "corr_pctr",
          discount_coef = "{{explore_fr_s1_realshow_emctr_unbias_discount_coef}}",
          boost_coef = "{{explore_fr_s1_realshow_emctr_unbias_boost_coef}}",
          realshow_threshold = "{{explore_fr_s1_realshow_emctr_unbias_realshow_threshold}}",
          click_threshold = "{{explore_fr_s1_realshow_emctr_unbias_click_threshold}}",
          cid_click_threshold = "{{explore_fr_s1_realshow_emctr_unbias_cid_click_threshold}}"
        ) \
      .end_() \
    .end_() \
    .if_("enable_eff_ctr_corr == 1") \
      .explore_eff_ctr_corr() \
    .end_() \
    .if_("enable_replace_ctr_corr == 1") \
      .explore_replace_ctr_corr() \
    .end_() \
    .if_("enable_fr_cal_quantile_relative_score == 1") \
      .fr_cal_quantile_relative_score() \
    .end_() \
    .if_("enable_fr_cal_no_ctr_score == 1 and user_age_segment >= explore_fr_cal_no_ctr_score_age_min and user_age_segment <= explore_fr_cal_no_ctr_score_age_max") \
      .fr_cal_no_ctr_score() \
    .end_() \
    .if_("explore_enable_fr_pctr_adjust_by_pcoc == 1 and user_age_segment >= explore_fr_pctr_adjust_by_pcoc_age_min and user_age_segment <= explore_fr_pctr_adjust_by_pcoc_age_max") \
      .fr_cal_ctr_adjust_by_pcoc_score() \
    .end_() \
    .if_("enable_fr_cal_svtr_rid_ctr_score == 1") \
      .fr_cal_svtr_rid_ctr_score() \
    .end_() \
    .if_("explore_enable_fr_cal_debias_xtr_by_pcoc_score == 1 and user_age_segment > 0 and user_age_segment <= explore_fr_cal_debias_xtr_by_pcoc_score_age_threshold") \
      .fr_cal_debias_xtr_by_pcoc_score("fr_ltr", "pltr", "debias_by_pcoc_ltr_score") \
      .fr_cal_debias_xtr_by_pcoc_score("fr_cltr", "pcltr", "debias_by_pcoc_cltr_score") \
      .fr_cal_debias_xtr_by_pcoc_score("fr_cmtr", "pcmtr", "debias_by_pcoc_cmtr_score") \
      .fr_cal_debias_xtr_by_pcoc_score("fr_wtr", "pwtr", "debias_by_pcoc_wtr_score") \
    .end_() \
    .enrich_attr_by_light_function(
      import_common_attr = [
        "explore_wtd_evtr_pctr_weight",
        "explore_wtd_lvtr_pctr_weight",
        "explore_future_xtr_pctr_weight",
        "explore_pic_wtd_pctr_weight",
        "explore_pic_lvtr_pctr_weight",
        "explore_pic_cpr_pctr_weight",
        "hot_fr_pic_cpr_max_pic_cnt",
      ],
      import_item_attr = [
        "corr_pctr",
        "wtd_evtr",
        "wtd_lvtr",
        "future_xtr",
        "pic_wtd",
        "pic_lvtr",
        "pic_cpr",
        "photo_picture_count",
      ],
      export_item_attr = [
        "corr_wtd_evtr",
        "corr_wtd_lvtr",
        "corr_future_xtr",
        "corr_pic_wtd",
        "corr_pic_lvtr",
        "corr_pic_cpr",
      ],
      function_name = "FrPxtrChange",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_lua(
      import_item_attr = [
        "corr_pctr",
        "pwtr",
        "fr_score2",
        "pltr",
        "pcmtr",
        "pepstr",
        "psvr",
        "plvtr"
      ],
      export_item_attr = [
        "svr_act_score"
      ],
      function_for_item = "svr_act_queue",
      lua_script_file = "explore/ranking/lua/module/ranking_score__pxtr_change.lua"
    )

    self.flow \
    .if_("enable_report_discount_cal == 1") \
      .calc_report_discount() \
    .end_() \
    .calc_hate_discount() \
    .if_("enable_pcmef_gender_debias == 1") \
      .calc_pcmef_gender_debias_score() \
    .end_() \
    .if_("enable_request_pctr_power_weight_adjust == 1") \
      .fr_ctr_ensemble_power_weight_adjust() \
    .end_() \
    .if_("enable_photo_history_interest_score_with_fr_ctr == 1 and interest_score_based_valid_user == 1") \
      .calc_by_simple_formula(
        formulas = [
          dict(
            expr = (
              "pow({{explore_history_interest_score_attr_x_bias}} + {{explore_history_interest_score_attr_x_alpha}} * [[photo_history_interest_score]], {{explore_history_interest_score_attr_x_pow}}) * "
              "pow({{explore_history_interest_score_fr_pctr_bias}} + {{explore_history_interest_score_fr_pctr_alpha}} * [[corr_pctr]], {{explore_history_interest_score_fr_pctr_pow}}) * "
              "pow({{explore_history_interest_score_empirical_ctr_bias}} + {{explore_history_interest_score_empirical_ctr_alpha}} * [[empirical_ctr]], {{explore_history_interest_score_empirical_ctr_pow}}) * "
              "pow({{explore_history_interest_score_fr_awesome_wtd_bias}} + {{explore_history_interest_score_fr_awesome_wtd_alpha}} * [[awesome_wtd]], {{explore_history_interest_score_fr_awesome_wtd_pow}})"
            ),
            output_attr = "photo_history_interest_score_with_fr_ctr"
          )
        ],
      ) \
    .end_() \
    .if_("enable_fr_user_age_interest_tagnex_tgi_product_pxtr_score == 1") \
      .calc_by_simple_formula(
        formulas = [
          dict(
            expr = (
              "{{explore_fr_age_tgi_product_global_coeff}} * "
              "pow(1 + {{explore_fr_age_tgi_product_tgi_alpha}} * [[user_age_interest_tagnex_tgi_score]], {{explore_fr_age_tgi_product_tgi_beta}}) * "
              "pow(1 + {{explore_fr_age_tgi_product_pctr_alpha}} * [[corr_pctr]], {{explore_fr_age_tgi_product_pctr_beta}}) * "
              "pow(1 + {{explore_fr_age_tgi_product_pwtr_alpha}} * [[corr_pwtr]], {{explore_fr_age_tgi_product_pwtr_beta}}) * "
              "pow(1 + {{explore_fr_age_tgi_product_fr_score1_alpha}} * [[fr_score1]], {{explore_fr_age_tgi_product_fr_score1_beta}})"
            ),
            output_attr = "user_age_interest_tagnex_tgi_product_fr_pxtr_score"
          )
        ]
      ) \
    .end_() \
    .if_("enable_fr_user_no_bias_interest_tagnex_tgi_product_pxtr_score == 1") \
      .calc_by_simple_formula(
        formulas = [
          dict(
            expr = (
              "{{explore_fr_no_bias_tgi_product_global_coeff}} * "
              "pow(1 + {{explore_fr_no_bias_tgi_product_tgi_alpha}} * [[user_no_bias_interest_tagnex_tgi_score]], {{explore_fr_no_bias_tgi_product_tgi_beta}}) * "
              "pow(1 + {{explore_fr_no_bias_tgi_product_pctr_alpha}} * [[corr_pctr]], {{explore_fr_no_bias_tgi_product_pctr_beta}}) * "
              "pow(1 + {{explore_fr_no_bias_tgi_product_fr_score1_alpha}} * [[fr_score1]], {{explore_fr_no_bias_tgi_product_fr_score1_beta}})"
            ),
            output_attr = "user_no_bias_interest_tagnex_tgi_product_fr_pxtr_score"
          )
        ]
      ) \
    .end_() \
    .if_("enable_fr_user_stage_interest_tagnex_tgi_product_pxtr_score == 1") \
      .calc_by_simple_formula(
        formulas = [
          dict(
            expr = (
              "{{explore_fr_stage_tgi_product_global_coeff}} * "
              "pow(1 + {{explore_fr_stage_tgi_product_tgi_alpha}} * [[user_stage_interest_tagnex_tgi_score]], {{explore_fr_stage_tgi_product_tgi_beta}}) * "
              "pow(1 + {{explore_fr_stage_tgi_product_pctr_alpha}} * [[corr_pctr]], {{explore_fr_stage_tgi_product_pctr_beta}}) * "
              "pow(1 + {{explore_fr_stage_tgi_product_pwtr_alpha}} * [[corr_pwtr]], {{explore_fr_stage_tgi_product_pwtr_beta}}) * "
              "pow(1 + {{explore_fr_stage_tgi_product_fr_score1_alpha}} * [[fr_score1]], {{explore_fr_stage_tgi_product_fr_score1_beta}})"
            ),
            output_attr = "user_stage_interest_tagnex_tgi_product_fr_pxtr_score"
          )
        ]
      ) \
    .end_() \
    .gen_score_stage1()

  def post_process(self) -> None:
    pass
