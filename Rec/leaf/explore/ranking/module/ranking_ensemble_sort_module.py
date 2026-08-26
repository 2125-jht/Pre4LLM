from ranking import CommonModule
from ranking.ranking_queues import all_ensemble_queues,zero_play_queues,zero_play_ctr_queues,fr_s2_weight_param_dict

class RankingEnsembleSortModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
    
  def process(self) -> None:
    self.flow \
    .if_("explore_enable_consume_pctr_x_pxtr_substitude == 1", to_be_delete = "date=2024-05-29;committer=zhaokangzhi") \
      .enrich_attr_by_light_function(
        import_item_attr = [
          {"name": "consume_time_ctr", "as": "pctr"},
          "pltr",
          {"name": "corr_pwtr", "as": "pwtr"},
          "pcmtr",
          "pcmef",
          {"name": "report_discount", "as": "discount"},
        ],
        export_item_attr = [
          {"name": "pltr_change", "as": "score_pltr_es"},
          {"name": "pwtr_change", "as": "score_pwtr_es"},
          {"name": "pcmtr_change", "as": "score_pcmtr_es"},
          {"name": "pcmef_change", "as": "score_pcmef_es"},
        ],
        function_name = "ExploreFrPxtrChangeDiscount",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .else_() \
      .copy_attr(
        attrs=[
          {"from_item": "score_pltr", "to_item": "score_pltr_es"},
          {"from_item": "score_pwtr", "to_item": "score_pwtr_es"},
          {"from_item": "score_pcmtr", "to_item": "score_pcmtr_es"},
          {"from_item": "score_pcmef", "to_item": "score_pcmef_es"},
        ]
      ) \
    .end_() \
    .if_("explore_enable_consume_pctr_substitude == 1") \
      .copy_attr(
        attrs=[
          {"from_item": "consume_time_ctr", "to_item": "score_pctr_es"},
        ]
      ) \
    .else_() \
      .copy_attr(
        attrs=[
          {"from_item": "score_pctr", "to_item": "score_pctr_es"},
        ]
      ) \
    .end_() \
    .if_("explore_enable_useful_author_revisit == 1") \
      .item_attr_operation(
        item_attr_a = "userfulness_author_score",
        common_attr_b = "{{explore_useful_author_revisit_score_coef}}",
        operator = "*",
        output_attr = "userfulness_author_score_tur"
      ) \
      .item_attr_operation(
        item_attr_a = "revisit_score_model",
        item_attr_b = "userfulness_author_score_tur",
        operator = "+",
        output_attr = "revisit_score"
      ) \
      .item_attr_operation(
        item_attr_a = "revisit_score_author_model",
        item_attr_b = "userfulness_author_score_tur",
        operator = "+",
        output_attr = "revisit_score_author"
      ) \
    .else_() \
      .copy_attr(
        attrs=[
          {"from_item": "revisit_score_model", "to_item": "revisit_score"},
          {"from_item": "revisit_score_author_model", "to_item": "revisit_score_author"},
        ]
      ) \
    .end_if_() \
    .pack_item_attr(
      item_source = {
        "reco_results": True,
      },
      mappings = [
        {
          "aggregator": "avg",
          "from_item_attr": "pwtr",
          "to_common_attr": "pwtr_avg"
        },
        {
          "aggregator": "dev",
          "from_item_attr": "pwtr",
          "to_common_attr": "pwtr_dev"
        },
        {
          "aggregator": "avg",
          "from_item_attr": "pltr",
          "to_common_attr": "pltr_avg"
        },
        {
          "aggregator": "dev",
          "from_item_attr": "pltr",
          "to_common_attr": "pltr_dev"
        },
        {
          "aggregator": "avg",
          "from_item_attr": "corr_pctr",
          "to_common_attr": "pctr_avg"
        },
        {
          "aggregator": "dev",
          "from_item_attr": "corr_pctr",
          "to_common_attr": "pctr_dev"
        },
        {
          "aggregator": "avg",
          "from_item_attr": "pcmtr",
          "to_common_attr": "pcmtr_avg"
        },
        {
          "aggregator": "dev",
          "from_item_attr": "pcmtr",
          "to_common_attr": "pcmtr_dev"
        },
        {
          "aggregator": "avg",
          "from_item_attr": "pepstr",
          "to_common_attr": "pepstr_avg"
        },
        {
          "aggregator": "dev",
          "from_item_attr": "pepstr",
          "to_common_attr": "pepstr_dev"
        },
      ],
    ) \
    .split_string(
      input_common_attr = "xtr_norm_str_rank",
      output_common_attr = "xtr_norm_str_rank_number",
      delimiters = ";",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_double = True,
    ) \
    .gen_score_stage2() \
    .if_("enable_corr_pxtr_formula == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "adjust_ctr_alpha"
        ],
        import_item_attr = [
          {"name": "score_pctr", "as": "score_pctr_input"},
          {"name": "score_pltr", "as": "score_pltr_input"},
          {"name": "fr_score2", "as": "fr_score2_input"},
          {"name": "cpr"},
        ], 
        export_item_attr = [
          {"name": "corr_pltr_formula_output", "as": "corr_pltr_formula"},
          {"name": "corr_fr_score2_formula_output", "as": "corr_fr_score2_formula"},
          {"name": "corr_pfvtr_formula_output", "as": "corr_pfvtr_formula"}  
        ],
        function_name = "CalEmsembleCorrXtrScore",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .if_("enable_pfrtr_fusion == 1") \
      .enrich_attr_by_light_function(
        import_item_attr = [
          {"name" : "corr_pctr", "as" : "pctr"},
          "pcltr",
          "consume_time_pf2r_score"
        ],
        import_common_attr = [
          "pfrtr_fusion_pctr_coffe",
          "pfrtr_fusion_pcltr_coffe"
        ],
        export_item_attr = [
          "pctr_pfr2r",
          "pcltr_pfr2r"
        ],
        function_name = "CalculatePfrtrFusionScore",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .enrich_attr_by_lua(
      import_common_attr = [
        "enable_act_combo_vtr_rank",
        "xtr_norm_str_rank_number",
        "vtr_pow_weight_rank",
        "enable_ranking_statistics_adaptive",
        "statistics_adaptive_xtr_norm_str",
        "statistics_adaptive_factor_bias",
        "max_tatistics_adaptive_factor",
        "pwtr_avg",
        "pwtr_dev",
        "pltr_avg",
        "pltr_dev",
        "pctr_avg",
        "pctr_dev",
        "pcmtr_avg",
        "pcmtr_dev",
        "pepstr_avg",
        "pepstr_dev",
        "explore_ensemble_power_weight_fullrank_pctr_score",
        "explore_ensemble_power_weight_fullrank_pltr_score",
        "explore_ensemble_power_weight_fullrank_pwtr_score",
        "explore_ensemble_power_weight_fullrank_pepstr_score",
        "fr_pmctr_rank_weight"
      ],
      export_common_attr = [
        "explore_ensemble_power_weight_fullrank_pctr_score",
        "explore_ensemble_power_weight_fullrank_pltr_score",
        "explore_ensemble_power_weight_fullrank_pwtr_score",
        "explore_ensemble_power_weight_fullrank_pepstr_score",
        "fr_pmctr_rank_weight"
      ],
      import_item_attr = [
        "corr_pctr",
        "pwtr",
        "pltr",
        "pcmtr",
        "pepstr",
        "fr_score2"
      ],
      export_item_attr = [
        "act_combo_vtr_score"
      ],
      function_for_item = "cal_act_vtr_combo",
      function_for_common = "cal_xtr_adaptive_weight",
      lua_script_file = "explore/ranking/lua/module/ranking_ensemble_sort__print.lua"
    ) \
    .if_("explore_enable_author_click_value_score == 1") \
      .gen_author_click_value_score() \
    .end_() \
    .if_("explore_enable_user_timely_diversity_entropy_score == 1") \
      .calculate_user_timely_diversity_score() \
    .end_() \
    .if_("enable_user_timely_diversity_entropy_fr_s2_pctr_weight_adjust == 1") \
      .user_timely_diversity_pxtr_weight_adjust() \
    .end_() \
    .if_("explore_user_group_consume_weight_adjust_fr_s2 == 1") \
      .user_group_consume_weight_adjust(fr_s2_weight_param_dict, "fr_s2") \
    .end_() \
    .if_("enable_interact_watchtime_cost_score == 1") \
      .enrich_attr_by_lua(
        import_common_attr = [
          "user_interact_watchtime_cost_score_pctr_weight",
          "user_interact_watchtime_cost_score_cost_weight",
          "user_interact_watchtime_cost_score_alpha",
          "user_interact_watchtime_cost_score_beta",
          "user_interact_watchtime_cost_score_weight_str",
          "user_interact_watchtime_cost_score_watchtime_weight",
          "enable_user_interact_watchtime_cost_score_cltr",
          "user_interact_watchtime_cost_score_cltr_weight", 
        ],
        import_item_attr = [
          "corr_pctr",
          "pltr",
          "pwtr",
          "pftr",
          "pcmtr",
          "pcmef",
          "pptr",
          "pepstr",
          "fetr",
          "fountain_eff",
          "fr_score2",
          "pcltr"
        ],
        export_item_attr = [
          "watchtime_interact_score",
        ],
        function_for_item = "watchtime_interact_score_calc",
        lua_script_file = "explore/ranking/lua/module/ranking_ensemble_sort__print.lua"
      ) \
    .end_() \
    .if_("enable_fr_explore_low_active_customization_view_score_weight == 1 and is_explore_new_la_user == 1") \
      .copy_attr(
        attrs=[{
          "from_common": "explore_fr_cover_view_predict_trans_score_ranking_weight_low_active",
          "to_common": "explore_fr_cover_view_predict_trans_score_ranking_weight"
        }, {
          "from_common": "explore_fr_sense_view_predict_trans_score_ranking_weight_low_active",
          "to_common": "explore_fr_sense_view_predict_trans_score_ranking_weight"
        }]
      ) \
    .end_() \
    .if_("enable_fr_watch_time_fusion_score_weight_divide_vv_adjust == 1 and active_days_avg_vv >= explore_fr_watch_time_fusion_vv_threshold") \
      .gen_common_attr_by_lua( # 根据活跃天均vv划分
        attr_map = {
            "fr_act_fusion_score_wtd_weight" : "explore_fr_watch_time_fusion_high_vv_weight * fr_act_fusion_score_wtd_weight",
        }
      ) \
    .end_() \
    .if_("enable_fr_watch_time_fusion_score_weight_divide_active_adjust == 1 and (find_user_active_degree == 3 or find_user_active_degree == 4)") \
      .gen_common_attr_by_lua( # 根据人群活跃度划分
        attr_map = {
            "fr_act_fusion_score_wtd_weight" : "explore_fr_watch_time_fusion_high_active_weight * fr_act_fusion_score_wtd_weight",
        }
      ) \
    .end_() \
    .if_("enable_explore_fr_diversity_interest_lma_score_divide_active_adjust == 1 and find_user_active_degree ~= 1 and find_user_active_degree ~= 2") \
      .gen_common_attr_by_lua( # explore_diversity_interest_lma_score 根据人群活跃度划分
        attr_map = {
          "explore_diversity_interest_lma_score_ranking_weight" : "explore_diversity_interest_lma_score_ranking_adjust_coeff * explore_diversity_interest_lma_score_ranking_weight",
          "explore_diversity_interest_lma_score_ranking_raw_weight" : "explore_diversity_interest_lma_score_ranking_adjust_coeff * explore_diversity_interest_lma_score_ranking_raw_weight",
          "explore_diversity_interest_lma_score_ranking_raw_power_weight" : "explore_diversity_interest_lma_score_ranking_adjust_coeff * explore_diversity_interest_lma_score_ranking_raw_power_weight",
        }
      ) \
    .end_() \
    .if_("enable_fr_user_age_tgi_product_first_refresh_weight_adjust == 1 and is_first_refresh == 1") \
      .gen_common_attr_by_lua( # user_age_interest_tagnex_tgi_product_fr_pxtr_score 首屏权重独立设置
        attr_map = {
          "explore_ensemble_power_weight_user_age_interest_tagnex_tgi_product_fr_pxtr_score" : "explore_fr_user_age_tgi_product_first_refresh_power_weight",
          "explore_user_age_interest_tagnex_tgi_product_fr_pxtr_score_queue_pow_t" : "explore_fr_user_age_tgi_product_first_refresh_queue_pow_t",
          "explore_fr_user_age_interest_tagnex_tgi_product_fr_pxtr_score_raw_weight" : "explore_fr_user_age_tgi_product_first_refresh_raw_weight",
          "explore_fr_user_age_interest_tagnex_tgi_product_fr_pxtr_score_raw_power_weight" : "explore_fr_user_age_tgi_product_first_refresh_raw_power_weight",
        }
      ) \
    .end_() \
    .if_("enable_user_age_tgi_score_population_weight_adjust == 1 and basic_info_age_segment_v2 > user_age_tgi_score_population_age_segment_threshold and active_days_gt_5min_rate < user_age_tgi_score_population_active_days_threshold") \
      .copy_attr(
        attrs = [{
          "from_common": "explore_ensemble_user_age_interest_tagnex_tgi_product_fr_pxtr_score_population_power_weight",
          "to_common": "explore_ensemble_power_weight_user_age_interest_tagnex_tgi_product_fr_pxtr_score"
        },
        {
          "from_common": "explore_ensemble_user_age_interest_tagnex_tgi_product_fr_pxtr_score_population_queue_pow_t",
          "to_common": "explore_user_age_interest_tagnex_tgi_product_fr_pxtr_score_queue_pow_t"
        },
        {
          "from_common": "explore_fr_user_age_interest_tagnex_tgi_product_fr_pxtr_score_population_raw_weight",
          "to_common": "explore_fr_user_age_interest_tagnex_tgi_product_fr_pxtr_score_raw_weight"
        },
        {
          "from_common": "explore_fr_user_age_interest_tagnex_tgi_product_fr_pxtr_score_population_raw_power_weight",
          "to_common": "explore_fr_user_age_interest_tagnex_tgi_product_fr_pxtr_score_raw_power_weight"
        }]
      ) \
    .end_() \
    .if_("enable_interact_fusion_score == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "fr_act_fusion_score_htr_weight",
          "fr_act_fusion_score_ctr_weight",
          "fr_act_fusion_score_interact_ctr_weight",
          "fr_act_fusion_score_ftr_weight",
          "fr_act_fusion_score_dtr_weight",
          "fr_act_fusion_score_cmtr_weight",
          "fr_act_fusion_score_ltr_weight",
          "fr_act_fusion_score_cltr_weight",
          "fr_act_fusion_score_wtr_weight",
          "fr_act_fusion_score_evtr_weight",
          "fr_act_fusion_score_lvtr_weight",
          "fr_act_fusion_score_fvtr_weight",
          "fr_act_fusion_score_epstr_weight",
          "fr_act_fusion_score_cmef_weight",
          "fr_act_fusion_score_fetr_weight",
          "fr_act_fusion_score_fr_score1_weight",
          "fr_act_fusion_score_fr_score2_weight",
          "fr_act_fusion_score_awesome_wtd_weight",
          "fr_act_fusion_score_wtd_weight",
          "fr_act_fusion_score_pvtr_weight",
          "fr_act_max_watchtime_threshold",
          "enable_pure_interact_fusion"
        ],
        import_item_attr = [
          "phtr",
          "pctr",
          "pftr",
          "pdtr",
          "pcmtr",
          "pltr",
          "pcltr",
          "pwtr",
          "pevtr",
          "plvtr",
          {"name": "consume_time_ptr", "as": "pfvtr"},
          "pepstr",
          "pcmef",
          "fetr",
          "fr_score1",
          "fr_score2",
          "pvtr",
          "awesome_wtd",
          "photo_history_interest_score_with_fr_ctr",
        ],
        export_item_attr = [
          "interact_fusion_score",
          "watch_time_fusion_score"
        ],
        function_name = "CalInteractFusionScore",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .if_("fullrank_enable_fusion_gen_l2r_score == 1") \
      .gen_l2r_score_fusion() \
    .end_() \
    .if_("fullrank_enable_gen_emp_action_score_fusion_score == 1") \
      .gen_emp_action_score_fusion() \
    .end_() \
    .if_("fullrank_enable_jarvis_param > 0") \
      .gen_common_attr_by_lua(
        attr_map = {
          "explore_ensemble_power_weight_fullrank_pctr_score" : "fullrank_ctr_adjust_ratio_attr * explore_ensemble_power_weight_fullrank_pctr_score",
          "explore_ensemble_power_weight_fullrank_pltr_score" : "fullrank_ltr_adjust_ratio_attr * explore_ensemble_power_weight_fullrank_pltr_score",
          "explore_ensemble_power_weight_fullrank_pwtr_score" : "fullrank_wtr_adjust_ratio_attr * explore_ensemble_power_weight_fullrank_pwtr_score",
          "explore_ensemble_power_weight_fullrank_pftr_score" : "fullrank_ftr_adjust_ratio_attr * explore_ensemble_power_weight_fullrank_pftr_score",
          "explore_ensemble_power_weight_fullrank_pcltr_score" : "fullrank_cltr_adjust_ratio_attr * explore_ensemble_power_weight_fullrank_pcltr_score",
          "explore_ensemble_power_weight_fullrank_pptr_score" : "fullrank_ptr_adjust_ratio_attr * explore_ensemble_power_weight_fullrank_pptr_score",
          "fr_pmctr_rank_weight" : "fullrank_cmtr_adjust_ratio_attr * fr_pmctr_rank_weight",
          "explore_ensemble_power_weight_fullrank_pcmef_score" : "fullrank_cmef_adjust_ratio_attr * explore_ensemble_power_weight_fullrank_pcmef_score",
          "explore_ensemble_power_weight_fullrank_pepstr_score" : "fullrank_epstr_adjust_ratio_attr * explore_ensemble_power_weight_fullrank_pepstr_score",
          "explore_ensemble_power_weight_fullrank_fr_score1_score" : "fullrank_fr_score1_adjust_ratio_attr * explore_ensemble_power_weight_fullrank_fr_score1_score",
          "explore_ensemble_power_weight_fullrank_fr_score2_score" : "fullrank_fr_score2_adjust_ratio_attr * explore_ensemble_power_weight_fullrank_fr_score2_score",
          "hot_fountain_fetr_weight_push" : "fullrank_fetr_adjust_ratio_attr * hot_fountain_fetr_weight_push",
          "hot_fountain_fountain_eff_weight_push" : "fullrank_fountain_eff_adjust_ratio_attr * hot_fountain_fountain_eff_weight_push",
          "explore_ensemble_weight_duration_ms" : "fullrank_duration_adjust_ratio_attr * explore_ensemble_weight_duration_ms",
          "explore_ensemble_power_weight_fullrank_l2r_score" : "fullrank_l2r_score_adjust_ratio_attr * explore_ensemble_power_weight_fullrank_l2r_score",
        }
      ) \
    .end_() \
    .if_("enable_fullrank_user_group_dynamic_weight == 1", to_be_delete = "date=2024-05-29;committer=xuwei09") \
      .gen_common_attr_by_lua(
        attr_map={
          "explore_weight_adjust_avg_emp_ltr": "explore_weight_adjust_avg_emp_ltr * user_group_emp_ltr",
          "explore_weight_adjust_avg_emp_wtr": "explore_weight_adjust_avg_emp_wtr * user_group_emp_wtr",
          "explore_weight_adjust_avg_emp_ftr": "explore_weight_adjust_avg_emp_ftr * user_group_emp_ftr",
          "explore_weight_adjust_avg_emp_cmtr": "explore_weight_adjust_avg_emp_cmtr * user_group_emp_cmtr",
        }
      ) \
    .end_() \
    .if_("enable_tnu_user_adjust_ranking_weight == 1 and uIsExploreTnuCrowdUser == 1") \
      .gen_common_attr_by_lua( # 显式判断新回人群逻辑删除 to_be_delete = 2024-09-20
        attr_map={
          "explore_ensemble_power_weight_fullrank_pctr_score": "explore_ensemble_power_weight_fullrank_pctr_score * explore_tnu_ctr_ranking_adjust_ratio",
          "explore_ensemble_power_weight_fullrank_pltr_score": "explore_ensemble_power_weight_fullrank_pltr_score * explore_tnu_ltr_ranking_adjust_ratio",
          "explore_ensemble_power_weight_fullrank_pwtr_score": "explore_ensemble_power_weight_fullrank_pwtr_score * explore_tnu_wtr_ranking_adjust_ratio",
          "explore_ensemble_power_weight_fullrank_pptr_score": "explore_ensemble_power_weight_fullrank_pptr_score * explore_tnu_ptr_ranking_adjust_ratio",
          "explore_ensemble_power_weight_fullrank_pepstr_score": "explore_ensemble_power_weight_fullrank_pepstr_score * explore_tnu_epstr_ranking_adjust_ratio",
        }
      ) \
    .end_() \
    .if_("explore_rank_enable_boost_user_group_emp_psvtr == 1") \
      .gen_common_attr_by_lua(
        attr_map={
          "explore_ensemble_power_weight_fullrank_psvr_score": "explore_ensemble_power_weight_fullrank_psvr_score * user_group_emp_rank_svtr",
        }
      ) \
    .end_() \
    .if_("enable_fullrank_alter_weight_cal == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "user_emp_watchtime",
          "user_emp_ltr",
          "user_emp_wtr",
          "user_emp_cmtr",
          "user_emp_ftr",
          {"name": "fr_alter_time_upper", "as": "time_upper"},
          {"name": "fr_alter_time_lower", "as": "time_lower"},
          {"name": "fr_alter_interact_upper", "as": "interact_upper"},
          {"name": "fr_alter_interact_lower", "as": "interact_lower"},
          {"name": "fr_alter_watchtime_thresh", "as": "time_thresh"},
          {"name": "fr_alter_ltr_thresh", "as": "ltr_thresh"},
          {"name": "fr_alter_wtr_thresh", "as": "wtr_thresh"},
          {"name": "fr_alter_cmtr_thresh", "as": "cmtr_thresh"},
          {"name": "fr_alter_ftr_thresh", "as": "ftr_thresh"},
        ],
        export_common_attr = [
          {"name": "time_coef", "as": "fr_alter_time_coef"},
          {"name": "interact_coef", "as": "fr_alter_interact_coef"},
        ],
        function_name = "UserAlterWeightCoefCalc",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .if_("enable_fullrank_alter_weight_time_adjust == 1") \
      .gen_common_attr_by_lua(
        attr_map={
          "awesome_wtd_weight_push": "awesome_wtd_weight_push * fr_alter_interact_coef * fullrank_alter_weight_time_boost",
        }
      ) \
    .end_() \
    .if_("enable_fullrank_boost_new_follow_adjust == 1") \
      .gen_common_attr_by_lua( #涨关摸高
        attr_map={
          "explore_ensemble_power_weight_fullrank_pwtr_score": "explore_ensemble_power_weight_fullrank_pwtr_score * fullrank_new_follow_touch_high_boost",
        }
      ) \
    .end_() \
    .if_("enable_fullrank_alter_weight_interact_adjust == 1") \
      .gen_common_attr_by_lua(
        attr_map={
          "explore_ensemble_power_weight_fullrank_pltr_score": "explore_ensemble_power_weight_fullrank_pltr_score * fr_alter_time_coef * fullrank_alter_weight_interact_boost",
          "explore_ensemble_power_weight_fullrank_pwtr_score": "explore_ensemble_power_weight_fullrank_pwtr_score * fr_alter_time_coef * fullrank_alter_weight_interact_boost",
          "fr_pmctr_rank_weight": "fr_pmctr_rank_weight * fr_alter_time_coef * fullrank_alter_weight_interact_boost",
          "explore_ensemble_power_weight_fullrank_pftr_score": "explore_ensemble_power_weight_fullrank_pftr_score * fr_alter_time_coef * fullrank_alter_weight_interact_boost",
        }
      ) \
    .end_() \
    .if_("explore_rank_sort_weight_adjust == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "enable_explore_weight_adjust_v2",
          "enable_explore_watchtime_weight_adjust",
          "enable_explore_fountain_weight_adjust",
          "enable_explore_ctr_weight_adjust",
          "enable_explore_adjust_weight_more",
          "explore_colossus_user_emp_xtr_map_ptr",
          "explore_weight_adjust_coeff_a",
          "explore_weight_adjust_coeff_b",
          "explore_weight_adjust_coeff_c",
          "explore_weight_adjust_coeff_d",
          "user_emp_ctr",
          "user_emp_ltr",
          "user_emp_wtr",
          "user_emp_ftr",
          "user_emp_cmtr",
          "user_emp_eptr",
          "user_emp_watchtime",
          "user_emp_fountain_time_ratio",
          {"name": "explore_weight_adjust_avg_emp_ctr", "as": "all_user_emp_ctr"},
          {"name": "explore_weight_adjust_avg_emp_ltr", "as": "all_user_emp_ltr"},
          {"name": "explore_weight_adjust_avg_emp_wtr", "as": "all_user_emp_wtr"},
          {"name": "explore_weight_adjust_avg_emp_ftr", "as": "all_user_emp_ftr"},
          {"name": "explore_weight_adjust_avg_emp_cmtr", "as": "all_user_emp_cmtr"},
          {"name": "explore_weight_adjust_avg_emp_eptr", "as": "all_user_emp_eptr"},
          {"name": "explore_weight_adjust_avg_emp_watchtime", "as": "all_user_emp_watchtime"},
          {"name": "explore_weight_adjust_avg_emp_fountain_time_ratio", "as": "all_user_emp_fountain_time_ratio"},
          {"name": "explore_ensemble_power_weight_fullrank_pltr_score", "as": "user_ori_ltr_weight"},
          {"name": "explore_ensemble_power_weight_fullrank_pwtr_score", "as": "user_ori_wtr_weight"},
          {"name": "explore_ensemble_power_weight_fullrank_pftr_score", "as": "user_ori_ftr_weight"},
          {"name": "fr_pmctr_rank_weight", "as": "user_ori_cmtr_weight"},
          {"name": "explore_ensemble_power_weight_fullrank_pepstr_score", "as": "user_ori_eptr_weight"},
          {"name": "awesome_wtd_weight_push", "as": "user_ori_watchtime_weight"},
          {"name": "hot_fountain_fountain_eff_weight_push", "as": "user_ori_fountain_time_weight"},
          {"name": "explore_ensemble_power_weight_fullrank_pctr_score", "as": "user_ori_ctr_weight"},
          {"name": "explore_ensemble_power_weight_fullrank_fr_score2_score", "as": "user_ori_frscore2_weight"},
          {"name": "hot_fountain_fetr_weight_push", "as": "user_ori_fetr_weight"},
          "explore_weight_adjust_coeff_min",
          "explore_weight_adjust_coeff_max"
        ],
        export_common_attr = [
          {"name": "user_ltr_weight", "as": "explore_ensemble_power_weight_fullrank_pltr_score"},
          {"name": "user_wtr_weight", "as": "explore_ensemble_power_weight_fullrank_pwtr_score"},
          {"name": "user_ftr_weight", "as": "explore_ensemble_power_weight_fullrank_pftr_score"},
          {"name": "user_cmtr_weight", "as": "fr_pmctr_rank_weight"},
          {"name": "user_eptr_weight", "as": "explore_ensemble_power_weight_fullrank_pepstr_score"},
          {"name": "user_watchtime_weight", "as": "awesome_wtd_weight_push"},
          {"name": "user_fountain_time_weight", "as": "hot_fountain_fountain_eff_weight_push"},
          {"name": "user_ctr_weight", "as": "explore_ensemble_power_weight_fullrank_pctr_score"},
          {"name": "user_frscore2_weight", "as": "explore_ensemble_power_weight_fullrank_fr_score2_score"},
          {"name": "user_fetr_weight", "as": "hot_fountain_fetr_weight_push"},
          "user_ltr_weight_adjust",
          "user_wtr_weight_adjust",
          "user_ftr_weight_adjust",
          "user_cmtr_weight_adjust",
          "user_eptr_weight_adjust",
        ],
        function_name = "UserSortWeightAdjust",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .if_("fullrank_enable_la_zero_click_optimized == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_vv_3d", "as": "origin_value"},
          {"name": "explore_ensemble_power_weight_fullrank_pctr_score", "as": "pctr_weight"},
          {"name": "explore_fullrank_la_ensemble_sort_pctr_weight_max", "as": "weight_max"},
          {"name": "explore_fullrank_la_ensemble_sort_pctr_weight_base", "as": "weight_base"}
        ],
        export_common_attr = [
          {"name": "new_pctr_weight", "as": "explore_ensemble_power_weight_fullrank_pctr_score"}
        ],
        function_name = "AdjustFullRankPxtrWeight",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .if_("enable_fullrank_htr_weight_adjust == 1", to_be_delete = "date=2024-05-29;committer=liucong03") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "htr_score_weight", "as": "raw_htr_weight"},
          {"name": "user_age_segment", "as": "age_segment"},
          {"name": "uCityLevelNew", "as": "city_level"},
          {"name": "explore_fr_htr_adjust_need_age_segment", "as": "need_age_segment"},
          {"name": "explore_fr_htr_adjust_need_city_level", "as": "need_city_level"},
          {"name": "explore_fr_htr_adjust_boost_htr_weight", "as": "boost_htr_weight"},
        ],
        export_common_attr = [
          {"name": "raw_htr_weight", "as": "htr_score_weight"},
        ],
        function_name = "AdjustHtrWeightPersonal",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .if_("enable_explore_fr_lvtr_weight_adjust_by_high_time_rate == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_emp_sharpe_score_weight", "as": "xtr_weight"},
          {"name": "active_days_gt_5min_rate", "as": "user_xtr"},
          {"name": "explore_fr_lvtr_weight_adjust_threshold", "as": "user_xtr_threshold"},
          {"name": "explore_fr_lvtr_weight_adjust_alpha", "as": "alpha"},
          {"name": "explore_fr_lvtr_weight_adjust_beta", "as": "beta"},
          {"name": "explore_fr_lvtr_weight_adjust_omega", "as": "omega"},
          {"name": "explore_fr_lvtr_weight_adjust_max", "as": "coeff_max"},
          {"name": "explore_fr_lvtr_weight_adjust_min", "as": "coeff_min"},
        ],
        export_common_attr = [
          {"name": "xtr_weight", "as": "explore_emp_sharpe_score_weight"},
        ],
        function_name = "AdjustXtrWeight",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .if_("enable_explore_ranking_weight_adjust_by_high_time_rate == 1") \
      .explore_ranking_low_time_active_weight_adjust() \
    .end_() \
    .if_("enable_fr_adjust_neg_similar_weight_by_entropy == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "user_info_ptr",
          {"name": "explore_fr_cal_weight_by_entropy_recent_minutes",  "as": "recent_minutes"},
          {"name": "explore_fr_cal_weight_by_entropy_entropy_threshold",  "as": "entropy_threshold"},
          {"name": "explore_fr_cal_weight_by_entropy_coeff_max",  "as": "coeff_max"},
          {"name": "explore_fr_cal_weight_by_entropy_coeff_min",  "as": "coeff_min"},
          {"name": "explore_fr_cal_weight_by_entropy_alpha",  "as": "alpha"},
          {"name": "explore_fr_cal_weight_by_entropy_beta",  "as": "beta"},
          {"name": "explore_fr_cal_weight_by_entropy_omega",  "as": "omega"},
          {"name": "hot_user_unexpected_score_weight_new",  "as": "xtr_weight"},
        ],
        export_common_attr = [
          {"name": "xtr_weight",  "as": "hot_user_unexpected_score_weight_new"},
        ],
        function_name = "AdjustWeightByEntropyScore",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .if_("enable_fr_adjust_neg_similar_weight_by_first_fresh == 1 and page_index >= 1 and page_index <= explore_fr_adjust_neg_weight_screen_threshold") \
      .gen_common_attr_by_lua(
        attr_map = {
          "explore_bad_sim_score_in_order_weight" : "hot_user_unexpected_score_weight_new_boost_weight * explore_bad_sim_score_in_order_weight",
        }
      ) \
    .end_() \
    .if_("explore_enable_request_pxtr_weight_adjust == 1") \
      .request_pxtr_weight_adjust() \
    .end_() \
    .if_("explore_enable_user_vv_weight_adjust == 1 and user_age_segment >= explore_fr_hate_like_weight_adjust_age_min and user_age_segment <= explore_fr_hate_like_weight_adjust_age_max") \
      .user_vv_weight_adjust() \
    .end_() \
    .if_("explore_enable_user_vv_ensemble_power_weight_adjust == 1") \
      .user_vv_ensemble_power_weight_adjust() \
    .end_() \
    .if_("explore_enable_user_active_days_ensemble_power_weight_adjust == 1") \
      .user_active_days_ensemble_power_weight_adjust() \
    .end_() \
    .if_("explore_enable_user_recent_hate_count_ensemble_power_weight_adjust == 1 and recent_hate_count > explore_koc_htr_count_threshold") \
      .user_active_days_ensemble_koc_cover_htr_power_weight_adjust() \
      .user_active_days_ensemble_koc_detail_htr_power_weight_adjust() \
    .end_() \
    .if_("explore_enable_user_active_days_ensemble_power_consume_time_slide_weight_adjust == 1") \
      .user_active_days_ensemble_power_consume_time_slide_weight_adjust() \
    .end_() \
    .if_("enable_fr_adjust_low_active_weight == 1 and find_user_active_degree >= explore_adjust_active_start and find_user_active_degree <= explore_adjust_active_end") \
      .gen_common_attr_by_lua(
        attr_map = {
          "explore_ensemble_power_weight_fullrank_pctr_score" : "explore_ctr_low_active_adjust_weight * explore_ensemble_power_weight_fullrank_pctr_score",
          "explore_ensemble_power_weight_fullrank_pltr_score" : "explore_ltr_low_active_adjust_weight * explore_ensemble_power_weight_fullrank_pltr_score",
          "explore_ensemble_power_weight_fullrank_pwtr_score" : "explore_wtr_low_active_adjust_weight * explore_ensemble_power_weight_fullrank_pwtr_score",
          "explore_ensemble_power_weight_fullrank_pftr_score" : "explore_ftr_low_active_adjust_weight * explore_ensemble_power_weight_fullrank_pftr_score",
          "explore_ensemble_power_weight_fullrank_pcltr_score" : "explore_cltr_low_active_adjust_weight * explore_ensemble_power_weight_fullrank_pcltr_score",
        }
      ) \
    .end_() \
    .if_("enable_explore_fr_boost_negative_feedback_weight == 1 and user_active_decline_score >= explore_fr_boost_negative_feedback_queue_of_user_active_decline_score_threshold and (find_user_active_degree == 3 or find_user_active_degree == 4)") \
      .gen_common_attr_by_lua( # 针对高活全勤人群如果用户活跃衰退分user_active_decline_score大于阈值对负反馈相关队列进行boost
        attr_map = {
          "htr_score_weight" : "explore_fr_boost_htr_score_weight_coefficient * htr_score_weight",
          "explore_bad_sim_score_in_order_weight" : "explore_fr_boost_bad_cover_similary_score_coefficient * explore_bad_sim_score_in_order_weight",
          "koc_cover_htr_score_in_order_weight" : "explore_fr_boost_koc_cover_htr_score_in_order_weight_coefficient * koc_cover_htr_score_in_order_weight",
          "koc_detail_htr_score_in_order_weight" : "explore_fr_boost_koc_detail_htr_score_in_order_weight_coefficient * koc_detail_htr_score_in_order_weight",
        }
      ) \
    .end_() \
    .if_("explore_enable_user_poor_quality_hate_reason_ensemble_power_weight_adjust == 1 and user_poor_quality_hate_reason_count ~= nil and user_poor_quality_hate_reason_count > explore_user_poor_quality_hate_reason_count_threshold") \
      .user_hate_reason_bad_score_power_weight_adjust() \
    .end_() \
    .if_("enable_fr_adjust_user_mau_weight == 1 and explore_enable_user_mau_emp_xtr == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "explore_ensemble_power_weight_fullrank_pctr_score" : "user_mau_emp_evtr * explore_ensemble_power_weight_fullrank_pctr_score",
          "explore_ensemble_power_weight_fullrank_pltr_score" : "user_mau_emp_ltr * explore_ensemble_power_weight_fullrank_pltr_score",
          "explore_ensemble_power_weight_fullrank_pwtr_score" : "user_mau_emp_wtr * explore_ensemble_power_weight_fullrank_pwtr_score",
          "explore_ensemble_power_weight_fullrank_pftr_score" : "user_mau_emp_ftr * explore_ensemble_power_weight_fullrank_pftr_score",
          "fr_pmctr_rank_weight" : "user_mau_emp_cmtr * fr_pmctr_rank_weight",
          "awesome_wtd_weight_push" : "user_mau_emp_rank_play * awesome_wtd_weight_push",
        }
      ) \
    .end_() \
    .if_("explore_is_low_diversity_status == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "explore_ensemble_power_weight_fullrank_pctr_score" : "explore_fullrank_pctr_low_diversity_ratio * explore_ensemble_power_weight_fullrank_pctr_score"
        }
      ) \
    .end_() \
    .if_("enable_explore_share_pull_ftr_weight_adjust_coef == 1") \
      .split_string(
        input_common_attr = "explore_user_ftr_weight_adjust_score_align_list",
        output_common_attr = "user_ftr_weight_adjust_score_align_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_double = True,
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "uShareidCntKV", "as": "shareid_cnt"},
          {"name": "uOpenShareidCntKV", "as": "open_shareid_cnt"},
          {"name": "uOpenDeviceCntKV", "as": "open_device_cnt"},
          {"name": "uPullNumKV", "as": "pull_num"},
          {"name": "uShareBringNewDeviceNumKV", "as": "share_bring_new_device_num"},
          {"name": "uAttributionPerShareKV", "as": "attribution_per_share"},
          {"name": "user_ftr_weight_adjust_score_align_list", "as": "score_align_list"},
          {"name": "explore_user_ftr_weight_adjust_upper", "as": "upper"},
          {"name": "explore_user_ftr_weight_adjust_lower", "as": "lower"},
          {"name": "explore_user_ftr_weight_adjust_score_avg", "as": "score_avg"},
        ],
        export_common_attr = [
          {"name": "coef", "as": "share_pull_ftr_adjust_coef"},
        ],
        function_name = "UserFtrWeightAdjustCoef",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .if_("enable_explore_cal_share_pull_ftr_full_rank == 1") \
      .gen_common_attr_by_lua(
        attr_map={
          "explore_ensemble_power_weight_fullrank_pftr_score": "explore_ensemble_power_weight_fullrank_pftr_score * share_pull_ftr_adjust_coef",
        }
      ) \
    .end_() \
    .if_("explore_fullrank_skip_zero_play_user_xtr_boost_handle == 1") \
      .set_attr_value( #和后面的pack_common_attr必须保持顺序一致性,非ctr相关队列，降权,zero play xtr begin
        no_overwrite=True,
        common_attrs=[
          {
            "name": "explore_fr_ensemble_xtr_name_list",
            "type": "string_list",
            "value": zero_play_queues
          }
        ]
      ) \
      .pack_common_attr(
        input_common_attrs = zero_play_queues,
        output_common_attr = "explore_fr_ensemble_xtr_value_list",
      ) \
      .enrich_attr_by_light_function( # suweiwei03 低活零播用户只保留ctr队列 begin
        import_common_attr = [
          {"name": "explore_zero_play_days_15d", "as": "explore_zero_play_days_15d"},
          {"name": "find_visit_days_30d", "as": "explore_visit_days_30d"},
          {"name": "explore_fr_ensemble_zero_play_days_threshold", "as": "zero_play_days_threshold"}, 
          {"name": "explore_fr_ensemble_zero_play_ratio_threshold", "as": "zero_play_ratio_threshold"},   
          {"name": "explore_fr_ensemble_zero_play_boost_type", "as": "boost_type"},
          {"name": "explore_fr_ensemble_zero_play_xtr_boost_weight", "as": "boost_weight"}, 
          {"name": "explore_fr_ensemble_xtr_name_list", "as": "attr_name_list"},
          {"name": "explore_fr_ensemble_xtr_value_list", "as": "attr_value_list"},
        ],
        export_common_attr = zero_play_queues,
        function_name = "CalculateZeroPlayLaXtrBoostWeight",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .set_attr_value( #和后面的pack_common_attr必须保持顺序一致性，ctr相关队列，单独boost
        no_overwrite=True,
        common_attrs=[
          {
            "name": "explore_fr_ensemble_ctr_name_list",
            "type": "string_list",
            "value": zero_play_ctr_queues
          }
        ]
      ) \
      .pack_common_attr(
        input_common_attrs = zero_play_ctr_queues,
        output_common_attr = "explore_fr_ensemble_ctr_value_list",
      ) \
      .enrich_attr_by_light_function( #zero play xtr end
        import_common_attr = [
          {"name": "explore_zero_play_days_15d", "as": "explore_zero_play_days_15d"},
          {"name": "find_visit_days_30d", "as": "explore_visit_days_30d"},
          {"name": "explore_fr_ensemble_zero_play_days_threshold", "as": "zero_play_days_threshold"}, 
          {"name": "explore_fr_ensemble_zero_play_ratio_threshold", "as": "zero_play_ratio_threshold"},   
          {"name": "explore_fr_ensemble_zero_play_boost_type", "as": "boost_type"},
          {"name": "explore_fr_ensemble_zero_play_ctr_boost_weight", "as": "boost_weight"}, 
          {"name": "explore_fr_ensemble_ctr_name_list", "as": "attr_name_list"},
          {"name": "explore_fr_ensemble_ctr_value_list", "as": "attr_value_list"},
        ],
        export_common_attr = zero_play_ctr_queues,
        function_name = "CalculateZeroPlayLaXtrBoostWeight",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_if_() \
    .if_("explore_fr_skip_infer_uv_ctr_boost_handle == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "infer_uv_ctr", 
          {"name": "refreshTimes", "as": "refresh_times"},
          {"name": "explore_fr_ensemble_infer_uv_ctr_refresh_times_threshold", "as": "refresh_times_threshold"},
          {"name": "explore_fr_ensemble_infer_uv_ctr_infer_uv_ctr_threshold", "as": "infer_uv_ctr_threshold"},
          {"name": "explore_fr_ensemble_infer_uv_ctr_weight_max", "as": "weight_max"},
          {"name": "explore_fr_ensemble_infer_uv_ctr_weight_min", "as": "weight_min"}, 
          {"name": "explore_fr_ensemble_infer_uv_ctr_alpha", "as": "alpha"},
          {"name": "explore_fr_ensemble_infer_uv_ctr_beta", "as": "beta"},
          {"name": "explore_fr_ensemble_infer_uv_ctr_omega", "as": "omega"},
          {"name": "explore_fr_ensemble_infer_uv_ctr_boost_type", "as": "boost_type"},
          {"name": "explore_ensemble_power_weight_fullrank_pctr_score", "as": "xtr_weight"},
        ],
        export_common_attr = [
          {"name": "xtr_weight", "as": "explore_ensemble_power_weight_fullrank_pctr_score"}
        ],
        function_name = "CalcXtrWeightByInferUvCtr",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .if_("explore_rank_social_pftr_queue_enable == 1 and (bid_follow_num == 0 or (u_inside_share_active_degree_detail_code > 3))") \
      .gen_common_attr_by_lua(
        attr_map = {
            "explore_ensemble_power_weight_fullrank_pftr_score_social" : "0.0",
        }
      ) \
    .end_() \
    .if_("explore_rank_social_pftr_queue_enable == 2 and (bid_follow_num == 0 or (u_share_num_30d == 0 and u_message_active_degree ~= 5 and u_message_active_degree ~= 6))") \
      .gen_common_attr_by_lua(
        attr_map = {
            "explore_ensemble_power_weight_fullrank_pftr_score_social" : "0.0",
        }
      ) \
    .end_() \
    .if_("explore_rank_social_pftr_queue_enable == 3 and (bid_follow_num == 0 or (user_msg_cnt_ssm_today + user_msg_cnt_gsm_today > 0))") \
      .gen_common_attr_by_lua(
        attr_map = {
            "explore_ensemble_power_weight_fullrank_pftr_score_social" : "0.0",
        }
      ) \
    .end_() \
    .if_("explore_enable_rank_stage2_ef_weight_adjust == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "hot_fountain_fetr_weight_push": "hot_fountain_fetr_weight_push * explore_fountain_view_weight",
          "hot_fountain_fountain_eff_weight_push": "hot_fountain_fountain_eff_weight_push * explore_fountain_view_weight",
        }
      ) \
    .end_() \
    .if_("explore_enable_rank_stage2_ft_preference_ctr_weight_adjust == 1 and explore_fountain_view_weight ~= nil and explore_fountain_view_weight > 0.0") \
      .gen_common_attr_by_lua(
        attr_map = {
          "explore_ensemble_power_weight_fullrank_corr_pctr_psvr": "explore_ensemble_power_weight_fullrank_corr_pctr_psvr / explore_fountain_view_weight",
        }
      ) \
      .if_("explore_enable_rank_stage2_ft_preference_ctr_svtr_weight_adjust == 1") \
        .gen_common_attr_by_lua(
          attr_map = {
            "explore_ensemble_power_weight_fullrank_psvr_score": "explore_ensemble_power_weight_fullrank_psvr_score / explore_fountain_view_weight * explore_enable_rank_stage2_ft_preference_ctr_svtr_weight_adjust_coef",
          }
        ) \
      .end_() \
    .end_() \
    .if_("explore_rank_social_pftr_dur_queue_enable == 1") \
      .explore_cal_rank_ensemble_pftr_dur() \
    .end_() \
    .if_("explore_enable_get_personal_requst_pxtr_coef == 1") \
      .rank_stage2_request_pxtr_boost_coef() \
    .end_() \
    .if_("explore_rank_enable_request_pxtr_adjust == 1") \
      .rank_stage2_request_personal_boost() \
    .end_() \
    .if_("explore_enable_get_emp_fetr_boost_coef == 1") \
      .rank_stage2_emp_fetr_boost_coef() \
    .end_() \
    .if_("explore_rank_enable_emp_fetr_adjust == 1") \
      .rank_stage2_fetr_adjust() \
    .end_() \
    .if_("enable_explore_rank_stage2_personal_cem == 1") \
      .rank_stage2_personal_cem() \
    .end_() \
    .if_("enable_explore_rank_stage2_personal_cem_es_weight == 1") \
      .rank_stage2_personal_cem_es_weight_adjust() \
    .end_() \
    .if_("enable_explore_rank_stage2_age_based_adjust == 1") \
      .user_age_based_weight_adjust_all() \
    .end_() \
    .if_("enable_calc_hetu_one_xtr_debias_fr_score == 1") \
      .explore_cal_hetu_one_debias_score_fr() \
    .end_() \
    .if_("enable_calc_fc_update_xtr_score == 1") \
      .explore_cal_update_xtr_score_rank() \
    .end_() \
    .if_("enable_calc_fc_upload_xtr_score == 1") \
      .explore_cal_upload_xtr_score_rank() \
    .end_() \
    .if_("enable_explore_fr_hot_ranking_retr_score == 1 and is_first_refresh == 1 and basic_info_age_segment_v2 < explore_fr_hot_ranking_retr_score_age_segment_threshold and active_days_gt_5min_rate < explore_fr_hot_ranking_retr_score_active_days_threshold") \
      .gen_hot_ranking_retr_score() \
    .end_() \
    .if_("enable_explore_fr_prefer_author_ranking_retr_score == 1 and is_first_refresh == 1 and basic_info_age_segment_v2 < explore_fr_prefer_author_ranking_retr_score_age_segment_threshold and active_days_gt_5min_rate < explore_fr_prefer_author_ranking_retr_score_active_days_threshold") \
      .gen_prefer_author_ranking_retr_score() \
    .end_() \
    .if_("enable_value_and_rank_score == 1 or (is_zero_play_user == 1 and enable_zero_user_value_and_rank_score == 1)") \
      .calc_value_and_rank_score() \
    .else_() \
      .explore_calc_ensemble_score(
        save_score_to_attr = "explore_fr_ensemble_score",
        save_ori_ensemble_score_to_attr = "original_explore_fr_ensemble_score",
        save_absolute_score_to_attr = "explore_fr_pxtr_absolute_score",
        user_power_calc = "{{explore_fr_fullrank_variant_enable_power_calc}}",
        rank_smooth = "{{explore_fr_fullrank_rank_smooth}}",
        rank_power_weight = "{{explore_fr_fullrank_rank_power_weight}}",
        use_reciprocal = "{{explore_fr_use_reciprocal}}",
        user_power_calc_v2 = "{{explore_fr_user_power_calc_v2}}",
        enable_use_reciprocal_duration_transform = "{{explore_fr_enable_duration_queue_transform}}",
        value_seq_fusion_status = "{{explore_fr_value_seq_fusion_status}}",
        duration_min = "{{explore_fr_duration_min}}",
        duration_max = "{{explore_fr_duration_max}}",
        enable_perf_pxtr_pic = "{{explore_enable_perf_pxtr_pic_limit}}",
        perf_pxtr_pic_num = "{{explore_perf_pxtr_pic_num}}",
        duration_add = 10,
        action_day = "{{rk_collect_queue_boost_active_day_num}}",
        enable_dynamic_weight_by_user_degree = "{{fr_enable_pcltr_adjust_by_gender_age}}",
        fr_rank_max_num = "{{explore_fr_rank_max_num}}",
        fr_rank_specified_num = "{{explore_fr_rank_specified_num}}",
        fr_rank_has_sec_str = "{{explore_fr_rank_has_sec_str}}",
        use_formula_pow_t = "{{explore_ensemble_score_use_new_formula}}",
        queues = all_ensemble_queues,
        two_way_sort = "{{explore_fr_use_two_way_sort}}",
        two_way_sort_total_size = "{{explore_fr_two_way_sort_total_size}}",
        two_way_sort_coeff = "{{explore_fr_two_way_sort_coeff}}",
        enable_power_weight_norm = "{{enable_power_weight_norm}}",
        power_weight_change_coeff = "{{power_weight_change_coeff}}",
        min_rank_weight = "{{explore_fr_fullrank_min_rank_weight}}",
        use_queue_smooth_as_rank_smooth = "{{explore_fr_use_queue_smooth_as_rank_smooth}}",
        use_queue_value_seq_fusion_status = "{{explore_fr_use_queue_value_seq_fusion_status}}",
        use_fractile_in_ensemble_sort = "{{explore_fr_use_fractile_in_ensemble_sort}}",
        queue_head_boost_index = "{{explore_fr_queue_head_boost_index}}",
        queue_tail_discount_index = "{{explore_fr_queue_tail_discount_index}}",
        use_rank_with_absolute_score = "{{explore_fr_use_rank_with_absolute_score}}",
        use_rank_sort_weight_adjust = "{{explore_rank_sort_weight_adjust}}",
        use_raw_bias_in_fusion = "{{explore_use_raw_bias_in_fusion}}",
      ) \
    .end_() \
    .if_("explore_enable_rank_ensemble_sort_f1 == 1") \
      .calc_by_formula1(
        kconf_key = "formula.scenarioKey64.ExploreRankEnsembleSort",
        import_item_attr = [
          "score_pctr_es",
          "interact_fusion_score", # action_once
          "score_pltr_es",
          "score_pwtr_es",
          "score_pcmtr_es",
          "score_pftr",
          "score_pcltr",
          "fr_score2",
          "bad_cover_similary_score",
          "bad_sense_similary_score",
          "score_psvr",
          "fr_score1",
          "corr_pctr_psvr",
          "global_emphtr_score",
          "score_phtr",
          "pctr",
          "pcmef",
          "pwtr",
          "pltr",
          "pcmtr",
          "awesome_wtd",
          "plsst",
          "pvtr",
          "pptr",
          "svtr_rid_ctr_score",
          "pevtr",
          "is_ugc_photo",
          "corr_fetr",
          "corr_fountain_eff",
          "koc_cover_htr",
          "koc_detail_htr",
          "esnn_model_score",
          "consume_time_slide",
          "consume_time_pf2r_score",
          "debias_by_pcoc_ltr_score",
          "debias_by_pcoc_cltr_score",
          "debias_by_pcoc_cmtr_score",
          "debias_by_pcoc_wtr_score",
          "pFindTotalSatisfactionScoreKV",
          "photo_history_interest_score_with_fr_ctr"
        ],
        import_common_attr = [
          "explore_today_vv",
          "active_days_avg_vv",
          "uExploreActiveDays",
          "user_age_segment",
          "basic_info_age_segment_v2",
          "page_index",
          "refreshTimes",
          "user_bad_sense_tolerance",
          "user_bad_cover_tolerance",
          "user_explore_last_like_gap_hour",
          "user_explore_last_follow_gap_hour",
          "user_explore_last_comment_gap_hour",
          "user_explore_last_collect_gap_hour",
          "user_emp_fountain_time_ratio"
        ],
        export_formula_value = [
          {"name": "final_score", "as": "rank_es_score_f1"}
        ],
        abtest_biz_name = "KUAISHOU_APPS"
      ) \
      .item_attr_operation(
        item_attr_a = "explore_fr_ensemble_score",
        item_attr_b = "rank_es_score_f1",
        operator = "*",
        output_attr = "explore_fr_ensemble_score"
      ) \
    .end_() \
    .if_("explore_enable_rank_unaudit_deboost_f1 == 1") \
      .calc_by_formula1(
        kconf_key = "formula.scenarioKey63.ExploreFrAuditScore",
        import_item_attr = [
          "explore_mc_bad_sense_similarity_score",
          "explore_mc_bad_hot_audit_similarity_score",
          "explore_mc_bad_cover_similarity_score"
        ],
        import_common_attr = [
          "active_days_gt_5min_rate",
          "uExploreActiveDays"
        ],
        export_formula_value = [
          {"name": "final_score", "as": "rank_unaudit_deboost_f1"}
        ],
        abtest_biz_name = "KUAISHOU_APPS"
      ) \
      .item_attr_operation(
        item_attr_a = "explore_fr_ensemble_score",
        item_attr_b = "rank_unaudit_deboost_f1",
        operator = "*",
        output_attr = "explore_fr_ensemble_score"
      ) \
    .end_() \
    .enrich_attr_by_lua(
      import_item_attr = [
        "pctr",
        "explore_fr_ensemble_score"
      ],
      export_item_attr = [
        "explore_fr_ensemble_score"
      ],
      function_for_item = "print1",
      lua_script_file = "explore/ranking/lua/module/ranking_ensemble_sort__print.lua"
    ) \
    .if_("ensemble_score_change_by_svr == 1") \
      .enrich_attr_by_light_function(
        import_item_attr = [
          "explore_fr_ensemble_score",
          "psvr"
        ],
        import_common_attr = [
          "ensemble_score_change_alpha",
          "ensemble_score_change_gamma",
          "ensemble_score_change_min_threshold",
          "ensemble_score_change_type"
        ],
        export_item_attr = [
          "explore_fr_ensemble_score"
        ],
        function_name = "EnsembleScoreChange",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .if_("ensemble_score_change_by_htr == 1") \
      .enrich_attr_by_light_function(
        import_item_attr = [
          "explore_fr_ensemble_score",
          {"name": "phtr", "as": "psvr"},
        ],
        import_common_attr = [
          {"name": "ensemble_score_change_alpha_htr", "as": "ensemble_score_change_alpha"},
          {"name": "ensemble_score_change_gamma_htr", "as": "ensemble_score_change_gamma"},
          {"name": "ensemble_score_change_min_threshold_htr", "as": "ensemble_score_change_min_threshold"},
          {"name": "ensemble_score_change_type_htr", "as": "ensemble_score_change_type"}
        ],
        export_item_attr = [
          "explore_fr_ensemble_score"
        ],
        function_name = "EnsembleScoreChange",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .if_("skip_hot_fr_ensemble_sort_boost == 0") \
      .if_("enable_fr_boost_loyal_fans_reason == 1") \
        .fr_boost_loyal_fans_reason() \
      .end_() \
      .if_("enable_ranking_personified_author_boost == 1", to_be_delete = "date=2024-05-29;committer=fenglei03") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "ranking_personified_author_boost_coef", "as": "personified_author_coeff"},
            {"name": "ranking_blacklist_author_boost_coef", "as": "blacklist_author_coeff"},
            {"name": "explore_personified_author_fans_thre_max", "as": "author_fans_thre_max"},
            {"name": "explore_personified_author_fans_thre_min", "as": "author_fans_thre_min"},
            {"name": "ranking_merchant_cart_boost_coef", "as": "merchant_cart_coeff"},
            {"name": "ranking_young_women_boost_coef", "as": "young_women_coeff"},
            {"name": "ranking_age_segment_18_23_coeff", "as": "age_segment_18_23_coeff"},
            "basic_info_gender_v2",
            "basic_info_age_segment_v2",
          ],
          import_item_attr = [
            {"name": "author__fans_count", "as": "author_fans_count"},
            {"name": "eyeshot_source", "as": "eyeshot_source"},
            {"name": "merchant_photo_cart_relation", "as": "cart_relation"},
            {"name": "merchant_item_info__item_id_list", "as": "cart_itemlist"},
            {"name": "live_photo_info__is_living", "as": "is_living"},
            {"name": "explore_fr_ensemble_score", "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": "explore_fr_ensemble_score"},
          ],
          function_name = "PersonifiedAuthorBoost",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_ranking_caption_boost == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "ranking_caption_boost_coef", "as": "caption_boost_coef"},
            {"name": "ranking_caption_boost_len_thresh", "as": "caption_boost_len_thresh"},
            {"name": "ranking_caption_boost_len_max", "as": "caption_boost_len_max"},
            {"name": "ranking_boost_only_xhs_photo", "as": "boost_only_xhs_photo"},
            {"name": "ranking_boost_only_picture", "as": "boost_only_picture"},
          ],
          import_item_attr = [
            {"name": "explore_fr_ensemble_score", "as": "score"},
            "caption_length",
            "is_xhs_type_photo",
            "is_picture",
          ],
          export_item_attr = [
            {"name": "score", "as": "explore_fr_ensemble_score"},
          ],
          export_common_attr = [
            {"name": "boost_count", "as": "ranking_caption_boost_count"},
            {"name": "total_count", "as": "ranking_caption_total_count"},
          ],
          function_name = "BoostWithCaption",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .perflog_attr_value(
          check_point = "ranking_caption_boost",
          common_attrs = [
            "ranking_caption_boost_count",
            "ranking_caption_total_count",
          ],
        ) \
      .end_() \
      .if_("explore_enable_fr_xhs_target_qualified_photo_boost == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_fr_xhs_target_qualified_photo_boost_coeff", "as": "boost_discount_coeff"},
          ],
          import_item_attr = [
            {"name": "is_xhs_target_qualified_photo", "as": "need_item_attr"},
            {"name": "explore_fr_ensemble_score", "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": "explore_fr_ensemble_score"},
          ],
          function_name = "BoostOrDiscount",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_() \
      .if_("explore_enable_gen_senseview_lowcost_photo_tag == 1") \
        .enrich_attr_by_light_function(
          import_item_attr = [
            "audit_b_second_tag"
          ],
          export_item_attr = [
            "is_senseview_lowcost_photo"
          ],
          function_name = "IsSenseviewLowcostPhoto",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .enrich_attr_by_light_function(
          item_list_from_attr = "explore_recent_play_list",
          import_item_attr = [
            "audit_b_second_tag"
          ],
          export_item_attr = [
            "is_senseview_lowcost_photo"
          ],
          function_name = "IsSenseviewLowcostPhoto",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .if_("explore_ranking_diversity_enable_standard_explore_realshow_pid_list == 1") \
          .enrich_attr_by_light_function(
            item_list_from_attr = "standard_explore_realshow_pid_list",
            import_item_attr = [
              "audit_b_second_tag"
            ],
            export_item_attr = [
              "is_senseview_lowcost_photo"
            ],
            function_name = "IsSenseviewLowcostPhoto",
            class_name = "ExploreLightFunctionSetV2",
          ) \
        .end_() \
        .if_("explore_enable_fr_senseview_lowcost_photo_adjust == 1") \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "explore_fr_senseview_lowcost_discount_coef", "as": "boost_discount_coeff"},
            ],
            import_item_attr = [
              {"name": "explore_fr_ensemble_score", "as": "score"},
            ],
            export_item_attr = [
              {"name": "score", "as": "explore_fr_ensemble_score"},
            ],
            function_name = "BoostOrDiscountV2",
            class_name = "ExploreLightFunctionSetV2",
            target_item = {
              "is_senseview_lowcost_photo": 1
            },
          ) \
        .end_() \
      .end_() \
      .if_("fr_enable_user_intrest_adjust == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "output_intrest_key_list", "as": "intrest_key_list"},
            {"name": "output_intrest_value_list", "as": "intrest_value_list"},
            {"name": "explore_fr_user_intrest_adjust_boost_coef", "as": "boost_coef"},
            {"name": "explore_fr_user_intrest_adjust_discount_coef", "as": "discount_coef"},
            {"name": "explore_enable_hetu1_user_intrest_adjust", "as": "enable_hetu1"}, 
          ],
          import_item_attr = [
            {"name": "explore_fr_ensemble_score", "as": "input_score"},
            "hetu_tag_level_info__hetu_level_two",
            "hetu_tag_level_info__hetu_level_one",
          ],
          export_item_attr = [
            {"name": "output_score", "as": "explore_fr_ensemble_score"},
          ],
          function_name = "IntrestAdjustScore",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_if_() \
      .if_("fr_enable_high_htr_discount == 1") \
        .enrich_attr_by_light_function(
          import_common_attr=[
            {"name": "fr_high_htr_discount_coef", "as": "high_htr_discount_coef"},
            {"name": "fr_high_htr_threshold", "as": "high_htr_threshold"},
            {"name": "fr_high_htr_discount_power", "as": "htr_discount_power"},
          ],
          import_item_attr = [
            {"name": "explore_fr_ensemble_score", "as": "es_score"},
            {"name": "phtr", "as": "htr_score"},
          ],
          export_item_attr = [
            {"name": "es_score", "as": "explore_fr_ensemble_score"},
          ],
          function_name = "HighHtrMixEsScore",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_if_() \
      .if_("fr_enable_la_follow_boost_fresh_thr >= (refreshTimes or 1000)") \
        .enrich_attr_by_light_function(
          target_item = {"is_long_view_author": 1},
          import_common_attr = [
            {"name": "fr_boost_follow_author_weight", "as": "boost_weight"},
            {"name": "fr_weaken_follow_author_weight", "as": "weaken_weight"},
          ],
          import_item_attr = [
            {"name": "explore_fr_ensemble_score", "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": "explore_fr_ensemble_score"},
          ],
          function_name = "EnsembleScoreBoost",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_if_() \
      .if_("enable_impression_audit_adjust == 1") \
        .impression_audit_adjust() \
      .end_if_() \
      .if_("explore_fr_enable_boost_recent_consume_photo == 1 and user_need_saving_flag == 1") \
        .boost_recent_consume_photo() \
      .end_if_() \
      .if_("explore_fr_enable_boost_audit_good_photo == 1 and user_need_saving_flag == 1") \
        .boost_audit_good_photo() \
      .end_if_() \
      .if_("explore_fr_enable_not_cover_audit_discount_for_first_page == 1") \
        .not_cover_audit_photo_discount() \
      .end_if_() \
      .if_("explore_fr_enable_cropped_photo_discount == 1") \
        .cropped_photo_discount() \
      .end_if_() \
      .if_("explore_fr_enable_merchant_photo_boost == 1") \
        .merchant_photo_boost_by_buyer_type() \
      .end_if_() \
      .if_("explore_fr_enable_merchant_live_boost == 1") \
        .merchant_live_boost_by_buyer_type() \
      .end_if_() \
      .if_("explore_fr_enable_merchant_price_inferior_reduce_weight == 1", to_be_delete = "date=2024-05-29;committer=zhanglinjiang") \
        .merchant_price_inferior_reduce_weight() \
      .end_if_() \
      .if_("enable_rank_refinement_boost_personified_author == 1", to_be_delete = "date=2024-05-29;committer=xubaoquan") \
        .refinement_boost_personified_author() \
      .end_if_() \
      .if_("explore_rank_enable_high_photo_count_author_adjust == 1") \
        .high_photo_count_author_adjust() \
      .end_() \
      .if_("explore_rank_enable_high_photo_count_author_adjust_v2 == 1") \
        .high_photo_count_author_adjust_v2() \
      .end_() \
      .if_("explore_fr_enable_new_interest_explore_boost == 1", to_be_delete = "date=2024-05-29;committer=wangziqi05") \
        .new_interest_explore_boost() \
      .end_() \
      .if_("explore_rank_enable_none_caption_discount == 1", to_be_delete = "date=2024-05-29;committer=liuhao07") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rank_none_caption_discount_coef", "as": "discount_coef"},
          ],
          import_item_attr = [
            {"name": "explore_fr_ensemble_score", "as": "score"},
            "caption_length",
          ],
          export_item_attr = [
            {"name": "score", "as": "explore_fr_ensemble_score"},
          ],
          function_name = "NoneCaptionDiscount",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("fr_enable_high_global_emphtr_discount == 1") \
        .high_global_emphtr_discount() \
      .end_() \
      .if_("enable_fr_search_score_boost == 1") \
        .fr_search_score_boost() \
      .end_() \
      .if_("enable_fr_boost_user_author_reason == 1", to_be_delete = "date=2024-05-29;committer=fengjingping") \
        .fr_boost_user_author_reason() \
      .end_() \
      .if_("enable_fr_boost_ua_long_view == 1") \
        .fr_boost_ua_long_view() \
      .end_() \
      .if_("enable_fr_boost_click_count == 1", to_be_delete = "date=2024-05-29;committer=fengjingping") \
        .fr_boost_click_count() \
      .end_() \
      .if_("explore_fr_timeliness_photo_boost == 1") \
        .fr_timeliness_photo_boost() \
      .end_() \
      .if_("enable_fr_boost_topk_high_ctr_photo == 1") \
        .fr_boost_topk_high_ctr_photo() \
      .end_() \
      .if_("enable_lower_avg_ctr_users == 1") \
        .is_lower_avg_ctr_users() \
        .if_("is_lower_avg_ctr_users == 1") \
          .fr_boost_lower_avg_ctr_photo() \
        .end_() \
      .end_() \
      .if_("enable_fr_marketing_compensation_discount == 1") \
        .fr_marketing_compensation_discount() \
      .end_() \
      .if_("enable_fr_marketing_compensation_personal_discount == 1") \
        .fr_marketing_compensation_personal_discount() \
      .end_() \
      .if_("enable_fr_s2_author_circle_cluster_id_boost == 1") \
        .author_circle_cluster_id_boost("explore_fr_ensemble_score", ["corr_pctr", "pwtr", "pltr", "pftr", "pcmtr", "pcltr", "fr_score2", "awesome_wtd", "explore_fr_ensemble_score"]) \
      .end_() \
      .if_("enable_fr_s2_interest_cluster_id_boost == 1") \
        .interest_cluster_id_boost("explore_fr_ensemble_score", ["corr_pctr", "pwtr", "pltr", "pftr", "pcmtr", "pcltr", "fr_score2", "awesome_wtd", "explore_fr_ensemble_score"]) \
      .end_() \
      .if_("enable_fr_s2_unbias_interest_photo_boost == 1") \
        .if_("enable_fr_s2_unbias_interest_photo_boost_vv_adjust == 1 and user_vv_flag == 1") \
          .user_vv_type_weight_adjust("explore_fr_s2_unbias_interest_photo_boost_coeff") \
        .end_() \
        .if_("enable_fr_s2_unbias_interest_photo_boost_cocoon_adjust == 1 and user_cocoon_flag == 1") \
          .user_cocoon_weight_adjust("explore_fr_s2_unbias_interest_photo_boost_coeff") \
        .end_() \
        .unbias_interest_photo_boost("explore_fr_ensemble_score", "fr_s2") \
      .end_() \
      .if_("enable_fr_s2_hot_list_photo_boost == 1") \
        .hot_list_photo_boost("explore_fr_ensemble_score", "fr_s2") \
      .end_() \
      .if_("enable_fr_s2_short_uninterest_photo_discount == 1") \
        .short_uninterest_photo_discount("explore_fr_ensemble_score", "fr_s2") \
      .end_() \
      .if_("enable_fr_s2_short_uninterest_decay_discount == 1") \
        .short_uninterest_decay_discount("explore_fr_ensemble_score", "fr_s2") \
      .end_() \
      .if_("enable_fr_s2_unbias_interest_cids_boost == 1") \
        .fr_unbias_interest_cluster_boost() \
      .end_() \
      .if_("enable_fr_s2_interest_generalization_boost == 1") \
        .fr_interest_generalization_boost("explore_fr_ensemble_score", "fr_s2") \
      .end_() \
      .if_("enable_fr_s2_interest_cid == 1") \
        .if_("enable_fr_s2_use_positive_interest_and_score_list == 1") \
          .fr_cal_interest_cid_coeff("user_postive_interest_score_list") \
        .else_() \
          .fr_cal_interest_cid_coeff() \
        .end_() \
        .fr_interest_score_cids_boost() \
      .end_() \
      .if_("enable_fr_s2_valid_interest_cid_boost == 1") \
        .fr_cal_valid_interest_cid_coeff() \
        .fr_valid_interest_score_cids_boost() \
      .end_() \
      .if_("enable_fr_s2_short_valid_interest_first_refresh_boost == 1 and is_first_refresh == 1") \
        .fr_cal_short_valid_interest_first_refresh_coeff() \
        .fr_short_valid_interest_first_refresh_boost() \
      .end_() \
      .if_("enable_fr_s2_interest_migration_photo_boost == 1") \
        .interest_migration_photo_boost("explore_fr_ensemble_score") \
      .end_() \
      .if_("enable_fr_protogenetic_advertise_discount == 1") \
        .fr_protogenetic_advertise_discount() \
      .end_() \
      .if_("enable_fr_category_boost == 1") \
        .fr_category_boost() \
      .end_() \
      .if_("enable_fr_boost_topk_hot_list_photo == 1") \
        .fr_boost_topk_hot_list_photo() \
      .end_() \
      .if_("enable_fr_boost_topk_prior_author_photo == 1") \
        .fr_boost_topk_prior_author_photo() \
      .end_() \
      .if_("enable_fr_boost_topk_life_prior_photo == 1") \
        .fr_boost_topk_life_prior_photo() \
      .end_() \
      .if_("enable_fr_boost_topk_original_author_photo == 1") \
        .fr_boost_topk_original_author_photo() \
      .end_() \
      .if_("enable_fr_deboost_over_distribute_photo == 1") \
        .fr_deboost_over_distribute_photo() \
      .end_() \
      .if_("enable_fr_boost_long_worth_author_photo == 1") \
        .fr_boost_long_worth_author_photo() \
      .end_() \
      .if_("explore_enable_fr_boost_useful_author == 1") \
        .fr_boost_useful_author() \
      .end_() \
      .if_("enable_fr_boost_top_and_deboost_reciprocal_action == 1") \
        .fr_boost_top_and_deboost_reciprocal_like_action() \
        .fr_boost_top_and_deboost_reciprocal_follow_action() \
      .end_() \
      .if_("explore_enable_rank_update_bar_boost == 1") \
        .rank_update_bar_boost() \
      .end_() \
      .if_("explore_rank_enable_llm_negative_photo_adjust == 1") \
        .llm_negative_photo_adjust() \
      .end_() \
      .if_("explore_rank_enable_llm_negative_photo_personal_adjust == 1") \
        .fr_llm_negative_photo_personal_adjust() \
      .end_() \
      .if_("explore_rank_enable_fr_poor_quality_author_personal_deboost == 1") \
        .fr_poor_quality_author_personal_deboost() \
      .end_() \
      .if_("explore_fr_enable_diversity_distribution_adjust == 1") \
        .diversity_distribution_adjust() \
      .end_() \
      .if_("enable_explore_fr_boost_authority_author == 1") \
        .fr_boost_authority_author() \
      .end_() \
      .if_("enable_explore_fr_boost_expertise_author == 1") \
        .fr_boost_expertise_author() \
      .end_() \
      .if_("enable_explore_fr_boost_original_submission_author == 1") \
        .fr_boost_original_submission_author() \
      .end_() \
      .if_("enable_explore_fr_boost_personalization_author == 1") \
        .fr_boost_personalization_author() \
      .end_() \
      .if_("enable_explore_fr_good_author_pool_photo_personal_adjust == 1") \
        .explore_fr_good_author_pool_photo_personal_adjust() \
      .end_() \
      .if_("enable_explore_fr_hetu_tag_time_preference_boost == 1") \
        .explore_fr_hetu_tag_time_preference_boost() \
      .end_() \
      .if_("enable_explore_partial_time_based_interest_boost_fr_s2 == 1") \
        .partial_time_based_interest_boost("explore_fr_ensemble_score", "fr_s2") \
      .end_() \
      .if_("enable_explore_partial_time_based_tagnex_boost_fr_s2 == 1") \
        .partial_time_based_tagnex_boost("explore_fr_ensemble_score", "fr_s2") \
      .end_() \
      .if_("enable_explore_fr_s2_boost_user_short_develop_interest == 1 and uExploreFountainPreferenceTypeKV == 1") \
        .boost_user_short_develop_interest("explore_fr_ensemble_score", stage_name="fr_s2") \
      .end_() \
      .if_("enable_explore_cs_boost_fr_s2 == 1") \
        .explore_ranking_cold_photo_boost() \
      .end_() \
      .if_("enable_fr_s2_short_term_photo_tagnex_score_adjust == 1") \
        .short_term_photo_tagnex_score_adjust() \
      .end_() \
      .if_("enable_fr_s2_short_term_photo_cluster_id_score_adjust == 1") \
        .short_term_photo_cluster_id_score_adjust() \
      .end_() \
      .if_("enable_fr_s2_short_term_photo_hetu_level2_score_adjust == 1") \
        .short_term_photo_hetu_level2_score_adjust() \
      .end_() \
      .if_("enable_fr_s2_short_term_photo_hashtag_score_adjust == 1") \
        .short_term_photo_hashtag_score_adjust() \
      .end_() \
      .if_("enable_fr_s2_short_term_photo_hetu_tag_score_adjust == 1") \
        .short_term_photo_hetu_tag_score_adjust() \
      .end_() \
      .if_("enable_fr_s2_short_term_photo_interest_community_tag_score_adjust == 1") \
        .short_term_photo_interest_community_tag_score_adjust() \
      .end_() \
      .if_("enable_fr_s2_short_term_photo_sid_score_adjust == 1") \
        .short_term_photo_sid_score_adjust() \
      .end_() \
      .if_("enable_explore_not_correlation_deboost_ranking_s2 == 1") \
        .explore_cover_video_not_correlation_ranking_deboost() \
      .end_() \
      .if_("enable_explore_fr_s2_good_author_show_case_boost == 1") \
        .explore_good_author_show_case_boost() \
      .end_() \
      .if_("enable_explore_fr_s2_good_author_e_commerce_boost == 1") \
        .explore_good_author_e_commerce_boost() \
      .end_() \
      .if_("enable_explore_fr_s2_interest_card_photo_score_adjust == 1") \
        .explore_fr_interest_card_photo_score_adjust() \
      .end_() \
    .end_if_() \
    .if_("enable_ranking_resort_new_interest_explore == 1") \
      .resort_new_interest_items() \
    .end_() \
    .sort(
      skip = "{{skip_fullrank_sort_by_ensemble_score}}",
      score_from_attr = "explore_fr_ensemble_score",
    ) \
    .copy_item_meta_info(
      save_item_seq_to_attr = "rank_final_index",
    ) \
    ._dump_attr_to_kafka(
      stage_name = "fr_s2_score", 
      dump_item_attr_list = [
        # ES 使用队列
        "score_pctr",
        "score_pltr",
        "score_pwtr",
        "score_pcmtr",
        "score_pptr",
        "score_pcmef",
        "score_pcltr",
        "corr_cpr",
        "score_pepstr",
        "score_phtr",
        "awesome_wtd_score",
        "photo_history_interest_score_with_fr_ctr",
        "fr_mc_embedding_score",
        "watch_time_fusion_score",
        "score_consume_time_ltr",
        "consume_time_pf2r_score",
        "watchtime_interact_score",
        "score_pftr",
        "score_pdtr",
        "corr_fetr",
        "svr_act_score",
        "pcmef_debias_score",
        "score_psvr",
        "corr_fountain_eff",
        "cascade_distill_play_7s",
        "multitask_ltr_pwtd",
        "multitask_ltr_pcvtr",
        "multitask_ltr_pctr",
        "explore_fr_ensemble_score",
        "save_es_pctr_score_to_kafka",
        "save_es_awesome_wtd_score_to_kafka",
        "is_long_view_author",
        "save_es_fountain_eff_score_to_kafka",
        "consume_time_ctr",
        "pic_ltr_acttr_db",
        "pic_ltr_ctr_db",
        "pic_ltr_fvtr",
        "pic_ltr_weighted_ctr",
        "fr_explore_cold_photo_score",
        "rank_final_index"
      ],
      dump_common_attr_list = [
        "explore_user_timely_diversity_click_entropy_score",
        "explore_user_timely_diversity_show_entropy_score"
      ]
    ) \
    .if_("explore_enable_rank_write_rank_neg_result_to_redis == 1") \
      .write_rank_neg_result_to_redis() \
    .end_() \
    .if_("explore_enable_rank_write_rank_pos_result_to_redis == 1") \
      .write_rank_pos_result_to_redis() \
    .end_() \

  def post_process(self) -> None:
      self.flow \
        .if_("enable_interact_fusion_score == 1") \
          .perflog_attr_value(
            check_point = "ranking_interact_fusion_score",
            item_attrs = [
              "phtr",
              "pctr",
              "pftr",
              "pdtr",
              "pcmtr",
              "pltr",
              "pcltr",
              "pwtr",
              "pevtr",
              "plvtr",
              "pfvtr",
              "pepstr",
              "pcmef",
              "fetr",
              "fr_score1",
              "interact_fusion_score",
              "watch_time_fusion_score"
            ],
          ) \
        .end_if_() \
        .log_debug_info(
          common_attrs = [
            "refreshTimes",
            "infer_uv_ctr",
            "explore_fr_ensemble_infer_uv_ctr_refresh_times_threshold",
            "explore_fr_ensemble_infer_uv_ctr_infer_uv_ctr_threshold",
            "explore_fr_ensemble_infer_uv_ctr_weight_max",
            "explore_fr_ensemble_infer_uv_ctr_weight_min",
            "explore_fr_ensemble_infer_uv_ctr_alpha",
            "explore_fr_ensemble_infer_uv_ctr_beta",
            "explore_fr_ensemble_infer_uv_ctr_omega",
            "explore_fr_ensemble_infer_uv_ctr_boost_type",
            "user_emp_ltr_fr_threshold",
            "user_emp_wtr_fr_threshold",
            "user_emp_ftr_fr_threshold",
            "user_emp_cmtr_fr_threshold"
          ],
          for_debug_request_only = True
        )
