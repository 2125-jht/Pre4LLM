from ranking import CommonModule
from ranking.ranking_queues import all_ensemble_queues,zero_play_queues,zero_play_ctr_queues
from ranking.xlife_ranking_queues import all_ensemble_queues_xlife

class RankingEnsembleSortModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
    
  def process(self) -> None:
    self.flow \
    .set_attr_value(
       no_overwrite = True,
       item_attrs = [
          {
             "name": "dis_fr_score1",
             "type": "double",
             "value": 0.0
          },
          {
             "name": "dis_fr_score2",
             "type": "double",
             "value": 0.0
          }
       ]
    ) \
    .if_("enable_explore_fr_pctr_fresh_request_adjust == 1 and is_fresh_request == 1") \
      .copy_attr(
        attrs=[{
          # 生活tab 首刷 pctr 调整
          "from_common": "explore_ensemble_power_weight_fullrank_pctr_score_fresh_request",
          "to_common": "explore_ensemble_power_weight_fullrank_pctr_score"
        }]
      ) \
    .end_() \
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
      lua_script_file = "life/ranking/lua/module/ranking_ensemble_sort__print.lua"
    ) \
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
        lua_script_file = "life/ranking/lua/module/ranking_ensemble_sort__print.lua"
      ) \
    .end_() \
    .if_("enable_awesome_wtd_push == 1") \
      .enrich_attr_by_lua(
        import_common_attr = [
          "awesome_wtd_pctr_weight",
          "awesome_wtd_awesome_wtd_weight"
        ],
        import_item_attr = [
          "awesome_wtd",
          "corr_pctr"
        ],
        export_item_attr = [
          "awesome_wtd_score"
        ],
        function_for_item = "awesome_wtd_score_change",
        lua_script_file = "life/ranking/lua/module/ranking_ensemble_sort__print.lua"
      ) \
    .end_() \
    .if_("enable_hot_fr_ewatch_score == 1") \
      .enrich_attr_by_lua(
        import_item_attr = [
          "duration_ms",
          "fr_score1",
        ],
        export_item_attr = [
          "ewatch_score"
        ],
        function_for_item = "ewatch_score_change",
        lua_script_file = "life/ranking/lua/module/ranking_ensemble_sort__print.lua"
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
          "user_group_emp_ltr",
          "user_group_emp_wtr",
          "user_group_emp_ftr",
          "user_group_emp_cmtr",
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
        ],
        function_name = "UserSortWeightAdjust",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .if_("enable_explore_rank_stage2_personal_cem == 1") \
      .rank_stage2_personal_cem() \
    .end_() \
    .if_("enable_explore_rank_stage2_personal_cem_es_weight == 1") \
      .rank_stage2_personal_cem_es_weight_adjust() \
    .end_() \
    .if_("enable_calc_coordinated_queues_score == 1") \
      .calc_coordinated_queues_score() \
    .end_() \
    .if_("explore_enable_request_pxtr_weight_adjust == 1") \
      .request_pxtr_weight_adjust() \
    .end_() \
    .if_("life_enable_calc_active_hetu_debias_score == 1") \
      .calc_active_hetu_debias_score() \
    .end_() \
    .if_("life_enable_active_hetu_pctr_debias_adjust == 1")\
      .life_ranking_s2_active_hetu_pctr_adjust()\
    .end_() \
    .if_("enable_life_fr_s2_hetu_debias_pctr == 1") \
      .explore_life_uninterest_hetu_exit_enricher(
        user_info_ptr_attr = "user_info_ptr",
        realshow_num_threshold = "{{life_fr_s2_hetu_debias_pctr_realshow_num_threshold}}",
        time_gap_s = "{{life_fr_s2_hetu_debias_pctr_time_gap_s}}",
        hetu_tag_attr = "hetu_tag_level_info__hetu_level_two",
        input_pctr_attr = "score_pctr",
        output_pctr_attr = "score_pctr",
        calculate_mode = "{{life_fr_s2_hetu_debias_pctr_calculate_mode}}",
        discount_coef = "{{life_fr_s2_hetu_debias_pctr_discount_coef}}",
        realshow_unclick_num_thr = "{{life_fr_s2_hetu_debias_pctr_realshow_unclick_num_thr}}",
      ) \
    .end_() \
    .if_("life_enable_f1_fr_first_page_adjust_score == 1") \
      .calc_by_formula1(
        kconf_key = "formula.scenarioKey33.LifeFrFirstPageAdjustScore",
        import_common_attr = [
          "page",
          "refreshTimes",
          "uNebulaXlifeVisitDays30dKV", 
          "uNebulaDoubleFindVisitDays30dKV",
        ],
        import_item_attr = [
          "hetu_tag_level1",
          "pctr", 
          "pltr", 
          "pevtr",
          "plvtr", 
          "score_pctr",
          "score_pltr",
          "report_discount",
        ],
        export_formula_value = [
          "fr_first_page_adjust_score"
        ],
        abtest_biz_name = "KUAISHOU_APPS"
      ) \
    .end_() \
    .if_("enable_life_fr_s2_diversity_weight_adjust == 1") \
      .life_fr_s2_diversity_weight_adjust() \
    .end_() \
    .if_("enable_life_fr_s2_neg_sim_weight_adjust == 1") \
      .life_fr_s2_neg_sim_weight_adjust() \
    .end_() \
    .if_("enable_value_and_rank_score == 1") \
      .calc_value_and_rank_score() \
    .else_if_("enable_life_fr_s2_use_new_es_func == 1") \
      .explore_life_calc_ensemble_score(
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
        queue_max_raw_score = "{{life_fr_queue_max_raw_score}}",
        queue_min_raw_score = "{{life_fr_queue_min_raw_score}}",
        enable_2sigma_range_control = "{{life_fr_enable_2sigma_range_control}}"
      ) \
    .else_if_("enable_life_fr_s2_use_simple_es_func == 1") \
      .explore_life_voyage_calc_ensemble_score(
        save_score_to_attr = "explore_fr_ensemble_score",
        save_ori_ensemble_score_to_attr = "original_explore_fr_ensemble_score",
        save_absolute_score_to_attr = "explore_fr_pxtr_absolute_score",
        queues = all_ensemble_queues_xlife,
      ) \
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
        queue_tail_discount_index = "{{explore_fr_queue_tail_discount_index}}"
      ) \
    .end_() \
    .log_debug_info(
          item_attrs = [
            "raw_similarity_score",
            "danlie_depress_score",
            "explore_fr_ensemble_score",
            "empirical_svtr",
            # "interest_migration_photo_coef",
            "long_term_nature_score",
          ],
          for_debug_request_only = True,
          respect_sample_loggging = True,
        ) \
    .if_("life_enable_f1_fr_s2_new_ensemble_sort_score == 1") \
    .calc_by_formula1(
      kconf_key = "formula.scenarioKey36.LifeFrS2EnsembleSortScore",
      import_item_attr = [
        "explore_fr_ensemble_score",
        "ctr_filter_ensemble_score",
        "report_discount",
        "hate_discount",
        "pctr",
        "corr_pctr",
        "pltr",
        "pwtr",
        "corr_pwtr",
        "pftr",
        "pcmtr",
        "pptr",
        "psvr",
        "pdtr",
        "pepstr",
        "pcltr",
        "pcmef",
        "phtr",
        "pevtr",
        "plvtr",
        "pfvtr",
        "score_psvr",
        "score_pctr",
        "score_pltr",
        "score_pwtr",
        "score_pftr",
        "score_pcmtr",
        "score_pptr",
        "score_pcmef",
        "score_pdtr",
        "score_pcltr",
        "score_phtr",
        "fr_mc_embedding_score",
        "ann_hetu_lvtr_score",
        "diversity_fr",
        "xlife_pantheon_model_score",
        "fr_score1",
        "fr_score2",
        "awesome_wtd_score",
        "score_consume_time_ltr",
        "consume_time_pf2r_score",
        "watch_time_fusion_score",
        "ctr_multy_wtd_sharpe_ratio_score",
        "cpr",
        "fetr",
        "fountain_eff",
        "corr_cpr",
        "corr_fetr",
        "corr_fountain_eff",
        "gen_l2r_fusion_score",
        "svr_act_score",
        "watchtime_interact_score",
        "min_act_rank_score",
        "life_truth_pctr",
        "global_emphtr_score",
        "longterm_cluster_score",
      ],
      export_formula_value = [
        "explore_fr_ensemble_score"
      ],
      abtest_biz_name = "KUAISHOU_APPS"
    ) \
    .end_() \
    .log_debug_info(
      item_attrs = [
        "explore_fr_ensemble_score",
        "original_explore_fr_ensemble_score",
        "explore_fr_pxtr_absolute_score",
      ],
      common_attrs = [
        # "xlife_filter_audit_b_second_tag_str",
      ],
      respect_sample_logging = True,
      for_debug_request_only = True,
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
        "fr_score2",
        "fr_score1",
        "score_psvr",
        "corr_fountain_eff",
        "fountain_eff",
        "cpr",
        "fetr",
        "debias_pctr",
        "debias_fetr",
        "debias_fountain_eff",
        "multitask_ltr_pfstr",
        "multitask_ltr_pwtd",
        "multitask_ltr_pcvtr",
        "multitask_ltr_pctr",
        "explore_fr_ensemble_score",
        "save_es_pctr_score_to_kafka",
        "save_es_awesome_wtd_score_to_kafka",
        "save_es_cpr_score_to_kafka",
        "save_es_fountain_eff_score_to_kafka",
        "pic_ltr_fvtr",
        "pic_ltr_weighted_ctr"
      ]
    ) \
    .if_("ensemble_score_change_by_svr == 1") \
      .enrich_attr_by_light_function(
        import_item_attr = [
          "explore_fr_ensemble_score",
          "psvr"
        ],
        import_common_attr = [
          "ensemble_score_change_alpha",
          "ensemble_score_change_gamma"
        ],
        export_item_attr = [
          "explore_fr_ensemble_score"
        ],
        function_name = "EnsembleScoreChange",
        class_name = "ExploreLightFunctionSetV2",
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
      lua_script_file = "life/ranking/lua/module/ranking_ensemble_sort__print.lua"
    ) \
    .if_("skip_hot_fr_ensemble_sort_boost == 0") \
      .if_("life_enable_fr_uninterest_deboost == 1 and page > 2") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "life_fr_uninterest_deboost_coeff", "as": "boost_discount_coeff"},
          ],
          import_item_attr = [
            {"name": "is_uninterest_depress", "as": "need_item_attr"},
            {"name": "explore_fr_ensemble_score", "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": "explore_fr_ensemble_score"},
          ],
          function_name = "BoostOrDiscount",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_() \
      .if_("enable_fr_search_score_boost == 1") \
        .fr_search_score_boost() \
      .end_() \
      .if_("enable_ranking_personified_author_boost == 1") \
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
      .if_("enable_ranking_hetu_v3_level2_discount == 1") \
        .discount_life_photo_hetu() \
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
      .if_("life_enable_fr_follow_author_photo_boost == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "life_fr_follow_author_photo_boost_coeff", "as": "boost_discount_coeff"},
          ],
          import_item_attr = [
            {"name": "is_follow_author", "as": "need_item_attr"},
            {"name": "explore_fr_ensemble_score", "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": "explore_fr_ensemble_score"},
          ],
          function_name = "BoostOrDiscount",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_() \
      .if_("fr_enable_user_intrest_adjust == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "output_intrest_key_list", "as": "intrest_key_list"},
            {"name": "output_intrest_value_list", "as": "intrest_value_list"}, 
          ],
          import_item_attr = [
            {"name": "explore_fr_ensemble_score", "as": "input_score"},
            "hetu_tag_level_info__hetu_level_two"
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
    .end_if_() \
    .if_("enable_life_fr_user_pos_hetu_boost == 1 and page == 1 and (life_user_pos_hetu_boost_limit_low_active ~= 1 or uIsLifeHighActive ~= 1)") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "life_fr_user_pos_hetu_boost_coeff", "as": "boost_coeff"},
          {"name": "user_positive_hetu2_list", "as": "pos_hetu_list"}
        ],
        import_item_attr = [
          {"name": "hetu_tag_level_info__hetu_level_two", "as": "hetu_list"},
          {"name": "explore_fr_ensemble_score", "as": "ensemble_score"}
        ],
        export_item_attr = [
          {"name": "ensemble_score", "as": "explore_fr_ensemble_score"}
        ],
        function_name = "UserPositiveHetuEsBoost",
        class_name = "ExploreLifeLightFunctionSet",
      ) \
    .end_() \
    .if_("enable_life_fr_hotfire_yellow_boost == 1 and (life_hotfire_yellow_boost_limit_low_active ~= 1 or uIsLifeHighActive ~= 1)") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "life_fr_hotfire_yellow_boost_coef", "as": "boost_discount_coeff"},
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
          "is_hotfire_yellow" : 1
        }
      ) \
    .end_() \
    .if_("life_enable_fr_marketing_compensation_discount == 1") \
      .fr_marketing_compensation_discount() \
    .end_() \
    .if_("life_enable_fr_search_topk_boost == 1") \
      .fr_search_topk_boost() \
    .end_() \
    .if_("enable_life_direct_tab_boost == 1") \
      .sort(
        score_from_attr = "explore_fr_ensemble_score",
        target_item = {
           "reason": 2416
        }
      ) \
      .limit(
         size = 1,
         target_item = {
            "reason": 2416
         }
      ) \
      .set_attr_value(
        item_attrs=[
          {
            "name": "explore_fr_ensemble_score",
            "type": "double",
            "value": 1000000000.0
          }
        ],
        target_item = {
          "reason": 2416
        }
      ) \
    .end_() \
    .sort(
      skip = "{{skip_fullrank_sort_by_ensemble_score}}",
      score_from_attr = "explore_fr_ensemble_score",
    ) \
    .if_("is_fresh_request == 1 and enable_life_active_interest_boost == 1 and (life_active_interest_boost_limit_low_active ~= 1 or uIsLifeHighActive ~= 1)") \
      .if_("enable_life_active_interest_insert_judge == 1") \
        .enrich_attr_by_light_function(
          import_item_attr = [
            "score_pctr"
          ],
          import_common_attr = [
            "life_author_insert_xtr_thres"
          ],
          export_common_attr = [
            "follow_author_insert_num"
          ],
          function_name = "CalFollowAuthorInsertNumV2",
          class_name = "ExploreLifeLightFunctionSet"
        ) \
      .else_() \
        .set_attr_value(
            common_attrs = [
              {
                "name": "follow_author_insert_num",
                "type": "int",
                "value": 0
              }
            ]
        ) \
        .enrich_attr_by_light_function(
          import_item_attr = [
            "score_pctr"
          ],
          import_common_attr = [
            "life_author_insert_xtr_thres"
          ],
          export_common_attr = [
            "follow_author_insert_num"
          ],
          function_name = "CalFollowAuthorInsertNum",
          class_name = "ExploreLifeLightFunctionSet"
        ) \
        .limit(
          size = "{{follow_author_insert_num}}",
          target_item = {
              "reason": [2422]
          }
        ) \
        .set_attr_value(
          item_attrs=[
            {
              "name": "explore_fr_ensemble_score",
              "type": "double",
              "value": 1000000000.0
            }
          ],
          target_item = {
            "reason": [2422]
          }
        ) \
        .sort(
          score_from_attr = "explore_fr_ensemble_score"
        ) \
      .end_() \
    .end_() \
    .copy_item_meta_info(
      save_item_seq_to_attr = "rank_final_index",
    ) \
    .if_("explore_enable_rank_write_rank_neg_result_to_redis == 1") \
      .write_rank_neg_result_to_redis() \
    .end_() \
    .if_("explore_enable_rank_write_rank_pos_result_to_redis == 1") \
      .write_rank_pos_result_to_redis() \
    .end_() \
    .log_debug_info(
       item_attrs = [
          "young_age_boost_rate",
          "hetu_tag_level_info__hetu_level_one",
          "active_hetu_pctr_pcoc_score",
          "corr_pctr",
          "pltr",
          "plvtr",
          "psvr",
          "pevtr",
          "report_discount",
          "hate_discount",
       ],
       common_attrs = [
          'boost_htr_weight_by_full_degree', 'boost_htr_weight_by_high_degree', 'boost_htr_weight_by_mid_degree', 'boost_htr_weight_by_p_idx', 'boost_htr_weight_p_idx_down', 'boost_htr_weight_p_idx_up',
          'enable_high_hot_audit_adjust', 'enable_impression_audit_adjust', 'enable_topk_audit_adjust', 'high_hot_audit_adjust_coeff_map_str', 'impression_audit_adjust_coeff_map_str', 'impression_audit_emp_ctr_avg', 
          'impression_audit_emp_watchtime_avg', 'impression_audit_emp_xtr_adjust_flag', 'impression_audit_emp_xtr_coeff_a', 'impression_audit_emp_xtr_coeff_b', 'topk_audit_adjust_coeff_map_str',
          'absolute_fusion_score_max_wtd', 'absolute_interact_score_cltr_cliff', 'absolute_interact_score_cltr_weight', 'absolute_interact_score_cmtr_cliff', 'absolute_interact_score_cmtr_weight', 
          'absolute_interact_score_ctr_cliff', 'absolute_interact_score_ctr_weight', 'absolute_interact_score_fscore1_cliff', 'absolute_interact_score_fscore1_weight', 'absolute_interact_score_ftr_cliff', 
          'absolute_interact_score_ftr_weight', 'absolute_interact_score_ltr_cliff', 'absolute_interact_score_ltr_weight', 'absolute_interact_score_wtd_cliff', 'absolute_interact_score_wtd_weight', 
          'absolute_interact_score_wtr_cliff', 'absolute_interact_score_wtr_weight', 'boost_htr_weight_by_full_degree', 'boost_htr_weight_by_high_degree', 'boost_htr_weight_by_mid_degree', 'boost_htr_weight_by_p_idx', 
          'boost_htr_weight_p_idx_down', 'boost_htr_weight_p_idx_up', 'boost_top_es_index', 'enable_absolute_interact_score_v2', 'enable_es_rank_mix_ltr_score', 'enable_explore_duration_debias_score', 
          'enable_explore_fr_click_boost_v2', 'enable_explore_fr_xhs_install_click_boost', 'enable_fr_boost_hetu_es_xhs', 'enable_fr_boost_top_es', 'enable_fr_top_interaction_boost', 'enable_fullrank_htr_weight_adjust', 
          'enable_fullrank_target_hetu_pic_boost', 'enable_high_hot_audit_adjust', 'enable_impression_audit_adjust', 'enable_life_young_age_boost_fr', 'enable_multiply_absolute_ctr_score', 
          'enable_rank_refinement_boost_personified_author', 'enable_ranking_heat_boost', 'enable_ranking_top_personified_author_boost', 'enable_specified_group_fr_boost_interactive', 'enable_topk_audit_adjust', 
          'enable_use_new_fr_score2_formula', 'enable_use_new_pfvtr_formula', 'enable_young_photo_fr_boost', 'ensemble_score_rank_smooth', 'es_rank_score_smooth', 'explore_dis_fr_score1_queue_pow_t', 
          'explore_dis_fr_score2_queue_pow_t', 'explore_ensemble_power_weight_fullrank_dis_fr_score1_score', 'explore_ensemble_power_weight_fullrank_dis_fr_score2_score', 'explore_ensemble_use_pure_value_hier_es', 
          'explore_fr_consumetime_alpha_weight', 'explore_fr_consumetime_beta_weight', 'explore_fr_enable_bot_content_retr_boost', 'explore_fr_enable_merchant_live_boost', 'explore_fr_enable_merchant_photo_boost', 
          'explore_fr_enable_smooth_fr_score2_formula', 'explore_fr_enable_smooth_pfvtr_formula', 'explore_fr_ensemble_zero_play_boost_type', 'explore_fr_ensemble_zero_play_ctr_boost_weight', 
          'explore_fr_ensemble_zero_play_days_threshold', 'explore_fr_ensemble_zero_play_ratio_threshold', 'explore_fr_ensemble_zero_play_xtr_boost_weight', 'explore_fr_hot_content_retr_boost_coef', 'explore_fr_lte_ctr_alpha_weight', 
          'explore_fr_lte_ctr_beta_weight', 'explore_fr_lte_ltr_alpha_weight', 'explore_fr_lte_ltr_beta_weight', 'explore_fr_merchant_live_boost_coef', 'explore_fr_merchant_photo_boost_coef', 'explore_fr_outflow_boost_click_count_alpha', 
          'explore_fr_outflow_boost_click_count_beta', 'explore_fr_outflow_boost_click_count_omega', 'explore_fr_pfvtr_alpha_weight', 'explore_fr_pfvtr_beta_weight', 'explore_fr_skip_infer_uv_ctr_boost_handle', 
          'explore_fr_smooth_fr_score2_formula_beta', 'explore_fr_smooth_pfvtr_formula_beta', 'explore_fr_whole_boost_click_count_alpha', 'explore_fr_whole_boost_click_count_beta', 'explore_fr_whole_boost_click_count_omega', 
          'explore_fr_wtd_alpha_weight', 'explore_fr_wtd_beta_weight', 'explore_fr_xhs_install_outflow_click_weight', 'explore_fr_xhs_install_whole_click_weight', 'explore_fullrank_emp_debias_fr_score1_alpha', 
          'explore_fullrank_emp_debias_fr_score1_beta', 'explore_fullrank_emp_debias_fr_score2_alpha', 'explore_fullrank_emp_debias_fr_score2_beta', 'explore_fullrank_emp_debias_pcltr_alpha', 'explore_fullrank_emp_debias_pcltr_beta', 
          'explore_fullrank_emp_debias_pcmtr_alpha', 'explore_fullrank_emp_debias_pcmtr_beta', 'explore_fullrank_emp_debias_pctr_alpha', 'explore_fullrank_emp_debias_pctr_beta', 'explore_fullrank_emp_debias_pftr_alpha', 
          'explore_fullrank_emp_debias_pftr_beta', 'explore_fullrank_emp_debias_phtr_alpha', 'explore_fullrank_emp_debias_phtr_beta', 'explore_fullrank_emp_debias_pltr_alpha', 'explore_fullrank_emp_debias_pltr_beta', 
          'explore_fullrank_emp_debias_psvr_alpha', 'explore_fullrank_emp_debias_psvr_beta', 'explore_fullrank_emp_debias_pwtr_alpha', 'explore_fullrank_emp_debias_pwtr_beta', 'explore_fullrank_emp_debias_wtd_alpha', 
          'explore_fullrank_emp_debias_wtd_beta', 'explore_fullrank_enable_picture_xtr_debias', 'explore_fullrank_fr_score1_debias_type', 'explore_fullrank_fr_score2_debias_type', 'explore_fullrank_la_ensemble_sort_pctr_weight_base', 
          'explore_fullrank_la_ensemble_sort_pctr_weight_max', 'explore_fullrank_pcltr_debias_type', 'explore_fullrank_pcmtr_debias_type', 'explore_fullrank_pctr_debias_type', 'explore_fullrank_pftr_debias_type', 
          'explore_fullrank_phtr_debias_type', 'explore_fullrank_pltr_debias_type', 'explore_fullrank_psvr_debias_type', 'explore_fullrank_pwtr_debias_type', 'explore_fullrank_skip_zero_play_user_xtr_boost_handle', 
          'explore_fullrank_wtd_debias_type', 'explore_fullrank_xtr_debias_dura_bucket_width', 'explore_hier_es_ensemble_power_ensemble_score_weight', 'explore_hier_es_ensemble_power_fullrank_pure_value_score_weight', 
          'explore_hier_es_ensemble_power_gen_l2r_fusion_score_weight', 'explore_hier_es_ensemble_score_skip_diff_judge', 'explore_la_fr_comment_boost_coeff', 'explore_la_fr_like_boost_coeff', 
          'explore_rank_enable_high_photo_count_author_adjust', 'explore_rank_high_photo_count_author_photo_coeff', 'explore_rank_high_photo_count_author_post_num_base', 'explore_rank_sort_weight_adjust_fetr', 
          'explore_rank_sort_weight_adjust_fountain_eff', 'explore_rank_sort_weight_low_follow', 'explore_rank_sort_weight_uplift', 'explore_smooth_ensemble_power_ensemble_score_weight', 'explore_smooth_ensemble_score_skip_diff_judge', 
          'explore_weight_adjust_avg_emp_fountain_time_ratio_fetr', 'explore_weight_adjust_avg_emp_fountain_time_ratio_fountain_eff', 'explore_weight_adjust_coeff_max_fetr', 'explore_weight_adjust_coeff_max_fountain_eff', 
          'explore_weight_adjust_coeff_min_fetr', 'explore_weight_adjust_coeff_min_fountain_eff', 'explore_weight_request_adjust_avg_coeff_max', 'explore_weight_request_adjust_avg_coeff_min', 'explore_weight_request_adjust_avg_emp_cmtr', 
          'explore_weight_request_adjust_avg_emp_ctr', 'explore_weight_request_adjust_avg_emp_eptr', 'explore_weight_request_adjust_avg_emp_fr_score1', 'explore_weight_request_adjust_avg_emp_fr_score2', 
          'explore_weight_request_adjust_avg_emp_ftr', 'explore_weight_request_adjust_avg_emp_ltr', 'explore_weight_request_adjust_avg_emp_wtd', 'explore_weight_request_adjust_avg_emp_wtr', 'explore_weight_request_adjust_avg_request_ratio', 
          'explore_xhs_hetu_boost_value', 'explore_xhs_whitelist_hetu_level_one_str', 'explore_xhs_whitelist_hetu_level_two_str', 'fr_absolute_score_weight', 'fr_alpha_for_top', 'fr_boost_follow_author_weight', 
          'fr_boost_top_es_weight', 'fr_cltr_boost_top_num', 'fr_cmtr_boost_top_num', 'fr_enable_la_follow_boost_fresh_thr', 'fr_enable_low_follow_boost', 'fr_enable_no_follow_boost', 'fr_hetu_distribution_colossus_total_count_threshold', 
          'fr_hetu_distribution_global_fuse_corr', 'fr_hetu_distribution_hetu_coef_alpha', 'fr_hetu_distribution_hetu_coef_beta', 'fr_hetu_distribution_hetu_discount_threshold', 'fr_hetu_distribution_hetu_encourage_threshold', 
          'fr_hetu_distribution_max_count', 'fr_low_follow_boost_threshold', 'fr_relative_score_weight', 'fr_weaken_follow_author_weight', 'fr_wtr_boost_top_num', 'fullrank_enable_comment_boost', 
          'fullrank_enable_comment_boost__god__coeff_max_w', 'fullrank_enable_comment_boost__god__coeff_min_w', 'fullrank_enable_comment_boost__god__coeff_p', 'fullrank_enable_comment_boost__god__coeff_w', 
          'fullrank_enable_comment_boost__hot__coeff_max_w', 'fullrank_enable_comment_boost__hot__coeff_min_w', 'fullrank_enable_comment_boost__hot__coeff_p', 'fullrank_enable_comment_boost__hot__coeff_w', 
          'fullrank_enable_follow_author_pic_boost', 'fullrank_enable_la_zero_click_optimized', 'fullrank_follow_author_pic_boost_coef', 'fullrank_target_hetu_pic_boost_coeff', 'high_hot_audit_adjust_coeff_map_str', 
          'impression_audit_adjust_coeff_map_str', 'impression_audit_emp_ctr_avg', 'impression_audit_emp_watchtime_avg', 'impression_audit_emp_xtr_adjust_flag', 'impression_audit_emp_xtr_coeff_a', 'impression_audit_emp_xtr_coeff_b', 
          'pic_xtr_quantile_rank__ranking__base_coef', 'pic_xtr_quantile_rank__ranking__enable', 'pic_xtr_quantile_rank__ranking__weights', 'rank_final_revisited_item_boost_coef', 'rank_low_follow_pwtr_weight', 'rank_low_follow_thres_s', 
          'rank_no_follow_pwtr_weight', 'rank_refinement_boost_personified_author_power_weight', 'rank_uplift_pcmtr_weight', 'rank_uplift_pftr_weight', 'rank_uplift_pwtr_weight', 'rank_uplift_wtd_weight', 'rank_valid_follow_pwtr_weight', 
          'rank_valid_high_follow_pwtr_weight', 'rank_valid_low_follow_pwtr_weight', 'rank_valid_media_follow_pwtr_weight', 'ranking_heat_boost_decay_coeff', 'ranking_heat_boost_init_heat', 'ranking_heat_boost_min_heat', 
          'ranking_top_personified_author_boost_cnt', 'ranking_top_personified_author_boost_coef', 'topk_audit_adjust_coeff_map_str', 'xtr_absolute_score_offset', 'xtr_absolute_score_pow_weight', 'young_photo_boost_fr_coeff',
          'fullrank_fr_score1_freq_idx_factor_list', 'fullrank_fr_score2_freq_idx_factor_list', 'fullrank_pcltr_freq_idx_factor_list', 'fullrank_pcmtr_freq_idx_factor_list', 'fullrank_pctr_freq_idx_factor_list', 'fullrank_pftr_freq_idx_factor_list', 
          'fullrank_phtr_freq_idx_factor_list', 'fullrank_pltr_freq_idx_factor_list', 'fullrank_psvr_freq_idx_factor_list', 'fullrank_pwtr_freq_idx_factor_list', 'fullrank_wtd_freq_idx_factor_list',
          'fullrank_cltr_adjust_ratio_attr', 'fullrank_cmef_adjust_ratio_attr', 'fullrank_cmtr_adjust_ratio_attr', 'fullrank_ctr_adjust_ratio_attr', 'fullrank_duration_adjust_ratio_attr', 'fullrank_epstr_adjust_ratio_attr', 
          'fullrank_expected_score_adjust_ratio_attr', 'fullrank_fetr_adjust_ratio_attr', 'fullrank_fountain_eff_adjust_ratio_attr', 'fullrank_fr_score1_adjust_ratio_attr', 'fullrank_fr_score2_adjust_ratio_attr', 
          'fullrank_ftr_adjust_ratio_attr', 'fullrank_l2r_score_adjust_ratio_attr', 'fullrank_ltr_adjust_ratio_attr', 'fullrank_ptr_adjust_ratio_attr', 'fullrank_wtr_adjust_ratio_attr',
          "uNebulaXlifeVisitDays30dKV","uNebulaDoubleFindVisitDays30dKV","shallow_consumer_user_pcoc","secondary_shallow_consumer_user_pcoc","moderate_user_pcoc","sub_deep_user_pcoc","deep_user_pcoc"
       ],
    )

  def calc_result_count_to_ab_metric(self):
      return self.flow \
        .count_reco_result(
          save_count_to = "ranking_ensemble_sort_top200_result_count",
          range_end = 200
        ) \
        .count_reco_result(
          save_count_to = "ranking_ensemble_sort_top200_pic_result_count",
          target_item = {"is_picture": 1},
          range_end = 200
        ) \
        .send_abtest_metrics(
          metrics = [
            "ranking_ensemble_sort_top200_result_count",
            "ranking_ensemble_sort_top200_pic_result_count"
          ],
          metric_name_prefix = "explore_reco_leaf_",
        )

  def post_process(self) -> None:
      self.calc_result_count_to_ab_metric()
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
        .if_("fullrank_enable_support_author_fans_profile_boost == 1") \
          .perflog_attr_value(
            check_point = "fullrank_support_author_boost",
            common_attrs = [
              "fullrank_support_author_boost_age_count",
              "fullrank_support_author_boost_gender_count",
              "fullrank_boost_hetu_lv1_inconsistent",
            ],
          ) \
        .end_() \
        .if_("explore_rank_sort_weight_adjust_request == 1") \
          .perflog_attr_value(
            check_point = "ranking_request_adap_weight",
            common_attrs = [
              "user_emp_ctr",
              "user_emp_lvtr",
              "user_emp_watchtime",
              "pctr_avg",
              "fr_score1_avg",
              "fr_score2_avg",
              "awesome_wtd_avg",
              "ctr_factor_for_request_weight_adjust",
              "fr_score1_factor_for_request_weight_adjust",
              "fr_score2_factor_for_request_weight_adjust",
              "wtd_factor_for_request_weight_adjust",
            ],
          ) \
        .end_() \
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
