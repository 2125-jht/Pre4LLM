from cascading import CommonModule
from cascading.module.queue.cascade_queues import hot_mc_pxtr_fractile_score_queues

class CascadingScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
    .explore_memory_data_enrich(
      data_key = "{{pptime_memory_data_key}}",
      data_type = "string_int32_map",
      save_data_ptr_to_attr = "pptime_memory_data"
    ) \
    .explore_custom_score_enricher(
      trans_type = "pptime",
      memory_data_name = "pptime_memory_data",
      export_item_attr = "pptime"
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        "cascade_fc_pctr",
        "cascade_fc_plvtr",
        "cascade_fc_psvtr",
        "cascade_fc_pvtr",
        "cascade_fc_pwtd",
      ],
      export_item_attr = [
        "cascade_pctr",
        "cascade_plvtr",
        "cascade_psvtr",
        "cascade_pwatch_time", # vtr可能不准
        "cascade_pcptr",
      ],
      function_name = "ReplaceMcPxtr",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .if_("enable_hot_fc_replace_pvtr2 == 1") \
      .copy_attr(
        attrs=[
          {
            "from_item": "cascade_fc_pvtr2",
            "to_item": "cascade_pwatch_time"
          }
        ]
      ) \
    .end_() \
    .if_("enable_hot_fc_replace_interface_interact == 1") \
      .copy_attr(
        attrs=[
          {
            "from_item": "cascade_fc_pltr",
            "to_item": "cascade_pltr"
          },
          {
            "from_item": "cascade_fc_pwtr",
            "to_item": "cascade_pwtr"
          },
          {
            "from_item": "cascade_fc_pftr",
            "to_item": "cascade_pftr"
          },
        ]
      ) \
    .end_() \
    .if_("enable_calc_cascade_wtd_inverse == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "cascade_wtd_table_0",
          "cascade_wtd_table_9",
          "cascade_wtd_table_13",
          "cascade_wtd_table_20",
          "cascade_wtd_table_38",
          "cascade_wtd_table_71",
          "cascade_wtd_table_118",
          "cascade_wtd_table_195",
          "cascade_wtd_table_inf",
        ],
        import_item_attr = [
          "cascade_pcptr",
          "duration_ms",
        ],
        export_item_attr = [
          "cascade_pwtd_inverse",
        ],
        function_name = "CalcCascadeWtdInverse",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .if_("enable_explore_cascade_pxtr_calibration == 1") \
      .explore_cascade_pxtr_calibration() \
    .else_() \
      .copy_attr(
        attrs = [{
          "from_item": "cascade_pctr",
          "to_item": "cascade_corr_pctr",
        }]
      ) \
    .end_() \
    .if_("enable_explore_cascade_cal_debias_xtr_by_pcoc_score == 1 and basic_info_age_segment_v2 > 0 and basic_info_age_segment_v2 <= explore_cascade_cal_debias_xtr_by_pcoc_score_age_threshold") \
      .explore_cascade_cal_debias_xtr_by_pcoc_score("cascade_pltr", "cascade_pltr", "cascade_debias_by_pcoc_pltr") \
      .explore_cascade_cal_debias_xtr_by_pcoc_score("cascade_pcltr", "cascade_pcltr", "cascade_debias_by_pcoc_pcltr") \
    .end_() \
    .if_("enable_gen_short_window_ctr_cali_coeff == 1")\
      .gen_short_window_ctr_coeff() \
    .end_() \
    .if_("enable_photo_history_interest_score_with_mc_ctr == 1 and interest_score_based_valid_user == 1") \
      .calc_by_simple_formula(
        formulas = [
          dict(
            expr = (
              "pow({{explore_history_interest_score_attr_x_bias}} + {{explore_history_interest_score_attr_x_alpha}} * [[photo_history_interest_score]], {{explore_history_interest_score_attr_x_pow}}) * "
              "pow({{explore_history_interest_score_pctr_bias}} + {{explore_history_interest_score_pctr_alpha}} * [[cascade_corr_pctr]], {{explore_history_interest_score_pctr_pow}}) * "
              "pow({{explore_history_interest_score_empirical_ctr_bias}} + {{explore_history_interest_score_empirical_ctr_alpha}} * [[empirical_ctr]], {{explore_history_interest_score_empirical_ctr_pow}}) * "
              "pow({{explore_history_interest_score_cascade_pwtd_inverse_bias}} + {{explore_history_interest_score_cascade_pwtd_inverse_alpha}} * [[cascade_pwtd_inverse]], {{explore_history_interest_score_cascade_pwtd_inverse_pow}})"
            ),
            output_attr = "photo_history_interest_score_with_mc_ctr"
          )
        ],
      ) \
    .end_() \
    .if_("enable_user_age_interest_tagnex_tgi_product_pxtr_score == 1") \
      .calc_by_simple_formula(
        formulas = [
          dict(
            expr = (
              "{{explore_mc_age_tgi_product_global_coeff}} * "
              "pow(1 + {{explore_mc_age_tgi_product_tgi_alpha}} * [[user_age_interest_tagnex_tgi_score]], {{explore_mc_age_tgi_product_tgi_beta}}) * "
              "pow(1 + {{explore_mc_age_tgi_product_pctr_alpha}} * [[cascade_corr_pctr]], {{explore_mc_age_tgi_product_pctr_beta}}) * "
              "pow(1 + {{explore_mc_age_tgi_product_pwtr_alpha}} * [[cascade_pwtr]], {{explore_mc_age_tgi_product_pwtr_beta}}) * "
              "pow(1 + {{explore_mc_age_tgi_product_pwatch_time_alpha}} * [[cascade_pwatch_time]], {{explore_mc_age_tgi_product_pwatch_time_beta}})"
            ),
            output_attr = "user_age_interest_tagnex_tgi_product_pxtr_score"
          )
        ]
      ) \
    .end_() \
    .if_("explore_enable_cascade_short_window_ctr_cali == 1") \
      .cascade_short_window_ctr_cali() \
    .end_() \
    .if_("enable_explore_cascade_eff_ctr_corr == 1") \
      .explore_cascade_eff_ctr_corr() \
    .end_() \
    .enrich_attr_by_light_function(
      import_common_attr = [
        "mille_avg_watch_time_upper_bound"
      ],
      import_item_attr = [
        "duration_ms",
        "explore_stat__click_count",
        "explore_stat__view_length_sum",
        "is_picture",
      ],
      export_item_attr = [
        "avg_watch_time"
      ],
      function_name = "McCalAvgWatchTime",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        "mille_l0_ctr_param",
        "mille_l0_ltr_param",
        "mille_l0_wtr_param",
        "mille_l0_ftr_param",
        "mille_l0_lvtr_param",
        "mille_l0_lvtr2_param",
        "mille_l0_svtr_param",
        "mille_l0_ptr_param",
        "mille_l0_watchtime_param",
        "mille_l0_epstr_param",
        "mille_l0_cestr_param",
        "mille_l0_cmtr_param",
        "mille_l0_livingtr_param",
        "mille_l0_svtr_power_param",
        "mille_l0_cas_xtr_param",
        "mille_l0_cas_final_param",
        "mc_picture_discount_param",
        "mille_l0_wtd_param",
        "hot_mc_cp_ctr_weight",
        "mc_mid_photo_boost_param",
        "mille_l0_phtr_param",
        "mille_l0_phtr_power_param",
        "mille_l0_cltr_param",
        "mille_l0_pfptr_param",
        "mille_l0_prerank_ctr_param",
        "mille_l0_prerank_ltr_param",
        "mille_l0_prerank_wtd_param",
        "mille_l0_cascase_life_ctr_param",
        "mille_l0_pwtd_inverse_param",
        "mille_l0_pcptr_param",
      ],
      import_item_attr = [
        {"name": "cascade_corr_pctr", "as": "cascade_pctr"},        #1
        "cascade_pltr",         #2
        "cascade_pwtr",         #3
        "cascade_pftr",         #4
        "cascade_plvtr",        #5
        "cascade_plvtr2",       #6
        "cascade_psvtr",        #7
        "cascade_ptr",          #8
        "cascade_pwatch_time",  #9
        "cascade_pepstr",       #10
        "cascade_pcestr",       #11
        "cascade_pcmtr",        #12
        "cascade_plivingtr",    #13
        "cascade_prerank_pctr",
        "cascade_prerank_pltr",
        "is_picture",
        "cascade_pwtd",
        "duration_ms",
        "cascade_phtr",
        "cascade_pcltr",
        "prerank_ltr",
        "prerank_ctr",
        "prerank_wtd",
        "cascade_pfptr",
        "cascase_life_ctr",
        "cascade_pwtd_inverse",
        "cascade_pcptr",
      ],
      export_item_attr = [
        "cascade_score",
      ],
      function_name = "CalMcMergedScore",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .if_("enable_calc_mc_pctr_adjust == 1", to_be_delete = "date=2024-05-29;committer=guohao") \
      .enrich_attr_by_light_function(
        import_item_attr = [
          {"name": "prerank_ctr_in_s1", "as": "xtr_boost_pred"},
          {"name": "cascade_pctr", "as": "origin_xtr"},
        ],
        export_item_attr = [
          {"name": "corr_xtr", "as": "cascade_pctr_corr"},
        ],
        function_name = "CalcPxtrAdjust",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .if_("enable_explore_mc_request_pctr_power_weight_adjust == 1") \
      .mc_user_vv_ensemble_power_weight_adjust("cascading_score_factor_power", "explore_mc_v3_request_pctr_weight_adjust_exp_upper", "explore_mc_v3_request_pctr_weight_adjust_alpha", "explore_mc_v3_request_pctr_weight_adjust_beta", "explore_mc_v3_request_pctr_weight_adjust_omega", "explore_mc_v3_request_pctr_weight_adjust_max", "explore_mc_v3_request_pctr_weight_adjust_min") \
    .end_() \
    .enrich_attr_by_light_function(
      import_common_attr = [
        "hot_fountain_eftr_pctr_weight",
        "hot_fountain_eftr_weight",
        "hot_fountain_efctr_pctr_weight",
        "hot_fountain_efctr_weight",
        "hot_mc_cp_ctr_weight",
        "hot_mc_pfptr_pctr_weight",
        "hot_mc_pic_wtd_pctr_weight",
        "hot_mc_pic_lvtr_pctr_weight",
        "hot_mc_pic_cpr_pctr_weight",
        "hot_mc_pic_cpr_max_pic_cnt",
        "hot_mc_wtd_ctr_weight", #wtd_pctr param
        "mc_ensemble_score_smooth",
        "hot_mc_ordinal_wtd_pctr_weight",
        "hot_mc_ordinal_prob_pctr_weight",
        "mc_enable_multiply_pctr",
        "wtd_factor_type",
        "prerank_wtd_power_weight",
        "cascading_score_factor_power",
        "enable_mc_pctr_corr_replace_pctr",
        "mc_pctr_corr_replace_pctr_weight",
        "enable_mc_pctr_corr_replace_pctr_in_factor",
        "mc_pctr_corr_replace_pctr_weight_in_factor",
        "enable_pic_mc_trans",
        "pic_cascading_score_factor_power",
        "pic_interact_cascading_score_factor_power",
        {"name": "explore_mc_ensemble_smooth_age_score_type", "as": "smooth_age_score_type"}
      ],
      import_item_attr = [
        {"name": "cascade_corr_pctr", "as": "cascade_pctr"},
        "cascade_pltr",
        "cascade_pwtr",
        "cascade_pftr",
        "cascade_plvtr",
        "cascade_plvtr2",
        "cascade_psvtr",
        "cascade_ptr",
        "cascade_pwatch_time",
        "cascade_pepstr",
        "cascade_pcestr",
        "cascade_pcmtr",
        "cascade_pcltr",
        "cascade_peftr",
        "cascade_pefctr",
        "cascade_fc_pevr",
        "cascade_pic_wtd",
        "cascade_pic_lvtr",
        "cascade_pic_cpr",
        "pptime",
        "avg_watch_time",
        "upload_time",
        "explore_stat__real_show_count",
        "cascade_pwtd",
        "duration_ms",
        "cascade_pfptr",
        "photo_picture_count",
        "prerank_ltr",
        "prerank_ctr",
        "cascade_pcptr",
        "cascade_pwtd_inverse",
        "empirical_watch_time",
        "empirical_ctr",
        "cascade_ordinal_wtd",
        "cascade_ordinal_prob",
        "prerank_wtd",
        "prerank_ctr_in_s1",
        "cascade_pctr_corr",
        "is_picture",
      ],
      export_item_attr = [
        "mc_ensemble_pctr",
        "mc_ensemble_pltr",
        "mc_ensemble_pwtr",
        "mc_ensemble_pftr",
        "mc_ensemble_pptime",
        "mc_ensemble_plvtr",
        "mc_ensemble_plvtr2",
        "mc_ensemble_psvtr",
        "mc_ensemble_ptr",
        "mc_ensemble_pwatch_time",
        "mc_ensemble_pepstr",
        "mc_ensemble_pcestr",
        "mc_ensemble_pcmtr",
        "mc_ensemble_pcltr",
        "mc_ensemble_peftr",
        "mc_ensemble_pefctr",
        "mc_ensemble_pevr",
        "mc_ensemble_age_score",
        "mc_ensemble_pwtd",
        "mc_ensemble_pfptr",
        "mc_ensemble_ordinal_wtd",
        "mc_ensemble_ordinal_prob",
        "mc_ensemble_pic_wtd",
        "mc_ensemble_pic_lvtr",
        "mc_ensemble_pic_cpr",
        "mc_ensemble_prerank_er",
        "mc_ensemble_pcptr",
        "mc_ensemble_pwtd_inverse",
        "mc_ensemble_smooth_age_score",
        "mc_ensemble_emp_pop_score",
        "mc_ensemble_psvtr2",
        "mc_ensemble_prerank_wtd"
      ],
      function_name = "CalMcEnsembleQueueScore",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .if_("explore_enable_update_xtr_cal_mc_s1 == 1") \
      .explore_cal_update_xtr_score_mc_s1() \
    .end_() \
    .if_("explore_enable_hetu_one_debias_score_cal_mc_s1 == 1") \
      .explore_cal_hetu_one_debias_score_mc_s1() \
    .end_() \
    .switch_("explore_mc_enable_xtr_debias") \
      .case_(2, to_be_delete = "date=2023-11-16;committer=yuyi03") \
        .explore_xtr_debias_v3_enricher(
          queues = [
            {
              "input_attr" : "mc_ensemble_pcmtr",
              "output_attr" : "mc_ensemble_pcmtr",
              "debias_type" : "explore_mc_pcmtr_debias_type",
              "dynamic_dura_key": "explore_mc_bias_pcmtr",
              "dura_factors": "pcmtr_dura_debias_factor_list",
              "freq_factors": "pcmtr_dura_debias_factor_list",
              "alpha" : "explore_mc_emp_debias_pcmtr_alpha",
              "beta" : "explore_mc_emp_debias_pcmtr_beta",
            },
            {
              "input_attr" : "mc_ensemble_pcltr",
              "output_attr" : "mc_ensemble_pcltr",
              "debias_type" : "explore_mc_pcltr_debias_type",
              "dynamic_dura_key": "explore_mc_bias_pcltr",
              "dura_factors": "pcltr_dura_debias_factor_list",
              "freq_factors": "pcltr_dura_debias_factor_list",
              "alpha" : "explore_mc_emp_debias_pcltr_alpha",
              "beta" : "explore_mc_emp_debias_pcltr_beta",
            },
            {
              "input_attr" : "mc_ensemble_pctr",
              "output_attr" : "mc_ensemble_pctr",
              "debias_type" : "explore_mc_pctr_debias_type",
              "dynamic_dura_key": "explore_mc_bias_pctr",
              "dura_factors": "pctr_dura_debias_factor_list",
              "freq_factors": "pctr_dura_debias_factor_list",
              "alpha" : "explore_mc_emp_debias_pctr_alpha",
              "beta" : "explore_mc_emp_debias_pctr_beta",
            },
            {
              "input_attr" : "mc_ensemble_pltr",
              "output_attr" : "mc_ensemble_pltr",
              "debias_type" : "explore_mc_pltr_debias_type",
              "dynamic_dura_key": "explore_mc_bias_pltr",
              "dura_factors": "pltr_dura_debias_factor_list",
              "freq_factors": "pltr_dura_debias_factor_list",
              "alpha" : "explore_mc_emp_debias_pltr_alpha",
              "beta" : "explore_mc_emp_debias_pltr_beta",
            },
            {
              "input_attr" : "mc_ensemble_pwtr",
              "output_attr" : "mc_ensemble_pwtr",
              "debias_type" : "explore_mc_pwtr_debias_type",
              "dynamic_dura_key": "explore_mc_bias_pwtr",
              "dura_factors": "pwtr_dura_debias_factor_list",
              "freq_factors": "pwtr_dura_debias_factor_list",
              "alpha" : "explore_mc_emp_debias_pwtr_alpha",
              "beta" : "explore_mc_emp_debias_pwtr_beta",
            },
            {
              "input_attr" : "mc_ensemble_plvtr",
              "output_attr" : "mc_ensemble_plvtr",
              "debias_type" : "explore_mc_plvtr_debias_type",
              "dynamic_dura_key": "explore_mc_bias_plvtr",
              "dura_factors": "plvtr_dura_debias_factor_list",
              "freq_factors": "plvtr_dura_debias_factor_list",
              "alpha" : "explore_mc_emp_debias_plvtr_alpha",
              "beta" : "explore_mc_emp_debias_plvtr_beta",
            },
            {
              "input_attr" : "mc_ensemble_pwtd",
              "output_attr" : "mc_ensemble_pwtd",
              "debias_type" : "explore_mc_pwtd_debias_type",
              "dura_factors": "pwtd_dura_debias_factor_list",
              "freq_factors": "pwtd_dura_debias_factor_list",
              "dynamic_dura_key": "explore_mc_bias_pwtd",
              "alpha" : "explore_mc_emp_debias_pwtd_alpha",
              "beta" : "explore_mc_emp_debias_pwtd_beta",
            },
            {
              "input_attr" : "mc_ensemble_pwatch_time",
              "output_attr" : "mc_ensemble_pwatch_time",
              "debias_type" : "explore_mc_pwatch_time_debias_type",
              "dynamic_dura_key": "explore_mc_bias_pwatch_time",
              "dura_factors": "pwatch_time_dura_debias_factor_list",
              "freq_factors": "pwatch_time_dura_debias_factor_list",
              "alpha" : "explore_mc_emp_debias_pwatch_time_alpha",
              "beta" : "explore_mc_emp_debias_pwatch_time_beta",
            },
            {
              "input_attr" : "mc_ensemble_psvtr",
              "output_attr" : "mc_ensemble_psvtr",
              "debias_type" : "explore_mc_psvtr_debias_type",
              "dynamic_dura_key": "explore_mc_bias_psvtr",
              "dura_factors": "psvtr_dura_debias_factor_list",
              "freq_factors": "psvtr_dura_debias_factor_list",
              "alpha" : "explore_mc_emp_debias_psvtr_alpha",
              "beta" : "explore_mc_emp_debias_psvtr_beta",
            },
            {
              "input_attr" : "mc_ensemble_pftr",
              "output_attr" : "mc_ensemble_pftr",
              "debias_type" : "explore_mc_pftr_debias_type",
              "dynamic_dura_key": "explore_mc_bias_pftr",
              "dura_factors": "pftr_dura_debias_factor_list",
              "freq_factors": "pftr_dura_debias_factor_list",
              "alpha" : "explore_mc_emp_debias_pftr_alpha",
              "beta" : "explore_mc_emp_debias_pftr_beta",
            },
            ],
          duration_attr = "duration_ms",
          picture_attr = "is_picture",
          enable_picture_xtr_debias = "{{explore_mc_enable_picture_xtr_debias}}",
          dura_bucket_width = "{{explore_mc_xtr_debias_dura_bucket_width}}",
          dura_xtr_debias_map_attr = "explore_mc_hourly_xtr_debias_map_ptr",
        ) \
      .case_(1) \
        .explore_xtr_debias_v3_enricher(
          queues = [
            {
              "input_attr" : "mc_ensemble_pcmtr",
              "output_attr" : "mc_ensemble_pcmtr",
              "debias_type" : "explore_mc_pcmtr_debias_type",
              "dynamic_dura_key": "explore_mc_bias_pcmtr",
              "dura_factors": "pcmtr_dura_debias_factor_list",
              "alpha" : "explore_mc_emp_debias_pcmtr_alpha",
              "beta" : "explore_mc_emp_debias_pcmtr_beta",
            },
            {
              "input_attr" : "mc_ensemble_pcltr",
              "output_attr" : "mc_ensemble_pcltr",
              "debias_type" : "explore_mc_pcltr_debias_type",
              "dynamic_dura_key": "explore_mc_bias_pcltr",
              "dura_factors": "pcltr_dura_debias_factor_list",
              "alpha" : "explore_mc_emp_debias_pcltr_alpha",
              "beta" : "explore_mc_emp_debias_pcltr_beta",
            },
            {
              "input_attr" : "mc_ensemble_pctr",
              "output_attr" : "mc_ensemble_pctr",
              "debias_type" : "explore_mc_pctr_debias_type",
              "dynamic_dura_key": "explore_mc_bias_pctr",
              "dura_factors": "pctr_dura_debias_factor_list",
              "alpha" : "explore_mc_emp_debias_pctr_alpha",
              "beta" : "explore_mc_emp_debias_pctr_beta",
            },
            {
              "input_attr" : "mc_ensemble_pltr",
              "output_attr" : "mc_ensemble_pltr",
              "debias_type" : "explore_mc_pltr_debias_type",
              "dynamic_dura_key": "explore_mc_bias_pltr",
              "dura_factors": "pltr_dura_debias_factor_list",
              "alpha" : "explore_mc_emp_debias_pltr_alpha",
              "beta" : "explore_mc_emp_debias_pltr_beta",
            },
            {
              "input_attr" : "mc_ensemble_pwtr",
              "output_attr" : "mc_ensemble_pwtr",
              "debias_type" : "explore_mc_pwtr_debias_type",
              "dynamic_dura_key": "explore_mc_bias_pwtr",
              "dura_factors": "pwtr_dura_debias_factor_list",
              "alpha" : "explore_mc_emp_debias_pwtr_alpha",
              "beta" : "explore_mc_emp_debias_pwtr_beta",
            },
            {
              "input_attr" : "mc_ensemble_plvtr",
              "output_attr" : "mc_ensemble_plvtr",
              "debias_type" : "explore_mc_plvtr_debias_type",
              "dynamic_dura_key": "explore_mc_bias_plvtr",
              "dura_factors": "plvtr_dura_debias_factor_list",
              "alpha" : "explore_mc_emp_debias_plvtr_alpha",
              "beta" : "explore_mc_emp_debias_plvtr_beta",
            },
            {
              "input_attr" : "mc_ensemble_plvtr2",
              "output_attr" : "mc_ensemble_plvtr2",
              "debias_type" : "explore_mc_plvtr2_debias_type",
              "dynamic_dura_key": "explore_mc_bias_plvtr2",
              "dura_factors": "plvtr2_dura_debias_factor_list",
              "alpha" : "explore_mc_emp_debias_plvtr2_alpha",
              "beta" : "explore_mc_emp_debias_plvtr2_beta",
            },
            {
              "input_attr" : "mc_ensemble_pwtd",
              "output_attr" : "mc_ensemble_pwtd",
              "debias_type" : "explore_mc_pwtd_debias_type",
              "dura_factors": "pwtd_dura_debias_factor_list",
              "dynamic_dura_key": "explore_mc_bias_pwtd",
              "alpha" : "explore_mc_emp_debias_pwtd_alpha",
              "beta" : "explore_mc_emp_debias_pwtd_beta",
            },
            {
              "input_attr" : "mc_ensemble_pwatch_time",
              "output_attr" : "mc_ensemble_pwatch_time",
              "debias_type" : "explore_mc_pwatch_time_debias_type",
              "dynamic_dura_key": "explore_mc_bias_pwatch_time",
              "dura_factors": "pwatch_time_dura_debias_factor_list",
              "alpha" : "explore_mc_emp_debias_pwatch_time_alpha",
              "beta" : "explore_mc_emp_debias_pwatch_time_beta",
            }
            ],
          duration_attr = "duration_ms",
          picture_attr = "is_picture",
          enable_picture_xtr_debias = "{{explore_mc_enable_picture_xtr_debias}}",
          dura_bucket_width = "{{explore_mc_xtr_debias_dura_bucket_width}}",
          dura_xtr_debias_map_attr = "explore_mc_hourly_xtr_debias_map_ptr",
        ) \
    .end_() \
    .if_("enable_user_group_interest_tgi_photo == 1 and user_age_segment >= explore_user_group_interest_tgi_photo_age_min and user_age_segment <= explore_user_group_interest_tgi_photo_age_max") \
      .get_user_group_interest_tgi() \
    .end_() \
    .if_("explore_mc_ensemble_pftr_dur_enable == 1") \
      .explore_cal_mc_ensemble_pftr_dur() \
    .end_() \
    .if_("mc_enable_fill_action_count == 1") \
      .explore_calc_user_xtr_enricher(
        user_info_ptr_attr = "user_info_ptr",
        realtime_ctr_attr = "realtime_ctr",
        realtime_ltr_attr = "realtime_ltr",
        realtime_wtr_attr = "realtime_wtr",
        realtime_ftr_attr = "realtime_ftr",
        realtime_cltr_attr = "realtime_cltr"
      ) \
    .end_() \
    .if_("cal_cascade_cascade_ctr_svr_corr == 1", to_be_delete = "date=2024-05-29;committer=xuwei09") \
      .explore_replace_cascade_ctr_corr() \
    .end_() \
    .if_("enable_user_no_bias_interest_tagnex_tgi_product_pxtr_score == 1") \
      .calc_by_simple_formula(
        formulas = [
          dict(
            expr = (
              "{{explore_mc_no_bias_tgi_product_global_coeff}} * "
              "pow(1 + {{explore_mc_no_bias_tgi_product_tgi_alpha}} * [[user_no_bias_interest_tagnex_tgi_score]], {{explore_mc_no_bias_tgi_product_tgi_beta}}) * "
              "pow(1 + {{explore_mc_no_bias_tgi_product_pctr_alpha}} * [[cascade_corr_pctr]], {{explore_mc_no_bias_tgi_product_pctr_beta}}) * "
              "pow(1 + {{explore_mc_no_bias_tgi_product_pwatch_time_alpha}} * [[cascade_pwatch_time]], {{explore_mc_no_bias_tgi_product_pwatch_time_beta}})"
            ),
            output_attr = "user_no_bias_interest_tagnex_tgi_product_pxtr_score"
          )
        ]
      ) \
    .end_() \
    .if_("enable_user_stage_interest_tagnex_tgi_product_pxtr_score == 1") \
      .calc_by_simple_formula(
        formulas = [
          dict(
            expr = (
              "{{explore_mc_stage_tgi_product_global_coeff}} * "
              "pow(1 + {{explore_mc_stage_tgi_product_tgi_alpha}} * [[user_stage_interest_tagnex_tgi_score]], {{explore_mc_stage_tgi_product_tgi_beta}}) * "
              "pow(1 + {{explore_mc_stage_tgi_product_pctr_alpha}} * [[cascade_corr_pctr]], {{explore_mc_stage_tgi_product_pctr_beta}}) * "
              "pow(1 + {{explore_mc_stage_tgi_product_pwtr_alpha}} * [[cascade_pwtr]], {{explore_mc_stage_tgi_product_pwtr_beta}}) * "
              "pow(1 + {{explore_mc_stage_tgi_product_pwatch_time_alpha}} * [[cascade_pwatch_time]], {{explore_mc_stage_tgi_product_pwatch_time_beta}})"
            ),
            output_attr = "user_stage_interest_tagnex_tgi_product_pxtr_score"
          )
        ]
      ) \
    .end_() \
    .if_("explore_enable_cal_rise_follow_boost_light_score == 1")\
      .cal_rise_follow_boost_light_score() \
    .end_() \

  def post_process(self) -> None:
    self.flow \
      .perflog_attr_value(
        check_point = "cascading_score",
        item_attrs=[
          "cascade_pctr",
          "cascade_score",
          "cascade_pfptr",
          "prerank_ltr",
          "prerank_ctr",
          "prerank_wtd",
          "cascase_life_ctr",
          "mc_ensemble_pctr",
          "mc_ensemble_pltr",
          "mc_ensemble_pwtr",
          "mc_ensemble_pftr",
          "mc_ensemble_plvtr",
          "mc_ensemble_plvtr2",
          "mc_ensemble_psvtr",
          "mc_ensemble_ptr",
          "mc_ensemble_pwatch_time",
          "mc_ensemble_pepstr",
          "mc_ensemble_pcestr",
          "mc_ensemble_pcmtr",
          "mc_ensemble_pptime",
          "mc_ensemble_pcltr",
          "mc_ensemble_age_score",
          "mc_ensemble_peftr",
          "mc_ensemble_pefctr",
          "mc_ensemble_pwtd",
          "mc_ensemble_pfptr",
          "mc_ensemble_ordinal_wtd",
          "mc_ensemble_ordinal_prob",
          "mc_ensemble_pic_wtd",
          "mc_ensemble_pic_lvtr",
          "mc_ensemble_pic_cpr",
          "mc_ensemble_prerank_er",
          "mc_ensemble_pcptr",
          "mc_ensemble_pwtd_inverse",
        ],
      ) \
      .log_debug_info(
        for_debug_request_only = True,
        item_attrs = ['cascade_fc_pctr', 'cascade_pctr', 'mc_ensemble_pwtd_inverse'],
      )
