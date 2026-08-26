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
    .explore_custom_embedding_score_enricher(
      skip = "{{explore_cascade_skip_mmu_embedding_score}}",
      user_info_ptr_attr = "user_info_ptr",
      embedding_list_attr = "mmu_embeddings",
      source_pids_list_attr = "embedding_source_pids", # 在 user_info_module 里产出
      calc_type = "action_bucket_dot",
      short_view_weight = 0.0,
      export_item_attr = "cascade_mmu_embedding_score",
      dim_size = 64
    ) \
    .if_("enable_hot_fc_exp == 1") \
      .enrich_attr_by_light_function(
        import_item_attr = [
          "cascade_fc_pctr",
          "cascade_fc_plvtr",
          "cascade_fc_psvtr",
          "cascade_fc_pvtr" #vtr可能不准
        ],
        export_item_attr = [
          "cascade_pctr",
          "cascade_plvtr",
          "cascade_psvtr",
          "cascade_pwatch_time" # vtr可能不准
        ],
        function_name = "ReplaceMcPxtr",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .else_() \
      .enrich_attr_by_lua(  # 粗排新模型下游服务不支持按 item 返回的方式，用 lua 处理一下
        import_common_attr = [
          "mc_pxtr_label",
          "mc_pxtr_value",
        ],
        export_item_attr = [
          "cascade_fc_pctr"
        ],
        function_for_item = "handle",
        lua_script_file = "life/cascading/lua/module/cascading_predict__fc_resp_handler.lua",
      ) \
      .enrich_attr_by_light_function(
        import_item_attr = [
          "cascade_fc_pctr",
        ],
        export_item_attr = [
          "cascade_pctr",
        ],
        function_name = "ReplaceMcPctr",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_()\
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
        "mille_l0_cascase_life_ctr_param"
      ],
      import_item_attr = [
        "cascade_pctr",         #1
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
      ],
      export_item_attr = [
        "cascade_score",
      ],
      function_name = "CalMcMergedScore",
      class_name = "ExploreLightFunctionSetV2",
      ) \
    .set_attr_default_value(
      item_attrs=[
        {
          "name": "mc_ensemble_opportunity_cost_score",
          "type": "double",
          "value": 0.0
        }
      ]
    ) \
    .enrich_attr_by_light_function(
      skip = "{{explore_cascade_skip_opportunity_cost_score}}",
      import_common_attr = [
        "mc_opportunity_cost_queue_cost_weight",
        "mc_opportunity_cost_queue_reward_weight",
        "mc_opportunity_cost_queue_pltr_weight",
        "mc_opportunity_cost_queue_pwtr_weight",
        "mc_opportunity_cost_queue_pftr_weight",
        "mc_opportunity_cost_queue_pcmtr_weight",
        "mc_opportunity_cost_queue_pepstr_weight",
        "mc_opportunity_cost_queue_pcltr_weight",
        "mc_opportunity_cost_queue_peftr_weight",
        "mc_opportunity_cost_queue_pefctr_weight",
        "mc_opportunity_cost_queue_ctr_power_weight",
      ],
      import_item_attr = [
        "cascade_pctr",
        "cascade_pltr",
        "cascade_pwtr",
        "cascade_pftr",
        "cascade_pcltr",
        "cascade_pepstr",
        "cascade_pcmtr",
        "cascade_peftr",
        "cascade_pefctr",
        "cascade_pwatch_time",
      ],
      export_item_attr = [
        "mc_ensemble_opportunity_cost_score",
      ],
      function_name = "CalcOpportunityCostScore",
      class_name = "ExploreLightFunctionSetV2",
    )\
    .if_("enable_calc_prerank_wtd_in_s1_score == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "prerank_wtd_in_s1_min_score", "as": "prerank_wtd_min_score"},
          {"name": "prerank_wtd_in_s1_max_score", "as": "prerank_wtd_max_score"},
          {"name": "prerank_wtd_in_s1_table_seg", "as": "prerank_wtd_table_seg"},
          {"name": "prerank_wtd_in_s1_table_0", "as": "prerank_wtd_table_0"},
          {"name": "prerank_wtd_in_s1_table_1", "as": "prerank_wtd_table_1"},
          {"name": "prerank_wtd_in_s1_table_2", "as": "prerank_wtd_table_2"},
          {"name": "prerank_wtd_in_s1_table_3", "as": "prerank_wtd_table_3"},
          {"name": "prerank_wtd_in_s1_table_4", "as": "prerank_wtd_table_4"},
          {"name": "prerank_wtd_in_s1_table_5", "as": "prerank_wtd_table_5"},
          {"name": "prerank_wtd_in_s1_table_6", "as": "prerank_wtd_table_6"},
          {"name": "prerank_wtd_in_s1_table_7", "as": "prerank_wtd_table_7"},
          {"name": "prerank_wtd_in_s1_table_8", "as": "prerank_wtd_table_8"},
          {"name": "prerank_wtd_in_s1_finish_pow_weight", "as": "prerank_wtd_finish_pow_weight"},
          {"name": "prerank_wtd_in_s1_finish_max", "as": "prerank_wtd_finish_max"},
          {"name": "prerank_wtd_in_s1_finish_min", "as": "prerank_wtd_finish_min"}
        ],
        import_item_attr = [
          {"name": "prerank_wtd_in_s1", "as": "prerank_wtd"},
          "duration_ms",
        ],
        export_item_attr = [
          {"name": "prerank_wtd", "as" : "prerank_wtd_in_s1_modified"},
        ],
        function_name = "CalcPreRankWtdScore",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .if_("enable_calc_mc_pctr_adjust == 1") \
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
        "wtd_in_s1_factor_type",
        "wtd_in_s1_power_weight",
        "cascading_score_factor_power",
        "enable_mc_pctr_corr_replace_pctr",
        "mc_pctr_corr_replace_pctr_weight",
        "enable_mc_pctr_corr_replace_pctr_in_factor",
        "mc_pctr_corr_replace_pctr_weight_in_factor",
      ],
      import_item_attr = [
        "cascade_pctr",
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
        "cascade_ordinal_wtd",
        "cascade_ordinal_prob",
        "prerank_wtd",
        "prerank_ctr_in_s1",
        {"name": "prerank_wtd_in_s1_modified", "as": "prerank_wtd_in_s1"},
        "cascade_pctr_corr",
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
        "mc_ensemble_prerank_wtd",
        "mc_ensemble_prerank_wtd_in_s1"
      ],
      function_name = "CalMcEnsembleQueueScore",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .if_("enable_mc_htr_cost_queue == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "htr_cost_queue_cost_weight",
          "htr_cost_queue_reward_weight",
          "htr_cost_queue_pctr_power_weight",
          "htr_cost_queue_pltr_weight",
          "htr_cost_queue_pwtr_weight",
          "htr_cost_queue_pftr_weight",
          "htr_cost_queue_pcmtr_weight",
          "htr_cost_queue_pcltr_weight",
          "htr_cost_queue_peftr_weight",
          "htr_cost_queue_pefctr_weight",
          "htr_cost_queue_wtd_weight",
          "htr_cost_queue_plvtr_weight",
          "htr_cost_queue_psvtr_weight",
        ],
        import_item_attr = [
          {"name": "cascade_pctr", "as": "pctr_input"},
          {"name": "cascade_pltr", "as": "pltr_input"},
          {"name": "cascade_pwtr", "as": "pwtr_input"},
          {"name": "cascade_pftr", "as": "pftr_input"},
          {"name": "cascade_pcltr", "as": "pcltr_input"},
          {"name": "cascade_pcmtr", "as": "pcmtr_input"},
          {"name": "cascade_peftr", "as": "peftr_input"},
          {"name": "cascade_pefctr", "as": "pefctr_input"},
          {"name": "cascade_pwtd", "as": "wtd_input"},
          {"name": "cascade_plvtr", "as": "plvtr_input"},
          {"name": "cascade_psvtr", "as": "psvtr_input"},
          {"name": "cascade_phtr", "as": "htr_input"},
        ],
        export_item_attr = [
          "mc_htr_cost_score",
        ],
        function_name = "HtrCostScore",
        class_name = "ExploreLightFunctionSetV2",
      )\
    .end_() \
    .if_("explore_mc_debias_score == 1") \
      .explore_memory_data_enrich(
        data_key = "{{explore_cascade_debias_xtr_map}}",
        data_type = "string_double_vector_map",
        save_data_ptr_to_attr = "cascade_debias_xtr_map_ptr",
      ) \
      .explore_user_debias_xtr_enricher(
        memory_data_map_ptr = "cascade_debias_xtr_map_ptr",
        debias_module_prefix = "cascade",
        gender_attr = "user_gender",
        age_segment_attr = "user_age_segment",
        hetu_tag_attr = "hetu_tag_level_info__hetu_level_one",
        ctr_attr = "mc_ensemble_pctr",
        ltr_attr = "mc_ensemble_pltr",
        wtr_attr = "mc_ensemble_pwtr",
        ftr_attr = "mc_ensemble_pftr",
        cmtr_attr = "mc_ensemble_pcmtr",
        epstr_attr = "mc_ensemble_pepstr",
        cltr_attr = "mc_ensemble_pcltr",
        debias_ctr_attr = "mc_ensemble_pctr",
        debias_ltr_attr = "mc_ensemble_pltr",
        debias_wtr_attr = "mc_ensemble_pwtr",
        debias_ftr_attr = "mc_ensemble_pftr",
        debias_cmtr_attr = "mc_ensemble_pcmtr",
        debias_epstr_attr = "mc_ensemble_pepstr",
        debias_cltr_attr = "mc_ensemble_pcltr",
      ) \
    .end_() \
    .switch_("explore_mc_enable_xtr_debias") \
      .case_(2) \
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
              "input_attr" : "cascade_phtr",
              "output_attr" : "cali_mc_phtr",
              "debias_type" : "explore_mc_phtr_debias_type",
              "dura_factors": "phtr_dura_debias_factor_list",
              "freq_factors": "phtr_dura_debias_factor_list",
              "dynamic_dura_key": "explore_mc_bias_phtr",
              "alpha" : "explore_mc_emp_debias_phtr_alpha",
              "beta" : "explore_mc_emp_debias_phtr_beta",
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
    .if_("enable_hot_fractile_abs_queue_after_debias == 1") \
      .explore_absolute_xtr_score_que_enricher(
        explore_absolute_xtr_boost_threshold = "{{hot_mc_pxtr_fractile_boost_threshold}}",
        explore_absolute_xtr_boost_weight = "{{hot_mc_pxtr_fractile_boost_weight}}",
        enable_explore_absolute_xtr_cliff = "{{enable_hot_mc_pxtr_fractile_cliff}}",
        pxtr_fractile_kconf_path = "reco.offline.hotMcPxtrFractile",
        absolute_xtr_score_que_attr = "cascade_pxtr_fractile_score",
        enable_explore_time_cost_optimal = "{{enable_hot_mc_time_cost_optimal}}",
        queues = hot_mc_pxtr_fractile_score_queues
      ) \
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
    .if_("enable_fill_pcltr_adjust_by_gender_age == 1") \
      .explore_fill_avg_xtr_enricher(
        user_info_ptr_attr = "user_info_ptr",
        avg_cltr_attr = "avg_cltr" 
      ) \
    .end_()

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
          "mc_ensemble_opportunity_cost_score",
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
