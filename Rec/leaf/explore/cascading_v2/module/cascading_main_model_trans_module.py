from cascading_v2 import CommonModule

class CascadingMainModelTransModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_cascade_downgrade == 1") \
        .enrich_attr_by_light_function(  # 确保此处 infer 已经返回，避免下面的 copy_attr 结果被再次覆盖
          import_item_attr = [
            "cascade_pltr",
            "cascade_pwtr",
            "cascade_pftr",
            "cascade_ptr",
            "cascade_phtr",
            "cascade_pcmtr",
            "cascade_pctr",
            "cascade_plvtr",
            "cascade_psvtr",
          ],
          function_name = "EmptyFunction",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .copy_attr(  # 降级需求，特批修改其他 module 产出的 attr
          attrs = [
            {
              "from_item": "empirical_ltr",
              "to_item": "cascade_pltr",
            },
            {
              "from_item": "empirical_wtr",
              "to_item": "cascade_pwtr",
            },
            {
              "from_item": "empirical_ftr",
              "to_item": "cascade_pftr",
            },
            {
              "from_item": "empirical_ptr",
              "to_item": "cascade_ptr",
            },
            {
              "from_item": "empirical_htr",
              "to_item": "cascade_phtr",
            },
            {
              "from_item": "empirical_cmtr",
              "to_item": "cascade_pcmtr",
            },
            {
              "from_item": "empirical_ctr",
              "to_item": "cascade_pctr",
            },
            {
              "from_item": "empirical_lvtr",
              "to_item": "cascade_plvtr",
            },
            {
              "from_item": "empirical_svtr",
              "to_item": "cascade_psvtr",
            },
          ],
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
      .if_("enable_explore_cascade_pxtr_calibration == 1") \
        .mc_cascade_pctr_calibration() \
      .else_() \
        .copy_attr(
          attrs = [{
            "from_item": "cascade_pctr",
            "to_item": "cascade_corr_pctr",
          }]
        ) \
      .end_() \
      .enrich_attr_by_light_function( # 非计算主模型平替分数的，都不要再改这里的逻辑了，像其他分数一样独立开发
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
          "cascading_score_factor_power",
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
          "avg_watch_time",
          "upload_time",
          "explore_stat__real_show_count",
          "cascade_pwtd",
          "duration_ms",
          "photo_picture_count",
          "cascade_pcptr",
          "cascade_pwtd_inverse",
          "empirical_watch_time",
          "empirical_ctr",
          "cascade_ordinal_wtd",
          "cascade_ordinal_prob",
          "is_picture",
        ],
        export_item_attr = [
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
          "mc_ensemble_pcltr",
          "mc_ensemble_peftr",
          "mc_ensemble_pefctr",
          "mc_ensemble_pwtd",
          "mc_ensemble_pfptr",
          "mc_ensemble_pic_wtd",
          "mc_ensemble_pic_lvtr",
          "mc_ensemble_pic_cpr",
          "mc_ensemble_pcptr",
          "mc_ensemble_pwtd_inverse",
          "mc_ensemble_smooth_age_score",
        ],
        function_name = "CalMcEnsembleQueueScore",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .if_("explore_mc_enable_xtr_debias == 1") \
        .explore_xtr_debias_v3_enricher(
          queues = [
            {
              "input_attr" : "mc_ensemble_pcmtr",
              "output_attr" : "mc_ensemble_pcmtr",
              "debias_type" : "explore_mc_pcmtr_debias_type",
              "dynamic_dura_key": "explore_mc_bias_pcmtr",
            },
            {
              "input_attr" : "mc_ensemble_pcltr",
              "output_attr" : "mc_ensemble_pcltr",
              "debias_type" : "explore_mc_pcltr_debias_type",
              "dynamic_dura_key": "explore_mc_bias_pcltr",
            },
            {
              "input_attr" : "mc_ensemble_pctr",
              "output_attr" : "mc_ensemble_pctr",
              "debias_type" : "explore_mc_pctr_debias_type",
              "dynamic_dura_key": "explore_mc_bias_pctr",
            },
            {
              "input_attr" : "mc_ensemble_pltr",
              "output_attr" : "mc_ensemble_pltr",
              "debias_type" : "explore_mc_pltr_debias_type",
              "dynamic_dura_key": "explore_mc_bias_pltr",
            },
            {
              "input_attr" : "mc_ensemble_pwtr",
              "output_attr" : "mc_ensemble_pwtr",
              "debias_type" : "explore_mc_pwtr_debias_type",
              "dynamic_dura_key": "explore_mc_bias_pwtr",
            },
            {
              "input_attr" : "mc_ensemble_plvtr",
              "output_attr" : "mc_ensemble_plvtr",
              "debias_type" : "explore_mc_plvtr_debias_type",
              "dynamic_dura_key": "explore_mc_bias_plvtr",
            },
            {
              "input_attr" : "mc_ensemble_plvtr2",
              "output_attr" : "mc_ensemble_plvtr2",
              "debias_type" : "explore_mc_plvtr2_debias_type",
              "dynamic_dura_key": "explore_mc_bias_plvtr2",
            },
            {
              "input_attr" : "mc_ensemble_pwtd",
              "output_attr" : "mc_ensemble_pwtd",
              "debias_type" : "explore_mc_pwtd_debias_type",
              "dura_factors": "pwtd_dura_debias_factor_list",
            },
            {
              "input_attr" : "mc_ensemble_pwatch_time",
              "output_attr" : "mc_ensemble_pwatch_time",
              "debias_type" : "explore_mc_pwatch_time_debias_type",
              "dura_factors": "pwatch_time_dura_debias_factor_list",
            }
          ],
          duration_attr = "duration_ms",
          picture_attr = "is_picture",
          enable_picture_xtr_debias = "{{explore_mc_enable_picture_xtr_debias}}",
          dura_bucket_width = "{{explore_mc_xtr_debias_dura_bucket_width}}",
          dura_xtr_debias_map_attr = "explore_mc_hourly_xtr_debias_map_ptr",
        ) \
      .end_()

  def post_process(self) -> None:
    self.flow \
      .perflog_attr_value(
        check_point = "cascading_score",
        item_attrs=[
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
          "mc_ensemble_pcltr",
          "mc_ensemble_peftr",
          "mc_ensemble_pefctr",
          "mc_ensemble_pwtd",
          "mc_ensemble_pfptr",
          "mc_ensemble_pic_wtd",
          "mc_ensemble_pic_lvtr",
          "mc_ensemble_pic_cpr",
          "mc_ensemble_pwtd_inverse",
        ],
      )
