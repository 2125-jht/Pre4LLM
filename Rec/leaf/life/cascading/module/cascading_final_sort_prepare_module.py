from cascading import CommonModule

# 视频图文共用的 processor 放这里
# 以及一些 common attr 的操作
class CascadingFinalSortPrepareModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def zero_play_xtr_name(self):
    queues = [
      "explore_mc_ensemble_s2_pltr_power_weight",
      "explore_mc_ensemble_s2_pwtr_power_weight",
      "explore_mc_ensemble_s2_pftr_power_weight",
      "explore_mc_ensemble_s2_pcmtr_power_weight",
      "explore_mc_ensemble_s2_pepstr_power_weight",
      "explore_mc_ensemble_s2_plvtr_power_weight",
      "explore_mc_ensemble_s2_plvtr2_power_weight",
      "explore_mc_ensemble_s2_pcestr_power_weight",
      "explore_mc_ensemble_s2_pptime_power_weight",
      "explore_mc_ensemble_s2_pwatch_time_power_weight"
    ]
    return queues

  def zero_play_ctr_name(self):
    queues = [
      "explore_mc_ensemble_s2_pctr_power_weight"
    ]
    return queues

  def process(self) -> None:
    self.flow \
      .if_("explore_mc_sort_weight_adjust_s2 == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "enable_explore_weight_adjust_v2",
            "explore_colossus_user_emp_xtr_map_ptr",
            "explore_weight_adjust_coeff_a",
            "explore_weight_adjust_coeff_b",
            "explore_weight_adjust_coeff_c",
            "explore_weight_adjust_coeff_d",
            "user_emp_ltr",
            "user_emp_wtr",
            "user_emp_ftr",
            "user_emp_cmtr",
            "user_emp_eptr",
            "user_group_emp_ltr",
            "user_group_emp_wtr",
            "user_group_emp_ftr",
            "user_group_emp_cmtr",
            {"name": "explore_weight_adjust_avg_emp_ltr", "as": "all_user_emp_ltr"},
            {"name": "explore_weight_adjust_avg_emp_wtr", "as": "all_user_emp_wtr"},
            {"name": "explore_weight_adjust_avg_emp_ftr", "as": "all_user_emp_ftr"},
            {"name": "explore_weight_adjust_avg_emp_cmtr", "as": "all_user_emp_cmtr"},
            {"name": "explore_weight_adjust_avg_emp_eptr", "as": "all_user_emp_eptr"},
            {"name": "explore_mc_ensemble_s2_pltr_power_weight", "as": "user_ori_ltr_weight"},
            {"name": "explore_mc_ensemble_s2_pwtr_power_weight", "as": "user_ori_wtr_weight"},
            {"name": "explore_mc_ensemble_s2_pftr_power_weight", "as": "user_ori_ftr_weight"},
            {"name": "explore_mc_ensemble_s2_pcmtr_power_weight", "as": "user_ori_cmtr_weight"},
            {"name": "explore_mc_ensemble_s2_pepstr_power_weight", "as": "user_ori_eptr_weight"},
            "explore_weight_adjust_coeff_min",
            "explore_weight_adjust_coeff_max"
          ],
          export_common_attr = [
            {"name": "user_ltr_weight", "as": "explore_mc_ensemble_s2_pltr_power_weight"},
            {"name": "user_wtr_weight", "as": "explore_mc_ensemble_s2_pwtr_power_weight"},
            {"name": "user_ftr_weight", "as": "explore_mc_ensemble_s2_pftr_power_weight"},
            {"name": "user_cmtr_weight", "as": "explore_mc_ensemble_s2_pcmtr_power_weight"},
            {"name": "user_eptr_weight", "as": "explore_mc_ensemble_s2_pepstr_power_weight"},
          ],
          function_name = "UserSortWeightAdjust",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_mc_ensemble_sort_ctr_weight_adjust == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "explore_vv_3d",
            "choose_type",
            {"name": "explore_mc_ensemble_sort_ctr_weight_max", "as": "explore_weight_adjust_coeff_max"},
            {"name": "explore_mc_ensemble_sort_ctr_weight_min", "as": "explore_weight_adjust_coeff_min"}
          ],
          export_common_attr = [
            {"name": "weight_value", "as": "explore_mc_ensemble_s2_pctr_power_weight"},
          ],
          function_name = "DynamicCalculateWeight",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_specified_group_mc_boost_interactive == 1 and is_la_correct_user == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_mc_ensemble_s2_pltr_power_weight", "as": "value"},
            {"name": "explore_la_mc_like_boost_coeff", "as": "weight"}
          ],
          export_common_attr = [
            {"name": "new_value", "as": "explore_mc_ensemble_s2_pltr_power_weight"},
          ],
          function_name = "CalExploreDoubleMultiDouble",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_mc_ensemble_s2_pcmtr_power_weight", "as": "value"},
            {"name": "explore_la_mc_comment_boost_coeff", "as": "weight"}
          ],
          export_common_attr = [
            {"name": "new_value", "as": "explore_mc_ensemble_s2_pcmtr_power_weight"},
          ],
          function_name = "CalExploreDoubleMultiDouble",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_cascade_s2_weight_low_follow == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "cascade_s2_low_follow_pwtr_weight", "as": "rank_low_follow_pwtr_weight"},
            {"name": "cascade_s2_low_follow_thres_s", "as": "rank_low_follow_thres_s"},
            {"name": "cascade_s2_enable_no_follow_boost", "as": "enable_no_follow_boost"},
            {"name": "cascade_s2_enable_low_follow_boost", "as": "enable_low_follow_boost"},
            {"name": "cascade_s2_low_follow_boost_threshold", "as": "low_follow_boost_threshold"},
            "follow_timestamps",
            {"name": "explore_mc_ensemble_s2_pwtr_power_weight", "as": "input_pwtr_score"},
            {"name": "user_follow_type", "as": "user_follow_type"},
            {"name": "cascade_s2_no_follow_pwtr_weight", "as": "no_follow_pwtr_weight"},
            {"name": "cascade_s2_valid_follow_pwtr_weight", "as": "valid_follow_pwtr_weight"},
            {"name": "cascade_s2_valid_low_follow_pwtr_weight", "as": "valid_low_follow_pwtr_weight"},
            {"name": "cascade_s2_valid_media_follow_pwtr_weight", "as": "valid_media_follow_pwtr_weight"},
            {"name": "cascade_s2_valid_high_follow_pwtr_weight", "as": "valid_high_follow_pwtr_weight"},
          ],
          export_common_attr = [
            {"name": "output_pwtr_score", "as": "explore_mc_ensemble_s2_pwtr_power_weight"},
          ],
          function_name = "UserSortWeightLowFollow",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .enrich_attr_by_light_function(
        skip = "{{explore_cascade_skip_user_emp_xtr_handle}}",
        import_common_attr = [
          "user_emp_ltr",
          "user_emp_wtr",
          "user_emp_ftr",
          "user_emp_cmtr",
          {"name": "explore_cascade_user_emp_xtr_coeff", "as": "emp_xtr_coeff"}
        ],
        export_common_attr = [
          {"name": "user_emp_ltr_new", "as": "user_emp_ltr_cas_threshold"},
          {"name": "user_emp_wtr_new", "as": "user_emp_wtr_cas_threshold"},
          {"name": "user_emp_ftr_new", "as": "user_emp_ftr_cas_threshold"},
          {"name": "user_emp_cmtr_new", "as": "user_emp_cmtr_cas_threshold"}
        ],
        function_name = "EmpXtrThreshold",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .if_("explore_cascade_skip_zero_play_user_xtr_boost_handle == 1") \
        .set_attr_value( #和后面的pack_common_attr必须保持顺序一致性,非ctr相关队列，降权,zero play xtr begin
          no_overwrite=True,
          common_attrs=[
            {
              "name": "explore_mc_ensemble_s2_name_list",
              "type": "string_list",
              "value": self.zero_play_xtr_name()
            }
          ]
        ) \
        .pack_common_attr(
          input_common_attrs = self.zero_play_xtr_name(),
          output_common_attr = "explore_mc_ensemble_s2_value_list",
        ) \
        .enrich_attr_by_light_function( # suweiwei03 低活零播用户只保留ctr队列 begin
          import_common_attr = [
            {"name": "explore_zero_play_days_15d", "as": "explore_zero_play_days_15d"},
            {"name": "find_visit_days_30d", "as": "explore_visit_days_30d"},
            {"name": "explore_mc_ensemble_s2_zero_play_days_threshold", "as": "zero_play_days_threshold"}, 
            {"name": "explore_mc_ensemble_s2_zero_play_ratio_threshold", "as": "zero_play_ratio_threshold"},
            {"name": "explore_mc_ensemble_s2_zero_play_boost_type", "as": "boost_type"},
            {"name": "explore_mc_ensemble_s2_zero_play_xtr_boost_weight", "as": "boost_weight"}, 
            {"name": "explore_mc_ensemble_s2_name_list", "as": "attr_name_list"},
            {"name": "explore_mc_ensemble_s2_value_list", "as": "attr_value_list"},
          ],
          export_common_attr = self.zero_play_xtr_name(),
          function_name = "CalculateZeroPlayLaXtrBoostWeight",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .set_attr_value( #和后面的pack_common_attr必须保持顺序一致性，ctr相关队列，单独boost
          no_overwrite=True,
          common_attrs=[
            {
              "name": "explore_mc_ensemble_s2_ctr_name_list",
              "type": "string_list",
              "value": self.zero_play_ctr_name()
            }
          ]
        ) \
        .pack_common_attr(
          input_common_attrs = self.zero_play_ctr_name(),
          output_common_attr = "explore_mc_ensemble_s2_ctr_value_list",
        ) \
        .enrich_attr_by_light_function( #zero play xtr end
          import_common_attr = [
            {"name": "explore_zero_play_days_15d", "as": "explore_zero_play_days_15d"},
            {"name": "find_visit_days_30d", "as": "explore_visit_days_30d"},
            {"name": "explore_mc_ensemble_s2_zero_play_days_threshold", "as": "zero_play_days_threshold"}, 
            {"name": "explore_mc_ensemble_s2_zero_play_ratio_threshold", "as": "zero_play_ratio_threshold"},       
            {"name": "explore_mc_ensemble_s2_zero_play_boost_type", "as": "boost_type"},
            {"name": "explore_mc_ensemble_s2_zero_play_xtr_boost_weight", "as": "boost_weight"}, 
            {"name": "explore_mc_ensemble_s2_ctr_name_list", "as": "attr_name_list"},
            {"name": "explore_mc_ensemble_s2_ctr_value_list", "as": "attr_value_list"},
          ],
          export_common_attr = self.zero_play_ctr_name(),
          function_name = "CalculateZeroPlayLaXtrBoostWeight",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_mc_htr_weight_adjust_s2 == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_mc_ensemble_s2_phtr_power_weight", "as": "raw_htr_weight"},
            "find_user_active_degree",
            "page_index",
            "boost_htr_weight_by_mid_degree",
            "boost_htr_weight_by_high_degree",
            "boost_htr_weight_by_full_degree",
            "boost_htr_weight_by_p_idx",
            "boost_htr_weight_p_idx_down",
            "boost_htr_weight_p_idx_up" 
          ],
          export_common_attr = [
            {"name": "raw_htr_weight", "as": "explore_mc_ensemble_s2_phtr_power_weight"},
          ],
          function_name = "HtrWeightAdjust",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
     .if_("explore_cascade_skip_infer_uv_ctr_boost_handle == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "infer_uv_ctr",
            {"name": "refreshTimes", "as": "refresh_times"}, 
            {"name": "explore_mc_ensemble_s2_infer_uv_ctr_refresh_times_threshold", "as": "refresh_times_threshold"},
            {"name": "explore_mc_ensemble_s2_infer_uv_ctr_infer_uv_ctr_threshold", "as": "infer_uv_ctr_threshold"},
            {"name": "explore_mc_ensemble_s2_infer_uv_ctr_weight_max", "as": "weight_max"},
            {"name": "explore_mc_ensemble_s2_infer_uv_ctr_weight_min", "as": "weight_min"}, 
            {"name": "explore_mc_ensemble_s2_infer_uv_ctr_alpha", "as": "alpha"},
            {"name": "explore_mc_ensemble_s2_infer_uv_ctr_beta", "as": "beta"},
            {"name": "explore_mc_ensemble_s2_infer_uv_ctr_omega", "as": "omega"},
            {"name": "explore_mc_ensemble_s2_infer_uv_ctr_boost_type", "as": "boost_type"},
            {"name": "explore_mc_ensemble_s2_pctr_power_weight", "as": "xtr_weight"},
          ],
          export_common_attr = [
            {"name": "xtr_weight", "as": "explore_mc_ensemble_s2_pctr_power_weight"}
          ],
          function_name = "CalcXtrWeightByInferUvCtr",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = [
          "_USER_ID_",
          "refresh_times",
          "infer_uv_ctr",
          "explore_mc_ensemble_s2_infer_uv_ctr_refresh_times_threshold",
          "explore_mc_ensemble_s2_infer_uv_ctr_infer_uv_ctr_threshold",
          "explore_mc_ensemble_s2_infer_uv_ctr_weight_max",
          "explore_mc_ensemble_s2_infer_uv_ctr_weight_min",
          "explore_mc_ensemble_s2_infer_uv_ctr_alpha",
          "explore_mc_ensemble_s2_infer_uv_ctr_beta",
          "explore_mc_ensemble_s2_infer_uv_ctr_omega",
          "explore_mc_ensemble_s2_pctr_power_weight",
          "explore_mc_ensemble_s2_zero_play_ctr_boost_weight" 
        ],
        for_debug_request_only = True
      )
