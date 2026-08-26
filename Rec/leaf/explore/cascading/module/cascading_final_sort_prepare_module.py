from cascading import CommonModule
from cascading.module.queue.cascade_final_queues import final_channel_sort_weight_param_dict

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
      .if_("explore_mc_s2_age_based_weight_adjust == 1", to_be_delete = "date=2024-05-29;committer=guohao") \
        .user_age_based_weight_adjust_all() \
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
      .if_("enable_tnu_user_adjust_cascading_weight == 1 and uIsExploreTnuCrowdUser == 1") \
        .gen_common_attr_by_lua( # 显式判断新回人群逻辑删除 to_be_delete = 2024-09-20
          attr_map={
            "explore_mc_ensemble_s2_pctr_power_weight": "explore_mc_ensemble_s2_pctr_power_weight * explore_tnu_ctr_cascading_adjust_ratio",
            "explore_mc_ensemble_s2_pltr_power_weight": "explore_mc_ensemble_s2_pltr_power_weight * explore_tnu_ltr_cascading_adjust_ratio",
          }
        ) \
      .end_() \
      .if_("enable_explore_mc_s2_interactive_weight_adjust == 1", to_be_delete = "date=2024-05-29;committer=wangziqi05") \
        ._diversify_interactive_power_weight_in_mc_s2() \
      .end_() \
      .if_("enable_explore_mc_s2_user_vv_weight_adjust == 1 and user_age_segment >= explore_mc_hate_like_weight_adjust_age_min and user_age_segment <= explore_mc_hate_like_weight_adjust_age_max") \
        .user_vv_weight_adjust_mc_s2() \
      .end_() \
      .if_("enable_explore_mc_s2_request_pxtr_weight_adjust == 1") \
        .mc_user_vv_ensemble_power_weight_adjust("explore_mc_ensemble_s2_pctr_power_weight", "explore_mc_v4_request_pctr_weight_adjust_exp_upper", "explore_mc_v4_request_pctr_weight_adjust_alpha", "explore_mc_v4_request_pctr_weight_adjust_beta", "explore_mc_v4_request_pctr_weight_adjust_omega", "explore_mc_v4_request_pctr_weight_adjust_max", "explore_mc_v4_request_pctr_weight_adjust_min") \
      .end_() \
      .if_("explore_enable_mc_s2_user_recent_hate_count_ensemble_power_weight_adjust == 1 and recent_hate_count <= explore_cascading_koc_htr_count_threshold") \
        .mc_s2_user_recent_hate_count_ensemble_koc_htr_power_weight_adjust(
          "explore_mc_ensemble_s2_cascade_explore_koc_cover_htr_weight",
          "explore_mc_s2_ensemble_power_koc_cover_htr_weight_adjust_exp_upper",
          "explore_mc_s2_ensemble_power_koc_cover_htr_weight_adjust_alpha",
          "explore_mc_s2_ensemble_power_koc_cover_htr_weight_adjust_beta",
          "explore_mc_s2_ensemble_power_koc_cover_htr_weight_adjust_omega",
          "explore_mc_s2_ensemble_power_koc_cover_htr_weight_adjust_max",
          "explore_mc_s2_ensemble_power_koc_cover_htr_weight_adjust_min"
        ) \
        .mc_s2_user_recent_hate_count_ensemble_koc_htr_power_weight_adjust(
          "explore_mc_ensemble_s2_cascade_explore_koc_detail_htr_weight",
          "explore_mc_s2_ensemble_power_koc_detail_htr_weight_adjust_exp_upper",
          "explore_mc_s2_ensemble_power_koc_detail_htr_weight_adjust_alpha",
          "explore_mc_s2_ensemble_power_koc_detail_htr_weight_adjust_beta",
          "explore_mc_s2_ensemble_power_koc_detail_htr_weight_adjust_omega",
          "explore_mc_s2_ensemble_power_koc_detail_htr_weight_adjust_max",
          "explore_mc_s2_ensemble_power_koc_detail_htr_weight_adjust_min"
        ) \
      .end_() \
      .if_("explore_enable_mc_s2_user_more_recent_hate_count_ensemble_power_weight_adjust == 1 and recent_hate_count > explore_cascading_koc_htr_count_threshold") \
        .mc_s2_user_recent_hate_count_ensemble_koc_htr_power_weight_adjust(
          "explore_mc_ensemble_s2_cascade_explore_koc_cover_htr_weight",
          "explore_mc_s2_ensemble_power_koc_cover_htr_more_hate_weight_adjust_exp_upper",
          "explore_mc_s2_ensemble_power_koc_cover_htr_more_hate_weight_adjust_alpha",
          "explore_mc_s2_ensemble_power_koc_cover_htr_more_hate_weight_adjust_beta",
          "explore_mc_s2_ensemble_power_koc_cover_htr_more_hate_weight_adjust_omega",
          "explore_mc_s2_ensemble_power_koc_cover_htr_more_hate_weight_adjust_max",
          "explore_mc_s2_ensemble_power_koc_cover_htr_more_hate_weight_adjust_min"
        ) \
        .mc_s2_user_recent_hate_count_ensemble_koc_htr_power_weight_adjust(
          "explore_mc_ensemble_s2_cascade_explore_koc_detail_htr_weight",
          "explore_mc_s2_ensemble_power_koc_detail_htr_more_hate_weight_adjust_exp_upper",
          "explore_mc_s2_ensemble_power_koc_detail_htr_more_hate_weight_adjust_alpha",
          "explore_mc_s2_ensemble_power_koc_detail_htr_more_hate_weight_adjust_beta",
          "explore_mc_s2_ensemble_power_koc_detail_htr_more_hate_weight_adjust_omega",
          "explore_mc_s2_ensemble_power_koc_detail_htr_more_hate_weight_adjust_max",
          "explore_mc_s2_ensemble_power_koc_detail_htr_more_hate_weight_adjust_min"
        ) \
      .end_() \
      .if_("enable_explore_mc_s2_boost_negative_feedback_weight == 1 and user_active_decline_score >= explore_mc_s2_boost_negative_feedback_queue_of_user_active_decline_score_threshold and (find_user_active_degree == 3 or find_user_active_degree == 4)") \
        .gen_common_attr_by_lua( # 针对高活全勤人群如果用户活跃衰退分user_active_decline_score大于阈值对负反馈相关队列进行boost
          attr_map = {
            "explore_mc_ensemble_s2_phtr_power_weight" : "explore_mc_s2_boost_phtr_power_weight_coefficient * explore_mc_ensemble_s2_phtr_power_weight",
            "explore_mc_ensemble_s2_cascade_explore_koc_cover_htr_weight" : "explore_mc_s2_boost_cascade_explore_koc_cover_htr_weight_coefficient * explore_mc_ensemble_s2_cascade_explore_koc_cover_htr_weight",
            "explore_mc_ensemble_s2_cascade_explore_koc_detail_htr_weight" : "explore_mc_s2_boost_cascade_explore_koc_detail_htr_weightt_coefficient * explore_mc_ensemble_s2_cascade_explore_koc_detail_htr_weight",
          }
        ) \
      .end_() \
      .if_("enable_mc_s2_user_age_tgi_product_first_refresh_weight_adjust == 1 and is_first_refresh == 1") \
        .gen_common_attr_by_lua( # user_age_interest_tagnex_tgi_product_pxtr_score 首屏权重独立设置
          attr_map = {
            "explore_mc_ensemble_s2_user_age_interest_tagnex_tgi_product_pxtr_score_weight" : "explore_mc_s2_user_age_tgi_product_first_refresh_weight",
            "explore_mc_ensemble_s2_user_age_interest_tagnex_tgi_product_pxtr_score_power_weight" : "explore_mc_s2_user_age_tgi_product_first_refresh_power_weight",
            "explore_mc_ensemble_s2_user_age_interest_tagnex_tgi_product_pxtr_score_raw_weight" : "explore_mc_s2_user_age_tgi_product_first_refresh_raw_weight",
            "explore_mc_ensemble_s2_user_age_interest_tagnex_tgi_product_pxtr_score_raw_power_weight" : "explore_mc_s2_user_age_tgi_product_first_refresh_raw_power_weight",
          }
        ) \
      .end_() \
      .if_("enable_user_age_tgi_score_population_weight_adjust == 1 and basic_info_age_segment_v2 > user_age_tgi_score_population_age_segment_threshold and active_days_gt_5min_rate < user_age_tgi_score_population_active_days_threshold") \
        .copy_attr(
          attrs = [{
            "from_common": "explore_mc_ensemble_s2_user_age_interest_tagnex_tgi_product_pxtr_score_population_weight",
            "to_common": "explore_mc_ensemble_s2_user_age_interest_tagnex_tgi_product_pxtr_score_weight"
          },
          {
            "from_common": "explore_mc_ensemble_s2_user_age_interest_tagnex_tgi_product_pxtr_score_population_power_weight",
            "to_common": "explore_mc_ensemble_s2_user_age_interest_tagnex_tgi_product_pxtr_score_power_weight"
          },
          {
            "from_common": "explore_mc_ensemble_s2_user_age_interest_tagnex_tgi_product_pxtr_score_population_raw_weight",
            "to_common": "explore_mc_ensemble_s2_user_age_interest_tagnex_tgi_product_pxtr_score_raw_weight"
          },
          {
            "from_common": "explore_mc_ensemble_s2_user_age_interest_tagnex_tgi_product_pxtr_score_population_raw_power_weight",
            "to_common": "explore_mc_ensemble_s2_user_age_interest_tagnex_tgi_product_pxtr_score_raw_power_weight"
          }]
        ) \
      .end_() \
      .if_("enable_mc_s2_explore_low_active_customization_view_score_weight == 1 and is_explore_new_la_user == 1") \
        .gen_common_attr_by_lua(
          attr_map={
            "explore_mc_ensemble_s2_pctr_power_weight": "explore_mc_ensemble_s2_pctr_power_weight * explore_low_active_ctr_cascading_adjust_ratio",
          }
        ) \
        .copy_attr(
          attrs=[{
            "from_common": "explore_mc_ensemble_s2_cover_view_predict_trans_score_weight_low_active",
            "to_common": "explore_mc_ensemble_s2_cover_view_predict_trans_score_weight"
          }, {
            "from_common": "explore_mc_ensemble_s2_sense_view_predict_trans_score_weight_low_active",
            "to_common": "explore_mc_ensemble_s2_sense_view_predict_trans_score_weight"
          }]
        ) \
      .end_() \
      .if_("explore_cascade_skip_user_emp_xtr_handle == 0", to_be_delete = "date=2024-05-29;committer=liuhao07") \
        .enrich_attr_by_light_function(
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
      .end_() \
      .if_("enable_first_screen_photo_discount_by_cid == 1") \
        .enrich_attr_by_light_function(
          item_list_from_attr = "explore_realshow_click_common_list",
          import_common_attr = [
            "explore_realshow_click_timestamp_common_list",
            "explore_click_common_list",
            {"name": "first_screen_photo_cid_timestamp_window_thred", "as": "timestamp_window_thred"},
            {"name": "first_screen_photo_cid_realshow_num_limit", "as": "realshow_num_limit"},
            {"name": "first_screen_photo_cid_only_no_click", "as": "only_no_click"},
          ],
          import_item_attr = [
            {"name": "hetu_sim_cluster_id", "as": "target_item_attr"},
          ],
          export_common_attr = [
            {"name": "real_show_attr_list", "as": "real_show_cid_list"},
          ],
          function_name = "GetTargetItemAttrListFromRealshow",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "real_show_cid_list",
          ],
          import_item_attr = [
            "hetu_sim_cluster_id",
          ],
          export_item_attr = [
            "is_first_screen_discount_by_cid",
          ],
          function_name = "AttrIsInSet",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_first_screen_photo_discount == 1 and page_index >= 1 and page_index <= explore_mc_s2_first_screen_discount_threshold") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "user_info_ptr",
            {"name": "first_screen_photo_hetu_level_type", "as": "hetu_level_type"},
            {"name": "first_screen_photo_recent_minutes", "as": "recent_minutes"},
            {"name": "explore_mc_s2_first_screen_photo_recent_counts", "as": "recent_counts"},
          ],
          import_item_attr = [
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_level_one"},
            {"name": "hetu_tag_level_info__hetu_level_two", "as": "hetu_level_two"},
          ],
          export_item_attr = [
            "is_first_screen_discount",
          ],
          function_name = "GetFirstScreenDiscountPhoto",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
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
      .if_("enable_cascading_mc_ensemble_s2_pwtr_weight_new_follow_adjust == 1") \
        .gen_common_attr_by_lua( # 粗排s2涨关摸高
          attr_map={
            "explore_mc_ensemble_s2_pwtr_power_weight": "explore_mc_ensemble_s2_pwtr_power_weight * explore_new_follow_pwtr_cascading_adjust_ratio",
          }
        ) \
      .end_() \
      .if_("explore_mc_ensemble_s2_pltr_social_enable == 1") \
        .if_("explore_mc_ensemble_s2_pltr_social_condition == 1 and (bid_follow_num == 0 or (u_inside_share_active_degree_detail_code > 3))") \
          .gen_common_attr_by_lua(
            attr_map = {
              "explore_mc_ensemble_s2_pftr_power_weight_social" : "0.0",
              "explore_mc_ensemble_pftr_raw_power_weight_social" : "0.0",
            }
          ) \
        .end_() \
        .if_("explore_mc_ensemble_s2_pltr_social_condition == 2 and (bid_follow_num == 0 or (u_share_num_30d == 0 and u_message_active_degree ~= 5 and u_message_active_degree ~= 6))") \
          .gen_common_attr_by_lua(
            attr_map = {
              "explore_mc_ensemble_s2_pftr_power_weight_social" : "0.0",
              "explore_mc_ensemble_pftr_raw_power_weight_social" : "0.0",
            }
          ) \
        .end_() \
        .if_("explore_mc_ensemble_s2_pltr_social_condition == 3 and (bid_follow_num == 0 or (user_msg_cnt_ssm_today + user_msg_cnt_gsm_today > 0))") \
          .gen_common_attr_by_lua(
            attr_map = {
              "explore_mc_ensemble_s2_pftr_power_weight_social" : "0.0",
              "explore_mc_ensemble_pftr_raw_power_weight_social" : "0.0",
            }
          ) \
        .end_() \
      .end_() \
      .if_("enable_explore_cascading_weight_adjust_by_high_time_rate == 1") \
        .explore_cascade_s2_low_time_active_weight_adjust() \
      .end_() \
      .if_("explore_enable_mc_s2_ef_weight_adjust == 1") \
        .gen_common_attr_by_lua(
          attr_map = {
            "explore_mc_ensemble_s2_eftr_score_power_weight": "explore_mc_ensemble_s2_eftr_score_power_weight * explore_fountain_view_weight",
            "explore_mc_ensemble_s2_efctr_score_power_weight": "explore_mc_ensemble_s2_efctr_score_power_weight * explore_fountain_view_weight",
          }
        ) \
      .end_() \
      .if_("explore_user_group_consume_weight_adjust_mc_s2 == 1") \
        .user_group_consume_weight_adjust(final_channel_sort_weight_param_dict, "mc_s2") \
      .end_() \

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
