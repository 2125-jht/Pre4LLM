from cascading import CommonModule

class CascadingOwtdPredictModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def consume_time_user_feture(self):
    features = [
      "uId",
      "dId",
      "uLikePids",
      "uFollowAids",
      "uClickPidsV1",
      "uFollowCount",
      "uFansCount",
      "uUploadCount",
      "uUploadRate",
      "uCityId",
      "uProvinceId",
      "uGender",
      "uTrueGender",
      "uInferYear",
      "uTrueYear",
      "uBasicAge",
      "uNetwork",
      "uExpCtr",
      "uExpLtr",
      "uExpWtr",
      "uExpFtr",
      "uExpLvtr",
      "uExpSvtr",
      "uAvgWatchTime",
    ]

    for key in ["uHotShow", "uHotClick", "uHotLike", "uHotFollow", "uHotHate"]:
        for suffix in ["5m", "1h", "1d"]:
            features.append(key + suffix)
    for key in ["1", "2", "3"]:
        features.append("uLongTermHetuLevel" + key + "topN")
    
    features.extend([
      "uClickPidsV1Hetu1",
      "uClickPidsV1Hetu2",
      "uLikePidsV1Hetu1",
      "uLikePidsV1Hetu2",
      "ufollowAidsV1Hetu1",
      "ufollowAidsV1Hetu2",
      "uRealShowNoActionPids",
      "uRealShowNoActionAids",
      "uRealshowNoActionHetu1",
      "uRealshowNoActionHetu2",
      "uRealshowNoActionHetu3",
      "uRealshowNoActionHetuTag",
      "uViewPidListV1",
      "uViewAidListV1",
      "uEffectiveViewLabelListV1",
      "uLongViewLabelListV1",
      "uShortViewLabelListV1",
      "uShortViewPidListV1",
      "uShortViewAidListV1",
      "uEffectiveViewPidListV1",
      "uEffectiveViewAidListV1",
      "uLongViewPidListV1",
      "uLongViewAidListV1",
      "uFinishViewPidListV1",
      "uFinishViewAidListV1",
      "uFinishViewHetu1ListV1",
      "uFinishViewHetu2ListV1",
      "uNonFinishViewPidListV1",
      "uNonFinishViewAidListV1",
      "uViewHetu1ListV1",
      "uViewHetu2ListV1",

      {"name": "user_emp_ltr", "as": "uColossusEmpLtr"},
      {"name": "user_emp_wtr", "as": "uColossusEmpWtr"},
      {"name": "user_emp_ftr", "as": "uColossusEmpFtr"},
      {"name": "user_emp_cmtr", "as": "uColossusEmpCmtr"},
      {"name": "user_emp_eptr", "as": "uColossusEmpPtr"},
      {"name": "user_emp_svtr", "as": "uColossusEmpSvtr"},
      {"name": "user_emp_evtr", "as": "uColossusEmpEvtr"},
      {"name": "user_emp_lvtr", "as": "uColossusEmpLvtr"},
      {"name": "user_emp_fintr", "as": "uColossusEmpFintr"},
      {"name": "user_emp_watch_time", "as": "uColossusAvgWatchTime"},
      {"name": "user_emp_finish_rate", "as": "uColossusAvgFinishRate"},
    ])
    
    return features
  

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_mc_owtd_predict == 1") \
        .explore_common_user_feature_enricher(
          user_info_attr = "user_info_ptr",
          user_uid_attr = "uId",
          user_did_attr = "dId",
          user_like_pids_attr = "uLikePids",
          user_profilev1_follow_aids_attr = "uFollowAids",
          user_follow_cnt_attr = "uFollowCount",
          user_fans_cnt_attr = "uFansCount",
          user_upload_cnt_attr = "uUploadCount",
          user_upload_rate_attr = "uUploadRate",
          user_city_attr = "uCityId",
          user_province_attr = "uProvinceId",
          user_gender_attr = "uGender",
          user_true_gender_attr = "uTrueGender",
          user_infer_year_attr = "uInferYear",
          user_true_year_attr = "uTrueYear",
          user_basic_age_attr = "uBasicAge",
          user_visit_net_attr = "uNetwork",
          user_longterm_hetu_level1_attr = "uLongTermHetuLevel1topN",
          user_longterm_hetu_level2_attr = "uLongTermHetuLevel2topN",
          user_longterm_hetu_level3_attr = "uLongTermHetuLevel3topN",
          user_count_action_attr = "uCount",
          user_exp_ctr_attr = "uExpCtr",
          user_exp_ltr_attr = "uExpLtr",
          user_exp_wtr_attr = "uExpWtr",
          user_exp_ftr_attr = "uExpFtr",
          user_exp_lvtr_attr = "uExpLvtr",
          user_exp_svtr_attr = "uExpSvtr",
          user_exp_avg_watchtime_attr = "uAvgWatchTime",
          # no action
          user_no_action_attr =  "uNoAction",
          user_no_action_pid_attr = "uRealShowNoActionPids",
          user_no_action_aid_attr =  "uRealShowNoActionAids",
          user_no_action_hetu1_attr = "uRealshowNoActionHetu1",
          user_no_action_hetu2_attr =  "uRealshowNoActionHetu2",
          user_no_action_hetu3_attr =  "uRealshowNoActionHetu3",
          user_no_action_hetu_tag_attr =  "uRealshowNoActionHetuTag",
          # action_list_long_version
          user_action_list_long_version_attr = "uActionListLongVersion",
          user_click_pids_v1_attr = "uClickPidsV1",
          user_click_pids_hetu1_attr = "uClickPidsV1Hetu1",
          user_click_pids_hetu2_attr = "uClickPidsV1Hetu2",
          user_like_pids_hetu1_attr =  "uLikePidsV1Hetu1",
          user_like_pids_hetu2_attr = "uLikePidsV1Hetu2",
          user_follow_pids_hetu1_attr = "ufollowAidsV1Hetu1",
          user_follow_pids_hetu2_attr = "ufollowAidsV1Hetu2",

          user_view_pids_attr = "uViewPidListV1",
          user_view_aids_attr = "uViewAidListV1",
          user_effective_view_label_attr = "uEffectiveViewLabelListV1",
          user_long_view_label_attr = "uLongViewLabelListV1",
          user_short_view_label_attr = "uShortViewLabelListV1",

          user_short_view_pids_attr = "uShortViewPidListV1",
          user_short_view_aids_attr = "uShortViewAidListV1",
          user_effective_view_pids_attr = "uEffectiveViewPidListV1",
          user_effective_view_aids_attr = "uEffectiveViewAidListV1",
          user_long_view_pids_attr = "uLongViewPidListV1",
          user_long_view_aids_attr = "uLongViewAidListV1",
          user_finish_view_pids_attr = "uFinishViewPidListV1",
          user_finish_view_aids_attr = "uFinishViewAidListV1",
          user_finish_view_hetu1_attr = "uFinishViewHetu1ListV1",
          user_finish_view_hetu2_attr = "uFinishViewHetu2ListV1",
          user_non_finish_view_pids_attr = "uNonFinishViewPidListV1",
          user_non_finish_view_aids_attr = "uNonFinishViewAidListV1",
          user_view_hetu1_attr = "uViewHetu1ListV1",
          user_view_hetu2_attr = "uViewHetu2ListV1"
        ) \
        .delegate_enrich(
          kess_service = "{{mc_tower_owtd_service}}",
          recv_item_attrs = ["owtd_label{}".format(i) for i in range(1,4)],
          timeout_ms = 50,
          send_item_attrs = [
            {"name": "photo_id", "as": "item_id"},
          ],
          send_common_attrs = self.consume_time_user_feture(),
          request_type = "{{mc_tower_owtd_request_type}}"
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "explore_mc_owtd_quantile_num",
            "explore_mc_owtd_label_threshold",
            "explore_mc_owtd_duration_max_limit",
            "explore_mc_owtd_duration_power_weight",
            "explore_mc_owtd_prob_duration_power_weight",
            "explore_mc_ordinal_duration_list",
            "explore_mc_ordinal_playtime_dist_list"
          ],
          import_item_attr = [
            "duration_ms",
            "owtd_label1",
            "owtd_label2",
            "owtd_label3"
          ],
          export_item_attr = [
            "cascade_ordinal_prob",
            "cascade_ordinal_wtd"
          ],
          function_name = "CalOrdinalWtdScore",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_() \

      
      
  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        item_attrs = [
            "duration_ms",
            "owtd_label1",
            "owtd_label2",
            "owtd_label3",
            "cascade_ordinal_prob",
            "cascade_ordinal_wtd"
        ],
        common_attrs = [
          "explore_mc_ordinal_duration_list",
          "explore_mc_ordinal_playtime_dist_list"
        ],
        for_debug_request_only = True,
        item_num_limit = 500
      )

