from cascading import CommonModule

class CascadingAdaXtrModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def user_feature(self):
    features = [
      "uId",
      "dId",
      "uFollowCount",
      "uFansCount",
      "uUploadCount",
      "uUploadRate",
      "uRiskLevel",
      "uClientId",
      "uVisitMod",
      "uNetwork",
      "uClickPids",
      "uLikePids",
      "uFollowAids",
      "uCityId",
      "uProvinceId",
      "uGender",
      "uInferGender",
      "uTrueGender",
      "uBasicGender",
      "uInferYear",
      "uTrueYear",
      "uBasicAge",
      "uAppList",
    ]

    for i in range(30):
      for suffix in ["", "aid_", "tag_", "play_"]:
        features.append("longview_" + suffix + str(i))
    
    for i in range(30):
      for suffix in ["", "aid_", "tag_", "play_"]:
        features.append("shortview_" + suffix + str(i))
    
    for key in ["uHotShow", "uHotClick", "uHotLike", "uHotFollow", "uHotHate", "uHotCollect", "uHotForward"]:
      for suffix in ["1m", "5m", "30m", "1h", "1d", "100n", "1000n"]:
        features.append(key + suffix)
    
    return features

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_mc_ada_ltr == 1") \
        .explore_common_user_feature_enricher(
          user_info_attr = "user_info_ptr",
          user_uid_attr = "uId",
          user_did_attr = "dId",
          user_province_attr =  "uProvinceId",
          user_city_attr = "uCityId",
          user_visit_mod_attr =  "uVisitMod",
          user_visit_net_attr = "uNetwork",
          user_follow_cnt_attr = "uFollowCount",
          user_upload_cnt_attr = "uUploadCount",
          user_risk_level_attr = "uRiskLevel",
          user_fans_cnt_attr =   "uFansCount",
          user_click_pids_attr = "uClickPids",
          user_like_pids_attr = "uLikePids",
          user_follow_pids_attr = "uFollowAids",
          user_upload_rate_attr = "uUploadRate", 
          user_gender_attr = "uGender",
          user_infer_gender_attr = "uInferGender", 
          user_ture_gender_attr = "uTrueGender",
          user_basic_gender_attr = "uBasicGender",
          user_infer_year_attr = "uInferYear", 
          user_true_year_attr = "uTrueYear",
          user_basic_age_attr = "uBasicAge",
          user_longview_action_attr = "longview_",
          user_shortview_action_attr = "shortview_", 
          user_count_action_attr = "cnt_",
        ) \
        .get_kuiba_user_embedding(
          tensor_request_layer='joint_mc_ltr',
          kess_service="{{hot_mc_ada_ltr_predict_service}}",
          timeout_ms=20,
          input_common_attr=self.user_feature(),
          output_tensor_attr='mc_user_ada_weight_tensor',
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "mc_user_ada_weight_tensor",
          ],
          import_item_attr = [
            "cascade_pctr",
            "cascade_pltr",
            "cascade_pwtr",
            "cascade_pftr",
            "cascade_plvtr",
            "cascade_plvtr2",
            "cascade_ptr",
            "cascade_pwatch_time",
            "cascade_pepstr",
            "cascade_pcmtr",
            "cascade_pcestr",
            "cascade_pcltr"
          ],
          export_item_attr = [
            "mc_ada_xtr_score",
          ],
          function_name = "CalMcAdaXtrScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = [
          "mc_user_ada_weight_tensor", "longview_", "shortview_", "cnt_"
        ] + self.user_feature(),
        for_debug_request_only = True
      )
