from cascading import CommonModule

class CascadingDistillLtrPredictModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def user_feature(self):
    features = [
      "uId",
      "dId",
      "uHourOfDay",
      "uDayOfWeek",
      "uExpFollowCount",
      "uFansCount",
      "uClickPids",
      "uLikePids",
      "uFollowAids",
      {"name": "uRealtimeForwardList", "as": "uForwardPids"},
      {"name": "uRealtimeNegativeList", "as": "uNegtivePids"},
      "uUploadCount",
      "uCityLevelNew",
      "uCityId",
      "uProvinceId",
      "uGender",
      "uTrueYear",
      "uBasicAge",
      "uNetwork",
      "uVisitMod",
      {"name": "refreshTimes", "as": "uRefreshTimes"}
    ]

    return features

  def photo_features(self):
    features = [
      {"name": "author__id", "as": "aId"},

      {"name": "duration_ms", "as": "pDurationMs"},
      {"name": "upload_type", "as": "pUploadType"},
      {"name": "location__city_id", "as": "pCityId"},
      {"name": "location__province_id", "as": "pProvinceId"},

      {"name": "hetu_tag_level_info__hetu_level_one", "as": "pHetuTagLevel1List"},
      {"name": "hetu_tag_level_info__hetu_level_two", "as": "pHetuTagLevel2List"},
      {"name": "hetu_tag_level_info__hetu_level_three", "as": "pHetuTagLevel3List"},

      {"name": "cascade_pctr", "as": "pMcPctr"},
      {"name": "cascade_pltr", "as": "pMcPltr"},
      {"name": "cascade_pwtr", "as": "pMcPwtr"},
      {"name": "cascade_plvtr", "as": "pMcPlvtr"},
      {"name": "cascade_psvtr", "as": "pMcPsvtr"},
      {"name": "cascade_pftr", "as": "pMcPftr"},
      {"name": "cascade_pcmtr", "as": "pMcPcmtr"},
      {"name": "cascade_plvtr2", "as": "pMcPlvtr2"},
      {"name": "cascade_pepstr", "as": "pMcPepstr"},
      {"name": "cascade_pcltr", "as": "pMcPcltr"},
      {"name": "cascade_phtr", "as": "pMcPhtr"},
      {"name": "cascade_pwatch_time", "as": "pMcPwatchTime"},
    ]
    return features
          
  def process(self) -> None:
    self.flow \
      .if_("enable_explore_mc_distill_ltr_predict == 1") \
        .explore_common_user_feature_enricher(
          user_info_attr = "user_info_ptr",
          user_uid_attr = "uId",
          user_did_attr = "dId",
          context_hour_of_day_attr = "uHourOfDay",
          context_day_of_week_attr = "uDayOfWeek",
          user_exp_follow_count_attr = "uExpFollowCount",
          user_fans_cnt_attr = "uFansCount",
          user_upload_cnt_attr = "uUploadCount",
          user_click_pids_attr = "uClickPids",
          user_like_pids_attr = "uLikePids",
          user_follow_pids_attr = "uFollowAids",
          user_city_level_attr = "uCityLevelNew",
          user_city_attr = "uCityId",
          user_province_attr = "uProvinceId",
          user_gender_attr = "uGender",
          user_true_year_attr = "uTrueYear",
          user_basic_age_attr = "uBasicAge",
          user_visit_net_attr = "uNetwork",
          user_visit_mod_attr =  "uVisitMod",
        ) \
        .delegate_enrich(
          kess_service = "{{mc_distill_ltr_service}}",
          recv_item_attrs = [
            {"name": "ctr", "as": "cascade_dstill_pctr"},
            {"name": "ltr", "as": "cascade_dstill_pltr"},
            {"name": "lvtr", "as": "cascade_dstill_plvtr"},
            {"name": "watch_time", "as": "cascade_dstill_pwatch_time"}
          ],
          timeout_ms = 100,
          send_item_attrs = self.photo_features(),
          send_common_attrs = self.user_feature(),
          request_type = "{{mc_distill_ltr_request_type}}",
        ) \
      .end_() \
