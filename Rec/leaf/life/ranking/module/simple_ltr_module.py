from ranking import CommonModule

class SimpleLtrModule(CommonModule):
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
        "uEnterProfilePhotoList",
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
        "uCollectPids"
      ]

      for i in range(30):
          for suffix in ["", "aid_", "tag_", "play_"]:
              features.append("realshow_" + suffix + str(i))
      
      for key in ["uHotShow", "uHotClick", "uHotLike", "uHotFollow", "uHotHate", "uHotForward", "uHotCollect"]:
          for suffix in ["1m", "5m", "30m", "1h", "1d", "100n", "1000n"]:
              features.append(key + suffix)
      
      return features

    def photo_features(self):
        features = [
          {"name": "photo_id", "as": "pId"},
          {"name": "author__id", "as": "aId"},
          {"name": "corr_pctr", "as": "pPctr"},
          {"name": "pltr", "as" : "pPltr"},
          {"name": "pwtr", "as" : "pPwtr"},
          {"name": "pftr", "as" : "pPftr"},
          {"name": "phtr", "as": "pPhtr"},
          {"name": "plvtr", "as": "pPlvtr"},
          {"name": "psvr", "as": "pPsvtr"},
          {"name": "pptr", "as" : "pPptr"},
          {"name": "pcmtr", "as": "pPcmtr"},
          {"name": "pcmef", "as" : "pPcmef"},
          {"name": "pepstr", "as": "pPepstr"},
          {"name": "pdtr", "as" : "pPdtr"},
          {"name": "fr_score1", "as" : "pPfrScore1"},
          {"name": "fr_score2", "as" : "pPfrScore2"},
          {"name": "fetr", "as" : "pPfetr"},
          {"name": "fountain_eff", "as": "pPfountainEff"},
          {"name": "cascade_pctr", "as": "pMcPctr"},
          {"name": "cascade_pltr", "as": "pMcPltr"},
          {"name": "cascade_pwtr", "as" : "pMcPwtr"},
          {"name": "cascade_plvtr", "as": "pMcPlvtr"},
          {"name": "cascade_psvtr", "as": "pMcPsvtr"},
          {"name": "cascade_pftr", "as" : "pMcPftr"},
          {"name": "cascade_pcmtr", "as" : "pMcPcmtr"},
          {"name": "cascade_plvtr2", "as": "pMcPlvtr2"},
          {"name": "cascade_pepstr", "as" : "pMcPepstr"},
          {"name": "cascade_pcestr", "as": "pMcPcestr"},
          {"name": "cascade_pwatch_time", "as" : "pMcPwatchTime"},
          {"name": "cascade_ptr", "as": "pMcPptr"},
          {"name": "empirical_ctr", "as": "pEmpCtr"},
          {"name": "empirical_ltr", "as": "pEmpLtr"},
          {"name": "empirical_wtr", "as": "pEmpWtr"},
          {"name": "empirical_ftr", "as": "pEmpFtr"},
          {"name": "empirical_ptr", "as": "pEmpPtr"},
          {"name": "empirical_cmtr", "as": "pEmpCmtr"},
          {"name": "empirical_htr", "as": "pEmpHtr"},
          {"name": "empirical_watch_time", "as": "pEmpWatchTime"},
          {"name": "author__fans_count", "as": "aFansCount"},
          {"name": "photoAgeHour", "as": "pAgeHour"},
          {"name": "duration_ms", "as": "pDurationMs"},
          {"name": "upload_type", "as": "pUploadType"},
          {"name": "explore_stat__show_count", "as": "pHotShow"},
          {"name": "explore_stat__click_count", "as": "pHotClick"},
          {"name": "explore_stat__like_count", "as": "pHotLike"},
          {"name": "explore_stat__follow_count", "as": "pHotFollow"},
          {"name": "explore_stat__negative_count", "as": "pHotHate"},
          {"name": "explore_stat__report_detail__total_report_count", "as": "pHotReport"},
          {"name": "click_upload_rate", "as": "pUploadRate"},
          {"name": "location__city_id", "as": "pCityId"},
          {"name": "location__province_id", "as": "pProvinceId"},
          {"name": "photo_dnn_cluster_id", "as": "pDnnClusterId"},
          {"name": "music", "as": "pMusic"},
          {"name": "hetu_tag_level_info__hetu_level_one", "as": "pHetuTagLevel1"},
          {"name": "hetu_tag_level_info__hetu_level_two", "as": "pHetuTagLevel2"},
          {"name": "hetu_tag_level_info__hetu_level_three", "as": "pHetuTagLevel3"},
          {"name": "hetu_tag_level_info__hetu_tag", "as": "pHetuTagLevelTag"},
          {"name": "mmu_img_cluster_v1", "as": "pMmuImgClusterV1"},
          {"name": "mmu_img_cluster_v3", "as": "pMmuImgClusterV3"},
          {"name": "reason", "as": "reason"}
        ]
        return features

    def log_photo_features(self):
      ans = []
      for item in self.photo_features():
        ans.append(item['name'])
      return ans

    def process(self) -> None:
        self.flow \
        .if_("enable_hot_deep_ltr_predict == 1") \
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
            global_collect_pids_attr = "uCollectPids",
            user_follow_pids_attr = "uFollowAids",
            user_profile_pids_attr = "uEnterProfilePhotoList",
            user_upload_rate_attr = "uUploadRate",
            user_gender_attr = "uGender",
            user_infer_gender_attr = "uInferGender",
            user_ture_gender_attr = "uTrueGender",
            user_basic_gender_attr = "uBasicGender",
            user_infer_year_attr = "uInferYear",
            user_true_year_attr = "uTrueYear",
            user_basic_age_attr = "uBasicAge",
            user_realshow_action_attr = "realshow_",
            user_count_action_attr = "cnt_",
            user_app_attr = "uAppList",
          ) \
          .explore_common_item_feature_enricher(
            item_age_hour_attr = "photoAgeHour",
           ) \
          .set_default_value(
            no_overwrite=True,
            item_attrs=[
              {
                  "name": "corr_pctr",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "pltr",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "pwtr",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "pftr",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "plvtr",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "psvr",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "pcmtr",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "pptr",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "pcmef",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "phtr",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "pepstr",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "fr_score1",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "fr_score2",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "pdtr",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "fetr",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "fountain_eff",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "cascade_pctr",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "cascade_pltr",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "cascade_pwtr",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "cascade_plvtr",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "cascade_psvtr",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "cascade_pftr",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "cascade_pcmtr",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "cascade_plvtr2",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "cascade_pepstr",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "cascade_pcestr",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "cascade_ptr",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "cascade_pwatch_time",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "empirical_ctr",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "empirical_ltr",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "empirical_wtr",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "empirical_ftr",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "empirical_ptr",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "empirical_cmtr",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "empirical_htr",
                  "type": "float",
                  "value": 0.0
              },
              {
                  "name": "empirical_watch_time",
                  "type": "float",
                  "value": 0.0
              },
            ]
          ) \
          .delegate_enrich(
            kess_service = "{{hot_deep_ltr_kai_predict_service}}",
            recv_item_attrs = [
              {"name": "joint_loss", "as": "deep_ltr_score"},
            ],
            timeout_ms = 120,
            send_item_attrs = self.photo_features(),
            send_common_attrs = self.user_feature(),
            request_type = "simple_ltr_predict",
            partition_size = "{{deep_ltr_predict_partition_size}}",
          ) \
        .end_()

    def post_process(self) -> None:
      self.flow \
        .log_debug_info(
          common_attrs = [
            "user_uid_attr", "user_did_attr", "user_province_attr", "user_city_attr", "user_visit_mod_attr",
            "user_click_pids_attr", "user_upload_rate_attr", "user_true_new_attr", "user_login_attr",
            "user_gender_attr", "user_infer_gender_attr", "user_ture_gender_attr", "user_basic_gender_attr",
            "user_infer_year_attr", "user_true_year_attr", "user_basic_age_attr", "user_visit_net_attr",
            "user_realshow_action_attr", "user_count_action_attr", "user_like_pids_attr", "user_follow_pids_attr",
            "user_fans_cnt_attr", "user_follow_cnt_attr", "user_upload_cnt_attr", "user_risk_level_attr",
            "user_app_attr", "user_city_level_attr", "user_longview_action_attr", "user_shortview_action_attr",
            "user_is_douyin_attr"
          ],
          item_attrs = self.log_photo_features() + ["deep_ltr_score"],
          for_debug_request_only = True
        )
