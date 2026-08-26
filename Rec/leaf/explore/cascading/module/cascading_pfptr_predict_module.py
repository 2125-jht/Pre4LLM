from cascading import CommonModule
from cascading.module.cascading_features import *

class CascadingPfptrPredictModule(CommonModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def user_feature(self):
      features = [
        "uId",
        "dId",
        "uGender",
        {"name": "uInferGenderInt", "as": "uInferGender"},
        {"name": "uTrueGenderInt", "as": "uTrueGender"},
        {"name": "uInferYearInt", "as": "uInferYear"},
        {"name": "uTrueYearInt", "as": "uTrueYear"},
        "uAgeSeg",
        "uMod",
        "uNetwork",
        "uClientId",
        "uVisitChannel",
        "hourOfDay",
        "dayOfWeek",
        {"name": "mc_wtd_uProvinceId", "as": "uProvinceId"},
        {"name": "mc_wtd_uCityId", "as": "uCityId"},
        "uFreqProvinceId",
        "uFreqCityId",
        "uGeoInfo",
        "uRequstProvinceId",
        "uRequstCityId",
        "uRequstPoi",
        "uRequestPoiType",
        "uRequestTown",
        "uRequestCityLevel",
        "uRequestCommuityType",
        "uRegion",
        "uFreRegion",
        {"name": "mc_wtd_uFollowCount", "as": "uFollowCount"},
        "uUploadCount",
        "uFansCount",
        "uCtr",
        "uLtr",
        "uWtr",
        "uFtr",
        "uLvtr",
        "uSvtr",
        "uAvgWatchTime",
        "uProfileV1ClickPidList",
        "uProfileV1ClickAidList",
        "uProfileV1LikePidList",
        "uProfileV1LikeAidList",
        "uProfileV1FollowPidList",
        "uProfileV1FollowAidList",
        "uProfileV1CommentPidList",
        "uProfileV1CommentAidList",
        "uProfileV1ProfileEnterPidList",
        "uProfileV1ProfileEnterAidList",
        "uProfileV1Play7SPidList",
        "uProfileV1Play18SPidList",
        "uProfileV1Play7SAidList",
        "uProfileV1Play18SAidList",
        "uLongTermHetuLevel1topN",
        "uLongTermHetuLevel2topN",
        "uAppNewCat1",
        "uAppNewCat2",
        "uAppNewCat3",
        "uAppNormNames"
      ]
      return features
    
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
        .if_("enable_explore_mc_pfptr_predict == 1", to_be_delete = "date=2023-11-16;committer=liuhao07") \
          .explore_common_user_feature_enricher(
            user_info_attr = "user_info_ptr",
            user_uid_attr = "uId",
            user_did_attr = "dId",
            user_gender_attr = "uGender",
            user_infer_gender_int_attr = "uInferGenderInt",
            user_true_gender_int_attr = "uTrueGenderInt",
            user_infer_year_int_attr = "uInferYearInt",
            user_true_year_int_attr = "uTrueYearInt",
            user_basic_age_attr = "uAgeSeg",
            user_visit_mod_attr =  "uMod",
            user_visit_net_attr = "uNetwork",
            user_client_id_attr = "uClientId",
            user_visit_channel_attr = "uVisitChannel",
            context_hour_of_day_attr = "hourOfDay",
            context_day_of_week_attr = "dayOfWeek",
            user_ori_province_attr = "mc_wtd_uProvinceId",
            user_ori_city_attr = "mc_wtd_uCityId",
            user_freq_province_attr = "uFreqProvinceId",
            user_freq_city_attr = "uFreqCityId",
            user_geo_info_attr = "uGeoInfo",
            user_province_attr = "uRequstProvinceId",
            user_city_attr = "uRequstCityId",
            user_request_poi_attr = "uRequstPoi",
            user_request_poi_type_attr = "uRequestPoiType",
            user_request_town_attr = "uRequestTown",
            user_request_city_level_attr = "uRequestCityLevel",
            user_request_community_type_attr = "uRequestCommuityType",
            user_region_attr = "uRegion",
            user_freq_region_attr = "uFreRegion",
            user_exp_follow_count_attr = "mc_wtd_uFollowCount",
            user_upload_cnt_attr = "uUploadCount",
            user_fans_cnt_attr = "uFansCount",
            user_exp_ctr_attr = "uCtr",
            user_exp_ltr_attr = "uLtr",
            user_exp_wtr_attr = "uWtr",
            user_exp_ftr_attr = "uFtr",
            user_exp_lvtr_attr = "uLvtr",
            user_exp_svtr_attr = "uSvtr",
            user_exp_avg_watchtime_attr = "uAvgWatchTime",
            user_profilev1_click_pids_attr = "uProfileV1ClickPidList",
            user_profilev1_click_aids_attr = "uProfileV1ClickAidList",
            user_profilev1_like_pids_attr = "uProfileV1LikePidList",
            user_profilev1_like_aids_attr = "uProfileV1LikeAidList",
            user_profilev1_follow_pids_attr = "uProfileV1FollowPidList",
            user_profilev1_follow_aids_attr = "uProfileV1FollowAidList",
            user_profilev1_comment_pids_attr = "uProfileV1CommentPidList",
            user_profilev1_comment_aids_attr = "uProfileV1CommentAidList",
            user_profilev1_profile_enter_pids_attr = "uProfileV1ProfileEnterPidList",
            user_profilev1_profile_enter_aids_attr = "uProfileV1ProfileEnterAidList",
            user_profilev1_play7s_pids_attr = "uProfileV1Play7SPidList",
            user_profilev1_play7s_aids_attr = "uProfileV1Play7SAidList",
            user_profilev1_play18s_pids_attr = "uProfileV1Play18SPidList",
            user_profilev1_play18s_aids_attr = "uProfileV1Play18SAidList",
            user_longterm_hetu_level1_attr = "uLongTermHetuLevel1topN",
            user_longterm_hetu_level2_attr = "uLongTermHetuLevel2topN",
            user_app_cate1_attr = "uAppNewCat1",
            user_app_cate2_attr = "uAppNewCat2",
            user_app_cate3_attr = "uAppNewCat3",
            user_app_norm_name_attr = "uAppNormNames"
          ) \
          .delegate_enrich(
            kess_service = "{{mc_tower_pftr_service}}",
            recv_item_attrs = [
              {"name": "explore_wtd", "as": "cascade_pfptr"},
            ],
            timeout_ms = 50,
            send_item_attrs = [
              {"name": "photo_id", "as": "item_id"},
            ],
            send_common_attrs = self.user_feature(),
            request_type = "{{mc_tower_pftr_request_type}}",
          ) \
        .end_() \
        
        


    def post_process(self) -> None:
      self.flow \
        .log_debug_info(
          item_attrs = ["duration_ms", "ori_mc_ftr", "cascade_pfptr"],
          for_debug_request_only = True,
          item_num_limit = 500
        )

