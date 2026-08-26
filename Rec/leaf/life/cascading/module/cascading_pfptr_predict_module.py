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
        .if_("enable_explore_mc_pfptr_predict == 1") \
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
        .else_() \
          .if_("enable_explore_mc_pftr_predict_v2 == 1") \
            .if_("enable_request_hot_fountain_model == 1") \
              .explore_common_user_feature_enricher(
                user_info_attr = "user_info_ptr",
                user_uid_attr = "uId",
                user_did_attr = "dId",
                user_gender_attr = "uGender",
                user_basic_age_attr = "uAgeSeg",
                user_ori_city_attr = "uCityId",
                user_client_id_attr = "uClientId",
                user_visit_mod_attr =  "uMod",
                user_visit_net_attr = "uNetwork",
                user_level_attr = "uUserLevel",
                user_active_days_attr = "uActiveDays",
                user_follow_cnt_attr = "uFollowCount",
                user_fans_cnt_attr = "uFansCount",
                user_upload_cnt_attr = "uUploadCount",
                user_province_attr = "uRequstProvinceId",
                user_city_attr = "uRequstCityId",
                user_request_poi_type_attr = "uRequestPoiType",
                user_request_city_level_attr = "uRequestCityLevel",
                user_request_community_type_attr = "uRequestCommuityType",
                user_region_attr = "uRegion",

                user_click_pids_attr = "uClickPids",
                user_like_pids_attr = "uLikePids",
                user_follow_pids_attr = "uFollowAids",
                user_profilev1_click_pids_attr = "uProfileV1ClickPidList",
                user_profilev1_click_aids_attr = "uProfileV1ClickAidList",
                user_profilev1_like_pids_attr = "uProfileV1LikePidList",
                user_profilev1_like_aids_attr = "uProfileV1LikeAidList",
                user_profilev1_follow_pids_attr = "uProfileV1FollowPidList",
                user_profilev1_follow_aids_attr = "uProfileV1FollowAidList",
                user_profilev1_profile_enter_pids_attr = "uProfileV1ProfileEnterPidList",
                user_profilev1_profile_enter_aids_attr = "uProfileV1ProfileEnterAidList",
                user_profilev1_play7s_pids_attr = "uProfileV1Play7SPidList",
                user_profilev1_play7s_aids_attr = "uProfileV1Play7SAidList",
                user_profilev1_play18s_pids_attr = "uProfileV1Play18SPidList",
                user_profilev1_play18s_aids_attr = "uProfileV1Play18SAidList",
                ft_click_pids_attr = "uClickPidsFountain",
                ft_click_aids_attr = "uClickAidsFountain",
                ft_like_pids_attr = "uLikePidsFountain",
                ft_like_aids_attr = "uLikeAidsFountain",
                ft_follow_pids_attr = "uFollowPidsFountain",
                ft_follow_aids_attr = "uFollowAidsFountain",
                ft_sv_pids_attr = "uShortViewPidsFountain",
                ft_sv_aids_attr = "uShortViewAidsFountain",
                ft_ev_pids_attr = "uEffViewPidsFountain",
                ft_ev_aids_attr = "uEffViewAidsFountain",
                ft_lv_pids_attr = "uLongViewPidsFountain",
                ft_lv_aids_attr = "uLongViewAidsFountain",
              ) \
              .gen_common_attr_by_lua(
                attr_map = {
                  "featureTab": "0",
                },
              ) \
              .delegate_enrich(
                kess_service = "{{mc_tower_pftr_service_v2}}",
                recv_item_attrs = [
                  {"name": "hot_finish_rate", "as": "ori_mc_ftr"},
                ],
                timeout_ms = 50,
                send_item_attrs = [
                  {"name": "photo_id", "as": "item_id"},
                ],
                send_common_attrs = simple_user_feture(),
                request_type = "{{mc_tower_pftr_request_type_v2}}"
              ) \
            .else_() \
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
                kess_service = "{{mc_tower_pftr_service_v2}}",
                recv_item_attrs = [
                  {"name": "ptr", "as": "ori_mc_ftr"},
                ],
                timeout_ms = 50,
                send_item_attrs = [
                  {"name": "photo_id", "as": "item_id"},
                ],
                send_common_attrs = self.consume_time_user_feture(),
                request_type = "{{mc_tower_pftr_request_type_v2}}"
              ) \
            .end_if_() \
            .explore_memory_data_enrich(
              data_key = "{{explore_cascade_pftr_debias_map}}",
              data_type = "string_string_map",
              save_data_ptr_to_attr = "cascade_pftr_debias_map_ptr",
            ) \
            .explore_trans_fintr_enricher(
              enable_transfer_sigmoid = "{{enable_transfer_sigmoid}}",
              get_fintr_quantile_mode = "{{explore_mc_enable_get_fintr_quantile_mode}}",
              fintr_debias_map_attr = "cascade_pftr_debias_map_ptr",
              fintr_redis_key_prefix = "{{explore_ftr_redis_key_prefix}}",
              fintr_short_photo_cluster_dist = "{{ftr_short_photo_cluster_dist}}",
              fintr_long_photo_threshold = "{{ftr_long_photo_threshold}}",
              fintr_long_photo_cluster_dist = "{{ftr_long_photo_cluster_dist}}",
              max_fintr_limit = "{{max_mc_ftr_limit}}",
              fintr_dist_reciprocal = "{{explore_mc_ftr_dist_reciprocal}}",
              enable_map_fintr_positive = "{{enable_map_ftr_positive}}",
              enable_multi_duration = "{{enable_multi_duration}}",
              fintr_duration_max_value = "{{explore_ftr_duration_max_value}}",
              fintr_duration_power_weight = "{{ftr_duration_power_weight}}",
              fintr_duration_offset = "{{ftr_duration_offset}}",
              duration_ms_attr = "duration_ms",
              fintr_attr = "ori_mc_ftr",
              save_fintr_quantile_to_attr = "cascade_pfptr"
            ) \
          .end_() \
        .end_() \
        
        


    def post_process(self) -> None:
      self.flow \
        .log_debug_info(
          item_attrs = ["duration_ms", "ori_mc_ftr", "cascade_pfptr"],
          for_debug_request_only = True,
          item_num_limit = 500
        )

