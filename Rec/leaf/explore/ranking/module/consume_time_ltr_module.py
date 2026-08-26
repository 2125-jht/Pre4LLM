from ranking import CommonModule
from ranking.module.fetch_user_colossus_info_module import photo_colossus_features

class ConsumeTimeLtrModule(CommonModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def uni_feature_trim_user_info(self):
        features = [
            "id",
            "active_days",
            "basic_info.age_segment",
            "location.city_id",
            "location.region_type",
            "client_id",
            "device_id",
            "gender",
            "infer_gender",
            "true_gender",
            "request_location.poi_type",
            "request_location.province_id",
            "request_location.city_id",
            "visit_mod",
            "user_profile.exp_stat.exp_click",
            "user_profile.exp_stat.exp_like",
            "user_profile.exp_stat.exp_follow",
            "user_profile.exp_stat.exp_realshow",
            "user_profile.exp_stat.exp_long_view",
            "user_profile.user_level",
            "fountain_reco_user_profile.click_list.author_id",
            "fountain_reco_user_profile.click_list.photo_id",
            "fountain_reco_user_profile.comment_list.author_id",
            "fountain_reco_user_profile.comment_list.photo_id",
            "fountain_reco_user_profile.follow_list.author_id",
            "fountain_reco_user_profile.follow_list.photo_id",
            "fountain_reco_user_profile.like_list.author_id",
            "fountain_reco_user_profile.like_list.photo_id",
            "fountain_reco_user_profile.video_play_stat.photo_id",
            "fountain_reco_user_profile.video_play_stat.author_id",
            "fountain_reco_user_profile.video_play_stat.video_duration",
            "fountain_reco_user_profile.video_play_stat.playing_time",
            "user_profile_v1.click_list.author_id",
            "user_profile_v1.click_list.photo_id",
            "user_profile_v1.follow_list.author_id",
            "user_profile_v1.follow_list.photo_id",
            "user_profile_v1.like_list.author_id",
            "user_profile_v1.like_list.photo_id",
            "user_profile_v1.video_playing_stat.playing_time",
            "user_profile_v1.video_playing_stat.author_id",
            "user_profile_v1.video_playing_stat.photo_id",
            "realtime_click_list",
            "realtime_follow_list",
            "realtime_forward_list",
            "realtime_like_list",
            "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_level_one",
            "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_level_two",
            "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_level_three",
            "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_level_five",
            "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_tag",
            "user_profile_v1.click_list.hetu_tag_level_info.hetu_level_one",
            "user_profile_v1.click_list.hetu_tag_level_info.hetu_level_two",
            "user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_one",
            "user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_two",
            "user_profile_v1.like_list.hetu_tag_level_info.hetu_level_one",
            "user_profile_v1.like_list.hetu_tag_level_info.hetu_level_two",
            "user_profile_v1.video_playing_stat.video_duration",
            "user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_one",
            "user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_two",
            "upload_count",
            "infer_year",
            "follow_count",
            "fans_count",
            "visit_net",
            "location.city_level",
            "is_douyin",
            "user_profile_v1.real_show_list.photo_id",
            "user_profile_v1.real_show_list.author_id",
            "user_profile_v1.real_show_list.time_ms",
            "user_profile_v1.real_show_list.page_type",
            "user_profile_v1.real_show_list.label.click",
            "user_profile_v1.real_show_list.label.like",
            "user_profile_v1.real_show_list.label.follow",
            "user_profile_v1.real_show_list.label.hate",
            "feature_collection.explore_low_active_level",
            "user_interest_profile.hetu_level_one_long_term_id",
            "user_interest_profile.hetu_level_one_long_term_score",
            "user_interest_profile.hetu_level_two_long_term_id",
            "user_interest_profile.hetu_level_two_long_term_score",
            "user_interest_profile.hetu_level_three_long_term_id",
            "user_interest_profile.hetu_level_three_long_term_score",
        ]

        return features

    def uni_feature_context_into(self):
      features = [
        "reason",
        "pctr",
        "pltr",
        "pwtr",
        "pftr",
        "phtr",
        "plvtr",
        "psvr",
        "pvtr",
        "awesome_wtd",
        "pptr",
        "pcmtr",
        "pcmef",
        "fr_score1",
        "fr_score2",
        "pdtr",
        "cascade_pctr",
        "cascade_pltr",
        "cascade_pwtr",
        "cascade_plvtr",
        "fetr",
        "fountain_eff",
        "pepstr",
        "pctr_index",
        "plvtr_index",
        "pvtr_index",
        "pltr_index",
        "pftr_index",
        "pwtr_index",
        "pesptr_index",
        "psvr_index",
        "pPctr",
        "pPltr",
        "pPwtr",
        "pPftr",
        "pPhtr",
        "pPlvtr",
        "pPsvtr",
        "pPvtr",
        "pPptr",
        "pPcmtr",
        "pPcmef",
        "pPfrScore1",
        "pPfrScore2",
        "pPdtr",
        "pMcPctr",
        "pMcPltr",
        "pMcPwtr",
        "pMcPlvtr",
        "pPfetr",
        "pPfountainEff",
        "bad_cover_similary_score"
      ]

      return features

    def user_feture(self):
      features = [
        "uId",
        "dId",
        "uTrueNewUser",
        "uLogin",
        "uFollowCount",
        "uFansCount",
        "uUploadCount",
        "uUploadRate",
        "uRiskLevel",
        "uNebula",
        "uClientId",
        "uVisitMod",
        "uVisitNet",
        "uVisitChannel",
        "uClickPids",
        "uLikePids",
        "uFollowAids",
        "uCityId",
        "uProvinceId",
        "uGender",
        "uInferGender",
        "uTrueGender",
        "uBasicGender",
        "uPromGender",
        "uInferYear",
        "uTrueYear",
        "uBasicAge",
        "hourOfDay",
        "dayOfWeek",
        "uAppList",
        "uCityLevelNew",
        "uIsDouyin",
        "uIsLowActiveUser",
        "uFindUserActiveDegree",
        "uExploreActiveDays",
        "uIsExploreLaUser"
      ]

      for key in ["uHotShow", "uHotClick", "uHotLike", "uHotFollow", "uHotHate"]:
          for suffix in ["5m", "1h", "1d", "100n", "1000n"]:
              features.append(key + suffix)
      for key in ["1", "2", "3"]:
          features.append("uLongTermHetuLevel" + key + "topN")
          features.append("uLongTermHetuLevel" + key + "Legal")
      
      features.extend([
        "uClickPidsV1",
        "uClickPidsV1Hetu1",
        "uClickPidsV1Hetu2",
        "uLikePidsV1Hetu1",
        "uLikePidsV1Hetu2",
        "ufollowAidsV1Hetu1", # lower case f
        "ufollowAidsV1Hetu2",
        "uViewPidListV1_LEN",
        "uViewPidListV1",
        "uViewAidListV1",
        "uEffectiveViewLabelListV1",
        "uLongViewLabelListV1",
        "uShortViewLabelListV1",
        "uViewHetu1ListV1",
        "uViewHetu2ListV1",
        "uNetwork",
        "uLongTermV4Hetu1",
        "uLongTermV4Hetu2",
        "uLongTermV4Hetu3",
        "uRealShowNoActionPids",
        "uRealShowNoActionPids_LEN",
        "uRealShowNoActionAids",
        "uRealshowNoActionHetu1", # lower case: s
        "uRealshowNoActionHetu2",
        "uRealshowNoActionHetu3",
        "uRealshowNoActionHetuTag"
      ])
      
      return features
    
    def gen_distill_user_features(self):
      features = [
        "uId",
        "dId",
        "uGender",
        "uAge",
        "uAgeSeg",
        "uBasicAge",
        "uBasicGender",
        "uProvinceId",
        "uCityId",
        "uMod",
        "uExpWatchTime",
        "uExpClick",
        "uExpLike",
        "uExpFollow",
        "uExpLongView",
        "uRequstProvinceId",
        "uRequstCityId",
        "uRealtimeClickList",
        "uRealtimeLikeList",
        "uRealtimeFollowList",
        "uRealtimeForwardList",
        "uRealtimeNegativeList",
        "uLikePhotoAuthorList",
        "uFollowPhotoAuthorList",
        "uRequestHour",
        "uRequestWeekday",
        "uViewPidListV1",
        "uViewAidListV1",
        "uEffectiveViewLabelListV1",
        "uLongViewLabelListV1",
        "uShortViewLabelListV1",
        "uViewHetu1ListV1",
        "uViewHetu2ListV1",
      ]

      for prefix in ['uPlayTimeFromNow', 'uDuration', 'uPlayTime', 'uActionLabel', 'uPlayActionLabel']:
        for suffix in ["1m", "5m", "10m", "30m", "1h", "2h"]:
          features.append(prefix + suffix)

      return features
    
    def gen_distill_photo_features(self):
      rerank_features_gen = [
      # 精排预测结果
        {"name": "pctr",  "as": "pPctr_idx0"},
        {"name": "pltr",  "as": "pPltr_idx0"},
        {"name": "pwtr",  "as": "pPwtr_idx0"},
        {"name": "pftr",  "as": "pPftr_idx0"},
        {"name": "phtr",  "as": "pPhtr_idx0"},
        {"name": "pvtr",  "as": "pPvtr_idx0"},
        {"name": "pptr",  "as": "pPptr_idx0"},
        {"name": "psvr", "as": "pPsvtr_idx0"},
        {"name": "pevtr", "as": "pPevtr_idx0"},
        {"name": "plvtr", "as": "pPlvtr_idx0"},
        {"name": "pcmtr",  "as": "pPcmtr_idx0"},
        {"name": "pdtr", "as": "pPdctr_idx0"},
        {"name": "pfvtr", "as": "pPfvtr_idx0"},
        {"name": "pcmef", "as": "pPcmef_idx0"},
        {"name": "fr_score1",  "as": "pPfrScore1_idx0"},
        {"name": "fr_score2",  "as": "pPfrScore2_idx0"},
        {"name": "pliving_ctr", "as": "pPlivingctr_idx0"},
        {"name": "pliving_wtr", "as": "pPlivingwtr_idx0"},
        {"name": "fetr", "as": "pPfetr_idx0"},
        {"name": "fountain_eff", "as": "pPfountainEff_idx0"},

        # 粗排预测结果
        {"name": "cascade_pctr", "as": "pMcPctr_idx0"},
        {"name": "cascade_pltr", "as": "pMcPltr_idx0"},
        {"name": "cascade_pwtr", "as": "pMcPwtr_idx0"},
        {"name": "cascade_plvtr", "as": "pMcPlvtr_idx0"},
        {"name": "cascade_psvtr", "as": "pMcPsvtr_idx0"},
        {"name": "cascade_pftr", "as": "pMcPftr_idx0"},
        {"name": "cascade_pcmtr", "as": "pMcPcmtr_idx0"},
        {"name": "cascade_plvtr2", "as": "pMcPlvtr2_idx0"},
        {"name": "cascade_pepstr", "as": "pMcPepstr_idx0"},
        {"name": "cascade_pcestr", "as": "pMcPcestr_idx0"},
        {"name": "cascade_pwatch_time", "as": "pMcPwatchTime_idx0"},
        {"name": "cascade_ptr", "as": "pMcPptr_idx0"},

        # emp xtr
        {"name": "empirical_ctr", "as": "pEmpCtr_idx0"},
        {"name": "empirical_ltr", "as": "pEmpLtr_idx0"},
        {"name": "empirical_wtr", "as": "pEmpWtr_idx0"},
        {"name": "empirical_ftr", "as": "pEmpFtr_idx0"},
        {"name": "empirical_ptr", "as": "pEmpPtr_idx0"},
        {"name": "empirical_htr", "as": "pEmpHtr_idx0"},
        {"name": "empirical_cmtr", "as": "pEmpCmtr_idx0"},
        {"name": "empirical_watch_time", "as": "pAvgWatchtime_idx0"},

        # photo info
        {"name": "photo_id", "as": "pId_idx0"},
        {"name": "author__id", "as": "aId_idx0"},
        {"name": "author__fans_count", "as": "pAuthorFansCount_idx0"},
        # 需要处理 打印看看
        {"name": "pAgeHour", "as": "pAgeHour_idx0"},
        {"name": "duration_ms", "as": "pDurationMs_idx0"},
        {"name": "upload_type", "as": "pUploadType_idx0"},
        {"name": "explore_stat__show_count", "as": "pHotShow_idx0"},
        {"name": "explore_stat__click_count", "as": "pHotClick_idx0"},
        {"name": "explore_stat__like_count", "as": "pHotLike_idx0"},
        {"name": "explore_stat__follow_count", "as": "pHotFollow_idx0"},
        {"name": "explore_stat__negative_count", "as": "pHotHate_idx0"},
        {"name": "explore_stat__report_detail__total_report_count", "as": "pHotReport_idx0"},
        {"name": "author__gender", "as": "pAuthorGender_idx0"},
        {"name": "author_age_info__age_segment", "as": "pAuthorAgeSeg_idx0"},
        {"name": "music", "as": "pMusic_idx0"},
        # 直播说这个更准确
        {"name": "live_photo_info__is_living", "as": "pHotLiving_idx0"},
        {"name": "mod", "as": "pPhoneMod_idx0"},
        # 需要处理 打log看一下是否ok
        {"name": "reason", "as": "pHotExptag_idx0"},
        {"name": "location__province_id", "as": "pProvinceId_idx0"},
        {"name": "location__city_id", "as": "pCityId_idx0"},
        {"name": "show_level_a", "as": "pLevelA_idx0"},
        {"name": "show_level_b", "as": "pLevelB_idx0"},
        {"name": "content_safety_level_with_namespace__level_hot_online", "as": "pContentLevel_idx0"},
        {"name": "tag", "as": "pTag_idx0"},
        {"name": "photo_dnn_cluster_id", "as": "pDnnClusterId_idx0"},
        {"name": "mmu_img_cluster_v3", "as": "pMmuImgClusterV3_idx0"},
        {"name": "mmu_img_cluster_v1", "as": "pMmuImgClusterV1_idx0"},
        {"name": "mmu_cluster_music_id", "as": "pMmuClusterMusicId_idx0"},
        {"name": "mmu_content_id", "as": "pMmuContentId_idx0"},
        {"name": "ocr_cover_text_word_count", "as": "pOcrCoverTextWordCount_idx0"},
        {"name": "music_info__music_combo_id", "as": "pMusicComboId_idx0"},
        {"name": "hetu_tag_level_info__hetu_level_one", "as": "pHetuTagLevel1Id_idx0"},
        {"name": "hetu_tag_level_info__hetu_level_two", "as": "pHetuTagLevel2Id_idx0"}
      ]
      return rerank_features_gen

    def photo_features_new_changed(self):
        features = [
            "photo_id",
            "author__id",
            "pPctr",
            "pPltr",
            "pPwtr",
            "pPftr",
            "pPhtr",
            "pPlvtr",
            "pPsvtr",
            "pPsvtr",
            "pPvtr",
            "pPptr",
            "pPcmtr",
            "pPdctr",
            "pPcmef",
            "pPfvtr",
            "pPdtr",
            "pPfrScore1",
            "pPfrScore2",
            "pPfrScore1_v1",
            "pPfrScore2_v1",
            "pPevtr",
            "pPfetr",
            "pPfountainEff",
            "pMcPctr",
            "pMcPltr",
            "pMcPwtr",
            "pMcPlvtr",
            "pMcPsvtr",
            "pEmpCtr",
            "pEmpLtr",
            "pEmpWtr",
            "pEmpFtr",
            "pEmpPtr",
            "pEmpCmtr",
            "pEmpHtr",
            "pEmpWatchTime",
            "pPctrFractile",
            "pPltrFractile",
            "pPlvtrFractile",
            "pPsvrFractile",
            "pPwtrFractile",
            "pPftrFractile",
            "pPptrFractile",
            "pPhtrFractile",
            "pPepstrFractile",
            "pPcmtrFractile",
            "pPcmefFractile",
            "pPFrScore1Fractile",
            "pPFrScore2Fractile",
            "pPfetrFractile",
            "pPFountainEffFractile",
            "author__fans_count",
            "pAgeHour",
            "duration_ms",
            "pUploadType",
            "explore_stat__show_count",
            "explore_stat__click_count",
            "explore_stat__like_count",
            "explore_stat__follow_count",
            "explore_stat__negative_count",
            "explore_stat__report_detail__total_report_count",
            "pHotExptag",
            "click_upload_rate",
            "pCityId",
            "pProvinceId",
            "infer_gender",
            "infer_gender_iter1",
            "infer_year",
            "infer_year_iter1",
            "photo_high_end_status_bits",
            "pPhotoPoi",
            "location__community_type",
            "photo_dnn_cluster_id",
            "pMusic",
            "explore_stat__external_download",
            "pHetuTagLevel1",
            "pHetuTagLevel2",
            "pHetuTagLevel3",
            "hetu_tag_level_info__hetu_level_four",
            "pHetuTagLevel5",
            "pHetuTagLevelTag",
            "pHetuTagFaceId",
            "author_high_score_v2",
            "content_safety_level_with_namespace__level_hot_online",
            "pAuditHotHighTagLevel",
            "mmu_img_cluster_v1",
            "pMmuImgClusterV3",
            "pMmuImgClusterV4",
            "pPhotoIndex",
            "pRealShowIndex",
            "user_hash_tag_id",
        ]
        for key in ["Hetu1", "Hetu2", "Hetu3", "Hetu4", "Hetu5", "HetuTag"]:
            for suffix in ["100n", "1000n"]:
                features.append("pShortStatShow" + key + suffix)
                features.append("pShortStatClick" + key + suffix)
                features.append("pShortStatClickRate" + key + suffix)
        return features

    def photo_features_changed(self):
        features = [
            {"name": "photo_id", "as": "pId"},
            {"name": "author__id", "as": "aId"},
            "pPctr",
            "pPltr",
            "pPwtr",
            "pPftr",
            "pPhtr",
            "pPlvtr",
            "pPsvtr",
            {"name": "pPsvtr", "as":"pPstr"},
            "pPvtr",
            "pPptr",
            "pPcmtr",
            "pPdctr",
            "pPcmef",
            "pPfvtr",
            "pPdtr",
            "pPfrScore1",
            "pPfrScore2",
            "pPfrScore1_v1",
            "pPfrScore2_v1",
            "pPevtr",
            "pPfetr",
            "pPfountainEff",
            {"name": "awesome_wtd", "as":"pPwtdScore_v2"},
            "pMcPctr",
            "pMcPltr",
            "pMcPwtr",
            "pMcPlvtr",
            "pMcPsvtr",
            "pEmpCtr",
            "pEmpLtr",
            "pEmpWtr",
            "pEmpFtr",
            "pEmpPtr",
            "pEmpCmtr",
            "pEmpHtr",
            "pEmpWatchTime",
            "pPctrFractile",
            "pPltrFractile",
            "pPlvtrFractile",
            "pPsvrFractile",
            "pPwtrFractile",
            "pPftrFractile",
            "pPptrFractile",
            "pPhtrFractile",
            "pPepstrFractile",
            "pPcmtrFractile",
            "pPcmefFractile",
            "pPFrScore1Fractile",
            "pPFrScore2Fractile",
            "pPfetrFractile",
            "pPFountainEffFractile",
            {"name": "author__fans_count", "as": "aFansCount"},
            "pAgeHour",
            {"name": "duration_ms", "as": "pDurationMs"},
            "pUploadType",
            {"name": "explore_stat__show_count", "as": "pHotShow"},
            {"name": "explore_stat__click_count", "as": "pHotClick"},
            {"name": "explore_stat__like_count", "as": "pHotLike"},
            {"name": "explore_stat__follow_count", "as": "pHotFollow"},
            {"name": "explore_stat__negative_count", "as": "pHotHate"},
            {"name": "explore_stat__report_detail__total_report_count", "as": "pHotReport"},
            "pHotExptag",
            {"name": "click_upload_rate", "as": "pUploadRate"},
            "pCityId",
            "pProvinceId",
            {"name": "infer_gender", "as": "pInferGender"},
            {"name": "infer_gender_iter1", "as": "pInferGender1"},
            {"name": "infer_year", "as": "pInferYear"},
            {"name": "infer_year_iter1", "as": "pInferYear1"},
            {"name": "photo_high_end_status_bits", "as": "pHighEndBits"},
            "pPhotoPoi",
            {"name": "location__community_type", "as": "pPhotoCommunityType"},
            {"name": "photo_dnn_cluster_id", "as": "pDnnClusterId"},
            "pMusic",
            {"name": "explore_stat__external_download", "as": "pHotDownload"},
            "pHetuTagLevel1",
            "pHetuTagLevel2",
            "pHetuTagLevel3",
            {"name": "hetu_tag_level_info__hetu_level_four", "as": "pHetuTagLevel4"},
            "pHetuTagLevel5",
            "pHetuTagLevelTag",
            "pHetuTagFaceId",
            {"name": "author_high_score_v2", "as": "pAuthorHighScoreV2"},
            {"name": "content_safety_level_with_namespace__level_hot_online", "as": "pLevelHotOnline"},
            "pAuditHotHighTagLevel",
            {"name": "mmu_img_cluster_v1", "as": "pMmuImgClusterV1"},
            "pMmuImgClusterV3",
            "pMmuImgClusterV4",
            "pPhotoIndex",
            "pRealShowIndex",
            {"name": "user_hash_tag_id", "as": "pHashTagIdList"}
        ]
        for key in ["Hetu1", "Hetu2", "Hetu3", "Hetu5", "HetuTag"]:
            for suffix in ["100n", "1000n"]:
                features.append("pShortStatShow" + key + suffix)
                features.append("pShortStatClick" + key + suffix)
                features.append("pShortStatClickRate" + key + suffix)
        return features

    def follow_long_term_user_feature(self):
      features = [
        {"name": "uId", "as": "user_id"},
        {"name": "tab_for_follow_long_term_model", "as": "tab"},
        {"name": "uTrueNewUser", "as": "true_new_user"},
        {"name": "uGender", "as": "user_gender"},
        {"name": "featureUserLevel", "as": "user_level"},
        "user_age_segment",
        {"name": "uCityLevelNew", "as": "city_level"},
        {"name": "uIsDouyin", "as": "is_douyin"},
        {"name": "uClickPids", "as": "click_list"},
        {"name": "uLikePids", "as": "like_list"},
        {"name": "uFollowAids", "as": "follow_list"},
      ]

      return features

    def follow_long_term_item_feature(self):
      features = [
        "photo_id",
        {"name": "author__id", "as": "author_id"},
        {"name": "pEmpCtr", "as": "emp_ctr"},
        {"name": "pEmpLtr", "as": "emp_ltr"},
        {"name": "pEmpWtr", "as": "emp_wtr"},
        {"name": "pEmpFtr", "as": "emp_ftr"},
        {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_level_one_tag"},
        {"name": "hetu_tag_level_info__hetu_level_two", "as": "hetu_level_two_tag"},
        {"name": "hetu_tag_level_info__hetu_level_three", "as": "hetu_level_three_tag"},
        {"name": "explore_stat__show_count", "as": "photo_exp_show"},
        {"name": "explore_stat__click_count", "as": "photo_exp_click"},
        {"name": "pUploadType", "as": "upload_type"},
        {"name": "author__gender", "as": "author_gender"},
        {"name": "pPctrOri", "as": "pctr"},
        {"name": "pPltr", "as": "pltr"},
        {"name": "pPwtr", "as": "pwtr"},
        {"name": "pPftr", "as": "pftr"},
        {"name": "pPlvtr", "as": "plvtr"},
        {"name": "pPvtr", "as": "pvtr"},
        {"name": "pPfrScore1", "as": "fr_score1"},
        {"name": "pPfrScore2", "as": "fr_score2"},
        {"name": "pMcPctr", "as": "mc_pctr"},
        {"name": "pMcPltr", "as": "mc_pltr"},
        {"name": "pMcPwtr", "as": "mc_pwtr"},
        {"name": "pMcPlvtr", "as": "mc_plvtr"},
      ]

      return features

    def revisit_model_user_feature(self):
      features = [
        {"name": "uId", "as": "user_id"},   
        {"name": "uGender", "as": "user_gender"},
        "user_age_segment",
        "tab"
      ]

      return features

    def revisit_model_item_feature(self):
      features = [
        "photo_id",
        {"name": "author__id", "as": "author_id"},
        {"name": "author__fans_count", "as": "author_fans_count"}, 

        {"name": "hetu_tag_level_info__hetu_cluster_id", "as": "hetu_cluster_id"},
        {"name": "hetu_tag_level_info__hetu_level_two", "as": "hetu_level_two_tag"},


        {"name": "pPltr", "as": "pltr"},
        {"name": "pPwtr", "as": "pwtr"},
        {"name": "pPlvtr", "as": "plvtr"},
        {"name": "pctr", "as": "pctr"},
        {"name": "pPcmtr", "as": "pcmtr"},
        {"name": "pPcmef", "as": "pcmef"},
        {"name": "pPptr", "as": "pptr"},
        {"name": "pPsvtr", "as": "psvtr"},

        "pctr_index",
        "pltr_index",
        "pwtr_index",
        "pvtr_index",
        "plvtr_index",

        {"name": "empirical_ctr", "as": "emp_ctr"},
        {"name": "empirical_ltr", "as": "emp_ltr"},
        {"name": "empirical_wtr", "as": "emp_wtr"}
      ]

      return features
    
    def rank_distill_ltr_user_features(self):
      features = [
        "uId",
        "dId",
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
        "uVisitMod",
        {"name": "uRealtimeClickList", "as": "uClickPids"},
        {"name": "uRealtimeLikeList", "as": "uLikePids"},
        {"name": "uFollowPhotoAuthorList", "as": "uFollowAids"},
        {"name": "uRealtimeForwardList", "as": "uForwardPids"},
        {"name": "uRealtimeNegativeList", "as": "uNegtivePids"},
      ]
      return features

    def process(self) -> None:
        self.flow \
          .explore_consume_time_ltr_attr(
            user_info_attr="user_info_ptr",
            item_attrs=[
                "upload_time",
                "explore_stat__report_detail__total_report_count",
                "explore_stat__show_count",
                "hetu_tag_level_info__hetu_level_one",
                "hetu_tag_level_info__hetu_level_two",
                "hetu_tag_level_info__hetu_level_three",
                "hetu_tag_level_info__hetu_level_four",
                "hetu_tag_level_info__hetu_level_five",
                "hetu_tag_level_info__hetu_tag",
                "hetu_tag_level_info__hetu_face_id",
                "pctr",
                "corr_pctr",
                "pltr",
                "plvtr",
                "psvr",
                "pwtr",
                "pftr",
                "pptr",
                "phtr",
                "pepstr",
                "pcmtr",
                "pcmef",
                "fr_score1",
                "fr_score2",
                "fetr",
                "fountain_eff"
            ],
            consume_time_ltr_need_long_action_list="{{consume_time_ltr_need_long_action_list_new}}",
            long_action_list_click_length="{{consume_time_ltr_long_action_list_click_length}}",
            long_action_list_action_length="{{consume_time_ltr_long_action_list_action_length}}",
            consume_time_view_list_length="{{consume_time_view_list_length}}",
            user_feature=self.user_feture()
          ) \
          .enrich_attr_by_lua(
            import_item_attr=[
                "pctr",
                "corr_pctr",
                "pltr",
                "pwtr",
                "pftr",
                "psvr",
                "pcmtr",
                "pptr",
                "pcmef",
                "phtr",
                "pevtr",
                "fr_score1",
                "fr_score2",
                "pdtr",
                "fetr",
                "fountain_eff",
                "cascade_pctr",
                "cascade_pltr",
                "cascade_pwtr",
                "cascade_plvtr",
                "cascade_psvtr",
                "empirical_ctr",
                "empirical_ltr",
                "empirical_wtr",
                "empirical_ftr",
                "empirical_ptr",
                "empirical_cmtr",
                "empirical_htr",
                "empirical_watch_time",
                "pdctr",
                "pvtr",
                "pfvtr",
                "plvtr",
                "location__poi",
                "music",
                "hetu_tag_level_info__hetu_face_id",
                "hetu_tag_level_info__hetu_level_three",
                "pShortStatShowHetu3100n",
                "pShortStatShowHetu31000n",
                "pShortStatClickHetu3100n",
                "pShortStatClickHetu31000n",
                "pShortStatClickRateHetu3100n",
                "pShortStatClickRateHetu31000n",
                "mmu_img_cluster_v3",
                "location__city_id",
                "location__province_id",
                "audit_hot_high_tag_level",
                "mmu_img_cluster_v4",
                "hetu_tag_level_info__hetu_level_five",
                "hetu_tag_level_info__hetu_level_two",
                "hetu_tag_level_info__hetu_tag",
                "upload_type",
                "hetu_tag_level_info__hetu_level_one",
                "hetu_tag_level_info__hetu_cluster_id",
                "explore_stat__report_detail__total_report_count",
                "infer_gender",
                "infer_year"
            ],
            export_item_attr=[
                "pPctr",
                "pPltr",
                "pPwtr",
                "pPftr",
                "pPsvtr",
                "pPcmtr",
                "pPptr",
                "pPcmef",
                "pPhtr",
                "pPevtr",
                "pPfrScore1",
                "pPfrScore2",
                "pPfrScore1_v1",
                "pPfrScore2_v1",
                "pPdtr",
                "pPfetr",
                "pPfountainEff",
                "pMcPctr",
                "pMcPltr",
                "pMcPwtr",
                "pMcPlvtr",
                "pMcPsvtr",
                "pPhotoIndex",
                "pRealShowIndex",
                "pEmpCtr",
                "pEmpLtr",
                "pEmpWtr",
                "pEmpFtr",
                "pEmpPtr",
                "pEmpCmtr",
                "pEmpHtr",
                "pEmpWatchTime",
                "pPdctr",
                "pPvtr",
                "pPfvtr",
                "pPlvtr",
                "pPhotoPoi",
                "pMusic",
                "pHetuTagFaceId",
                "pHetuTagLevel3",
                "pShortStatShowHetu3100n",
                "pShortStatShowHetu31000n",
                "pShortStatClickHetu3100n",
                "pShortStatClickHetu31000n",
                "pShortStatClickRateHetu3100n",
                "pShortStatClickRateHetu31000n",
                "pMmuImgClusterV3",
                "pCityId",
                "pProvinceId",
                "pAuditHotHighTagLevel",
                "pMmuImgClusterV4",
                "pHetuTagLevel5",
                "pHetuTagLevel2",
                "pHetuTagLevelTag",
                "pUploadType",
                "pHetuTagLevel1",
                "pHetuClusterId"
            ],
            function_for_item = "calculate_changed",
            lua_script_file = "explore/ranking/lua/module/ranking_score__consume_time_ltr_attr.lua",
          ) \
          .if_("enable_gen_distill_ltr_in_consume_time_ltr == 1") \
            .explore_rerank_attr(
              user_info_attr = "user_info_ptr"
            ) \
          .end_if_() \
          .if_("use_explore_fusion_ltr_model == 1") \
            .explore_custom_trim_user_info(
              user_info_attr="userInfo",
              save_trimed_user_info_to_attr="explore_ltr_uni_feature_trimmed_user_info",
              trim_user_info=self.uni_feature_trim_user_info(),
            ) \
            .delegate_enrich(
              name = "explore_ranking_ltr_model",
              kess_service="{{explore_ltr_uni_feature_kess_service}}",
              recv_item_attrs=[
                {"name": "ptr", "as": "consume_time_ptr"},
                {"name": "reward", "as": "consume_time_ltr"},
                {"name": "ctr", "as": "consume_time_ctr"},
                {"name": "pltr", "as": "consume_time_pltr"},
                {"name": "lph", "as": "consume_time_lph"},
                {"name": "pstd", "as": "consume_time_pstd"},
                {"name": "slide", "as": "consume_time_slide"},
                {"name": "evtr", "as": "consume_time_evtr"},
                {"name": "lvtr", "as": "consume_time_lvtr"}
              ],
              timeout_ms=100,
              send_item_attrs=self.uni_feature_context_into(),
              send_common_attrs=[
                {"name": "explore_ltr_uni_feature_trimmed_user_info", "as": "user_info_str"},
                {"name": "uOldMmuClusterId300ListList", "as": "user_feasury_cluster_id_list"},
                "active_days_gt_5min_rate"
              ],
              partition_size="{{explore_ltr_uni_feature_partition_size}}",
            ) \
          .end_if_() \
          .if_("enable_gen_distill_ltr_in_consume_time_ltr == 1") \
            .delegate_enrich(  # TODO(xuwei09) 对比验证耗时问题，预计 5 月末删除
              name = "explore_rank_gen_distill",
              kess_service = "{{explore_rank_gen_distill_kai_kess_service}}",
              recv_item_attrs = [
                {"name": "gen_l2r_pos", "as": "gen_l2r_score"},
              ],
              timeout_ms = 100,
              send_item_attrs = self.gen_distill_photo_features(),
              send_common_attrs = self.gen_distill_user_features(),
              partition_size = "{{explore_rank_gen_distill_kai_partition_size}}",
              request_type = "default"
            ) \
          .end_if_() \
          .if_("explore_revisit_model == 1") \
            .delegate_enrich(
              name = "explore_revisit_model",
              kess_service = "{{explore_revisit_model_kess_service}}",
              recv_item_attrs = [
                {"name": "revisit", "as": "revisit_score_model"},
                {"name": "revisit2", "as": "revisit_score_author_model"},
              ],
              timeout_ms = 80,
              send_item_attrs = self.revisit_model_item_feature(),
              send_common_attrs = self.revisit_model_user_feature(),
              request_type = "default",
              partition_size = "{{revisit_model_predict_partition_size}}",
            ) \
          .end_()


        self.flow \
        .enrich_attr_by_lua(
            import_item_attr=[
               "consume_time_ltr"
            ],
            export_item_attr=[
                "consume_time_ltr"
            ],
            function_for_item = "transfer_ltr",
            lua_script_file = "explore/ranking/lua/module/ranking_score__consume_time_ltr_attr.lua",
        ) \
        .if_("use_explore_fusion_ltr_model == 1") \
            .enrich_attr_by_lua(
                import_common_attr = [
                  "enable_sk_pftr_st_feed"
                ],
                import_item_attr=[
                    "duration_ms"
                ],
                export_item_attr=[
                    "duration_cluster",
                    "duration_low_0"
                ],
                function_for_item = "transfer_duration",
                lua_script_file = "explore/ranking/lua/module/ranking_score__consume_time_ltr_attr.lua",
            ) \
            .enrich_attr_by_lua(
                import_common_attr=[
                    "enable_sk_pftr_negative"
                ],
                import_item_attr=[
                    "consume_time_ptr"
                ],
                export_item_attr=[
                    "consume_time_pftr_score",
                    "pftr_low_0",
                ],
                function_for_item = "transfer_ftr",
                lua_script_file = "explore/ranking/lua/module/ranking_score__consume_time_ltr_attr.lua",
            ) \
            .if_("enable_zip_process == 0") \
              .enrich_attr_by_lua(
                  import_common_attr = [
                      "enable_modified_trunc",
                      "pftr_prefx"
                  ],
                  import_item_attr = [
                      "duration_cluster",
                      "consume_time_pftr_score"
                  ],
                  export_item_attr = [
                      "redis_key_pf2r"
                  ],
                  function_for_item = "transfer_key",
                  lua_script_file = "explore/ranking/lua/module/ranking_score__consume_time_ltr_attr.lua",
              ) \
              .get_item_attr_from_redis(
                  cluster_name = "recoMmuLongTermNum",
                  redis_key_from="redis_key_pf2r",
                  save_value_to="consume_time_pf2r",
              ) \
            .end_() \
            .enrich_attr_by_lua(
                import_item_attr = [
                    "consume_time_pf2r",
                    "duration_low_0",
                    "pftr_low_0",
                    "consume_time_ptr"
                ],
                import_common_attr=[
                    "enable_zip_process"
                ],
                export_item_attr = [
                    "consume_time_pf2r_score"
                ],
                function_for_item = "transfer_pf2r",
                lua_script_file = "explore/ranking/lua/module/ranking_score__consume_time_ltr_attr.lua",
            ) \
            .if_("use_explore_pf2r_calc == 1") \
              .explore_calc_pf2r_score_by_multiple_bucket_enricher(
                pxtr_fractile_map_attr = "explore_pftr_fractile_score_attr_from_redis_ptr",
                raw_consume_time_ptr_attr = "consume_time_ptr",
                duration_ms_attr = "duration_ms",
                duration_ms_upper_bound = "{{explore_pftr_fractile_score_duration_ms_upper_bound}}",
                active_days_attr = "active_days_gt_5min_rate",
                active_days_split_conf_attr = "explore_fr_active_days_split_conf",
                pf2r_score_attr = "debias_pf2r_score"
              ) \
            .end_() \
        .end_() \

    def post_process(self) -> None:
      self.flow \
        .log_debug_info(
          item_attrs = self.photo_features_new_changed(),
          common_attrs = self.user_feture(),
            for_debug_request_only = True
        )
