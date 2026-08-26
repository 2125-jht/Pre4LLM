#!/usr/bin/env python3
# coding=utf-8

fullrank_common_attrs = [
  # 模型 feature 需要的字段
  "author__gender",
  "infer_year",
  "author_age_info__age_segment",
  "location__province_id",
  "location__city_id",
  "mod",
  "chn",
  "from_client",
  "mmu_cluster_music_id",
  "music_info__music_combo_id",
  "ocr_cover_text_word_count",
  "definition_level",
  "content_safety_level",
  "author__reg_time",
  "author__upload_count",
  "explore_stat__like_count",
  "explore_stat__follow_count",
  "explore_stat__forward_count",
  "explore_stat__long_play_count",
  "explore_stat__short_play_count",
  "explore_stat__profile_enter_count",
  "explore_stat__comment_count",
  "author__exp_stat__exp_click",
  "author__exp_stat__exp_like",
  "author__exp_stat__exp_follow",
  "author__exp_stat__exp_long_view",
  "author__exp_stat__exp_realshow",
  "author__exp_stat__exp_forward",
  "author__exp_stat__exp_short_view",
  "author__exp_stat__exp_watch_time",
  "music",
  "collect_count",
  "mmu_img_cluster_v1",
  "mmu_content_id"
]

fullrank_common_copy_attrs = [
  {
    "from_item": "photo_id",
    "to_item": "featurePId"
  },
  {
    "from_item": "author__id",
    "to_item": "featurePAId"
  },
  {
    "from_item": "author__gender",
    "to_item": "featurePGender"
  },
  {
    "from_item": "infer_year",
    "to_item": "featurePInferYear"
  },
  {
    "from_item": "author_age_info__age_segment",
    "to_item": "featurePAgeSegment"
  },
  {
    "from_item": "location__province_id",
    "to_item": "featurePProvinceId"
  },
  {
    "from_item": "location__city_id",
    "to_item": "featurePCityId"
  },
  {
    "from_item": "mod",
    "to_item": "featurePMod"
  },
  {
    "from_item": "chn",
    "to_item": "featurePChn"
  },
  {
    "from_item": "upload_type",
    "to_item": "featurePUploadType"
  },
  {
    "from_item": "from_client",
    "to_item": "featurePFromClient"
  },
  {
    "from_item": "tag",
    "to_item": "featurePTag"
  },
  {
    "from_item": "photo_dnn_cluster_id",
    "to_item": "featurePDnnClusterId"
  },
  {
    "from_item": "mmu_img_cluster_v3",
    "to_item": "featurePMmuImgClusterV3"
  },
  {
    "from_item": "mmu_cluster_music_id",
    "to_item": "featurePMmuClusterMusicId"
  },
  {
    "from_item": "music_info__music_combo_id",
    "to_item": "featurePMusicComboId"
  },
  {
    "from_item": "ocr_cover_text_word_count",
    "to_item": "featurePOcrCoverTextWordCount"
  },
  {
    "from_item": "definition_level",
    "to_item": "featurePDefinitionLevel"
  },
  {
    "from_item": "content_safety_level",
    "to_item": "featurePContentSafetyLevel"
  },
  {
    "from_item": "duration_ms",
    "to_item": "featurePDurationMs"
  },
  {
    "from_item": "author__reg_time",
    "to_item": "featurePARegTime"
  },
  {
    "from_item": "author__upload_count",
    "to_item": "featurePAUploadCount"
  },
  {
    "from_item": "author__fans_count",
    "to_item": "featurePAFansCount"
  },
  {
    "from_item": "music",
    "to_item": "featurePMusic"
  },
  {
    "from_item": "hetu_tag_level_info__hetu_level_one",
    "to_item": "featurePHetuTagLevel1"
  },
  {
    "from_item": "hetu_tag_level_info__hetu_level_two",
    "to_item": "featurePHetuTagLevel2"
  },
  {
    "from_item": "hetu_tag_level_info__hetu_level_three",
    "to_item": "featurePHetuTagLevel3"
  },
  {
    "from_item": "hetu_tag_level_info__hetu_level_four",
    "to_item": "featurePHetuTagLevel4"
  },
  {
    "from_item": "hetu_tag_level_info__hetu_level_five",
    "to_item": "featurePHetuTagLevel5"
  },
  {
    "from_item": "hetu_tag_level_info__hetu_face_id",
    "to_item": "featurePHetuFaceId"
  },
  {
    "from_item": "author__category_detail__first_level_id",
    "to_item": "featureACategoryLevelOne"
  },
  {
    "from_item": "author__category_detail__second_level_id",
    "to_item": "featureACategoryLevelTwo"
  },
  {
    "from_item": "author__category_detail__third_level_id",
    "to_item": "featureACategoryLevelThree"
  },
  {
    "from_item": "author__category_detail__fourth_level_id",
    "to_item": "featureACategoryLevelFour"
  },
  {
    "from_item": "online_lda_topic__ids",
    "to_item": "featurePOnlineLdaTopic"
  }
]

fullrank_splash_attrs = [
  "explore_stat__report_detail__total_report_count",
]

fullrank_fast_attrs = [
  # mmr 依赖字段
  "tag",
  "mmu_img_cluster_v3",
  "photo_dnn_cluster_id",
  "mmu_text_cluster",
  "GE_cluster_id",
  "mmu_text_lda_topic",
  "author__category_detail__first_level_id",
  "author__category_detail__second_level_id",
  "author__category_detail__third_level_id",
  # 打散依赖字段
  "similar_event_id",
  "author__is_pr_account",
  "author__is_gr_account",
  "online_lda_topic__ids",
  "hetu_tag_level_info__hetu_level_four",
  "author__category_detail__fourth_level_id",
]

photo_features = [
  { "name": "featurePId", "type": "int" },
  { "name": "featurePAId", "type": "int" },
  { "name": "featurePGender", "type": "int" },
  { "name": "featurePInferYear", "type": "float" },
  { "name": "featurePAgeSegment", "type": "int" },
  { "name": "featurePProvinceId", "type": "int" },
  { "name": "featurePCityId", "type": "int" },
  { "name": "featurePMod", "type": "string" },
  { "name": "featurePChn", "type": "string" },
  { "name": "featurePUploadType", "type": "int" },
  { "name": "featurePFromClient", "type": "int" },
  { "name": "featurePTag", "type": "int" },
  { "name": "featurePDnnClusterId", "type": "int" },
  { "name": "featurePOnlineLdaTopic", "type": "int_list" },
  { "name": "featurePMmuImgClusterV3", "type": "int" },
  { "name": "featurePMmuClusterMusicId", "type": "int" },
  { "name": "featurePMusicComboId", "type": "string" },
  { "name": "featurePOcrCoverTextWordCount", "type": "int" },
  { "name": "featurePDefinitionLevel", "type": "int" },
  { "name": "featurePContentSafetyLevel", "type": "int" },
  { "name": "featurePUploadTimeDiff", "type": "int" },
  { "name": "featurePDurationMs", "type": "int" },
  { "name": "featurePARegTime", "type": "int" },
  { "name": "featurePAUploadCount", "type": "int" },
  { "name": "featurePAFansCount", "type": "int" },
  { "name": "featurePHotClickCount", "type": "int" },
  { "name": "featurePHotLikeCount", "type": "int" },
  { "name": "featurePHotFollowCount", "type": "int" },
  { "name": "featurePHotLongViewCount", "type": "int" },
  { "name": "featurePHotCtr", "type": "float" },
  { "name": "featurePHotLtr", "type": "float" },
  { "name": "featurePHotWtr", "type": "float" },
  { "name": "featurePHotFtr", "type": "float" },
  { "name": "featurePHotLvtr", "type": "float" },
  { "name": "featurePHotSvtr", "type": "float" },
  { "name": "featurePHotAvgWatchTime", "type": "float" },
  { "name": "featurePAClickCount", "type": "int" },
  { "name": "featurePALikeCount", "type": "int" },
  { "name": "featurePAFollowCount", "type": "int" },
  { "name": "featurePALongViewCount", "type": "int" },
  { "name": "featurePACtr", "type": "float" },
  { "name": "featurePALtr", "type": "float" },
  { "name": "featurePAWtr", "type": "float" },
  { "name": "featurePAFtr", "type": "float" },
  { "name": "featurePALvtr", "type": "float" },
  { "name": "featurePASvtr", "type": "float" },
  { "name": "featurePAAvgWatchTime", "type": "float" },
  { "name": "featurePMusic", "type": "string" },
  { "name": "featurePHetu0", "type": "int" },
  { "name": "featurePHetuTagLevel1", "type": "int_list" },
  { "name": "featurePHetuTagLevel2", "type": "int_list" },
  { "name": "featurePHetuTagLevel3", "type": "int_list" },
  { "name": "featurePHetuTagLevel5", "type": "int_list" },
  { "name": "fullrank_detail_pcmtr", "type": "float" },
  { "name": "fullrank_detail_pptr", "type": "float" },
  { "name": "fullrank_detail_pctr", "type": "float" },
  { "name": "fullrank_detail_pltr", "type": "float" },
  { "name": "fullrank_detail_pwtr", "type": "float" },
  { "name": "fullrank_detail_pftr", "type": "float" },
  { "name": "fullrank_detail_plvtr", "type": "float" },
  { "name": "fullrank_detail_psvr", "type": "float" },
  { "name": "fullrank_detail_pvtr", "type": "float" },
  { "name": "fullrank_detail_pwtd", "type": "float" },
  { "name": "fullrank_sim_pfintr", "as": "fountainPfintrFeature", "type": "float" },
  { "name": "cascade_pctr", "type": "float" },
  { "name": "cascade_pltr", "type": "float" },
  { "name": "cascade_pwtr", "type": "float" },
  { "name": "cascade_pftr", "type": "float" },
  { "name": "cascade_plvtr", "type": "float" },
  { "name": "cascade_psvtr", "type": "float" },
  { "name": "featureDurationSId", "type": "int" },
  { "name": "faActionL2rV4DurationId", "type": "int" },
  { "name": "fountainDurationPercent", "type": "int"},
]

rr_photo_features = [
    # 调用精排 or 通过rankResult传入
    {"name": "fullrank_sim_pevtr",  "as" : "pPctr_idx0"},
    {"name": "fullrank_sim_pltr",  "as" : "pPltr_idx0"},
    {"name": "fullrank_sim_pwtr",  "as" : "pPwtr_idx0"},
    {"name": "fullrank_sim_pftr",  "as" : "pPftr_idx0"},
    {"name": "fullrank_sim_phtr",  "as" : "pPhtr_idx0"},
    {"name": "fullrank_sim_plvtr",  "as" : "pPlvtr_idx0"},
    {"name": "fullrank_sim_out_pctr",  "as" : "pPsvtr_idx0"},
    {"name": "fullrank_sim_pvtr",  "as" : "pPvtr_idx0"},
    {"name": "fullrank_sim_pptr",  "as" : "pPptr_idx0"},
    {"name": "fullrank_sim_pcmtr",  "as" : "pPcmtr_idx0"},
    #{"name": "fullrank_plivewtr",  "as" : "pPlivingwtr_idx0"},
    {"name": "fullrank_sim_pcmef",  "as" : "pPcmef_idx0"},
    {"name": "fullrank_sim_pepstr",  "as" : "pPepstr_idx0"},
    {"name": "fullrank_sim_pfintr",  "as" : "pPwtd_idx0"},

    # 通过分布式索引
    {"name": "photo_id",  "as" : "pId_idx0"},
    {"name": "author__id",  "as" : "aId_idx0"},
    {"name": "author__fans_count",  "as" : "pAuthorFansCount_idx0"},
    {"name": "featurePUploadType",  "as" : "pUploadType_idx0"},
    {"name": "explore_stat__show_count",  "as" : "pHotShow_idx0"},
    {"name": "explore_stat__click_count",  "as" : "pHotClick_idx0"},
    {"name": "explore_stat__like_count",  "as" : "pHotLike_idx0"},
    {"name": "explore_stat__follow_count",  "as" : "pHotFollow_idx0"},
    {"name": "explore_stat__negative_count",  "as" : "pHotHate_idx0"},
    {"name": "explore_stat__report_detail__total_report_count",  "as" : "pHotReport_idx0"},
    #{"name": "click_upload_rate",  "as" : "pUploadRate_idx0"},
    {"name": "featurePCityId",  "as" : "pCityId_idx0"},
    {"name": "featurePProvinceId",  "as" : "pProvinceId_idx0"},
    {"name": "featurePDurationMs",  "as" : "pDurationMs_idx0"},
    {"name": "content_safety_level_with_namespace__level_hot_online",  "as" : "pContentLevel_idx0"},
    #{"name": "author.gender",  "as" : "pAuthorGender_idx0"},
    {"name": "featurePHetuTagLevel1",  "as" : "pHetuTagLevel1Id_idx0"},
    {"name": "featurePHetuTagLevel2",  "as" : "pHetuTagLevel2Id_idx0"},
    {"name": "featurePDnnClusterId",  "as" : "pDnnClusterId_idx0"},
    {"name": "mmu_img_cluster_v1",  "as" : "pMmuImgClusterV1_idx0"},
    {"name": "featurePMmuImgClusterV3",  "as" : "pMmuImgClusterV3_idx0"},
    {"name": "mmu_content_id",  "as" : "pMmuContentId_idx0"},
    {"name": "featurePMusic",  "as" : "pMusic_idx0"},
    {"name": "featurePMusicComboId",  "as" : "pMusicComboId_idx0"},
    {"name": "featurePOcrCoverTextWordCount",  "as" : "pOcrCoverTextWordCount_idx0"},
    {"name": "featurePUploadTimeDiff",  "as" : "pAgeHour_idx0"},

    # 通过rpc传入rankResult对齐recoPhotoInfo的口径
    {"name": "cascade_pctr",  "as" : "pMcPctr_idx0"}, # 粗排分从RPC传过来的RankResult里拿
    {"name": "cascade_pltr",  "as" : "pMcPltr_idx0"},
    {"name": "cascade_pwtr",  "as" : "pMcPwtr_idx0"},
    {"name": "cascade_plvtr",  "as" : "pMcPlvtr_idx0"},
    {"name": "cascade_psvtr",  "as" : "pMcPsvtr_idx0"},
    {"name": "fullrank_empirical_ctr",  "as" : "pEmpCtr_idx0"},
    {"name": "fullrank_empirical_ltr",  "as" : "pEmpLtr_idx0"},
    {"name": "fullrank_empirical_wtr",  "as" : "pEmpWtr_idx0"},
    {"name": "fullrank_empirical_ftr",  "as" : "pEmpFtr_idx0"},
    {"name": "fullrank_empirical_ptr",  "as" : "pEmpPtr_idx0"},
    {"name": "fullrank_empirical_cmtr",  "as" : "pEmpCmtr_idx0"},
    {"name": "fullrank_empirical_htr",  "as" : "pEmpHtr_idx0"},
    {"name": "fullrank_empirical_watchtime",  "as" : "pAvgWatchtime_idx0"},
    # 暂时没找到, 获取相对比较麻烦 先不传入
    # {"name": "living",  "as" : "pHotLiving_idx0"},
    # 这个需要问一下
    {"name": "reason",  "as" : "pHotExptag_idx0"},
]

rr_user_feature = [
  {"name": "featureUId",  "as" : "uId"},
  {"name": "featureDeviceId",  "as" : "dId"},
  {"name": "featureGender",  "as" : "uBasicGender"},
  {"name": "featureAgeSegment",  "as" : "uBasicAge"},
  {"name": "featureProvinceId",  "as" : "uRequstProvinceId"},
  {"name": "featureCityId",  "as" : "uCityId"},
  {"name": "featureClientId",  "as" : "uClientId"},
  {"name": "featureVisitMod",  "as" : "uMod"},
  {"name": "featureVisitNet",  "as" : "uNetwork"},
  {"name": "featureRealtimeClickList",  "as" : "uRealtimeClickList"},
  {"name": "featureRealtimeLikeList",  "as" : "uRealtimeLikeList"},
  {"name": "featureRealtimeFollowList",  "as" : "uRealtimeFollowList"},
  {"name": "featureRealtimeForwardList",  "as" : "uRealtimeForwardList"},
  {"name": "uRealtimeNegativeList", "as" : "uRealtimeNegativeList"},
  {"name": "featureUserProfileV1ClickPidList",  "as" : "uClickPhotoList"},
  {"name": "featureUserProfileV1ClickAidList",  "as" : "uClickPhotoAuthorList"},
  {"name": "featureUserProfileV1LikePidList",  "as" : "uLikePhotoList"},
  {"name": "featureUserProfileV1LikeAidList",  "as" : "uLikePhotoAuthorList"},
  {"name": "featureUserProfileV1CommentPidList",  "as" : "uCommentPhotoList"},
  {"name": "featureUserProfileV1CommentAidList",  "as" : "uCommentPhotoAuthorList"},
  {"name": "featureUserProfileV1FollowPidList",  "as" : "uFollowPhotoList"},
  {"name": "featureUserProfileV1FollowAidList",  "as" : "uFollowPhotoAuthorList"},
  {"name": "uExpClick",  "as" : "uExpClick"},
  {"name": "uExpLike",  "as" : "uExpLike"},
  {"name": "uExpFollow",  "as" : "uExpFollow"},
  {"name": "uExpLongView",  "as" : "uExpLongView"},
  {"name": "uExpWatchTime",  "as" : "uExpWatchTime"},
  {"name": "uRequestHour",  "as" : "uRequestHour"},
  {"name": "uRequestWeekday",  "as" : "uRequestWeekday"},
  {"name": "featureFountainProfileClickAidList",  "as" : "featureFountainProfileClikAidList"},
  {"name": "featureFountainProfileClickPidList",  "as" : "featureFountainProfileClikPidList"},
  {"name": "featureFountainProfileLikeAidList",  "as" : "featureFountainProfileLikeAidList"},
  {"name": "featureFountainProfileLikePidList",  "as" : "featureFountainProfileLikePidList"},
  {"name": "featureFountainProfileFollowAidList",  "as" : "featureFountainProfileFollowAidList"},
  {"name": "featureFountainProfileFollowPidList",  "as" : "featureFountainProfileFollowPidList"},
  {"name": "featureFountainProfileEffViewPidList",  "as" : "featureFountainProfileEffViewPidList"},
  {"name": "featureFountainProfileEffViewAidList",  "as" : "featureFountainProfileEffViewAidList"},
  {"name": "featureFountainProfileLongViewPidList",  "as" : "featureFountainProfileLongViewPidList"},
  {"name": "featureFountainProfileLongViewAidList",  "as" : "featureFountainProfileLongViewAidList"}, 
]

features = [
  "uViewPidListV1",
  "uViewAidListV1",
  "uEffectiveViewLabelListV1",
  "uLongViewLabelListV1",
  "uShortViewLabelListV1",
  "uViewHetu1ListV1",
  "uViewHetu2ListV1",
]
rr_user_feature.extend(features)
for i in range(30):
  for suffix in ["", "aid_", "tag_", "play_"]:
    rr_user_feature.append("realshow_" + suffix + str(i))

photo_pxtr_features = [
  "fullrank_sim_pfintr",
  "fullrank_detail_pctr",
  "fullrank_detail_pltr",
  "fullrank_detail_pwtr",
  "fullrank_detail_pftr",
  "fullrank_detail_plvtr",
  "fullrank_detail_psvr",
  "fullrank_detail_pvtr",
  "fullrank_detail_pcmtr",
  "fullrank_detail_pptr",
  "fullrank_detail_pwtd"
]

item_features = [
  "featurePId",
  "featurePAId",
  "featurePGender",
  "featurePAgeSegment",
  "featurePProvinceId",
  "featurePCityId",
  "featurePMod",
  "featurePChn",
  "featurePUploadType",
  "featurePFromClient",
  "featurePUploadTimeDiff",
  "featurePDurationMs",
  "featurePAUploadCount",
  "featurePAFansCount",
  "featurePMusic",
  "featurePHetuTagLevel1",
  "featurePHetuTagLevel2",
  "featurePHetuTagLevel3",
  "featurePHetuTagLevel4",
  "featurePHetuTagLevel5",
  "featurePHetuFaceId",
  "featureACategoryLevelOne",
  "featureACategoryLevelTwo",
  "featureACategoryLevelThree",
  "featureACategoryLevelFour",
  "featurePTag",
  "featurePDnnClusterId",
  "featurePOnlineLdaTopic",
  "featurePMmuImgClusterV3",
  "featurePHotCtr",
  "featurePHotLtr",
  "featurePHotWtr",
  "featurePHotFtr",
  "featurePHotLvtr",
  "featurePHotSvtr",
  "featurePACtr",
  "featurePALtr",
  "featurePAWtr",
  "featurePAFtr",
  "featurePALvtr",
  "featurePASvtr",
]

user_features = [
  "featureUId",
  "featureDeviceId",
  "featureDeviceModel",
  "featureGender",
  "featureAge",
  "featureAgeSegment",
  "featureProvinceId",
  "featureCityId",
  "featureClientId",
  "featureVisitMod",
  "featureVisitNet",
  "featureUserLevel",
  "featureRiskLevel",
  "featureRegTime",
  "featureFollowCount",
  "featureFansCount",
  "featureUploadCount",
  "featureUserCtr",
  "featureUserLtr",
  "featureUserWtr",
  "featureUserFtr",
  "featureUserLvtr",
  "featureUserSvtr",
  "featureUserAvgWatchTime",
  "featureUserRequestProvinceId",
  "featureUserRequestCityId",
  "featureUserRequestPoiType",
  "featureUserRegion",
  "featureUserPhonePriceLevel",
  "featureUserAppNewCat1",
  "featureUserAppNewCat2",
  "featureUserAppNewCat3",
  "featureUserAppNormNames",
  "featureUserAppNameList",
  "featureRealtimeClickList",
  "featureRealtimeLikeList",
  "featureRealtimeFollowList",
  "featureRealtimeForwardList",
  "featureAppSignList",
  "featureLongTermInterestPhotoDnnClusterId",
  "featureTopDislikeTopic",
  "featureTopRateTopic",
  "featureActiveDays",
  "featureUserClickCount",
  "featureUserProfileV1ClickPidList",
  "featureUserProfileV1ClickAidList",
  "featureUserProfileV1LikePidList",
  "featureUserProfileV1LikeAidList",
  "featureUserProfileV1CommentPidList",
  "featureUserProfileV1CommentAidList",
  "featureUserProfileV1DownloadPidList",
  "featureUserProfileV1DownloadAidList",
  "featureUserProfileV1FollowPidList",
  "featureUserProfileV1FollowAidList",
  "featureUserProfileV1ForwardPidList",
  "featureUserProfileV1ForwardAidList",
  "featureUserProfileV1SearchClickPidList",
  "featureUserProfileV1SearchClickAidList",
  "featureUserProfileV1ProfileEnterPidList",
  "featureUserProfileV1ProfileEnterAidList",
  "featureUserProfileV1Play11SAidList",
  "featureUserProfileV1Play3SPidList",
  "featureUserProfileV1Play11SPidList",
  "featureUserProfileV1Play3SAidList",
  "featureUserProfileV1Play7SPidList",
  "featureUserProfileV1Play18SPidList",
  "featureUserProfileV1Play7SAidList",
  "featureUserProfileV1Play18SAidList",
  "featureSourcePId",
  "sourcePidAuthorId",
  "sourcePidFirstLevelCategory",
  "sourcePidSecondLevelCategory",
  "sourcePidThirdLevelCategory",
  "sourcePidFourthLevelCategory",
  "sourcePidHetuLevelOneList",
  "sourcePidHetuLevelTwoList",
  "sourcePidHetuLevelThreeList",
  "sourcePidHetuLevelFourList",
  "sourcePidHetuLevelFiveList",
  "sourcePidHetu0",
  "sourcePidHetuLevelTwo0",
  "sourcePidHetuTagList",
  "sourcePidHetuFaceIdList",
  "sourcePidMmuImgClusterV3",
  "sourcePidDnnCluster",
  "sourcePidTagId",
  "sourcePidDuration",
]

item_features_rerank = [
  "fullrank_sim_out_pctr",
  "fullrank_sim_pltr",
  "fullrank_sim_pwtr",
  "fullrank_sim_pftr",
  "fullrank_sim_psvr",
  "fullrank_sim_plvtr",
  "fullrank_sim_pcmtr",
  "fullrank_sim_pptr",
  "fullrank_sim_pcmef",
  "fullrank_sim_phtr",
  "fullrank_sim_pevtr",
  "fullrank_sim_pvtr",
  "fullrank_sim_pepstr",
  "fullrank_sim_pfintr",
  "fullrank_sim_pcltr",
  "fullrank_sim_pcpr",
  "fullrank_sim_lstr",
  "fullrank_ori_pswptr",
  "fullrank_detail_new_pevtr_v2"
]

rerank_listwise_distill_item_features =[
  {"name": "featurePId", "as": "featurePIdRerankList"},
  {"name": "featurePAId", "as": "featurePAIdRerankList"},
  {"name": "featurePGender", "as": "featurePGenderRerankList"},
  {"name": "featurePProvinceId", "as": "featurePProvinceIdRerankList"},
  {"name": "featurePCityId", "as": "featurePCityIdRerankList"},
  {"name": "featurePUploadType", "as": "featurePUploadTypeRerankList"},
  {"name": "featurePDurationMs", "as": "featurePDurationMsRerankList"},
  {"name": "item_seq", "as": "item_seqRerankList"},
  {"name": "fullrank_sim_out_pctr", "as": "fullrank_sim_out_pctrRerankList"},
  {"name": "fullrank_sim_pltr", "as": "fullrank_sim_pltrRerankList"},
  {"name": "fullrank_sim_pwtr", "as": "fullrank_sim_pwtrRerankList"},
  {"name": "fullrank_sim_pftr", "as": "fullrank_sim_pftrRerankList"},
  {"name": "fullrank_sim_psvr", "as": "fullrank_sim_psvrRerankList"},
  {"name": "fullrank_sim_plvtr", "as": "fullrank_sim_plvtrRerankList"},
  {"name": "fullrank_sim_pcmtr", "as": "fullrank_sim_pcmtrRerankList"},
  {"name": "fullrank_sim_pptr", "as": "fullrank_sim_pptrRerankList"},
  {"name": "fullrank_sim_pcmef", "as": "fullrank_sim_pcmefRerankList"},
  {"name": "fullrank_sim_phtr", "as": "fullrank_sim_phtrRerankList"},
  {"name": "fullrank_sim_pevtr", "as": "fullrank_sim_pevtrRerankList"},
  {"name": "fullrank_sim_pvtr", "as": "fullrank_sim_pvtrRerankList"},
  {"name": "fullrank_sim_pepstr", "as": "fullrank_sim_pepstrRerankList"},
  {"name": "fullrank_sim_pfintr", "as": "fullrank_sim_pfintrRerankList"},
  {"name": "fullrank_sim_pcltr", "as": "fullrank_sim_pcltrRerankList"},
  {"name": "fullrank_sim_pcpr", "as": "fullrank_sim_pcprRerankList"},
  {"name": "fullrank_sim_lstr", "as": "fullrank_sim_lstrRerankList"},
  {"name": "fullrank_ori_pswptr", "as": "fullrank_ori_pswptrRerankList"},
  {"name": "fullrank_detail_new_pevtr_v2", "as": "fullrank_detail_new_pevtr_v2RerankList"},
]

user_features_fulllink = [
    {"name": "featureUId", "as": "uId"},
    {"name": "featureDeviceId",  "as": "dId"},
    "uGender",
    "uBasicAge",
    {"name": "featureProvinceId",  "as": "uProvinceId"},
    {"name": "featureCityId",  "as": "uCityId"},
    {"name": "featureRealtimeClickList", "as": "uClickPids"},
    {"name": "featureRealtimeLikeList", "as": "uLikePids"},
    {"name": "uRealtimeNegativeList", "as": "uNegativePids"},
    {"name": "featureUserProfileV1FollowAidList", "as": "uFollowAids"},
    {"name": "uRequestHour", "as": "requestTime"},
    "uRequestWeekday"
  ]

rerank_features_fulllink = [
    # 精排预测结果
    {"name": "fullrank_sim_pevtr",  "as": "pPctrRerankList"},
    {"name": "fullrank_sim_pltr",  "as": "pPltrRerankList"},
    {"name": "fullrank_sim_pwtr",  "as": "pPwtrRerankList"},
    {"name": "fullrank_sim_pftr",  "as": "pPftrRerankList"},
    {"name": "fullrank_sim_pvtr",  "as": "pPvtrRerankList"},
    {"name": "fullrank_sim_pptr",  "as": "pPptrRerankList"},
    {"name": "fullrank_sim_plvtr", "as": "pPlvtrRerankList"},
    {"name": "fullrank_sim_pcmtr",  "as": "pPcmtrRerankList"},

    # 粗排预测结果
    {"name": "cascade_pctr", "as": "pMcPctrRerankList"},
    {"name": "cascade_pltr", "as": "pMcPltrRerankList"},
    {"name": "cascade_pwtr", "as": "pMcPwtrRerankList"},
    {"name": "cascade_plvtr", "as": "pMcPlvtrRerankList"},
    {"name": "cascade_pftr", "as": "pMcPftrRerankList"},
    {"name": "cascade_pcmtr", "as": "pMcPcmtrRerankList"},
    {"name": "cascade_ptr", "as": "pMcPptrRerankList"},

    # emp xtr
    {"name": "fullrank_empirical_ctr", "as": "pEmpCtrRerankList"},
    {"name": "fullrank_empirical_ltr", "as": "pEmpLtrRerankList"},
    {"name": "fullrank_empirical_wtr", "as": "pEmpWtrRerankList"},
    {"name": "fullrank_empirical_ftr", "as": "pEmpFtrRerankList"},
    {"name": "fullrank_empirical_ptr", "as": "pEmpPtrRerankList"},
    {"name": "fullrank_empirical_htr", "as": "pEmpHtrRerankList"},
    {"name": "fullrank_empirical_cmtr", "as": "pEmpCmtrRerankList"},
    {"name": "fullrank_empirical_watchtime", "as": "pEmpWatchTimeRerankList"},

    # photo info
    {"name": "photo_id", "as": "pidRerankList"},
    {"name": "author__id", "as": "aidRerankList"},
    # 需要处理 打印看看
    {"name": "duration_ms", "as": "pDurationMsRerankList"},
    "pHetuTagLevel1RerankList",
    "pHetuTagLevel2RerankList",
    "pHetuTagLevel3RerankList",
]

rerank_distill_fullchain_user_feature = [
  {"name": "featureUId", "as": "uId"},
  {"name": "featureDeviceId",  "as": "dId"},
  {"name": "featureFollowCount",  "as": "uFollowCount"},
  {"name": "featureFansCount",  "as": "uFansCount"},
  {"name": "featureUploadCount",  "as": "uUploadCount"},
  "uGender",
  {"name": "featureTrueGender", "as": "uTrueGender"},
  {"name": "featureInferYear", "as": "uInferYear"},
  {"name": "featureTrueYear", "as": "uTrueYear"},
  "uBasicAge",
  {"name": "featureVisitMod", "as": "uVisitMod"},
  {"name": "featureUserProfileV1ClickPidList", "as": "uClickPids"},
  {"name": "featureUserProfileV1LikePidList", "as": "uLikePids"},
  {"name": "featureUserProfileV1FollowAidList", "as": "uFollowAids"},
  {"name": "featureUserProfileV1ForwardPidList", "as": "uForwardPids"},
]

rerank_distill_fullchain_item_feature = [
  {"name": "featurePId", "as": "pid"},
  {"name": "featurePAId",  "as": "aid"},
  {"name": "featurePDurationMs",  "as": "pDurationMs"},
  {"name": "featurePHetuTagLevel1",  "as": "pHetuTagLevel1List"},
  {"name": "featurePHetuTagLevel2",  "as": "pHetuTagLevel2List"},
  {"name": "featurePHetuTagLevel3",  "as": "pHetuTagLevel3List"},
  {"name": "fullrank_sim_out_pctr", "as": "pPctr"},
  {"name": "fullrank_sim_pltr", "as": "pPltr"},
  {"name": "fullrank_sim_pwtr", "as": "pPwtr"},
  {"name": "fullrank_sim_pftr", "as": "pPftr"},
  {"name": "fullrank_sim_pptr", "as": "pPptr"},
  {"name": "fullrank_sim_pcmtr", "as": "pPcmtr"},
  {"name": "fullrank_sim_plvtr", "as": "pPlvtr"},
  {"name": "cascade_pctr", "as": "pMcPctr"},
  {"name": "cascade_pltr", "as": "pMcPltr"},
  {"name": "cascade_pwtr", "as": "pMcPwtr"},
  {"name": "cascade_pftr", "as": "pMcPftr"},
  {"name": "cascade_ptr", "as": "pMcPptr"},
  {"name": "cascade_pcmtr", "as": "pMcPcmtr"},
  {"name": "cascade_plvtr", "as": "pMcPlvtr"},
]


simple_ltr_photo_feature = [
  "featurePId",
  "featurePAId",
  "fullrank_detail_pctr",
  "fullrank_detail_pltr",
  "fullrank_detail_pwtr",
  "fullrank_detail_pftr",
  "fullrank_detail_plvtr",
  "fullrank_detail_psvr",
  "fullrank_detail_pvtr",
  "cascade_pctr",
  "cascade_pltr",
  "cascade_pwtr",
  "cascade_pftr",
  "cascade_plvtr",
  "cascade_psvtr"
]

simple_ltr_user_feature = [
  "featureUId",
  "featureDeviceId",
  "featureUserLtrNew", 
  "featureUserWtrNew", 
  "featureUserFtrNew", 
  "featureUserCmtrNew",
  "featureUserEptrNew",
  "featureUserHtrNew",
  "featureSourcePId",
  "sourcePidAuthorId"
]

user_features_v2 = [{"name": "featureTabId", "as": "featureTab"},]
user_features_v2.extend(user_features)
user_features_v2.extend([
  "featureFountainProfileClickPidList",
  "featureFountainProfileClickAidList",
  "featureFountainProfileClickDnnClusterIds",
  "featureFountainProfileClickImgV3List",
  "featureFountainProfileLikePidList",
  "featureFountainProfileLikeAidList",
  "featureFountainProfileCommentPidList",
  "featureFountainProfileCommentAidList",
  "featureFountainProfileFollowPidList",
  "featureFountainProfileFollowAidList",
  "featureFountainProfileShortViewPidList",
  "featureFountainProfileShortViewAidList",
  "featureFountainProfileEffViewPidList",
  "featureFountainProfileEffViewAidList",
  "featureFountainProfileLongViewPidList",
  "featureFountainProfileLongViewAidList",
  "featureUserRequestCityLevel",
  "featureUserRequestCommunityType",

  {"name": "user_emp_ltr", "as": "featureColossusEmpLtr"},
  {"name": "user_emp_wtr", "as": "featureColossusEmpWtr"},
  {"name": "user_emp_ftr", "as": "featureColossusEmpFtr"},
  {"name": "user_emp_htr", "as": "featureColossusEmpHtr"},
  {"name": "user_emp_cmtr", "as": "featureColossusEmpCmtr"},
  {"name": "user_emp_eptr", "as": "featureColossusEmpPtr"},
  {"name": "user_emp_svtr", "as": "featureColossusEmpSvtr"},
  {"name": "user_emp_evtr", "as": "featureColossusEmpEvtr"},
  {"name": "user_emp_lvtr", "as": "featureColossusEmpLvtr"},
  {"name": "user_emp_fintr", "as": "featureColossusEmpFintr"},
  {"name": "user_emp_watch_time", "as": "featureColossusAvgWatchTime"},
  {"name": "user_emp_finish_rate", "as": "featureColossusAvgFinishRate"},
])
user_features_v2.append("featureUserSeqBehavior")

user_features_v3 = [
  "userFountainSessionRealshowList",
  "userRequestHour",
  "userRequestDayOfWeek",
  "uFindUserActiveDegree",
  "uIsLowActiveUser",
  "uFountainRealtimeLikeCountAttr",
  "uFountainRealtimeFollowCountAttr",
  "uFountainRealtimeForwardCountAttr",
  "uFountainRealtimeCommentCountAttr",
  "uFountainRealtimeShortViewCountAttr",
  "uFountainRealtimeLongViewCountAttr",
  "uFountainRealtimeEffectiveViewCountAttr",
  "uFountainRealtimeFinishViewCountAttr",
  {"name": "page", "as": "featureFountainPageNew"},
  "morePage",
  "featureFountainIsFirstPage",
  "featureUserProfileV1ClickTimestampList",
  "featureUserProfileV1LikeTimestampList",
  "featureUserProfileV1CommentTimestampList",
  "featureUserProfileV1DownloadTimestampList",
  "featureUserProfileV1FollowTimestampList",
  "featureUserProfileV1ForwardTimestampList",
  "featureUserProfileV1ProfileEnterTimestampList",
  "featureSimilarUserList",
  "featureUserIsFountainSplash",
  "featureUserIsFountainRequest",
]
user_features_v3.extend(user_features_v2)

user_ada_weight_feature = [
  "uId",
  "dId",
  "uClickPids",
  "uLikePids",
  "uFollowAids",
  "uFollowCount",
  "uFansCount",
  "uUploadCount",
  "uUploadRate",
  "uTrueNewUser",
  "uLogin",
  "uRiskLevel",
  "uVisitMod",
  "uNetwork",
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
  "uCityLevelNew",
  "uIsDouyin",
]
play_action_feature = [tag + suffix + str(i) for tag in ["longview_", "shortview_"] for suffix in ["", "aid_", "tag_", "play_"] for i in range(30)]
hotshow_cnt_feature = [tag + suffix for tag in ["uHotShow", "uHotClick", "uHotLike", "uHotFollow", "uHotHate", "uHotCollect", "uHotForward"] for suffix in ["1m", "5m", "30m", "1h", "1d", "100n", "1000n"]]
user_ada_weight_feature.extend(play_action_feature)
user_ada_weight_feature.extend(hotshow_cnt_feature)

item_sim_gsu_feature = [
  "gsu_signs",
  "gsu_slots",
  "gsu_bias_signs",
  "gsu_bias_slots"
]