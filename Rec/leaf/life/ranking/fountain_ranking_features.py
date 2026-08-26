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
  # "explore_stat__comment_count",
  "author__exp_stat__exp_click",
  "author__exp_stat__exp_like",
  "author__exp_stat__exp_follow",
  "author__exp_stat__exp_long_view",
  "author__exp_stat__exp_realshow",
  "author__exp_stat__exp_forward",
  "author__exp_stat__exp_short_view",
  "author__exp_stat__exp_watch_time",
  "music",
  # "collect_count",
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
  # {
  #   "from_item": "hetu_tag_level_info__hetu_level_four",
  #   "to_item": "featurePHetuTagLevel4"
  # },
  {
    "from_item": "hetu_tag_level_info__hetu_level_five",
    "to_item": "featurePHetuTagLevel5"
  },
  # {
  #   "from_item": "hetu_tag_level_info__hetu_face_id",
  #   "to_item": "featurePHetuFaceId"
  # },
  # {
  #   "from_item": "author__category_detail__first_level_id",
  #   "to_item": "featureACategoryLevelOne"
  # },
  # {
  #   "from_item": "author__category_detail__second_level_id",
  #   "to_item": "featureACategoryLevelTwo"
  # },
  # {
  #   "from_item": "author__category_detail__third_level_id",
  #   "to_item": "featureACategoryLevelThree"
  # },
  # {
  #   "from_item": "author__category_detail__fourth_level_id",
  #   "to_item": "featureACategoryLevelFour"
  # },
  {
    "from_item": "online_lda_topic__ids",
    "to_item": "featurePOnlineLdaTopic"
  }
]

fullrank_splash_attrs = [
  # "is_three_stage_photo_prob",
  # "explore_stat__report_detail__total_report_count",
]

fullrank_fast_attrs = [
  # mmr 依赖字段
  "tag",
  "mmu_img_cluster_v3",
  "photo_dnn_cluster_id",
  # "mmu_text_cluster",
  # "GE_cluster_id",
  # "mmu_text_lda_topic",
  # "author__category_detail__first_level_id",
  # "author__category_detail__second_level_id",
  # "author__category_detail__third_level_id",
  # 打散依赖字段
  # "similar_event_id",
  # "author__is_pr_account",
  # "author__is_gr_account",
  "online_lda_topic__ids",
  "hetu_tag_level_info__hetu_level_four",
  # "author__category_detail__fourth_level_id",
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