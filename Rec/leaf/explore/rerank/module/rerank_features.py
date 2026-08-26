rerank_features_new_v2 = [
    # 精排预测结果
    {"name": "pctr",  "as" : "pPctr"},
    {"name": "pltr",  "as" : "pPltr"},
    {"name": "pwtr",  "as" : "pPwtr"},
    {"name": "pftr",  "as" : "pPftr"},
    {"name": "phtr",  "as" : "pPhtr"},
    {"name": "pvtr",  "as" : "pPvtr"},
    {"name": "pptr",  "as" : "pPptr"},
    {"name": "psvr", "as" : "pPsvtr" },
    {"name": "plvtr", "as" : "pPlvtr"},
    {"name": "pcmtr",  "as" : "pPcmtr"},
    {"name": "pcmef", "as" : "pPcmef"},
    {"name": "fr_score1",  "as" : "pPfrScore1"},
    {"name": "fr_score2",  "as" : "pPfrScore2"},
    {"name": "fetr", "as" : "pPfetr" },
    {"name": "fountain_eff", "as" : "pPfountainEff"},

    # 粗排预测结果
    {"name": "cascade_pctr", "as" : "pMcPctr"},
    {"name": "cascade_pltr", "as" : "pMcPltr"},
    {"name": "cascade_pwtr", "as" : "pMcPwtr"},
    {"name": "cascade_plvtr", "as" : "pMcPlvtr"},
    {"name": "cascade_psvtr", "as" : "pMcPsvtr"},
    {"name": "cascade_plvtr2", "as" : "pMcPlvtr2"},
    {"name": "cascade_pepstr", "as" : "pMcPepstr"},
    {"name": "cascade_pwatch_time", "as" : "pMcPwatchTime"},

    # emp xtr
    {"name": "empirical_ctr", "as" : "pEmpCtr"},
    {"name": "empirical_ltr", "as" : "pEmpLtr"},
    {"name": "empirical_wtr", "as" : "pEmpWtr"},
    {"name": "empirical_ftr", "as" : "pEmpFtr"},
    {"name": "empirical_ptr", "as" : "pEmpPtr"},
    {"name": "empirical_htr", "as" : "pEmpHtr"},
    {"name": "empirical_cmtr", "as" : "pEmpCmtr"},
    {"name": "empirical_watch_time", "as" : "pAvgWatchtime"},

    # photo info
    {"name": "photo_id", "as" : "pId"},
    {"name": "author__id", "as" : "aId"},
    # 需要处理 打印看看
    {"name": "photoAgeHour", "as" : "pAgeHour"},
    {"name": "duration_ms", "as" : "pDurationMs"},
    {"name": "upload_type", "as" : "pUploadType"},
    {"name": "author__gender", "as" : "pAuthorGender"},
    {"name": "music", "as" : "pMusic"},
    # 直播说这个更准确
    {"name": "live_photo_info__is_living", "as" : "pHotLiving"},
    # 需要处理 打log看一下是否ok
    {"name": "location__province_id", "as" : "pProvinceId"},
    {"name": "location__city_id", "as" : "pCityId"},
    {"name": "content_safety_level_with_namespace__level_hot_online", "as" : "pContentLevel"},
    {"name": "tag", "as" : "pTag"},
    {"name": "mmu_img_cluster_v3", "as" : "pMmuImgClusterV3"},
    {"name": "mmu_img_cluster_v1", "as" : "pMmuImgClusterV1"},
    {"name": "mmu_content_id", "as" : "pMmuContentId"},
    {"name": "ocr_cover_text_word_count", "as" : "pOcrCoverTextWordCount"},
    {"name": "music_info__music_combo_id", "as" : "pMusicComboId"},
    {"name": "hetu_tag_level_info__hetu_level_one", "as" : "pHetuTagLevel1Id"},
    {"name": "hetu_tag_level_info__hetu_level_two", "as" : "pHetuTagLevel2Id"}
]

rerank_features_new = [
    # 精排预测结果
    {"name": "corr_pctr",  "as" : "pPctr"},
    {"name": "pltr",  "as" : "pPltr"},
    {"name": "pwtr",  "as" : "pPwtr"},
    {"name": "pftr",  "as" : "pPftr"},
    {"name": "phtr",  "as" : "pPhtr"},
    {"name": "pvtr",  "as" : "pPvtr"},
    {"name": "pptr",  "as" : "pPptr"},
    {"name": "psvr", "as" : "pPsvtr" },
    {"name": "plvtr", "as" : "pPlvtr"},
    {"name": "pcmtr",  "as" : "pPcmtr"},
    {"name": "pcmef", "as" : "pPcmef"},
    {"name": "fr_score1",  "as" : "pPfrScore1"},
    {"name": "fr_score2",  "as" : "pPfrScore2"},
    {"name": "pliving_ctr", "as" : "pPlivingctr"},
    {"name": "pliving_wtr", "as" : "pPlivingwtr"},
    {"name": "fetr", "as" : "pPfetr" },
    {"name": "fountain_eff", "as" : "pPfountainEff"},

    # 粗排预测结果
    {"name": "cascade_pctr", "as" : "pMcPctr"},
    {"name": "cascade_pltr", "as" : "pMcPltr"},
    {"name": "cascade_pwtr", "as" : "pMcPwtr"},
    {"name": "cascade_plvtr", "as" : "pMcPlvtr"},
    {"name": "cascade_psvtr", "as" : "pMcPsvtr"},
    {"name": "cascade_plvtr2", "as" : "pMcPlvtr2"},
    {"name": "cascade_pepstr", "as" : "pMcPepstr"},
    {"name": "cascade_pwatch_time", "as" : "pMcPwatchTime"},

    # emp xtr
    {"name": "empirical_ctr", "as" : "pEmpCtr"},
    {"name": "empirical_ltr", "as" : "pEmpLtr"},
    {"name": "empirical_wtr", "as" : "pEmpWtr"},
    {"name": "empirical_ftr", "as" : "pEmpFtr"},
    {"name": "empirical_ptr", "as" : "pEmpPtr"},
    {"name": "empirical_htr", "as" : "pEmpHtr"},
    {"name": "empirical_cmtr", "as" : "pEmpCmtr"},
    {"name": "empirical_watchtime", "as" : "pAvgWatchtime"},
    # photo info
    {"name": "photo_id", "as" : "pId"},
    {"name": "author__id", "as" : "aId"},
    # 需要处理 打印看看
    {"name": "photo_age_hour", "as" : "pAgeHour"},
    {"name": "duration_ms", "as" : "pDurationMs"},
    {"name": "upload_type", "as" : "pUploadType"},
    {"name": "author__gender", "as" : "pAuthorGender"},
    {"name": "music", "as" : "pMusic"},
    # 直播说这个更准确
    {"name": "live_photo_info__is_living", "as" : "pHotLiving"},
    # 需要处理 打log看一下是否ok
    {"name": "location__province_id", "as" : "pProvinceId"},
    {"name": "location__city_id", "as" : "pCityId"},
    {"name": "content_safety_level_with_namespace__level_hot_online", "as" : "pContentLevel"},
    {"name": "tag", "as" : "pTag"},
    {"name": "mmu_img_cluster_v3", "as" : "pMmuImgClusterV3"},
    {"name": "mmu_img_cluster_v1", "as" : "pMmuImgClusterV1"},
    {"name": "mmu_content_id", "as" : "pMmuContentId"},
    {"name": "ocr_cover_text_word_count", "as" : "pOcrCoverTextWordCount"},
    {"name": "music_info__music_combo_id", "as" : "pMusicComboId"},
    {"name": "hetu_tag_level_info__hetu_level_one", "as" : "pHetuTagLevel1Id"},
    {"name": "hetu_tag_level_info__hetu_level_two", "as" : "pHetuTagLevel2Id"}
]

rerank_features_gen = [
    # 精排预测结果
    {"name": "pctr",  "as" : "pPctrRerankList"},
    {"name": "pltr",  "as" : "pPltrRerankList"},
    {"name": "pwtr",  "as" : "pPwtrRerankList"},
    {"name": "pftr",  "as" : "pPftrRerankList"},
    {"name": "pvtr",  "as" : "pPvtrRerankList"},
    {"name": "pptr",  "as" : "pPptrRerankList"},
    {"name": "plvtr", "as" : "pPlvtrRerankList"},
    {"name": "pcmtr",  "as" : "pPcmtrRerankList"},

    # 粗排预测结果
    {"name": "cascade_pctr", "as" : "pMcPctrRerankList"},
    {"name": "cascade_pltr", "as" : "pMcPltrRerankList"},
    {"name": "cascade_pwtr", "as" : "pMcPwtrRerankList"},
    {"name": "cascade_plvtr", "as" : "pMcPlvtrRerankList"},
    {"name": "cascade_pftr", "as" : "pMcPftrRerankList"},
    {"name": "cascade_pcmtr", "as" : "pMcPcmtrRerankList"},
    {"name": "cascade_ptr", "as" : "pMcPptrRerankList"},

    # emp xtr
    {"name": "empirical_ctr", "as" : "pEmpCtrRerankList"},
    {"name": "empirical_ltr", "as" : "pEmpLtrRerankList"},
    {"name": "empirical_wtr", "as" : "pEmpWtrRerankList"},
    {"name": "empirical_ftr", "as" : "pEmpFtrRerankList"},
    {"name": "empirical_ptr", "as" : "pEmpPtrRerankList"},
    {"name": "empirical_htr", "as" : "pEmpHtrRerankList"},
    {"name": "empirical_cmtr", "as" : "pEmpCmtrRerankList"},
    {"name": "empirical_watch_time", "as" : "pEmpWatchTimeRerankList"},

    # photo info
    {"name": "photo_id", "as" : "pidRerankList"},
    {"name": "author__id", "as" : "aidRerankList"},
    # 需要处理 打印看看
    {"name": "duration_ms", "as" : "pDurationMsRerankList"},
    "pHetuTagLevel1RerankList",
    "pHetuTagLevel2RerankList",
    "pHetuTagLevel3RerankList",
]

rerank_features = [
    # 精排预测结果
    {"name": "corr_pctr",  "as" : "pPctr"},
    {"name": "pltr",  "as" : "pPltr"},
    {"name": "pwtr",  "as" : "pPwtr"},
    {"name": "pftr",  "as" : "pPftr"},
    {"name": "phtr",  "as" : "pPhtr"},
    {"name": "pvtr",  "as" : "pPvtr"},
    {"name": "pptr",  "as" : "pPptr"},
    {"name": "psvr", "as" : "pPsvtr" },
    {"name": "pevtr", "as" : "pPevtr"},
    {"name": "plvtr", "as" : "pPlvtr"},
    {"name": "pcmtr",  "as" : "pPcmtr"},
    {"name": "pdtr", "as" : "pPdctr"},
    {"name": "pfvtr", "as" : "pPfvtr"},
    {"name": "pcmef", "as" : "pPcmef"},
    {"name": "fr_score1",  "as" : "pPfrScore1"},
    {"name": "fr_score2",  "as" : "pPfrScore2"},
    {"name": "pliving_ctr", "as" : "pPlivingctr"},
    {"name": "pliving_wtr", "as" : "pPlivingwtr"},

    # 粗排预测结果
    {"name": "cascade_pctr", "as" : "pMcPctr"},
    {"name": "cascade_pltr", "as" : "pMcPltr"},
    {"name": "cascade_pwtr", "as" : "pMcPwtr"},
    {"name": "cascade_plvtr", "as" : "pMcPlvtr"},
    {"name": "cascade_psvtr", "as" : "pMcPsvtr"},
    {"name": "cascade_pftr", "as" : "pMcPftr"},
    {"name": "cascade_pcmtr", "as" : "pMcPcmtr"},
    {"name": "cascade_plvtr2", "as" : "pMcPlvtr2"},
    {"name": "cascade_pepstr", "as" : "pMcPepstr"},
    {"name": "cascade_pcestr", "as" : "pMcPcestr"},
    {"name": "cascade_pwatch_time", "as" : "pMcPwatchTime"},
    {"name": "cascade_ptr", "as" : "pMcPptr"},

    # emp xtr
    {"name": "empirical_ctr", "as" : "pEmpCtr"},
    {"name": "empirical_ltr", "as" : "pEmpLtr"},
    {"name": "empirical_wtr", "as" : "pEmpWtr"},
    {"name": "empirical_ftr", "as" : "pEmpFtr"},
    {"name": "empirical_ptr", "as" : "pEmpPtr"},
    {"name": "empirical_htr", "as" : "pEmpHtr"},
    {"name": "empirical_cmtr", "as" : "pEmpCmtr"},
    {"name": "empirical_watchtime", "as" : "pAvgWatchtime"},
    {"name": "empirical_rrr", "as" : "pHotRRR"},

    # photo info
    {"name": "photo_id", "as" : "pId"},
    {"name": "author__id", "as" : "aId"},
    {"name": "author__fans_count", "as" : "pAuthorFansCount"},
    # 需要处理 打印看看
    {"name": "photo_age_hour", "as" : "pAgeHour"},
    {"name": "duration_ms", "as" : "pDurationMs"},
    {"name": "upload_type", "as" : "pUploadType"},
    {"name": "explore_stat__show_count", "as" : "pHotShow"},
    {"name": "explore_stat__click_count", "as" : "pHotClick"},
    {"name": "explore_stat__like_count", "as" : "pHotLike"},
    {"name": "explore_stat__follow_count", "as" : "pHotFollow"},
    {"name": "explore_stat__negative_count", "as" : "pHotHate"},
    {"name": "explore_stat__report_detail__total_report_count", "as" : "pHotReport"},
    {"name": "author__gender", "as" : "pAuthorGender"},
    {"name": "author_age_info__age_segment", "as" : "pAuthorAgeSeg"},
    {"name": "music", "as" : "pMusic"},
    # 直播说这个更准确
    {"name": "live_photo_info__is_living", "as" : "pHotLiving"},
    {"name": "mod", "as" : "pPhoneMod"},
    # 需要处理 打log看一下是否ok
    {"name": "reason", "as" : "pHotExptag"},
    {"name": "location__province_id", "as" : "pProvinceId"},
    {"name": "location__city_id", "as" : "pCityId"},
    {"name": "show_level_a", "as" : "pLevelA"},
    {"name": "show_level_b", "as" : "pLevelB"},
    {"name": "content_safety_level_with_namespace__level_hot_online", "as" : "pContentLevel"},
    {"name": "tag", "as" : "pTag"},
    {"name": "photo_dnn_cluster_id", "as" : "pDnnClusterId"},
    {"name": "mmu_img_cluster_v3", "as" : "pMmuImgClusterV3"},
    {"name": "mmu_img_cluster_v1", "as" : "pMmuImgClusterV1"},
    {"name": "mmu_cluster_music_id", "as" : "pMmuClusterMusicId"},
    {"name": "mmu_content_id", "as" : "pMmuContentId"},
    {"name": "ocr_cover_text_word_count", "as" : "pOcrCoverTextWordCount"},
    {"name": "music_info__music_combo_id", "as" : "pMusicComboId"},
    {"name": "hetu_tag_level_info__hetu_level_one", "as" : "pHetuTagLevel1Id"},
    {"name": "hetu_tag_level_info__hetu_level_two", "as" : "pHetuTagLevel2Id"}
]

rerank_revisit_user_features = [
    {"name": "uId", "as": "user_id"},
    {"name": "dId", "as": "device_id"},
    {"name": "uGender", "as": "user_gender"},
    {"name": "user_age_segment", "as": "user_age_segment"},
]

rerank_revisit_item_features = [
    {"name": "photo_id", "as": "photo_id"},
    {"name": "author__id", "as": "author_id"},
    {"name": "hetu_tag_level_info__hetu_cluster_id", "as": "hetu_cluster_id"},
    {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_level_one_tag"},
    {"name": "hetu_tag_level_info__hetu_level_two", "as": "hetu_level_two_tag"},
    {"name": "hetu_tag_level_info__hetu_level_three", "as": "hetu_level_three_tag"},

    {"name": "pPctrOri", "as": "pctr"},
    {"name": "pPltr", "as": "pltr"},
    {"name": "pPwtr", "as": "pwtr"},
    {"name": "pPftr", "as": "pftr"},
    {"name": "pPlvtr", "as": "plvtr"},
    {"name": "pctr", "as": "pctr_true"},
    {"name": "pPcmtr", "as": "pcmtr"},
    {"name": "pPcmef", "as": "pcmef"},
    {"name": "pPptr", "as": "pptr"},
    {"name": "pPsvtr", "as": "psvtr"},
    {"name": "pPhtr", "as": "phtr"},
    {"name": "pctr_index", "as": "pctr_index"},
    {"name": "pltr_index", "as": "pltr_index"},
    {"name": "pwtr_index", "as": "pwtr_index"},
    {"name": "pftr_index", "as": "pftr_index"},
    {"name": "pvtr_index", "as": "pvtr_index"},
    {"name": "plvtr_index", "as": "plvtr_index"},

    {"name": "empirical_ctr", "as": "emp_ctr"},
    {"name": "empirical_ltr", "as": "emp_ltr"},
    {"name": "empirical_wtr", "as": "emp_wtr"},
    {"name": "empirical_ftr", "as": "emp_ftr"},
    {"name": "empirical_lvtr", "as": "emp_lvtr"},
]

rerank_flash_eval_model_send_item_feas = [
  "cascade_pctr",
  "cascade_pltr",
  "cascade_pwtr",
  "cascade_plvtr",
  "cascade_psvtr",
  "cascade_ptr",
  "cascade_pcmtr",
  "cascade_pftr",
  "corr_pctr_psvr",
  "plvtr",
  "awesome_wtd",
  "pctr",
  "pltr",
  "pwtr",
  "pftr",
  "pcmtr",
  "pdtr",
  "pcltr",
  "pptr",
  "pcmef",
  "pepstr",
  "pevtr",
  "fr_score1",
  "fr_score2",
  "phtr",
  "fetr",
  "fountain_eff",
  "is_picture",
  "fr_pic_ensemble_score",
  "explore_fr_ensemble_score",
  "psvr",
]

rerank_flash_eval_model_send_common_feas = [
  { "name": "rerank_deep_ltr_trimmed_user_info", "as": "user_info_str" },
  { "name": "featureSimilarUserList", "as": "similar_user_list" },
  "rerank_list_item_idx_flat_list",
  "page",
]

rerank_flash_eval_model_send_user_feas = [
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
  "user_profile_v1.click_list.author_id",
  "user_profile_v1.click_list.photo_id",
  "user_profile_v1.follow_list.author_id",
  "user_profile_v1.follow_list.photo_id",
  "user_profile_v1.like_list.author_id",
  "user_profile_v1.like_list.photo_id",
  "user_profile_v1.hate_list.photo_id",
  "user_profile_v1.video_playing_stat.playing_time",
  "user_profile_v1.video_playing_stat.author_id",
  "user_profile_v1.video_playing_stat.photo_id",
  "user_profile_v1.video_playing_stat.client_timestamp",
  "user_profile.user_level",
  "realtime_click_list",
  "realtime_follow_list",
  "realtime_forward_list",
  "realtime_like_list"
]

def rerank_list_fea():
  rerank_list_features = [
    # 调用精排 or 通过rankResult传入
    {"name": "corr_pctr",  "as" : "Pctr_context"},
    {"name": "pltr",  "as" : "Pltr_context"},
    {"name": "pwtr",  "as" : "Pwtr_context"},
    {"name": "pftr",  "as" : "Pftr_context"},
    {"name": "pvtr",  "as" : "Pvtr_context"},
    {"name": "pptr",  "as" : "Pptr_context"},
    {"name": "pepstr",  "as" : "Pepstr_context"},
    {"name": "pcmtr",  "as" : "Pcmtr_context"},
    {"name": "pcmef", "as" : "Pcmef_context"},
    {"name": "pcltr", "as" : "Pcltr_context"},
    {"name": "psvr", "as" : "Psvtr_context" },
    {"name": "plvtr", "as" : "Plvtr_context"},
    {"name": "fetr", "as" : "Pfetr_context" },
    {"name": "fountain_eff", "as" : "Pfeff_context"},
    {"name": "fr_score1",  "as" : "Pfrscore1_context"},
    {"name": "fr_score2",  "as" : "Pfrscore2_context"},
  ]
  return rerank_list_features

def rerank_list_fea_v2():
  rerank_list_features = [
    # 调用精排 or 通过rankResult传入
    {"name": "pctr",  "as" : "Pctr_context"},
    {"name": "pltr",  "as" : "Pltr_context"},
    {"name": "pwtr",  "as" : "Pwtr_context"},
    {"name": "pftr",  "as" : "Pftr_context"},
    {"name": "pvtr",  "as" : "Pvtr_context"},
    {"name": "pptr",  "as" : "Pptr_context"},
    {"name": "pepstr",  "as" : "Pepstr_context"},
    {"name": "pcmtr",  "as" : "Pcmtr_context"},
    {"name": "pcmef", "as" : "Pcmef_context"},
    {"name": "pcltr", "as" : "Pcltr_context"},
    {"name": "psvr", "as" : "Psvtr_context" },
    {"name": "plvtr", "as" : "Plvtr_context"},
    {"name": "fetr", "as" : "Pfetr_context" },
    {"name": "fountain_eff", "as" : "Pfeff_context"},
    {"name": "fr_score1",  "as" : "Pfrscore1_context"},
    {"name": "fr_score2",  "as" : "Pfrscore2_context"},
  ]
  return rerank_list_features


def user_features():
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
        "uRealShowNoActionPids",
        "uRealShowNoActionPids_LEN",
        "uRealShowNoActionAids",
        "uRealshowNoActionHetu1", # lower case: s
        "uRealshowNoActionHetu2",
        "uRealshowNoActionHetu3",
        "uRealshowNoActionHetuTag"
    ]

    for suffix in ['', 'aid_', 'tag_', 'play_']:
        for i in range(30):
            features.append('click_' + suffix + str(i))

    for suffix in ['', 'aid_']:
        for i in range(30):
            features.append('realshow_' + suffix + str(i))

    for prefix in ['uPlayTimeFromNow', 'uDuration', 'uPlayTime', 'uActionLabel', 'uPlayActionLabel']:
      for suffix in ["1m", "5m", "10m", "30m", "1h", "2h"]:
        features.append(prefix + suffix)

    return features

def user_features_full_link():
  features = [
    "uId",
    "dId",
    "uBasicAge",
    "uGender",
    "uBasicGender",
    "uProvinceId",
    "uCityId",
    {"name": "uRealtimeClickList", "as": "uClickPids"},
    {"name": "uRealtimeLikeList", "as": "uLikePids"},
    {"name": "uRealtimeForwardList", "as": "uForwardPids"},
    {"name": "uRealtimeNegativeList", "as": "uNegativePids"},
    {"name": "uFollowPhotoAuthorList", "as" : "uFollowAids"},
    {"name": "uRequestHour", "as": "requestTime"},
    "uRequestWeekday"
  ]
  return features

def gen_photo_features_for_idx_position(idx):
    return [
        "pId_idx{}".format(idx),
        "aId_idx{}".format(idx),
        "pPctr_idx{}".format(idx),
        "pPltr_idx{}".format(idx),
        "pPwtr_idx{}".format(idx),
        "pPftr_idx{}".format(idx),
        "pPhtr_idx{}".format(idx),
        "pPvtr_idx{}".format(idx),
        "pPptr_idx{}".format(idx),
        "pPsvtr_idx{}".format(idx),
        "pPevtr_idx{}".format(idx),
        "pPlvtr_idx{}".format(idx),
        "pPcmtr_idx{}".format(idx),
        "pPdctr_idx{}".format(idx),
        "pPfvtr_idx{}".format(idx),
        "pPcmef_idx{}".format(idx),
        "pPfrScore1_idx{}".format(idx),
        "pPfrScore2_idx{}".format(idx),
        "pPlivingctr_idx{}".format(idx),
        "pPlivingwtr_idx{}".format(idx),
        "pPfetr_idx{}".format(idx),
        "pPfountainEff_idx{}".format(idx),

        "pMcPctr_idx{}".format(idx),
        "pMcPltr_idx{}".format(idx),
        "pMcPwtr_idx{}".format(idx),
        "pMcPlvtr_idx{}".format(idx),
        "pMcPsvtr_idx{}".format(idx),
        "pMcPftr_idx{}".format(idx),
        "pMcPcmtr_idx{}".format(idx),
        "pMcPlvtr2_idx{}".format(idx),
        "pMcPepstr_idx{}".format(idx),
        "pMcPcestr_idx{}".format(idx),
        "pMcPwatchTime_idx{}".format(idx),
        "pMcPptr_idx{}".format(idx),

        "pEmpCtr_idx{}".format(idx),
        "pEmpLtr_idx{}".format(idx),
        "pEmpWtr_idx{}".format(idx),
        "pEmpFtr_idx{}".format(idx),
        "pEmpPtr_idx{}".format(idx),
        "pEmpHtr_idx{}".format(idx),
        "pEmpCmtr_idx{}".format(idx),
        "pAvgWatchtime_idx{}".format(idx),
        "pHotRRR_idx{}".format(idx),
        
        "pAuthorFansCount_idx{}".format(idx),
        "pAgeHour_idx{}".format(idx),
        "pDurationMs_idx{}".format(idx),
        "pUploadType_idx{}".format(idx),

        "pHotShow_idx{}".format(idx),
        "pHotClick_idx{}".format(idx),
        "pHotLike_idx{}".format(idx),
        "pHotFollow_idx{}".format(idx),
        "pHotHate_idx{}".format(idx),
        "pHotReport_idx{}".format(idx),
        "pAuthorGender_idx{}".format(idx),
        "pAuthorAgeSeg_idx{}".format(idx),
        "pMusic_idx{}".format(idx),
        "pHotLiving_idx{}".format(idx),
        "pPhoneMod_idx{}".format(idx),
        "pHotExptag_idx{}".format(idx),
        "pProvinceId_idx{}".format(idx),
        "pCityId_idx{}".format(idx),
        "pLevelA_idx{}".format(idx),
        "pLevelB_idx{}".format(idx),
        "pContentLevel_idx{}".format(idx),
        "pTag_idx{}".format(idx),
        "pDnnClusterId_idx{}".format(idx),
        "pMmuImgClusterV3_idx{}".format(idx),
        "pMmuImgClusterV1_idx{}".format(idx),
        "pMmuClusterMusicId_idx{}".format(idx),
        "pMmuContentId_idx{}".format(idx),
        "pOcrCoverTextWordCount_idx{}".format(idx),
        "pMusicComboId_idx{}".format(idx),
        "pHetuTagLevel1Id_idx{}".format(idx),
        "pHetuTagLevel2Id_idx{}".format(idx),
    ]

def gen_photo_context_feature():
  return [
    "maxPctr_context",
    "maxPltr_context",
    "maxPwtr_context",
    "maxPftr_context",
    "maxPvtr_context",
    "maxPptr_context",
    "maxPepstr_context",
    "maxPcltr_context",
    "maxPcmtr_context",
    "maxPcmef_context",
    "maxPsvtr_context",
    "maxPlvtr_context",
    "maxPfetr_context",
    "maxPfeff_context",
    "maxPfrscore1_context",
    "maxPfrscore2_context",
    "avgPctr_context",
    "avgPltr_context",
    "avgPwtr_context",
    "avgPftr_context",
    "avgPvtr_context",
    "avgPptr_context",
    "avgPepstr_context",
    "avgPcltr_context",
    "avgPcmtr_context",
    "avgPcmef_context",
    "avgPsvtr_context",
    "avgPlvtr_context",
    "avgPfetr_context",
    "avgPfeff_context",
    "avgPfrscore1_context",
    "avgPfrscore2_context",
    "avg_duration_context",
    "hetu_level_one_count",
    "hetu_level_two_count",
    "0_9s_duration_photo_count",
    "9_15s_duration_photo_count",
    "15_20s_duration_photo_count",
    "20_58s_duration_photo_count",
    "gt_58s_duration_photo_count",
  ]

def gen_photo_features_for_all_position(n):
    return [attr for idx in range(n) for attr in gen_photo_features_for_idx_position(idx)] + gen_photo_context_feature()

def gen_photo_revisit_item_feature():
    features = [
      "photo_id_list",
      "author_id_list",
      "hetu_cluster_id_list",
      "hetu_level_one_tag_list",
      "hetu_level_two_tag_list",
      "hetu_level_three_tag_list",
      "pctr_list",
      "pltr_list",
      "pwtr_list",
      "plvtr_list",
      "pcmtr_list",
      "pcmef_list",
      "pptr_list",
      "pctr_index_list",
      "pltr_index_list",
      "pwtr_index_list",
      "pvtr_index_list",
      "plvtr_index_list",
      "emp_ctr_list",
      "emp_ltr_list",
      "emp_wtr_list",
      "emp_lvtr_list",
    ]

    return features
