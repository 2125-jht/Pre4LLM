from ranking import CommonModule


class LifeLtrModule(CommonModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.photo_features = [
            # photo & author basic features
            {"name": "photo_id", "as": "pId"},
            {"name": "author__id", "as": "aId"},
            {"name": "upload_type", "as": "pUploadType"},
            {"name": "location__province_id", "as": "pProvinceId"},
            {"name": "author__fans_count", "as": "aFansCount"},
            {"name": "photo_picture_count", "as": "pPictureCount"},
            {"name": "caption_length", "as": "pCaptionLength"},
            {"name": "location__city_id", "as": "pCityId"},
            {"name": "location__province_id", "as": "pProvinceId"},
            {"name": "duration_ms", "as": "pDurationMs"},
            {"name": "upload_type", "as": "pUploadType"},
            {"name": "author__gender", "as": "pAuthorGender"},
            {"name": "author_age_info__age_segment", "as": "pAuthorAgeSeg"},
            "is_picture",
            "upload_time",
            # mmu & hetu features
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "pHetuTagLevel1"},
            {"name": "hetu_tag_level_info__hetu_level_two", "as": "pHetuTagLevel2"},
            {"name": "hetu_tag_level_info__hetu_level_three", "as": "pHetuTagLevel3"},
            {"name": "hetu_tag_level_info__hetu_level_five", "as": "pHetuTagLevel5"},
            {"name": "hetu_tag_level_info__hetu_tag", "as": "pHetuTagLevelTag"},
            {"name": "hetu_tag_level_info__hetu_face_id", "as": "pHetuTagFaceId"},
            {"name": "hetu_tag_level_info__hetu_cluster_id", "as": "pHetuClusterId"},
            {"name": "mmu_img_cluster_v3", "as": "pMmuImgClusterV3"},
            {"name": "music", "as": "pMusic"},
            # fr pxtrs
            {"name": "pctr", "as": "pPctr"},
            {"name": "pltr", "as": "pPltr"},
            {"name": "pwtr", "as": "pPwtr"},
            {"name": "pftr", "as": "pPftr"},
            {"name": "phtr", "as": "pPhtr"},
            {"name": "plvtr", "as": "pPlvtr"},
            {"name": "psvr", "as": "pPsvtr"},
            {"name": "pptr", "as": "pPptr"},
            {"name": "pcmtr", "as": "pPcmtr"},
            {"name": "pcmef", "as": "pPcmef"},
            {"name": "pepstr", "as": "pPepstr"},
            {"name": "pdtr", "as": "pPdtr"},
            {"name": "pcltr", "as": "pPcltr"},
            {"name": "fr_score1", "as": "fr_score1"},
            {"name": "fr_score2", "as": "fr_score2"},
            # mc pxtrs
            {"name": "cascade_pctr", "as": "pMcPctr"},
            {"name": "cascade_pltr", "as": "pMcPltr"},
            {"name": "cascade_pwtr", "as": "pMcPwtr"},
            {"name": "cascade_plvtr", "as": "pMcPlvtr"},
            {"name": "cascade_psvtr", "as": "pMcPsvtr"},
            {"name": "cascade_pftr", "as": "pMcPftr"},
            {"name": "cascade_pcmtr", "as": "pMcPcmtr"},
            {"name": "cascade_plvtr2", "as": "pMcPlvtr2"},
            {"name": "cascade_pepstr", "as": "pMcPepstr"},
            {"name": "cascade_pcestr", "as": "pMcPcestr"},
            # empirical xtrs
            {"name": "empirical_ctr", "as": "pEmpCtr"},
            {"name": "empirical_ltr", "as": "pEmpLtr"},
            {"name": "empirical_wtr", "as": "pEmpWtr"},
            {"name": "empirical_ftr", "as": "pEmpFtr"},
            {"name": "empirical_ptr", "as": "pEmpPtr"},
            {"name": "empirical_cmtr", "as": "pEmpCmtr"},
            {"name": "empirical_htr", "as": "pEmpHtr"},
            {"name": "empirical_watch_time", "as": "pEmpWatchTime"},
            # 计数特征
            {"name": "explore_stat__show_count", "as": "pHotShow"},
            {"name": "explore_stat__click_count", "as": "pHotClick"},
            {"name": "explore_stat__like_count", "as": "pHotLike"},
            {"name": "explore_stat__follow_count", "as": "pHotFollow"},
            {"name": "explore_stat__negative_count", "as": "pHotHate"},
            {"name": "explore_stat__report_detail__total_report_count", "as": "pHotReport"},
        ]

    def process(self) -> None:
      """leave empty function by AutoDelete"""

    def post_process(self) -> None:
        pass
