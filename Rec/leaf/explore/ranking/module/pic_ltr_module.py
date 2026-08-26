from ranking import CommonModule

class PicLtrModule(CommonModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def user_feature(self):
        features = [
            {"name": "featureUId", "as": "uId"},
            {"name": "did", "as": "dId"},
            {"name": "user_age_segment", "as": "uAgeSeg"},
            {"name": "featureClientId", "as": "uClientId"},
            {"name": "user_gender", "as": "uGender"},
            {"name": "infer_gender", "as": "uInferGender"},
            {"name": "true_gender", "as": "uTrueGender"},
            {"name": "featureUserRequestProvinceId", "as": "uProvinceId"},
            {"name": "featureUserRequestCityId", "as": "uCityId"},
            {"name": "featureVisitMod", "as": "uVisitMod"},
            {"name": "click_list", "as": "uClickPids"},
            {"name": "forward_aids", "as": "uFollowAids"},
            {"name": "like_list", "as": "uLikePids"},
            "uStandardRealShowPicAllIdList",
            "uStandardClickPicAllIdList",
            "uStandardLongviewPicAllIdList",
            "uDoubleOutsideValidPicCluster7dList",
            "uSingleValidPicCluster7dList",
            "uPicGrowthCidList"
        ]
        return features

    def photo_features(self):
        features = [
            {"name": "photo_id", "as": "pId"},
            {"name": "author__id", "as": "aId"},
            {"name": "upload_type", "as": "pUploadType"},
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "pHetuTagLevel1"},
            {"name": "hetu_tag_level_info__hetu_level_two", "as": "pHetuTagLevel2"},
            {"name": "hetu_tag_level_info__hetu_level_five", "as": "pHetuTagLevel5"},
            {"name": "mmu_img_cluster_v3", "as": "pMmuImgClusterV3"},
            {"name": "author__fans_count", "as": "aFansCount"},
            {"name": "pctr", "as": "pPctr"},
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
            {"name": "pcltr", "as" : "pPcltr"},
            {"name": "fr_score1", "as" : "fr_score1"},
            {"name": "fr_score2", "as" : "fr_score2"},
            {"name": "empirical_ctr", "as": "pEmpCtr"},
            {"name": "empirical_ltr", "as": "pEmpLtr"},
            {"name": "empirical_wtr", "as": "pEmpWtr"},
            {"name": "empirical_ftr", "as": "pEmpFtr"},
            {"name": "empirical_ptr", "as": "pEmpPtr"},
            {"name": "empirical_cmtr", "as": "pEmpCmtr"},
            {"name": "empirical_htr", "as": "pEmpHtr"},
            {"name": "empirical_watch_time", "as": "pEmpWatchTime"},
            {"name": "photo_picture_count", "as": "pPictureCount"},
            {"name": "caption_length", "as": "pCaptionLength"},
            {"name": "location__city_id", "as": "pCityId"},
            {"name": "location__province_id", "as": "pProvinceId"},
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
            {"name": "explore_stat__click_count", "as": "pHotClick"},
            {"name": "explore_stat__real_show_count", "as": "pHotRealShow"},
            {"name": "explore_stat__like_count", "as": "pHotLike"},
            {"name": "explore_stat__follow_count", "as": "pHotFollow"},
            {"name": "explore_stat__comment_count", "as": "pHotComment"},
            {"name": "explore_stat__negative_count", "as": "pHotHate"},
            {"name": "picture_type", "as": "pPictureType"},
        ]
        return features

    def div_ctr_trim_user_features(self):
        return [
            "id",
            "device_id",
            "basic_info.age_segment",
            "location.city_id",
            "client_id",
            "gender",
            "infer_gender",
            "true_gender",
            "request_location.province_id",
            "request_location.city_id",
            "visit_mod",
            "user_profile_v1.click_list.photo_id",
        ]

    def div_ctr_photo_features(self):
        return [
            {"name": "author__id", "as": "aId"},
            {"name": "upload_type", "as": "pUploadType"},
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "pHetuTagLevel1"},
            {"name": "hetu_tag_level_info__hetu_level_two", "as": "pHetuTagLevel2"},
            {"name": "pctr", "as": "pPctr"},
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
            {"name": "pcltr", "as" : "pPcltr"},
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
            {"name": "explore_stat__real_show_count", "as": "pHotRealShow"},
            {"name": "explore_stat__click_count", "as" : "pHotClick"},
            {"name": "explore_stat__like_count", "as" : "pHotLike"},
            {"name": "explore_stat__follow_count", "as": "pHotFollow"},
            {"name": "explore_stat__comment_count", "as" : "pHotComment"},
            {"name": "explore_stat__negative_count", "as": "pHotHate"},
        ]
    
    def interest_photo_features(self):
        return [
            {"name": "author__gender", "as": "author_gender"},
            {"name": "hetu_tag_level_info__hetu_cluster_id", "as": "hetu_cluster_id"},
            {"name": "mmu_img_cluster_v3", "as": "mmu_imgcluster_v3"},
            {"name": "picture_type", "as": "picture_type"},
            {"name": "photo_picture_count", "as": "picture_count"},
            {"name": "upload_type", "as": "upload_type"},
            {"name": "author__fans_count", "as": "author_fans_count"},
            {"name": "empirical_ctr", "as": "emp_ctr"},
            {"name": "empirical_ltr", "as": "emp_ltr"},
            {"name": "empirical_wtr", "as": "emp_wtr"},
            {"name": "empirical_ftr", "as": "emp_ftr"},
            {"name": "pctr", "as": "pctr_"},
            {"name": "pltr", "as": "pltr_"},
            {"name": "pwtr", "as": "pwtr_"},
            {"name": "pcmtr", "as": "pcmtr_"},
            {"name": "pcltr", "as": "pcltr_"},
            {"name": "pcmef", "as": "pcmef_"},
            {"name": "pptr", "as": "pptr_"},
            {"name": "plvtr", "as": "plvtr_"},
            {"name": "psvr", "as": "psvtr_"},

            {"name": "cascade_pctr", "as": "mc_pctr_"},
            {"name": "cascade_pltr", "as": "mc_pltr_"},
            {"name": "cascade_pwtr", "as": "mc_pwtr_"},
            {"name": "cascade_pcmtr", "as": "mc_pcmtr_"},
            {"name": "cascade_pcltr", "as": "mc_pcltr_"},
            {"name": "cascade_pepstr", "as": "mc_pepstr_"},
            {"name": "cascade_plvtr", "as": "mc_plvtr_"},
            {"name": "cascade_psvtr", "as": "mc_psvtr_"},
            {"name": "cascade_phtr", "as": "mc_htr_"},
        ]

    def log_photo_features(self):
        ans = []
        for item in self.photo_features():
            ans.append(item['name'])
        return ans
    
    def ui_ltv_photo_feature(self):
        return [
            {"name": "author__id", "as": "author_id"},
            {"name": "author__gender", "as": "author_gender"},
            {"name": "author_age_info__age_segment", "as": "author_age_segment"},
            {"name": "author__fans_count", "as": "author_fans_count"},
            {"name": "hetu_tag_level_info__hetu_cluster_id", "as": "hetu_cluster_id"},
            {"name": "upload_type", "as": "upload_type"},

            {"name": "pctr", "as": "pctr_orig"},
            {"name": "pltr", "as": "pltr_orig"},
            {"name": "pwtr", "as": "pwtr_orig"},
            {"name": "pcmtr", "as": "pcmtr_orig"},
            {"name": "pcltr", "as": "pcltr_orig"},
            {"name": "pcmef", "as": "pcmef_orig"},
            {"name": "pptr", "as": "pptr_orig"},
            {"name": "plvtr", "as": "plvtr_orig"},
            {"name": "psvr", "as": "psvtr_orig"},

            {"name": "cascade_pltr", "as": "mc_pltr_orig"},
            {"name": "cascade_pctr", "as": "mc_pctr_orig"},
            {"name": "cascade_pwtr", "as": "mc_pwtr_orig"},
            {"name": "cascade_pcmtr", "as": "mc_pcmtr_orig"},
            {"name": "cascade_pcltr", "as": "mc_pcltr_orig"},
            {"name": "cascade_pepstr", "as": "mc_pepstr_orig"},
            {"name": "cascade_plvtr", "as": "mc_plvtr_orig"},
            {"name": "cascade_psvtr", "as": "mc_psvtr_orig"},
            {"name": "cascade_phtr", "as": "mc_htr_orig"},
        ]

    def ltv_photo_feature(self):
        return [
            "photo_id",
            {"name": "author__id", "as": "author_id"},
            {"name": "author__gender", "as": "author_gender"},
            {"name": "author_age_info__age_segment", "as": "author_age_segment"},
            {"name": "author__fans_count", "as": "author_fans_count"},
            {"name": "hetu_tag_level_info__hetu_cluster_id", "as": "hetu_cluster_id"},

            {"name": "pctr", "as": "pctr_"},
            {"name": "pltr", "as": "pltr_"},
            {"name": "pwtr", "as": "pwtr_"},
            {"name": "pcmtr", "as": "pcmtr_"},
            {"name": "pcltr", "as": "pcltr_"},
            {"name": "pcmef", "as": "pcmef_"},
            {"name": "pptr", "as": "pptr_"},
            {"name": "plvtr", "as": "plvtr_"},
            {"name": "psvr", "as": "psvtr_"},

            {"name": "cascade_pctr", "as": "mc_pctr_"},
            {"name": "cascade_pltr", "as": "mc_pltr_"},
            {"name": "cascade_pwtr", "as": "mc_pwtr_"},
            {"name": "cascade_pcmtr", "as": "mc_pcmtr_"},
            {"name": "cascade_pcltr", "as": "mc_pcltr_"},
            {"name": "cascade_pepstr", "as": "mc_pepstr_"},
            {"name": "cascade_plvtr", "as": "mc_plvtr_"},
            {"name": "cascade_psvtr", "as": "mc_psvtr_"},
            {"name": "cascade_phtr", "as": "mc_htr_"},
        ]

    def u2u_user_feature(self):
        features = [
            {"name": "featureUId", "as": "uId"},
            {"name": "did", "as": "dId"},
            {"name": "user_age_segment", "as": "uAgeSeg"},
            {"name": "featureClientId", "as": "uClientId"},
            {"name": "user_gender", "as": "uGender"},
            {"name": "infer_gender", "as": "uInferGender"},
            {"name": "true_gender", "as": "uTrueGender"},
            {"name": "featureUserRequestPoiType", "as": "uPoiType"},
            {"name": "featureUserRequestProvinceId", "as": "uProvinceId"},
            {"name": "featureUserRequestCityId", "as": "uCityId"},
            {"name": "featureCityId", "as": "uLCityId"},
            {"name": "location_city_level_v2", "as": "uLCityLevel"},
            {"name": "featureVisitMod", "as": "uVisitMod"},
            {"name": "is_douyin", "as": "uIsDouyin"},
            {"name": "app_list", "as": "apps_package"},
            {"name": "click_list", "as": "uClickPids"},
            {"name": "forward_aids", "as": "uFollowAids"},
            {"name": "like_list", "as": "uLikePids"},
            {"name": "follow_list", "as": "uFollowPids"},
            {"name": "videoPlayingPid", "as": "uPlayStatPids"},
            {"name": "playstat_playtimes", "as": "uPlayStatTimes"},
            "uDoubleOutsideValidPicCluster7dList",
            "uSingleValidPicCluster7dList",
            "uPicGrowthCidList"
        ]
        return features

    def u2u_photo_features(self):
        features = [
            {"name": "pctr", "as": "pPctr"},
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
            {"name": "pcltr", "as" : "pPcltr"},
            {"name": "fr_score1", "as" : "fr_score1"},
            {"name": "fr_score2", "as" : "fr_score2"},
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
        ]
        return features
    
    def ua_score_user_feature(self):
        features = [
            {"name": "featureUId", "as": "uId"},
            {"name": "did", "as": "dId"},
            {"name": "user_gender", "as": "uGender"},
            {"name": "user_age_segment", "as": "uAgeSeg"},
            {"name": "featureUserRequestProvinceId", "as": "uProvinceId"},
            {"name": "featureUserRequestCityId", "as": "uCityId"},
            {"name": "location_city_level_v2", "as": "uLCityLevel"},
            {"name": "app_list", "as": "uAppListHash"},
            {"name": "click_aids", "as": "click_author_list"},
            {"name": "like_aids", "as": "like_author_list"},
            {"name": "follow_aids", "as": "follow_author_list"},
            {"name": "forward_aids", "as": "forward_author_list"},
            {"name": "collect_aids", "as": "collect_author_list"}
        ]
        return features
    
    def ua_score_author_feature(self):
        features = [
            {"name": "author__id", "as": "aId"},
            {"name": "author__gender", "as": "aGender"},
            {"name": "author_age_info__age_segment", "as": "pAuthorAgeSegment"},
        ]
        return features

    def process(self) -> None:
        self.flow \
        .delegate_enrich(
            kess_service="{{explore_pic_ltr_predict_service}}",
            recv_item_attrs=[
                {"name": "weighted_ctr", "as": "pic_ltr_weighted_ctr"},
                {"name": "lvtr", "as": "pic_ltr_lvtr"},
                {"name": "fvtr", "as": "pic_ltr_fvtr"},
                {"name": "wtd", "as": "pic_ltr_wtd"},
                {"name": "acttr", "as": "pic_ltr_acttr"},
                {"name": "ctr_db", "as": "pic_ltr_ctr_db"},
                {"name": "acttr_db", "as": "pic_ltr_acttr_db"},
                {"name": "hot_collect", "as": "pic_ltr_collect"},
                {"name": "like_timing", "as": "pic_ltr_like_timing"},
                {"name": "action_twice", "as": "pic_ltr_action_twice"},
            ],
            timeout_ms=100,
            send_item_attrs=self.photo_features(),
            send_common_attrs=self.user_feature(),
            request_type="default",
            partition_size="{{explore_pic_ltr_predict_partition_size}}",
            target_item={
                "is_picture": 1
            }
        ) \
        .if_("enable_explore_pic_diversity_ltr_predict == 1") \
            .switch_("explore_pic_diversity_ltr_mode") \
                .case_(1) \
                    .delegate_enrich(
                        kess_service = "{{explore_pic_diversity_ltr_predict_service}}",
                        recv_item_attrs = [
                            {"name": "diversity_score", "as": "pic_diversity_score"},
                        ],
                        timeout_ms = 100,
                        send_item_attrs = self.photo_features(),
                        send_common_attrs = [
                            { "name": "userInfo", "as": "user_info_str" },
                            "uStandardRealShowPicAllIdList",
                            "uStandardClickPicAllIdList",
                            "uStandardLongviewPicAllIdList"
                        ],
                        request_type = "{{explore_pic_diversity_ltr_predict_request_type}}",
                        partition_size = "{{explore_pic_diversity_ltr_predict_partition_size}}",
                        target_item = {
                            "is_picture" : 1
                        }
                    ) \
                .default_() \
                    .delegate_enrich(
                        kess_service = "{{explore_pic_diversity_ltr_predict_service}}",
                        recv_item_attrs = [
                            {"name": "diversity_score", "as": "pic_diversity_score"},
                        ],
                        timeout_ms = 100,
                        send_item_attrs = self.photo_features(),
                        send_common_attrs = [
                            { "name": "userInfo", "as": "user_info_str" },
                        ],
                        request_type = "{{explore_pic_diversity_ltr_predict_request_type}}",
                        partition_size = "{{explore_pic_diversity_ltr_predict_partition_size}}",
                        target_item = {
                            "is_picture" : 1
                        }
                    ) \
            .end_() \
        .end_() \
        .if_("enable_explore_pic_ltv_predict == 1") \
            .delegate_enrich(
                kess_service="{{explore_pic_ltv_predict_service}}",
                recv_item_attrs=[
                    {"name": "ltv1", "as": "pic_ltv1"},
                    {"name": "ltv2", "as": "pic_ltv2"},
                ],
                timeout_ms=100,
                send_common_attrs=[
                    {"name": "uId", "as": "user_id"},
                ],
                send_item_attrs=self.ltv_photo_feature(),
                request_type="{{explore_pic_ltv_predict_request_type}}",
                partition_size="{{explore_pic_ltv_predict_partition_size}}",
                target_item={
                    "is_picture": 1
                }
            ) \
        .end_() \
        .if_("enable_explore_pic_ui_ltv_predict == 1") \
            .delegate_enrich(
                kess_service="{{explore_pic_ui_ltv_predict_service}}",
                recv_item_attrs=[
                    {"name": "ui_ltv_over_show", "as": "pic_ui_ltv_over_show"},
                    {"name": "ui_ltv_over_click", "as": "pic_ui_ltv_over_click"},
                ],
                timeout_ms=100,
                send_common_attrs=[
                    {"name": "uId", "as": "user_id"},
                    {"name": "user_age_segment", "as": "user_age_segment"},
                    {"name": "user_gender", "as": "user_gender"},
                ],
                send_item_attrs=self.ui_ltv_photo_feature(),
                request_type="{{explore_pic_ui_ltv_predict_request_type}}",
                partition_size="{{explore_pic_ui_ltv_predict_partition_size}}",
                target_item={
                    "is_picture": 1
                }
            ) \
        .end_() \
        .if_("enable_explore_pic_good_comment_ltr_predict == 1") \
            .delegate_enrich(
                kess_service = "{{explore_pic_good_comment_ltr_predict_service}}",
                recv_item_attrs = [
                    {"name": "pic_good_comment_ltr", "as": "pic_ltr_for_good_comment"}
                ],
                timeout_ms = 100,
                send_common_attrs = [
                    {"name": "uId", "as": "user_id"},
                    {"name": "user_age_segment", "as": "user_age_segment"},
                    {"name": "user_gender", "as": "user_gender"},
                ],
                send_item_attrs = self.ui_ltv_photo_feature(),
                request_type = "{{explore_pic_good_comment_ltr_predict_request_type}}",
                partition_size = "{{explore_pic_good_comment_ltr_predict_partition_size}}",
                target_item = {
                    "is_picture": 1
                }
            ) \
        .end_() \
        .if_("enable_explore_pic_fr_div_ctr_predict == 1") \
            .explore_custom_trim_user_info(
              user_info_attr="userInfo",
              save_trimed_user_info_to_attr="pic_div_ctr_trimmed_user_info",
              trim_user_info=self.div_ctr_trim_user_features(),
            ) \
            .delegate_enrich(
                kess_service="{{explore_pic_fr_div_ctr_predict_service}}",
                recv_item_attrs=[
                    {"name": "pic_div_ctr", "as": "fr_pic_div_ctr"},
                ],
                timeout_ms=100,
                send_common_attrs=[
                    {"name": "pic_div_ctr_trimmed_user_info", "as": "user_info_str"},
                ],
                send_item_attrs=self.div_ctr_photo_features(),
                request_type="{{explore_pic_fr_div_ctr_predict_request_type}}",
                partition_size="{{explore_pic_fr_div_ctr_predict_partition_size}}",
                target_item={
                    "is_picture": 1
                }
            ) \
        .end_() \
        .if_("enable_explore_pic_fr_interest_predict == 1") \
            .delegate_enrich(
                kess_service="{{explore_pic_fr_interest_predict_service}}",
                recv_item_attrs=[
                    {"name": "pic_interest_ctr", "as": "fr_pic_interest_ctr"},
                    {"name": "pic_interest_acttr", "as": "fr_pic_interest_acttr"},
                ],
                timeout_ms=50,
                send_common_attrs=self.user_feature(),
                send_item_attrs=self.photo_features(),
                request_type="{{explore_pic_fr_interest_predict_request_type}}",
                partition_size="{{explore_pic_fr_interest_predict_partition_size}}",
                target_item={
                    "is_picture": 1
                }
            ) \
        .end_() \
        .if_("enable_explore_pic_u2u_ltr_predict == 1") \
            .delegate_enrich(
                kess_service = "{{explore_pic_u2u_ltr_service}}",
                recv_item_attrs = [
                    {"name": "pic_u2u_acttr", "as": "fr_pic_u2u_acttr"},
                    {"name": "pic_u2u_evtr", "as": "fr_pic_u2u_evtr"},
                ],
                timeout_ms = 50,
                send_common_attrs = self.u2u_user_feature(),
                send_item_attrs = self.u2u_photo_features(),
                request_type = "default",
                partition_size = "{{explore_pic_u2u_ltr_partition_size}}",
                target_item = {
                    "is_picture": 1
                }
            ) \
        .end_() \
        .if_("enable_explore_pic_ua_score_predict == 1") \
            .delegate_enrich(
                kess_service = "{{explore_pic_ua_score_service}}",
                recv_item_attrs = [
                    {"name": "ua_action_score", "as": "fr_pic_ua_action_score"},
                    {"name": "ua_click_score", "as": "fr_pic_ua_click_score"},
                ],
                timeout_ms = 50,
                send_common_attrs = self.ua_score_user_feature(),
                send_item_attrs = self.ua_score_author_feature(),
                request_type = "default",
                partition_size = "{{explore_pic_ua_score_partition_size}}",
                target_item = {
                    "is_picture": 1
                }
            ) \
        .end_()

    def post_process(self) -> None:
        self.flow \
            .log_debug_info(
                item_attrs = self.log_photo_features(),
                for_debug_request_only = True,
                target_item = {
                    "is_picture" : 1
                }
            )
