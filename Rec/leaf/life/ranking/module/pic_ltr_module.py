from ranking import CommonModule

class PicLtrModule(CommonModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def user_feature(self):
        features = [
            "uId",
            "dId",
            "uCityId",
            "uProvinceId",
            "uVisitMod",
            "uNetwork",
            "uGender",
            "uTrueGender",
            "uBasicGender",
            "uTrueYear",
            "uInferYear",
            "uBasicAge",
            "uFollowCount",
            "uClickPids",
            "uLikePids",
            "uFollowAids",
            "uAgeSeg",
            "uRealShowPidList",
            "uStandardRealShowPicAllIdList",
            "uStandardClickPicAllIdList",
            "uStandardLongviewPicAllIdList"
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
        ]
        return features

    def log_photo_features(self):
        ans = []
        for item in self.photo_features():
            ans.append(item['name'])
        return ans

    def process(self) -> None:
        self.flow \
        .switch_("explore_pic_ltr_mode") \
            .case_(1) \
                .delegate_enrich(
                    kess_service="{{explore_pic_mix_ltr_predict_service}}",
                    recv_item_attrs=[
                        {"name": "weighted_ctr", "as": "pic_ltr_weighted_ctr"},
                        {"name": "lvtr", "as": "pic_ltr_lvtr"},
                        {"name": "fvtr", "as": "pic_ltr_fvtr"},
                        {"name": "wtd", "as": "pic_ltr_wtd"},
                        {"name": "acttr", "as": "pic_ltr_acttr"},
                    ],
                    timeout_ms=100,
                    send_item_attrs=self.photo_features(),
                    send_common_attrs=[
                        {"name": "userInfo", "as": "user_info_str"},
                    ],
                    request_type="{{explore_pic_mix_ltr_predict_request_type}}",
                    partition_size="{{explore_pic_mix_ltr_predict_partition_size}}",
                    target_item={
                        "is_picture": 1
                    }
                ) \
            .default_() \
                .if_("skip_explore_pic_ltr_predict == 0") \
                    .explore_common_user_feature_enricher(
                        user_info_attr="user_info_ptr",
                        user_uid_attr="uId",
                        user_did_attr="dId",
                        user_province_attr="uProvinceId",
                        user_city_attr="uCityId",
                        user_visit_mod_attr="uVisitMod",
                        user_visit_net_attr="uNetwork",
                        user_follow_cnt_attr="uFollowCount",
                        user_gender_attr="uGender",
                        user_ture_gender_attr="uTrueGender",
                        user_basic_gender_attr="uBasicGender",
                        user_infer_year_attr="uInferYear",
                        user_true_year_attr="uTrueYear",
                        user_basic_age_attr="uAgeSeg",
                        user_click_pids_attr="uClickPids",
                        user_like_pids_attr="uLikePids",
                        user_follow_pids_attr="uFollowAids",
                        user_profilev1_real_show_pids_attr="uRealShowPidList",
                    ) \
                    .delegate_enrich(
                        kess_service="{{explore_pic_ltr_predict_service}}",
                        recv_item_attrs=[
                            {"name": "weighted_ctr", "as": "pic_ltr_weighted_ctr"},
                            {"name": "lvtr", "as": "pic_ltr_lvtr"},
                            {"name": "fvtr", "as": "pic_ltr_fvtr"},
                            {"name": "wtd", "as": "pic_ltr_wtd"},
                            {"name": "acttr", "as": "pic_ltr_acttr"},
                        ],
                        timeout_ms=100,
                        send_item_attrs=self.photo_features(),
                        send_common_attrs=self.user_feature(),
                        request_type="pic_ltr_predict",
                        partition_size="{{explore_pic_ltr_predict_partition_size}}",
                        target_item={
                            "is_picture": 1
                        }
                    ) \
                    .if_("skip_explore_pic_revisit_ltr_predict == 0") \
                        .delegate_enrich(
                            kess_service="{{explore_pic_revisit_ltr_predict_service}}",
                            recv_item_attrs=[
                                {"name": "bpr_ctr", "as": "pic_ltr_bpr_ctr"},
                                {"name": "bpr_cltr", "as": "pic_ltr_bpr_cltr"},
                                {"name": "bpr_revisittr", "as": "pic_ltr_bpr_revisittr"},
                            ],
                            timeout_ms=100,
                            send_item_attrs=self.photo_features(),
                            send_common_attrs=self.user_feature(),
                            request_type="{{explore_pic_revisit_ltr_predict_request_type}}",
                            partition_size=48,
                            target_item={
                                "is_picture": 1
                            }
                        ) \
                    .end_() \
                .end_() \
        .end_() \
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

    def post_process(self) -> None:
        self.flow \
            .log_debug_info(
                common_attrs = self.user_feature(),
                item_attrs = self.log_photo_features(),
                for_debug_request_only = True,
                target_item = {
                    "is_picture" : 1
                }
            )
