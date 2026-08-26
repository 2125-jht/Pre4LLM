from ranking import CommonModule


class MMOEMACRModelModule(CommonModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.photo_features = [
            {"name": "photo_id", "as": "photo_id"},
            {"name": "author__id", "as": "author_id"},
            {"name": "hetu_tag_level_info__hetu_cluster_id", "as": "cluster_id"},
            #{"name": "upload_type", "as": "upload_type"},
            {"name": "picture_type", "as": "picture_type"},
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_tag_level_info__hetu_level_one"},
            {"name": "hetu_tag_level_info__hetu_level_two", "as": "hetu_tag_level_info__hetu_level_two"},
            {"name": "pctr", "as": "pctr"},
            {"name": "plvtr", "as": "plvtr"},
            {"name": "psvr", "as": "psvr"},
            {"name": "pcmtr", "as": "pcmtr"},
            {"name": "reason", "as": "reason"},
            {"name": "pltr", "as": "pltr"},
            {"name": "pftr", "as": "pftr"},
            {"name": "pwtr", "as": "pwtr"},
            {"name": "pvtr", "as": "pvtr"},
            {"name": "pptr", "as": "pptr"},
            {"name": "pevtr", "as": "pevr"},
            {"name": "pcltr", "as": "pcltr"},
            {"name": "empirical_ctr", "as": "empirical_ctr"},
            {"name": "empirical_ltr", "as": "empirical_ltr"},
            {"name": "empirical_ftr", "as": "empirical_ftr"},
            {"name": "empirical_ptr", "as": "empirical_ptr"},
            {"name": "empirical_cmtr", "as": "empirical_cmtr"},
            {"name": "cascade_pctr", "as": "cascade_pctr"},
            {"name": "cascade_plvtr", "as": "cascade_plvtr"},
            {"name": "cascade_psvtr", "as": "cascade_psvr"},
            {"name": "cascade_pltr", "as": "cascade_pltr"},
            {"name": "cascade_pwtr", "as": "cascade_pwtr"}
        ]

    def process(self) -> None:
        self.flow \
            .if_("enable_mmoe_macr_model == 1") \
                .delegate_enrich(
                    kess_service="{{mmoe_macr_model_service}}",
                    recv_item_attrs=[
                        {"name": "final_click", "as": "mmoe_macr_model_final_pctr_pred"},
                        {"name": "final_like", "as": "mmoe_macr_model_final_pltr_pred"},
                        {"name": "final_follow", "as": "mmoe_macr_model_final_pwtr_pred"},
                        {"name": "item_click_pred", "as": "mmoe_macr_model_item_pctr_pred"},
                        {"name": "item_like_pred", "as": "mmoe_macr_model_item_pltr_pred"},
                        {"name": "item_follow_pred", "as": "mmoe_macr_model_item_pwtr_pred"},
                        {"name": "user_click_pred", "as": "mmoe_macr_model_user_pctr_pred"},
                        {"name": "user_like_pred", "as": "mmoe_macr_model_user_pltr_pred"},
                        {"name": "user_follow_pred", "as": "mmoe_macr_model_user_pwtr_pred"},
                    ],
                    timeout_ms=100,
                    send_item_attrs=self.photo_features,
                    send_common_attrs=[
                        {"name": "userInfo", "as": "user_info_str"},
                        {"name": "virtualTabId", "as": "tab_id"}
                    ],
                    request_type="default",
                    partition_size="{{mmoe_macr_model_partition_size}}",
                ) \
            .calc_by_formula1(
                kconf_key = "formula.scenarioKey81.life_ranking_mmoe_macr_model_pred",
                import_item_attr = [
                    "pctr",
                    "pltr",
                    "pwtr",
                    "mmoe_macr_model_final_pctr_pred",
                    "mmoe_macr_model_final_pltr_pred",
                    "mmoe_macr_model_final_pwtr_pred",
                    "mmoe_macr_model_item_pctr_pred",
                    "mmoe_macr_model_item_pltr_pred",
                    "mmoe_macr_model_item_pwtr_pred",
                    "mmoe_macr_model_user_pctr_pred",
                    "mmoe_macr_model_user_pltr_pred",
                    "mmoe_macr_model_user_pwtr_pred",
                ],
                export_formula_value = [
                    "xlife_mmoe_macr_pctr_model_score",
                    "xlife_mmoe_macr_pltr_model_score",
                    "xlife_mmoe_macr_pwtr_model_score"
                ],
                abtest_biz_name = "KUAISHOU_APPS"
                )\
            .end_()

    def post_process(self) -> None:
        self.flow \
        .log_debug_info(
          common_attrs = ["userInfo","virtualTabId"],
          item_attrs = ["pctr","pltr","pwtr","mmoe_macr_model_final_pctr_pred","mmoe_macr_model_final_pltr_pred","mmoe_macr_model_final_pwtr_pred","mmoe_macr_model_item_pctr_pred","mmoe_macr_model_item_pltr_pred","mmoe_macr_model_item_pwtr_pred","mmoe_macr_model_user_pctr_pred","mmoe_macr_model_user_pltr_pred","mmoe_macr_model_user_pwtr_pred","xlife_mmoe_macr_pctr_model_score","xlife_mmoe_macr_pltr_model_score","xlife_mmoe_macr_pwtr_model_score"],
          for_debug_request_only = True
        )
