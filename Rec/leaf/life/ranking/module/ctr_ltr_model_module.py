from ranking import CommonModule


class CtrLtrModelModule(CommonModule):
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
            .if_("enable_ctr_ltr_model == 1") \
                .delegate_enrich(
                    kess_service="{{ctr_ltr_model_service}}",
                    recv_item_attrs=[
                        {"name": "final_pred", "as": "ctr_ltr_model_final_pred"},
                        {"name": "item_pred", "as": "ctr_ltr_model_item_pred"},
                        {"name": "user_pred", "as": "ctr_ltr_model_user_pred"},
                    ],
                    timeout_ms=100,
                    send_item_attrs=self.photo_features,
                    send_common_attrs=[
                        {"name": "userInfo", "as": "user_info_str"},
                        {"name": "virtualTabId", "as": "tab_id"}
                    ],
                    request_type="default",
                    partition_size="{{ctr_ltr_model_partition_size}}",
                ) \
            .calc_by_formula1(
                kconf_key = "formula.scenarioKey92.life_ranking_ctr_ltr_model_pred",
                import_item_attr = [
                    "pctr",
                    "ctr_ltr_model_final_pred",
                    "ctr_ltr_model_item_pred",
                    "ctr_ltr_model_user_pred"
                ],
                export_formula_value = [
                    "xlife_ctr_ltr_model_score"
                ],
                abtest_biz_name = "KUAISHOU_APPS"
                )\
            .end_()

    def post_process(self) -> None:
        self.flow \
        .log_debug_info(
          common_attrs = ["userInfo","virtualTabId"],
          item_attrs = ["pctr","ctr_ltr_model_final_pred","ctr_ltr_model_item_pred","ctr_ltr_model_user_pred","xlife_ctr_ltr_model_score"],
          for_debug_request_only = True
        )
