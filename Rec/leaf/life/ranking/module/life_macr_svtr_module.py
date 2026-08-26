from ranking import CommonModule


class LifeMacrSvtrModule(CommonModule):
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
            .if_("enable_svtr_macr_model_v2 == 1") \
                .delegate_enrich(
                    kess_service="{{svtr_macr_model_service}}",
                    recv_item_attrs=[
                        {"name": "svtr_pred", "as": "life_dnn_svtr"},
                        {"name": "lvtr_pred", "as": "life_dnn_lvtr"},
                        {"name": "evtr_pred", "as": "life_dnn_evtr"},
                    ],
                    timeout_ms=100,
                    send_item_attrs=self.photo_features,
                    send_common_attrs=[
                        {"name": "userInfo", "as": "user_info_str"},
                        {"name": "virtualTabId", "as": "tab_id"}
                    ],
                    request_type="default",
                    partition_size="{{macr_mt_model_partition_size}}",
                ) \
            .set_attr_default_value(
                item_attrs=[
                    {
                    "name": "one_default_value",
                    "type": "int",
                    "value": 1
                    }
                ]
                )\
            .item_attr_operation(
                item_attr_a="one_default_value",
                item_attr_b="life_dnn_svtr",
                operator="-",
                output_attr="life_dnn_svtr"
                )\
            .copy_attr(
                attrs=[{
                    "from_item": "life_dnn_svtr",
                    "to_item": "macr_svtr"
                }]
            )\
            .switch_("score_psvr_version") \
            .case_(3) \
            .copy_attr(
                attrs=[{
                    "from_item": "empirical_svtr",
                    "to_item": "macr_svtr"
                }]
                )\
            .end_() \
            .end_() \

    def post_process(self) -> None:
        self.flow \
        .log_debug_info(
          common_attrs = ["virtualTabId"],
          item_attrs = ["pctr","pltr","pwtr","life_dnn_svtr",
                        "life_dnn_lvtr","life_dnn_evtr"
                        ],
          for_debug_request_only = True
        )
