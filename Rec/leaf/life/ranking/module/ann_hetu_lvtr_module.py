from ranking import CommonModule


class ANNHetuLvtrModule(CommonModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.photo_features = [
            # photo & author basic features
            # mmu & hetu features
            {"name": "hetu_tag_level_info__hetu_level_two", "as": "hetu_tag_level_two_list"},
            # fr pxtrs
            {"name": "pctr", "as": "pctr"},
            {"name": "plvtr", "as": "plvtr"},
            {"name": "psvr", "as": "psvr"},
            {"name": "pcmtr", "as": "pcmtr"}
        ]

    def process(self) -> None:
        self.flow \
            .if_("enable_ann_hetu_lvtr == 1") \
                .delegate_enrich(
                    kess_service="{{ann_hetu_lvtr_service}}",
                    recv_item_attrs=[
                        {"name": "accumulate_lvtr_score", "as": "ann_hetu_lvtr_score"}
                    ],
                    timeout_ms=100,
                    send_item_attrs=self.photo_features,
                    send_common_attrs=[
                        {"name": "userInfo", "as": "user_info_str"},
                    ],
                    request_type="default",
                    partition_size="{{ann_hetu_lvtr_partition_size}}",
                ) \
            .end_()

    def post_process(self) -> None:
        pass