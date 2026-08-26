from rerank import CommonModule

class RerankGenModule(CommonModule):
    def __init__(self, name: str) -> None:
      super().__init__(name)

    def enrich_fulllink_kai2_feature(self):
        self.flow \
        .explore_custom_trim_user_info(
            user_info_attr = "userInfo",
            save_trimed_user_info_to_attr = "rerank_deep_ltr_trimmed_user_info",
            trim_user_info = [
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
                "user_profile.user_level",
                "realtime_click_list",
                "realtime_follow_list",
                "realtime_forward_list",
                "realtime_like_list",
            ],
        )
    
    def rerank_gen_model(self):
        self.flow \
        .delegate_enrich(
            name = "explore_rerank_gen_model",
            kess_service = "{{explore_rerank_gen_model_kess_service}}",
            recv_item_attrs = [
                "explore_rerank_gen_score"
            ],
            timeout_ms = 100,
            send_item_attrs = [
                "cascade_pctr",
                "cascade_pltr",
                "cascade_pwtr",
                "cascade_plvtr",
                "cascade_psvtr",
                "pctr",
                "pltr",
                "pwtr",
                "pftr",
                "plvtr",
                "pvtr",
                "psvr",
                "pcmtr",
                "pptr",
                "awesome_wtd",
            ],
            send_common_attrs = [
                { "name": "rerank_deep_ltr_trimmed_user_info", "as": "user_info_str" },
            ],
            range_end = "{{explore_rerank_gen_model_limit_num}}",
            request_type = "{{explore_rerank_gen_model_request_type}}",
            partition_size = "{{explore_rerank_gen_model_partition_size}}",
        )
    
    def process(self) -> None:
        self.flow.if_("explore_rerank_enable_gen_model == 1")

        self.enrich_fulllink_kai2_feature()
        
        self.rerank_gen_model()
        
        self.flow.end_()