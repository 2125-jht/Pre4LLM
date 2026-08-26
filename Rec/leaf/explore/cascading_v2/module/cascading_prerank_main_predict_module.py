from cascading_v2 import CommonModule

class CascadingPrerankMainPredictModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .switch_("explore_prerank_model") \
        .case_(2) \
          .explore_custom_trim_user_info(
            user_info_attr = "userInfo",
            save_trimed_user_info_to_attr = "explore_prerank_trimmed_user_info",
            trim_user_info = [
              "id",
              "device_id",
              "gender",
              "location.city_id",
              "basic_info.age_segment",
              "visit_mod",
              "follow_count",
              "upload_count",
              "request_location.city_id",
              "request_location.province_id",
              "realtime_click_list",
              "realtime_follow_list",
              "realtime_like_list",
              "user_profile_v1.click_list.author_id",
              "user_profile_v1.click_list.photo_id",
              "user_profile_v1.follow_list.author_id",
              "user_profile_v1.follow_list.photo_id",
              "user_profile_v1.like_list.author_id",
              "user_profile_v1.like_list.photo_id",
            ]
          ) \
          .delegate_enrich(
            kess_service = "{{cascade_prerank_tower_service_v1}}",
            recv_item_attrs = [
              {"name": "distill_fr", "as": "cascade_prerank_pctr"},
              {"name": "distill_rerank", "as": "cascade_prerank_pltr"},
              {"name": "distill_show", "as": "cascade_prerank_prstr"}
            ],
            timeout_ms = 50,
            send_common_attrs = [
              "explore_prerank_trimmed_user_info",
              "find_user_active_degree",
            ],
            request_type = "predict_by_mc_prerank_v0",
          ) \
        .default_() \
          .explore_prerank_trim_userinfo(
            user_info_ptr_attr = "user_info_ptr",
            output_user_info_attr = "prerank_trim_user_info",
          ) \
          .get_item_attr_by_predict_fetcher_v2(
            kess_service = "{{cascade_prerank_tower_service}}",
            service_group = "PRODUCTION",
            timeout_ms = 100,
            user_info_attr = "prerank_trim_user_info",
            try_parse_user_info = False,
            output_prefix = "cascade_prerank_",
            tower_request_type = "{{cascade_prerank_tower_request_type}}",
            pxtr = ["pctr", "pltr"],
          ) \
      .end_()
