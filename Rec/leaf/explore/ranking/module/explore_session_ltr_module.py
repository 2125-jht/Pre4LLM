from ranking import CommonModule

class ExploreSessionLtrModule(CommonModule):
    def __init__(self, name: str) -> None:
      super().__init__(name)

    def process(self) -> None:
      self.flow \
      .if_("enable_explore_fullrank_sess_model == 1") \
        .explore_custom_trim_user_info(
          user_info_attr = "userInfo",
          save_trimed_user_info_to_attr = "explore_fullrank_sess_trimmed_user_info",
          trim_user_info = [
            "id",
            "basic_info.age_segment",
            "request_location.province_id",
            "request_location.city_id",
            "gender",
            "follow_count",
          ],
        ) \
        .delegate_enrich(
          kess_service="{{explore_fullrank_sess_model_kess_service}}",
          recv_item_attrs = [
            {"name": "reward", "as": "fullrank_sess_reward_score"},
          ],
          timeout_ms = 100,
          send_item_attrs = [
              {"name": "pctr", "as": "pctr_leaf"},
              {"name": "awesome_wtd", "as": "awesome_wtd_leaf"},
              {"name": "pltr", "as": "pltr_leaf"},
              {"name": "pwtr", "as": "pwtr_leaf"},
              {"name": "pcmtr", "as": "pcmtr_leaf"},
              {"name": "pcltr", "as": "pcltr_leaf"},
          ],
          send_common_attrs = [
            {"name": "explore_fullrank_sess_trimmed_user_info", "as": "user_info_str"},
          ],
          partition_size="{{explore_fullrank_sess_model_partition_size}}",
        ) \
      .end_()
