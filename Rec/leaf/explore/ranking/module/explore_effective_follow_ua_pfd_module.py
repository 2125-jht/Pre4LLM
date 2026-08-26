from ranking import CommonModule

class ExploreEffectiveFollowUAPfdModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def uni_feature_trim_user_info(self):
    features = [
      "device_stat.human_action",
      "device_stat.device_status_flags",
      "id",
      "device_id",
      "gender"
      "user_active_level",
      "location.province_id",
      "location.city_id",
      "basic_info.age_segment",
    ]
    return features

  def uni_feature_context_info(self):
    features = [
      {"name": "pctr", "as": "pctr"},
      {"name": "pltr", "as": "pltr"},
      {"name": "pftr", "as": "pftr"},
      {"name": "pwtr", "as": "pwtr"},
      {"name": "plvtr", "as": "plvtr"},
      {"name": "pvtr", "as": "pvtr"},
      {"name": "pptr", "as": "pptr"},
      {"name": "cascade_pctr", "as": "cascade_pctr"},
      {"name": "cascade_pltr", "as": "cascade_pltr"},
      {"name": "cascade_pftr", "as": "cascade_pftr"},
      {"name": "cascade_pwtr", "as": "cascade_pwtr"},
    ]
    return features

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_effective_follow_ua_pfd_model == 1") \
        .explore_custom_trim_user_info(
          user_info_attr = "userInfo",
          save_trimed_user_info_to_attr = "explore_effective_follow_feature_trimmed_user_info",
          trim_user_info = self.uni_feature_trim_user_info(),
        ) \
        .delegate_enrich(
          kess_service = "{{explore_effective_follow_ua_pfd_kess_service_name}}",
          request_type = "{{explore_effective_follow_ua_pfd_model_request_type}}",
          recv_item_attrs = [
            {"name": "is_effective_follow", "as": "effective_follow_ua_pfd_score"},
          ],
          timeout_ms = 100,
          send_item_attrs = self.uni_feature_context_info(),
          send_common_attrs = [
            {"name": "explore_effective_follow_feature_trimmed_user_info", "as": "user_info_str"},
          ],
          partition_size = "{{explore_effective_follow_ua_pfd_partition_size}}",
        ) \
      .end_()