from ranking import CommonModule

class ExploreUATrackingScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def uni_feature_trim_user_info(self):
    features = [
      "id",
      "device_id",
      "gender",
      "user_profile_v1.like_list.photo_id",
      "user_profile_v1.follow_list.photo_id",
      "user_profile_v1.click_list.photo_id"
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
      {"name": "cascade_plvtr", "as": "cascade_plvtr"},
      {"name": "cascade_psvtr", "as": "cascade_psvr"},
      {"name": "cascade_pftr", "as": "cascade_pftr"},
      {"name": "cascade_pwtr", "as": "cascade_pwtr"}
    ]
    return features

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_ua_tracking_model == 1") \
        .explore_custom_trim_user_info(
          user_info_attr = "userInfo",
          save_trimed_user_info_to_attr = "explore_ua_tracking_feature_trimmed_user_info",
          trim_user_info = self.uni_feature_trim_user_info(),
        ) \
        .delegate_enrich(
          kess_service = "{{explore_ua_tracking_model_kess_service_name}}",
          request_type = "{{explore_ua_tracking_model_request_type}}",
          recv_item_attrs = [
            {"name": "active_tracking_score", "as": "ua_active_tracking_score"},
            {"name": "author_follow_long_reward_value", "as": "ua_tracking_score"},
          ],
          timeout_ms = 100,
          send_item_attrs = self.uni_feature_context_info(),
          send_common_attrs = [
            {"name": "explore_ua_tracking_feature_trimmed_user_info", "as": "user_info_str"},
          ],
          partition_size = "{{explore_ua_tracking_model_partition_size}}",
        ) \
      .end_()