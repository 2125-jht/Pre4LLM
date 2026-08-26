from ranking import CommonModule

class ExploreEffectiveFollowModule(CommonModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def uni_feature_trim_user_info(self):
        features = [
            "device_stat.human_action",
            "device_stat.device_status_flags",
            "id",
            "gender",
            "device_id",
            "user_active_level",
            "location.province_id",
            "location.city_id",
            "user_profile_v1.click_list.photo_id",
            "user_profile_v1.like_list.photo_id",
            "user_profile_v1.follow_list.photo_id",
        ]

        return features

    def uni_feature_context_into(self):
        features = [
            "pctr",
            "pltr",
            "pftr",
            "pwtr",
            "plvtr",
            "pvtr",
            "pptr",
            "cascade_pctr",
            "cascade_pltr",
            "cascade_pftr",
            "cascade_pwtr",
        ]

        return features      

    def process(self) -> None:
        self.flow \
          .if_("enable_explore_effective_follow_model == 1") \
            .explore_custom_trim_user_info(
              user_info_attr = "userInfo",
              save_trimed_user_info_to_attr = "explore_effective_follow_feature_trimmed_user_info",
              trim_user_info = self.uni_feature_trim_user_info(),
            ) \
            .delegate_enrich(
              name = "explore_effective_follow_model",
              kess_service = "{{explore_effective_follow_kess_service}}",
              recv_item_attrs = [
                {"name": "is_effective_follow", "as": "effective_follow_rate_score"},
                {"name": "effective_follow_value", "as": "effective_follow_value_score"}
              ],
              timeout_ms = 100,
              send_item_attrs = self.uni_feature_context_into(),
              send_common_attrs = [
                {"name": "explore_effective_follow_feature_trimmed_user_info", "as": "user_info_str"},
              ],
              partition_size = "{{explore_effective_follow_partition_size}}",
            ) \
          .else_if_("enable_explore_effective_interact_model == 1") \
            .explore_custom_trim_user_info(
              user_info_attr = "userInfo",
              save_trimed_user_info_to_attr = "explore_effective_interact_feature_trimmed_user_info",
              trim_user_info = self.uni_feature_trim_user_info(),
            ) \
            .delegate_enrich(
              kess_service = "{{explore_effective_interact_kess_service}}",
              recv_item_attrs = [
                {"name": "is_effective_follow", "as": "effective_follow_rate_score"},
                {"name": "effective_follow_value", "as": "effective_follow_value_score"},
                {"name": "is_effective_interact", "as": "effective_interact_rate_score"},
                {"name": "effective_interact_value", "as": "effective_interact_value_score"},
              ],
              timeout_ms = 100,
              send_item_attrs = self.uni_feature_context_into(),
              send_common_attrs = [
                {"name": "explore_effective_interact_feature_trimmed_user_info", "as": "user_info_str"},
              ],
              partition_size = "{{explore_effective_interact_partition_size}}",
            ) \
          .end_()