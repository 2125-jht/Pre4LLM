from cascading import CommonModule

class CascadingExploreKocHtrModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def uni_feature_trim_user_info(self):
    features = [
      "id",
      "device_id",
      "request_location.city_id",
      "request_location.province_id",
      "gender",
      "infer_year",
      "basic_info.age_segment",
      "visit_net",
      "location.city_level",
      "is_douyin",
      "active_days",
      "follow_count",
      "fans_count",
      "upload_count",
      "user_profile_v1.click_list.photo_id",
      "realtime_follow_list",
      "realtime_like_list",
      "user_profile_v1.click_list.hetu_tag_level_info.hetu_level_one",
      "user_profile_v1.click_list.hetu_tag_level_info.hetu_level_two",
      "user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_one",
      "user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_two",
      "user_profile_v1.like_list.hetu_tag_level_info.hetu_level_one",
      "user_profile_v1.like_list.hetu_tag_level_info.hetu_level_two",
      "user_profile_v1.video_playing_stat.author_id",
      "user_profile_v1.video_playing_stat.photo_id",
      "user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_one",
      "user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_two"
    ]

    return features

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_cascading_koc_htr_model == 1 and recent_hate_count > explore_cascading_koc_htr_count_threshold") \
        .explore_custom_trim_user_info(
          user_info_attr = "userInfo",
          save_trimed_user_info_to_attr = "explore_cascading_explore_koc_htr_feature_trimmed_user_info",
          trim_user_info = self.uni_feature_trim_user_info(),
        ) \
        .delegate_enrich(
          name = "explore_cascading_koc_htr",
          kess_service = "{{explore_cascading_koc_htr_kess_service}}",
          recv_item_attrs = [
            {"name": "cover_htr", "as": "cascading_cover_htr"},
            {"name": "detail_htr", "as": "cascading_detail_htr"},
          ],
          timeout_ms = 100,
          send_common_attrs = [
            {"name": "explore_cascading_explore_koc_htr_feature_trimmed_user_info", "as": "user_info_str"},
            {"name": "uMultiDimensionGroupKV", "as": "user_feasury_multi_dimension_group"},
            {"name": "uMultiDimensionGroupDetailKV", "as": "user_feasury_multi_dimension_group_detail"},
          ],
          request_type = "{{explore_cascading_koc_htr_request_type}}"
        ) \
      .end_()