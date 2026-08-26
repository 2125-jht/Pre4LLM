from cascading_v2 import CommonModule

class CascadingDistillLtrPredictModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def uni_feature_trim_user_info(self):
    features = [
      "id",
      "basic_info.age_segment",
      "location.city_id",
      "gender",
      "infer_gender",
      "true_gender",
      "request_location.province_id",
      "request_location.city_id",
      "fountain_reco_user_profile.follow_list.author_id",
      "fountain_reco_user_profile.like_list.photo_id",
      "fountain_reco_user_profile.video_play_stat.photo_id",
      "fountain_reco_user_profile.video_play_stat.video_duration",
      "fountain_reco_user_profile.video_play_stat.playing_time",
      "user_profile_v1.video_playing_stat.playing_time",
      "user_profile_v1.video_playing_stat.photo_id",
      "realtime_click_list",
      "realtime_follow_list",
      "realtime_like_list",
      "upload_count",
      "infer_year",
      "follow_count",
      "fans_count",
      "visit_net",
      "location.city_level",
      "is_douyin",
      "user_interest_profile.hetu_level_one_long_term_id",
      "user_interest_profile.hetu_level_one_long_term_score",
      "user_interest_profile.hetu_level_two_long_term_id",
      "user_interest_profile.hetu_level_two_long_term_score",
      "user_interest_profile.hetu_level_three_long_term_id",
      "user_interest_profile.hetu_level_three_long_term_score",
    ]
    return features

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_mc_distill_ltr_predict == 1") \
        .explore_custom_trim_user_info(
          user_info_attr = "userInfo",
          save_trimed_user_info_to_attr = "explore_cascade_trimmed_user_info",
          trim_user_info = self.uni_feature_trim_user_info(),
        ) \
        .delegate_enrich(
          kess_service = "{{mc_distill_ltr_service}}",
          recv_item_attrs = [
            {"name": "read", "as": "cascade_distill_read"},
            {"name": "finish", "as": "cascade_distill_finish"},
            {"name": "play_7s", "as": "cascade_distill_play_7s"},
            {"name": "play_60s", "as": "cascade_distill_play_60s"}
          ],
          timeout_ms = 100,
          send_common_attrs = [
            "explore_cascade_trimmed_user_info"
          ],
          request_type = "{{mc_distill_ltr_request_type}}"
        ) \
      .end_()