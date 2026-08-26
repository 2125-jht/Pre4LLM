from ranking import CommonModule

class ExploreTeenagerLtrModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def uni_feature_trim_user_info(self):
    features = [
      "id",
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
      "fountain_reco_user_profile.click_list.author_id",
      "fountain_reco_user_profile.click_list.photo_id",
      "fountain_reco_user_profile.comment_list.author_id",
      "fountain_reco_user_profile.comment_list.photo_id",
      "fountain_reco_user_profile.follow_list.author_id",
      "fountain_reco_user_profile.follow_list.photo_id",
      "fountain_reco_user_profile.like_list.author_id",
      "fountain_reco_user_profile.like_list.photo_id",
      "fountain_reco_user_profile.video_play_stat.photo_id",
      "fountain_reco_user_profile.video_play_stat.author_id",
      "fountain_reco_user_profile.video_play_stat.video_duration",
      "fountain_reco_user_profile.video_play_stat.playing_time",
      "user_profile_v1.click_list.author_id",
      "user_profile_v1.click_list.photo_id",
      "user_profile_v1.follow_list.author_id",
      "user_profile_v1.follow_list.photo_id",
      "user_profile_v1.like_list.author_id",
      "user_profile_v1.like_list.photo_id",
      "user_profile_v1.video_playing_stat.playing_time",
      "user_profile_v1.video_playing_stat.author_id",
      "user_profile_v1.video_playing_stat.photo_id",
      "realtime_click_list",
      "realtime_follow_list",
      "realtime_forward_list",
      "realtime_like_list",
      "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_level_one",
      "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_level_two",
      "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_level_three",
      "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_level_five",
      "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_tag",
      "user_profile_v1.click_list.hetu_tag_level_info.hetu_level_one",
      "user_profile_v1.click_list.hetu_tag_level_info.hetu_level_two",
      "user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_one",
      "user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_two",
      "user_profile_v1.like_list.hetu_tag_level_info.hetu_level_one",
      "user_profile_v1.like_list.hetu_tag_level_info.hetu_level_two",
      "user_profile_v1.video_playing_stat.video_duration",
      "user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_one",
      "user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_two",
      "upload_count",
      "infer_year",
      "follow_count",
      "fans_count",
      "visit_net",
      "location.city_level",
      "is_douyin",
      "user_profile_v1.real_show_list.photo_id",
      "user_profile_v1.real_show_list.author_id",
      "user_profile_v1.real_show_list.time_ms",
      "user_profile_v1.real_show_list.page_type",
      "user_profile_v1.real_show_list.label.click",
      "user_profile_v1.real_show_list.label.like",
      "user_profile_v1.real_show_list.label.follow",
      "user_profile_v1.real_show_list.label.hate",
      "feature_collection.explore_low_active_level",
      "user_interest_profile.hetu_level_one_long_term_id",
      "user_interest_profile.hetu_level_one_long_term_score",
      "user_interest_profile.hetu_level_two_long_term_id",
      "user_interest_profile.hetu_level_two_long_term_score",
      "user_interest_profile.hetu_level_three_long_term_id",
      "user_interest_profile.hetu_level_three_long_term_score",
    ]

    return features

  def uni_feature_context_into(self):
    features = [
      "reason",
      "pctr",
      "pltr",
      "pwtr",
      "pftr",
      "phtr",
      "plvtr",
      "psvr",
      "pvtr",
      "awesome_wtd",
      "pptr",
      "pcmtr",
      "pcmef",
      "fr_score1",
      "fr_score2",
      "pdtr",
      "cascade_pctr",
      "cascade_pltr",
      "cascade_pwtr",
      "cascade_plvtr",
      "fetr",
      "fountain_eff",
      "pepstr",
      "pctr_index",
      "plvtr_index",
      "pvtr_index",
      "pltr_index",
      "pftr_index",
      "pwtr_index",
      "pesptr_index",
      "psvr_index"
    ]

    return features

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_teenager_ltr_model == 1 and user_age_segment > 0 and user_age_segment <= explore_teenager_ltr_age_threshold") \
        .explore_custom_trim_user_info(
          user_info_attr = "userInfo",
          save_trimed_user_info_to_attr = "explore_ltr_uni_feature_trimmed_user_info",
          trim_user_info = self.uni_feature_trim_user_info(),
        ) \
        .delegate_enrich(
          kess_service = "{{explore_teenager_ltr_kess_service}}",
          recv_item_attrs = [
            {"name": "ctr", "as": "teenager_ctr"},
            {"name": "wtd", "as": "teenager_wtd"},
            {"name": "ltr", "as": "teenager_ltr"}
          ],
          timeout_ms = 100,
          send_item_attrs = self.uni_feature_context_into(),
          send_common_attrs = [
            {"name": "explore_ltr_uni_feature_trimmed_user_info", "as": "user_info_str"},
            {"name": "uOldMmuClusterId300ListList", "as": "user_feasury_cluster_id_list"},
          ],
          partition_size = "{{explore_teenager_ltr_partition_size}}",
        ) \
      .end_if_()
