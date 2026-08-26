from ranking import CommonModule

class ExplorePreferXtrModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def uni_feature_trim_user_info(self):
    features = [
      "id",
      "active_days",
      "basic_info.age_segment",
      "location.city_id",
      "location.region_type",
      "device_id",
      "gender",
      "infer_gender",
      "true_gender",
      "request_location.poi_type",
      "request_location.province_id",
      "request_location.city_id",
      "user_profile.exp_stat.exp_click",
      "user_profile.exp_stat.exp_like",
      "user_profile.exp_stat.exp_follow",
      "user_profile.exp_stat.exp_realshow",
      "user_profile.exp_stat.exp_long_view",
      "user_profile.user_level",
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
      "user_profile_v1.click_list.hetu_tag_level_info.hetu_level_one",
      "user_profile_v1.click_list.hetu_tag_level_info.hetu_level_two",
      "user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_one",
      "user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_two",
      "user_profile_v1.like_list.hetu_tag_level_info.hetu_level_one",
      "user_profile_v1.like_list.hetu_tag_level_info.hetu_level_two",
      "user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_one",
      "user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_two",
      "upload_count",
      "infer_year",
      "follow_count",
      "fans_count",
      "location.city_level",
      "feature_collection.explore_low_active_level",
    ]

    return features

  def uni_feature_context_into(self):
    features = [
      "awesome_wtd",
      "corr_pctr",
      "pepstr",
      {"name": "pctr", "as": "pPctr"},
      {"name": "pltr", "as": "pPltr"},
      {"name": "pwtr", "as": "pPwtr"},
      {"name": "pftr", "as": "pPftr"},
      {"name": "pcltr", "as": "pPcltr"},
      {"name": "phtr", "as": "pPhtr"},
      {"name": "plvtr", "as": "pPlvtr"},
      {"name": "psvr", "as": "pPsvtr"},
      {"name": "pvtr", "as": "pPvtr"},
      {"name": "pptr", "as": "pPptr"},
      {"name": "pcmtr", "as": "pPcmtr"},
      {"name": "pcmef", "as": "pPcmef"},
      {"name": "fr_score1", "as": "pPfrScore1"},
      {"name": "fr_score2", "as": "pPfrScore2"},
      {"name": "fetr", "as": "pPfetr"}
    ]

    return features

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_prefer_xtr_model == 1") \
        .explore_custom_trim_user_info(
          user_info_attr = "userInfo",
          save_trimed_user_info_to_attr = "explore_ltr_uni_feature_trimmed_user_info",
          trim_user_info = self.uni_feature_trim_user_info(),
        ) \
        .delegate_enrich(
          kess_service = "{{explore_prefer_xtr_kess_service}}",
          recv_item_attrs = [
            {"name": "ctr", "as": "prefer_xtr_score"},
          ],
          timeout_ms = 100,
          send_item_attrs = self.uni_feature_context_into(),
          send_common_attrs = [
            {"name": "explore_ltr_uni_feature_trimmed_user_info", "as": "user_info_str"},
            {"name": "uMultiDimensionGroupKV", "as": "user_feasury_multi_dimension_group"},
            {"name": "uMultiDimensionGroupDetailKV", "as": "user_feasury_multi_dimension_group_detail"},
          ],
          partition_size = "{{explore_prefer_xtr_partition_size}}"
        ) \
      .end_if_()
