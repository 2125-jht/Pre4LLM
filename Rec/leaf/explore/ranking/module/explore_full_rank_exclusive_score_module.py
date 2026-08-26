from ranking import CommonModule

class ExploreFullRankExclusiveScoreModule(CommonModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def process(self) -> None:
        self.flow \
        .if_("enable_explore_full_rank_exclusive_score == 1") \
          .explore_custom_trim_user_info(
            user_info_attr = "userInfo",
            save_trimed_user_info_to_attr = "explore_full_rank_exclusive_score_trimmed_user_info",
            trim_user_info = self.user_info_list()
          ) \
          .delegate_enrich(
            kess_service = "{{explore_full_rank_exclusive_score_infer_name}}",
            recv_item_attrs = [
              {"name": "explore_ctr", "as": "exclusive_ctr"},
              {"name": "explore_vtr", "as": "exclusive_vtr"},
              {"name": "explore_wtd", "as": "exclusive_wtd"}
            ],
            timeout_ms = 100,
            send_common_attrs = [
              "explore_full_rank_exclusive_score_trimmed_user_info",
              "uStandardExploreRealshowLabelList",
              "uStandardExploreRealshowHetuTag1List",
              "uStandardExploreRealshowHetuTag2List",
              "uStandardExploreRealshowHetuTag5List",
              "uStandardExploreRealshowTimestampList",
              "uOldMmuClusterId300ListList",
            ],
            send_item_attrs = [
              "cascade_psvtr",
              "cascade_pctr",
              "cascade_plvtr",
              "cascade_pwtr",
              "cascade_pltr",
              "cascade_pftr",
              "cascade_ptr",
              "cascade_pcmtr",
              "cascade_pctr_index",
              "cascade_plvtr_index",
              "cascade_pvtr_index",
              "cascade_pltr_index",
              "cascade_pftr_index",
              "cascade_pwtr_index",
              "cascade_pesptr_index",
              "cascade_psvr_index",
            ],
            request_type = "default",
            partition_size = "{{explore_full_rank_exclusive_score_partition_size}}"
          ) \
        .end_()

    def user_info_list(self):
      useful_user_info_data = [
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
        "request_location.province_id",
        "request_location.city_id",
        "visit_mod",
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
        "user_profile_v1.video_playing_stat.client_timestamp",
        "realtime_click_list",
        "realtime_like_list",
        "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_level_one",
        "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_level_two",
        "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_level_three",
        "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_tag",
        "user_profile_v1.click_list.hetu_tag_level_info.hetu_level_one",
        "user_profile_v1.click_list.hetu_tag_level_info.hetu_level_two",
        "user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_one",
        "user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_two",
        "user_profile_v1.like_list.hetu_tag_level_info.hetu_level_one",
        "user_profile_v1.like_list.hetu_tag_level_info.hetu_level_two",
        "user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_one",
        "user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_two",
        "upload_count",
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
        "user_interest_profile.hetu_level_one_long_term_id",
        "user_interest_profile.hetu_level_one_long_term_score",
        "user_interest_profile.hetu_level_two_long_term_id",
        "user_interest_profile.hetu_level_two_long_term_score",
        "user_interest_profile.hetu_level_three_long_term_id",
        "user_interest_profile.hetu_level_three_long_term_score",
        "feature_collection.explore_low_active_level",
        "apps.app.package",
      ]
      return useful_user_info_data
