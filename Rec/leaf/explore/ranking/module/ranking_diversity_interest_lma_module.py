from ranking import CommonModule

class RankingDiversityInterestLmaModule(CommonModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def uni_feature_trim_user_info(self):
        features = [
            "id",
            "device_id",
            "request_location.city_id",
            "request_location.province_id",
            "request_location.poi_type",
            "location.city_id",
            "location.region_type",
            "gender",
            "infer_gender",
            "true_gender",
            "infer_year",
            "basic_info.age_segment",
            "location.city_level",
            "is_douyin",
            "feature_collection.explore_low_active_level",
            "active_days", 
            "client_id",
            "visit_net",
            "follow_count",
            "fans_count",
            "upload_count",
            "user_profile_v1.real_show_list.time_ms",
            "user_profile_v1.real_show_list.photo_id",
            "user_profile_v1.real_show_list.page_type",
            "user_profile_v1.real_show_list.label.click",
            "user_profile_v1.real_show_list.label.like",
            "user_profile_v1.real_show_list.label.follow",
            "user_profile_v1.real_show_list.label.hate",
            "user_profile_v1.real_show_list.author_id",
            "user_profile_v1.click_list.photo_id",
            "user_profile_v1.click_list.page_type",
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
            "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_level_one",
            "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_level_two",
            "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_level_three",
            "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_level_five",
            "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_tag",
            "user_profile_v1.real_show_list.time_ms",
            "user_interest_profile.hetu_level_one_long_term_id",
            "user_interest_profile.hetu_level_one_long_term_score",
            "user_interest_profile.hetu_level_two_long_term_id",
            "user_interest_profile.hetu_level_two_long_term_score",
            "user_interest_profile.hetu_level_three_long_term_id",
            "user_interest_profile.hetu_level_three_long_term_score",
            "user_profile_v1.video_playing_stat.playing_time",
            "user_profile_v1.video_playing_stat.author_id",
            "user_profile_v1.video_playing_stat.photo_id",       
            "user_profile_v1.video_playing_stat.video_duration",
            "user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_one",
            "user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_two",
            "user_profile.exp_stat.exp_click",
            "user_profile.exp_stat.exp_like",
            "user_profile.exp_stat.exp_follow",
            "user_profile.exp_stat.exp_realshow",
            "user_profile.exp_stat.exp_long_view",
            "user_profile.user_level",
            "user_profile_v1.click_list.author_id",
            "user_profile_v1.follow_list.author_id",
            "user_profile_v1.follow_list.photo_id",
            "user_profile_v1.like_list.author_id",
            "user_profile_v1.like_list.photo_id",   
        ]

        return features

    def uni_feature_context_info(self):
      features = [
        "reason",
        "pctr",
        "pltr",
        "pwtr",
        "pftr",
        "phtr",
        "plvtr",
        {"name": "psvr", "as": "psvtr"},
        "pvtr",
        {"name": "awesome_wtd", "as": "pwtd"},
        "pptr",
        "pcmtr",
        "pcmef",
        "fr_score1",
        "fr_score2",
        "pdtr",
        "fetr",
        "pctr_index",
        "plvtr_index",
        "pvtr_index",
        "pltr_index",
        "pftr_index",
        "pwtr_index",
        "pesptr_index",
        "psvr_index",
        "cascade_pctr",
        "cascade_pltr",
        "cascade_pwtr",
        "cascade_plvtr",
        "fountain_eff",
        "pepstr",
      ]

      return features

    def process(self) -> None:
        self.flow \
        .if_("enable_explore_diversity_interest_lma == 1") \
          .if_("enable_explore_diversity_interest_lma_divide_active_degree == 0 or (find_user_active_degree == 1 or find_user_active_degree == 2)") \
            .explore_custom_trim_user_info(
              user_info_attr = "userInfo",
              save_trimed_user_info_to_attr = "explore_diversity_interest_lma_trimmed_user_info",
              trim_user_info = self.uni_feature_trim_user_info(),
            ) \
            .delegate_enrich(
              kess_service = "{{explore_diversity_interest_lma_infer_name}}",
              recv_item_attrs = [
                {"name": "ctr", "as": "explore_diversity_interest_lma_score"},
                {"name": "group_ctr", "as": "explore_diversity_interest_group_ctr_score"}
              ],
              timeout_ms = 100,
              send_common_attrs = [
                {"name": "explore_diversity_interest_lma_trimmed_user_info", "as": "user_info_str"},
                {"name": "uMultiDimensionGroupKV", "as": "user_feasury_multi_dimension_group"},
                {"name": "uMultiDimensionGroupDetailKV", "as": "user_feasury_multi_dimension_group_detail"},
              ],
              send_item_attrs = self.uni_feature_context_info(),
              request_type = "default",
              partition_size = "{{explore_diversity_interest_lma_ctr_partition_size}}"
            ) \
          .end_() \
        .end_()