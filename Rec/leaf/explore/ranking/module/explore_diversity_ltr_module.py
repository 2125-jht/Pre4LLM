from ranking import CommonModule

class ExploreDiversityLtrModule(CommonModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def process(self) -> None:
        self.flow \
        .if_("enable_explore_diversity_ltr == 1") \
          .explore_custom_trim_user_info(
            user_info_attr = "userInfo",
            save_trimed_user_info_to_attr = "explore_diversity_ltr_trimmed_user_info",
            trim_user_info = [
              "id",
              "device_id",
              "request_location.city_id",
              "request_location.province_id",
              "gender",
              "infer_gender",
              "true_gender",
              "infer_year",
              "basic_info.age_segment",
              "location.city_id",
              "client_id",
              "visit_mod",
              "user_profile.user_level",
              "active_days",
              "user_profile.exp_stat.exp_click",
              "user_profile.exp_stat.exp_like",
              "user_profile.exp_stat.exp_follow",
              "user_profile.exp_stat.exp_realshow",
              "user_profile.exp_stat.exp_long_view",
              "user_profile_v1.real_show_list.photo_id",
              "user_profile_v1.real_show_list.author_id",
              "user_profile_v1.real_show_list.time_ms",
              "user_profile_v1.real_show_list.page_type",
              "user_profile_v1.real_show_list.label.click",
              "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_level_one",
              "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_level_two",
              "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_level_three",
              "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_level_four",
              "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_level_five",
              "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_tag",
              "user_profile_v1.click_list.photo_id",
              "user_profile_v1.click_list.hetu_tag_level_info.hetu_level_one",
              "user_profile_v1.click_list.hetu_tag_level_info.hetu_level_two",
              "user_profile_v1.follow_list.author_id",
              "user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_one",
              "user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_two",
              "user_profile_v1.like_list.photo_id",
              "user_profile_v1.like_list.hetu_tag_level_info.hetu_level_one",
              "user_profile_v1.like_list.hetu_tag_level_info.hetu_level_two",
            ]
          ) \
          .delegate_enrich(
            kess_service = "{{explore_diversity_ltr_infer_name}}",
            recv_item_attrs = [
              {"name": "diversity_ltr", "as": "explore_diversity_ltr_score"}
            ],
            timeout_ms = 100,
            send_common_attrs = [
              "explore_diversity_ltr_trimmed_user_info",
              "uOldMmuClusterId300ListList",
            ],
            send_item_attrs = [
              "pcmtr",
              "pctr",
              "pftr",
              "pltr",
              "plvtr",
              "pptr",
              "psvr",
              "pvtr",
              {"name": "awesome_wtd", "as": "pwtd"},
              "pwtr",
              "fr_score1",
              "fr_score2",
              "fetr",
              "fountain_eff",
              "pcltr",
              "pdtr",
              "pcmef",
              "pepstr",
              
              "pctr_index",
              "plvtr_index",
              "pvtr_index",
              "pltr_index",
              "pftr_index",
              "pwtr_index",
              "pesptr_index",
              "psvr_index",
            ],
            request_type = "default",
            partition_size = "{{explore_diversity_ltr_partition_size}}"
          ) \
        .end_()
