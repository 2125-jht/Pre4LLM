from ranking import CommonModule

class RankingCorrectionPctrnModule(CommonModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def process(self) -> None:
        self.flow \
        .if_("enable_explore_ranking_correction_pctr == 1") \
          .explore_custom_trim_user_info(
            user_info_attr = "userInfo",
            save_trimed_user_info_to_attr = "explore_correction_pctr_trimmed_user_info",
            trim_user_info = [
              "id",
              "device_id",
              "gender",
              "infer_gender",
              "true_gender",
              "basic_info.age_segment",
              "request_location.city_id",
              "request_location.province_id",
              "user_profile.exp_stat.exp_click",
              "user_profile.exp_stat.exp_realshow",
              "user_profile.exp_stat.exp_like",
              "user_profile_v1.click_list.photo_id",
              "user_profile_v1.click_list.page_type",
              "user_profile_v1.real_show_list.photo_id",
              "user_profile_v1.real_show_list.author_id",
              "user_profile_v1.real_show_list.page_type",
              "user_profile_v1.real_show_list.label.click",
              "user_profile_v1.real_show_list.time_ms",
              "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_level_one",
              "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_level_two",
              "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_level_three",
              "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_level_five",
              "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_tag",
              "user_profile_v1.click_list.hetu_tag_level_info.hetu_level_one",
              "user_profile_v1.click_list.hetu_tag_level_info.hetu_level_two",
              "user_profile_v1.follow_list.author_id",
              "user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_one",
              "user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_two",
              "user_profile_v1.like_list.photo_id",
              "user_profile_v1.like_list.hetu_tag_level_info.hetu_level_one",
              "user_profile_v1.like_list.hetu_tag_level_info.hetu_level_two",
              "user_profile_v1.video_playing_stat.playing_time",
              "user_profile_v1.video_playing_stat.author_id",
              "user_profile_v1.video_playing_stat.photo_id", 
            ]
          ) \
          .delegate_enrich(
            kess_service = "{{explore_correction_pctr_infer_name}}",
            recv_item_attrs = [
              {"name": "diversity_ctr", "as": "explore_correction_pctr_score"}
            ],
            timeout_ms = 100,
            send_common_attrs = [
              "explore_correction_pctr_trimmed_user_info",
              {"name": "uMultiDimensionGroupKV", "as": "user_feasury_multi_dimension_group"},
              {"name": "uMultiDimensionGroupDetailKV", "as": "user_feasury_multi_dimension_group_detail"},
              {"name": "uStandardExploreRealshowAuthorIdList", "as": "user_feasury_explore_realshow_author_id_list"},
              {"name": "uStandardExploreRealshowPhotoIdList", "as": "user_feasury_explore_realshow_photo_id_list"},
              {"name": "uStandardExploreRealshowHetuTag1List", "as": "user_feasury_explore_realshow_hetu_level_one_list"},
              {"name": "uStandardExploreRealshowHetuTag2List", "as": "user_feasury_explore_realshow_hetu_level_two_list"},
              {"name": "uStandardExploreRealshowHetuTag5List", "as": "user_feasury_explore_realshow_hetu_level_five_list"},
              {"name": "uStandardExploreRealshowLabelList", "as": "user_feasury_explore_realshow_label_list"},
              {"name": "uStandardExploreRealshowTimestampList", "as": "user_feasury_explore_realshow_timestamp_list"},
            ],
            send_item_attrs = [
              "pctr",
              "pltr",
              {"name": "psvr", "as": "psvtr"},
              "pctr_index",
              "pltr_index",
              "psvr_index",
              "cascade_pctr",
              "cascade_pctr_index",
            ],
            request_type = "default",
            partition_size = "{{explore_correction_pctr_partition_size}}"
          ) \
        .end_()