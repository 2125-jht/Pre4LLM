from ranking import CommonModule

class RankingUninterestCtrModule(CommonModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def process(self) -> None:
        self.flow \
        .if_("enable_explore_ranking_get_history_attr_stats_count == 1") \
          .explore_history_attr_stats_enricher(
            prev_item_from_attr = "standard_explore_realshow_pid_list",
            prev_item_from_attr_timestamp = "uStandardExploreRealshowTimestampList",
            prev_item_label_from_attr = "uStandardExploreRealshowLabelList",
            time_window = "{{explore_ranking_history_attr_stats_timestamp_threshold}}",
            realshow_num_threshold = "{{explore_ranking_history_attr_stats_realshow_num_threshold}}",
            cluster_id_attr = "cluster_id_632",
            hetu_level_five_attr = "hetu_tag_level_info__hetu_level_five",
            hetu_level_two_attr = "hetu_tag_level_info__hetu_level_two",
            tagnex_attr = "hetu_tag_level_info__hetu_tag",
            enable_cluster_id = "{{explore_ranking_history_attr_stats_enable_cluster_id}}",
            enable_hetu_level_five = "{{explore_ranking_history_attr_stats_enable_hetu_level_five}}",
            enable_hetu_level_two = "{{explore_ranking_history_attr_stats_enable_hetu_level_two}}",
            enable_tagnex = "{{explore_ranking_history_attr_stats_enable_tagnex}}",
            output_cluster_id_count_attr = "cluster_id_packed_count",
            output_hetu_level_five_count_attr = "hetu5_packed_count",
            output_hetu_level_two_count_attr = "hetu2_packed_count",
            output_tagnex_count_attr = "tagnex_packed_count",
            output_zero_exposure_flag_attr = "zero_exposure_flag",
          ) \
        .end_() \
        .if_("enable_explore_uninterest_ctr == 1") \
          .explore_custom_trim_user_info(
            user_info_attr = "userInfo",
            save_trimed_user_info_to_attr = "explore_uninterest_ctr_trimmed_user_info",
            trim_user_info = [
              "id",
              "device_id",
              "gender",
              "basic_info.age_segment",
              "location.city_level",
              "user_active_level",
              "active_days",
            ]
          ) \
          .delegate_enrich(
            kess_service = "{{explore_uninterest_ctr_infer_name}}",
            recv_item_attrs = [
              {"name": "ctr", "as": "explore_uninterest_ctr_score"},
            ],
            timeout_ms = 100,
            send_common_attrs = [
              {"name": "explore_uninterest_ctr_trimmed_user_info", "as": "user_info_str"},
              {"name": "standard_explore_realshow_pid_list", "as": "user_explore_realshow_pid_list"},
              {"name": "uStandardExploreRealshowTimestampList", "as": "user_explore_realshow_timestamp_list"},
              {"name": "uStandardExploreRealshowLabelList", "as": "user_explore_realshow_label_list"},
            ],
            send_item_attrs = [
              "cluster_id_packed_count",
              "hetu5_packed_count",
              "hetu2_packed_count",
              "tagnex_packed_count",
            ],
            request_type = "default",
            partition_size = "{{explore_uninterest_ctr_partition_size}}"
          ) \
        .end_()