from retrieval.retrieval_module import RetrievalModule

life_related_source2pid_feature = [
  {"name": "featureUId", "as": "uId"},
  {"name": "sensi_photo_id", "as": "featureSourcePId"},
  {"name": "sensiPidAuthorId", "as": "sourcePidAuthorId"},
  {"name": "sensiPidMmuImgClusterV3", "as": "sourcePidMmuImgClusterV3"},
  {"name": "sensiPidMmuImgClusterV4", "as": "sourcePidMmuImgClusterV4"},
  {"name": "sensi_duration_ms", "as": "sourcePidDuration"},
  {"name": "sensi_hetu_level_two",   "as": "SourcePidHetuTagLevel2"},
  {"name": "sensi_hetu_level_three", "as": "SourcePidHetuTagLevel3"},
  {"name": "sensi_hetu_level_four", "as": "SourcePidHetuTagLevel4"},
  {"name": "sensi_hetu_level_five",  "as": "SourcePidHetuTagLevel5"},
  {"name": "videoPlayingPid", "as": "playstat_pids"},
  {"name": "profile_v1_click_trigger_aids", "as": "playstat_aids"},
  {"name": "playstat_playtimes", "as": "playstat_playtimes"},
  {"name": "playstat_durations", "as": "playstat_durations"},
  {"name": "playstat_hetu1s", "as": "playstat_hetu1s"},
  {"name": "playstat_hetu2s", "as": "playstat_hetu2s"},
  {"name": "playstat_hetu3s", "as": "playstat_hetu3s"},
  {"name": "playstat_hetu4s", "as": "playstat_hetu4s"},
  {"name": "userRecentViewTimeListRaw", "as": "playstat_timestamps"},
  {"name": "userRecentViewPageListRaw", "as": "playstat_pages"},
  {"name": "realtime_photo_id_list", "as": "user_fountain_play_id_list"},
  {"name": "realtime_list_aid", "as": "user_fountain_play_aid_list"},
#   {"name": "user_fountain_play_time_list", "as": "user_fountain_play_time_list"},
  {"name": "realtime_list_duration", "as": "user_fountain_play_duration_list"},
#   {"name": "user_fountain_play_timestamp_list", "as": "user_fountain_play_timestamp_list"},
#   {"name": "user_fountain_play_page_list", "as": "user_fountain_play_page_list"},
  {"name": "realtime_list_hetu_level_one1", "as": "user_fountain_play_hetu_l1_top1_list"},
  {"name": "realtime_list_hetu_level_two1", "as": "user_fountain_play_hetu_l2_top1_list"},
  {"name": "realtime_list_hetu_level_three1", "as": "user_fountain_play_hetu_l3_top1_list"},
  {"name": "realtime_list_hetu_level_four1", "as": "user_fountain_play_hetu_l4_top1_list"},
]

class LifeRelatedSource2pidRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .get_item_attr_by_distributed_flat_index(
          photo_store_kconf_key = "reco.distributedIndex.hotPhotoInfoCommonIndex",
          use_dynamic_photo_store = True,
          photo_store_request_data_set_tags_attr = 'explore_request_data_set_tags',
          attrs = [
            "hetu_tag_level_info_v2__hetu_level_one", 
            "hetu_tag_level_info_v2__hetu_level_two",
            "hetu_tag_level_info_v2__hetu_level_three",
            "hetu_tag_level_info_v2__hetu_level_four",
            "hetu_tag_level_info_v2__hetu_tag",
            "hetu_tag_level_info_v2__hetu_face_id",
            "hetu_sim_cluster_id",
            "hetu_tag_level_info_v2__hetu_cluster_id",
            "mmu_img_cluster_v3",
            "mmu_img_cluster_v4",
            "mmu_text_cluster",
            "author__id",
            "author__category_detail__first_level_id",
            "author__category_detail__second_level_id",
            "author__category_detail__third_level_id",
            "tag",
            "upload_type",
            "photo_id",
            "duration_ms",
            "hetu_tag_level_info__hetu_level_one",
            "hetu_tag_level_info__hetu_level_two",
            "hetu_tag_level_info__hetu_level_three",
            "hetu_tag_level_info__hetu_level_four",
            "hetu_tag_level_info__hetu_level_five",
            ],
          additional_item_source={
            "reco_results": False,
            "common_attr": ["realtime_photo_id_list"],
          },
        ) \
      .enrich_attr_by_lua(
            item_list_from_attr = "realtime_photo_id_list",
            import_item_attr = ["hetu_tag_level_info__hetu_level_one","hetu_tag_level_info__hetu_level_two","hetu_tag_level_info__hetu_level_three","hetu_tag_level_info__hetu_level_four"],
            export_item_attr = ["hetu_tag_level_info__hetu_level_one1","hetu_tag_level_info__hetu_level_two1","hetu_tag_level_info__hetu_level_three1","hetu_tag_level_info__hetu_level_four1"],
            function_for_item = 'get_one_hetu',
            lua_script = """
            function get_one_hetu()
                local hetu_one = hetu_tag_level_info__hetu_level_one or {}
                local hetu_two = hetu_tag_level_info__hetu_level_two or {}
                local hetu_three = hetu_tag_level_info__hetu_level_three or {}
                local hetu_four = hetu_tag_level_info__hetu_level_four or {}
                local hetu_one1 = 0
                local hetu_two1 = 0
                local hetu_three1 = 0
                local hetu_four1 = 0
                if #hetu_one >= 1 then
                    hetu_one1 = hetu_one[1]
                end
                if #hetu_two >= 1 then
                    hetu_two1 = hetu_two[1]
                end
                if #hetu_three >= 1 then
                    hetu_three1 = hetu_three[1]
                end
                if #hetu_four >= 1 then
                    hetu_four1 = hetu_four[1]
                end
                return hetu_one1, hetu_two1, hetu_three1, hetu_four1
            end
            """
      )\
      .pack_item_attr(
            item_source={
            "reco_results": False,
            "common_attr": ["realtime_photo_id_list"],
            },
            mappings=[
            {
                "aggregator": "copy",
                "from_item_attr": "hetu_tag_level_info_v2__hetu_level_one",
                "to_common_attr": "sensi_hetu_level_one_v2",
            },
            {
                "aggregator": "copy",
                "from_item_attr": "hetu_tag_level_info_v2__hetu_level_two",
                "to_common_attr": "sensi_hetu_level_two_v2",
            },
            {
                "aggregator": "copy",
                "from_item_attr": "hetu_tag_level_info_v2__hetu_level_three",
                "to_common_attr": "sensi_hetu_level_three_v2",
            },
            {
                "aggregator": "copy",
                "from_item_attr": "hetu_tag_level_info_v2__hetu_level_four",
                "to_common_attr": "sensi_hetu_level_four_v2",
            },
            {
                "aggregator": "copy",
                "from_item_attr": "hetu_tag_level_info_v2__hetu_tag",
                "to_common_attr": "sensi_hetu_tag_v2",
            },
            {
                "aggregator": "copy",
                "from_item_attr": "hetu_tag_level_info_v2__hetu_face_id",
                "to_common_attr": "sensi_hetu_face_id_v2",
            },
            {
                "aggregator": "copy",
                "from_item_attr": "hetu_sim_cluster_id",
                "to_common_attr": "sensi_hetu_sim_cluster_id",
            },
            {
                "aggregator": "copy",
                "from_item_attr": "hetu_tag_level_info_v2__hetu_cluster_id",
                "to_common_attr": "sensi_hetu_cluster_id_v2",
            },
            {
                "aggregator": "copy",
                "from_item_attr": "mmu_img_cluster_v3",
                "to_common_attr": "sensiPidMmuImgClusterV3",
            },
            {
                "aggregator": "copy",
                "from_item_attr": "mmu_text_cluster",
                "to_common_attr": "sensiPidMmuTextCluster",
            },
            {
                "aggregator": "copy",
                "from_item_attr": "author__id",
                "to_common_attr": "sensiPidAuthorId",
            },
            {
                "aggregator": "copy",
                "from_item_attr": "author__category_detail__first_level_id",
                "to_common_attr": "sensiPidFirstLevelCategory",
            },
            {
                "aggregator": "copy",
                "from_item_attr": "author__category_detail__second_level_id",
                "to_common_attr": "sensiPidSecondLevelCategory",
            },
            {
                "aggregator": "copy",
                "from_item_attr": "author__category_detail__third_level_id",
                "to_common_attr": "sensiPidThirdLevelCategory",
            },
            {
                "aggregator": "copy",
                "from_item_attr": "tag",
                "to_common_attr": "sensiPidTagId",
            },
            {
                "aggregator": "copy",
                "from_item_attr": "upload_type",
                "to_common_attr": "sensiPidUploadType",
            },
            {
                "aggregator": "copy",
                "from_item_attr": "photo_id",
                "to_common_attr": "sensi_photo_id",
            },
            {
                "aggregator": "copy",
                "from_item_attr": "mmu_img_cluster_v4",
                "to_common_attr": "sensiPidMmuImgClusterV4",
            },
            {
                "aggregator": "copy",
                "from_item_attr": "hetu_tag_level_info__hetu_level_two",
                "to_common_attr": "sensi_hetu_level_two",
            },
            {
                "aggregator": "copy",
                "from_item_attr": "hetu_tag_level_info__hetu_level_three",
                "to_common_attr": "sensi_hetu_level_three",
            },
            {
                "aggregator": "copy",
                "from_item_attr": "hetu_tag_level_info__hetu_level_four",
                "to_common_attr": "sensi_hetu_level_four",
            },
            {
                "aggregator": "copy",
                "from_item_attr": "hetu_tag_level_info__hetu_level_five",
                "to_common_attr": "sensi_hetu_level_five",
            },
            {
                "aggregator": "copy",
                "from_item_attr": "duration_ms",
                "to_common_attr": "sensi_duration_ms",
            },
            {
                "aggregator": "concat",
                "from_item_attr": "photo_id",
                "to_common_attr": "realtime_list_photo_id",
                "default_val" : 0
            },
            {
                "aggregator": "concat",
                "from_item_attr": "author__id",
                "to_common_attr": "realtime_list_aid",
                "default_val" : 0
            },
            {
                "aggregator": "concat",
                "from_item_attr": "duration_ms",
                "to_common_attr": "realtime_list_duration",
                "default_val" : 0
            },
            {
                "aggregator": "concat",
                "from_item_attr": "hetu_tag_level_info__hetu_level_one1",
                "to_common_attr": "realtime_list_hetu_level_one1",
                "default_val" : 0
            },
            {
                "aggregator": "concat",
                "from_item_attr": "hetu_tag_level_info__hetu_level_two1",
                "to_common_attr": "realtime_list_hetu_level_two1",
                "default_val" : 0
            },
            {
                "aggregator": "concat",
                "from_item_attr": "hetu_tag_level_info__hetu_level_three1",
                "to_common_attr": "realtime_list_hetu_level_three1",
                "default_val" : 0
            },
            {
                "aggregator": "concat",
                "from_item_attr": "hetu_tag_level_info__hetu_level_four1",
                "to_common_attr": "realtime_list_hetu_level_four1",
                "default_val" : 0
            },
            ]
      ) \
      .if_("enable_life_related_source2pid_retr_real == 1") \
      .delegate_retrieve(
            kess_service = "{{life_related_s2p_retr_service_name}}",
            timeout_ms = 50,
            reason = self.reason,
            request_type = "default",
            request_num = "{{life_related_s2p_retr_retrieve_num}}",
            send_common_attrs = life_related_source2pid_feature,
            send_common_attrs_in_request = False,
            # save_result_to_common_attr="life_related_s2p_list_4090"
      ) \
      .end_()
  
  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = ["realtime_photo_id_list", "sensi_photo_id", "life_related_s2p_list_4090","realtime_list_photo_id",
                        "realtime_list_aid", "realtime_list_duration","realtime_list_hetu_level_one1","realtime_list_hetu_level_three1",
                        "sensi_duration_ms","userRecentViewTimeListRaw", 
                        "sensiPidFirstLevelCategory",
                        "sensiPidMmuTextCluster",
                        "sensiPidSecondLevelCategory",
                        "sensiPidTagId",
                        "sensiPidThirdLevelCategory",
                        "sensiPidUploadType",
                        "sensi_hetu_cluster_id_v2",
                        "sensi_hetu_face_id_v2",
                        "sensi_hetu_level_four_v2",
                        "sensi_hetu_level_one_v2",
                        "sensi_hetu_level_three_v2",
                        "sensi_hetu_level_two_v2",
                        "sensi_hetu_sim_cluster_id",
                        "sensi_hetu_tag_v2",
                        "featureUId", "playstat_playtimes", "profile_v1_click_trigger_aids"
                        ],
        for_debug_request_only = True,
        respect_sample_loggging = True,
      )