from retrieval.retrieval_module import RetrievalModule

class MemoryU2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self):
    self.flow \
      .delegate_enrich(
        kess_service = "{{infer_service_name}}",
        shard_num = 1,
        timeout_ms = 50,
        send_common_attrs = [
          "featureUId",
          "gender",
          "true_gender",
          "infer_gender",
          "true_year",
          "infer_year",
          "featureAgeSegment",
          "featureProvinceId",
          "featureCityId",
          "featureClientId",
          "featureVisitMod",
          "featureVisitNet",
          "app_list",
          "featureUserLevel",
          "featureActiveDays",
          "featureTopDislikeTopic",
          "featureRiskLevel",
          "featureLongTermInterestPhotoDnnClusterId",
          "featureUserRequestProvinceId",
          "featureUserRequestCityId",
          "exp_show",
          "exp_click",
          "exp_like",
          "exp_follow",
          "exp_forward",
          "exp_long_view",
          "exp_short_view",
          "exp_watch_time",
          "userRecentViewTimeListRaw",
          "userRecentViewPageListRaw",
          {"name": "videoPlayingPid", "as": "userRecentViewPidListRaw"},
          {"name": "profile_v1_click_trigger_aids", "as": "userRecentViewAidListRaw"},
          {"name": "playstat_durations", "as": "userRecentViewDurationListRaw"},
          {"name": "playstat_playtimes", "as": "userRecentViewPlayTimeListRaw"},
          {"name": "playstat_hetu1s", "as": "userRecentViewHetuOneListRaw"},
          {"name": "playstat_hetu2s", "as": "userRecentViewHetuTwoListRaw"},
          {"name": "playstat_hetu3s", "as": "userRecentViewHetuThreeListRaw"},
          {"name": "playstat_hetu4s", "as": "userRecentViewHetuFourListRaw"},
          {"name": "like_list", "as": "userLikePidList"},
          {"name": "follow_list", "as": "userFollowPidList"},
          {"name": "forward_list", "as": "userForwardPidList"},
          {"name": "profile_enter_list", "as": "userProfileEnterPidList"},
          {"name": "collect_list", "as": "userCollectPidList"}
        ],
        recv_common_attrs = ["user_top_layer"],
        for_predict = False
      ) \
      .gen_common_attr_by_lua(
        attr_map = {
          "uid_1": "(1 << 54) | featureUId",
          "uid_2": "(2 << 54) | featureUId",
          "uid_3": "(3 << 54) | featureUId",
          "uid_4": "(4 << 54) | featureUId",
          "uid_5": "(5 << 54) | featureUId",
          "uid_6": "(6 << 54) | featureUId",
          "uid_7": "(7 << 54) | featureUId",
          "uid_8": "(8 << 54) | featureUId",
        }
      ) \
      .pack_common_attr(
        input_common_attrs = ["uid_1", "uid_2", "uid_3", "uid_4", "uid_5", "uid_6", "uid_7", "uid_8"],
        output_common_attr = "uid_list"
      ) \
      .retrieve_by_ann_embedding(
        reason = self.reason,
        kess_service = "{{ann_service_name}}",
        space = "ip",
        timeout_ms = 50,
        items_from_attr = ["uid_list"],
        embeddings_from_attr = ["user_top_layer"],
        bound_type = {
          "total_limit": "{{request_num}}"
        },
        algo_type = {
          "scann": {}
        },
        src_data_type = "photo",
        src_bucket = "photo",
        dest_bucket = "{{ann_dest_bucket}}",
        save_source_item_to_attr = "src_id_list",
        save_distance_to_attr = "src_dist_list"
      ) \
      .deduplicate() \
      .if_("enable_sort_by_ann_score ~= nil and enable_sort_by_ann_score > 0", to_be_delete = "date=2023-11-16;committer=shaolei") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "ann_dist_threshold"
          ],
          import_item_attr = [
            "src_id_list",
            "src_dist_list"
          ],
          export_item_attr = [
            {"name": "final_score", "as": "ann_score"}
          ],
          function_name = "CalcAnnResultFinalScore",
          class_name = "ExploreLightFunctionSetV2"
        ) \
        .sort(
          score_from_attr = "ann_score"
        ) \
      .else_() \
        .shuffle() \
      .end_() \
      .limit("{{retrieve_num}}")