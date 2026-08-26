from retrieval import RetrievalModule

class PicMcI2IRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  @classmethod
  def is_retrieval(cls) -> bool:
    return True
    
  def process(self) -> None:
    self.flow \
      .if_("enable_global_trigger ~= nil and enable_global_trigger > 0")
    ## 从 global trigger 里完成抽取
    self._sample_pic_global_triggers("seed_photos_", "trigger_weight_")
    self.flow \
      .else_() \
        .enrich_with_protobuf(
          from_extra_var = "user_info_ptr",
          attrs = [
            dict(name="videoPlayingPid", path="user_profile_v1.video_playing_stat.photo_id"),
            dict(name="videoPlayingDuration", path="user_profile_v1.video_playing_stat.playing_time"),
            dict(name="realtimeClickList", path="realtime_click_list"),
            dict(name="searchList", path="user_profile_v1.search_click_photo_list.photo_id")
          ]
        ) \
        .enrich_attr_by_lua(
          import_common_attr = ["videoPlayingPid","videoPlayingDuration", "realtimeClickList", "searchList", "enable_nouse_clicklist_trigger", "enable_duration_more_trigger", "enable_search_more_trigger", "trigger_num"],
          export_common_attr = ["seed_photos_","trigger_weight_"],
          function_for_common = "calculate",
          lua_script_file = "explore/retrieval/lua/module/pic_mc_i2i_retr__key_generator.lua"
        ) \
      .end_() \
      .get_remote_embedding(
        kess_service = "{{embedding_service_name}}",
        shard_num = 8,
        timeout_ms = 20,
        slot = 0,
        id_converter = {
          "type_name": "kuibaEmbeddingIdConverter"
        },
        item_list_from_attr = "seed_photos_",
        save_to_common_attr = True,
        output_item_list_attr = "mc_trigger_list",
        output_embedding_list_attr = "mc_trigger_embedding",
        query_source_type = "item_key",
        client_side_shard = True,
        raw_data_type = "uint16",
        is_raw_data = False,
        is_raw_data_list = False,
      ) \
      .retrieve_by_ann_embedding(
        kess_service = "{{ann_service_name}}",
        space = "cosine",
        timeout_ms = 100,
        reason = self.reason,
        shard_num = 1,
        items_from_attr = ["mc_trigger_list"],
        embeddings_from_attr = ["mc_trigger_embedding"],
        bound_type = {
          "top_k": "{{retrieve_num_per_trigger}}"
        },
        algo_type = {
          "scann": {},
        },
        src_bucket = "{{src_bucket}}",
        dest_bucket = "{{dest_bucket}}",
        save_distance_to_attr = "ann_dist_list",
        save_source_item_to_attr = "ann_src_list"
      ) \
      .deduplicate() \
      .filter_by_common_attr(
        common_attr = ["browse_screen__pid_list"]
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "ann_dist_threshold",
          {"name": "seed_photos_", "as": "trigger_list"},
          {"name": "trigger_weight_", "as": "trigger_weight_list"},
        ],
        import_item_attr = [
          {"name": "ann_src_list", "as": "src_id_list"},
          {"name": "ann_dist_list", "as": "src_dist_list"}
        ],
        export_item_attr = [
          {"name": "src_id", "as": "i2i_retr__trigger_pid"},
          {"name": "final_score", "as": "ann_dist"}
        ],
        function_name = "CalcAnnResultFinalScore",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .filter_by_attr(
        attr_name = "ann_dist",
        remove_if = "<",
        compare_to = "{{ann_dist_threshold}}"
      ) \
      .sort(score_from_attr = "ann_dist") \
      .limit(size = "{{result_num}}") \
      .log_debug_info(
        common_attrs = [
          "seed_photos_",
          "trigger_weight_",
        ],
        item_attrs = [
          "i2i_retr__trigger_pid",
          "ann_dist",
        ],
        for_debug_request_only=True
      )