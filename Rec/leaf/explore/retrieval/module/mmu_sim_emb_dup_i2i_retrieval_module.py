from retrieval import RetrievalModule

class MmuSimEmbDupI2IRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
    
  def process(self) -> None:
    self.flow \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "browse_trigger_num",
          "hate_trigger_num",
          "short_view_trigger_num",
          "latest_play_minute",
          "hate_list",
          {"name": "browse_screen__pid_list", "as": "browse_set"},
          {"name": "videoPlayingPid", "as": "play_list"},
          {"name": "playstat_playtimes", "as": "playtime_list"},
          {"name": "playstat_durations", "as": "duration_list"},
          {"name": "userRecentViewTimeListRaw", "as": "timestamp_list"}
        ],
        export_common_attr = [
          {"name": "trigger_list", "as": "dup_seed_photos"}
        ],
        function_name = "TriggerFromBrowseSetAndHateList",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .if_("dup_seed_photos == nil or #dup_seed_photos == 0") \
        .return_() \
      .end_() \
      .get_remote_embedding_lite(
        kess_service = "{{mmu_sim_emb_dup_i2i_embedding_service_name}}",
        shard_num = 4,
        id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
        input_attr_name = "dup_seed_photos",
        output_attr_name = "dup_mmu_embeddings",
        query_source_type = "common_attr",
        size = 64,
        client_side_shard = True
      ) \
      .if_("dup_mmu_embeddings == nil or #dup_mmu_embeddings ~= 64 * #dup_seed_photos") \
        .return_() \
      .end_() \
      .set_attr_value(
        common_attrs = [
          {
            "name": "dup_mmu_embedding_dim",
            "type": "int",
            "value": 64
          }
        ]
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "dup_seed_photos", "as": "trigger_list"},
          {"name": "dup_mmu_embeddings", "as": "trigger_embedding_list"},
          {"name": "dup_mmu_embedding_dim", "as": "dim"}
        ],
        export_common_attr = [
          {"name": "trigger_list", "as": "dup_seed_photos"},
          {"name": "trigger_embedding_list", "as": "dup_mmu_embeddings"}
        ],
        function_name = "GetValidEmbeddings",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .retrieve_by_ann_embedding(
        kess_service = "{{mmu_sim_emb_dup_i2i_ann_service_name}}",
        space = "cosine",
        timeout_ms = 50,
        reason = self.reason,
        shard_num = 1,
        items_from_attr = ["dup_seed_photos"],
        embeddings_from_attr = ["dup_mmu_embeddings"],
        bound_type = {
          "total_limit": "{{retrieve_num}}",
        },
        algo_type = {
          "scann": {},
        },
        src_bucket = "photo",
        dest_bucket = "{{mmu_sim_emb_dup_i2i_retr_dest_bucket}}",
        save_distance_to_attr = "ann_dist_list"
      ) \
      .deduplicate() \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "ann_dist_threshold"
        ],
        import_item_attr = [
          "ann_dist_list"
        ],
        export_common_attr = [
          {"name": "remain_pid_list", "as": "mmu_sim_emb_dup_pid_list"}
        ],
        function_name = "GetPidListWithAnnDistThreshold",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .limit(size = 0)

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = ["mmu_sim_emb_dup_pid_list"],
        for_debug_request_only = True
      )