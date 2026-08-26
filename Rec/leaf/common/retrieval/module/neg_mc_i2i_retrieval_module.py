from common.retrieval import RetrievalModule

class NegMCI2IRetrievalModule(RetrievalModule):
  def __init__(self, name: str, config_dir: str) -> None:
    super().__init__(name, config_dir)

  def process(self):
    self.flow \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "hate_list",
          "hate_list_timestamps",
          {"name": "trigger_time_gap_day", "as": "time_gap_day"}
        ],
        export_common_attr = [
          {"name": "hate_trigger_list", "as": "trigger_list"}
        ],
        function_name = "GenSwingNegI2IHateList",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .if_("trigger_list == nil or #trigger_list == 0") \
        .return_() \
      .end_() \
      .pack_common_attr(
        input_common_attrs = ["trigger_list"],
        output_common_attr = "trigger_list",
        limit_num = "{{trigger_size}}",
      ) \
      .get_remote_embedding_lite(
        kess_service = "{{embedding_service_name}}",
        shard_num = 8,
        timeout_ms = 20,
        id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
        input_attr_name = "trigger_list",
        output_attr_name = "trigger_embedding_list",
        query_source_type = "common_attr",
        size = 128,
        client_side_shard = True
      ) \
      .if_("trigger_list == nil or #trigger_embedding_list ~= 128 * #trigger_list") \
        .return_() \
      .end_() \
      .set_attr_value(
        common_attrs = [
          {
            "name": "dim",
            "type": "int",
            "value": 128
          }
        ]
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "trigger_list",
          "trigger_embedding_list",
          "dim"
        ],
        export_common_attr = [
          "trigger_list",
          "trigger_embedding_list"
        ],
        function_name = "GetValidEmbeddings",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .retrieve_by_ann_embedding(
        kess_service = "{{ann_service_name}}",
        space = "cosine",
        timeout_ms = 50,
        reason = self.reason,
        shard_num = 1,
        items_from_attr = ["trigger_list"],
        embeddings_from_attr = ["trigger_embedding_list"],
        bound_type = {
          "total_limit": "{{retr_num}}",
        },
        algo_type = {
          "scann": {},
        },
        src_bucket = "photo",
        dest_bucket = "{{dest_bucket}}",
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
          {"name": "remain_pid_list", "as": "mc_i2i_neg_pid_list"}
        ],
        function_name = "GetPidListWithAnnDistThreshold",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .limit(0)