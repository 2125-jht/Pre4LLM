from retrieval.retrieval_module import RetrievalModule

class AugmentedTtU2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  @classmethod
  def is_retrieval(cls) -> bool:
    return True
    
  def process(self) -> None:
    self.flow \
      .get_remote_embedding_lite(
        kess_service = "{{embedding_service_name}}",
        timeout_ms = 10,
        id_converter = {
          "type_name": "mioEmbeddingIdConverter"
        },
        size = 64,
        slot = 301,
        query_source_type = "user_id",
        output_attr_name = "user_embedding_list"
      ) \
      .retrieve_by_ann_embedding(
        kess_service = "{{ann_service_name}}",
        timeout_ms = "{{ann_service_timeout}}",
        reason = self.reason,
        items_from_attr = ["_USER_ID_"],
        embeddings_from_attr = ["user_embedding_list"],
        bound_type = {
          "total_limit": "{{ann_service_retr_count}}",
        },
        algo_type = {
          "faiss": {},
        },
        src_bucket = "{{ann_service_data_type}}",
        src_data_type = "{{ann_service_data_type}}",
        dest_bucket = "{{ann_service_bucket}}",
        save_distance_to_attr = "ann_dist_list"
      ) \
      .deduplicate(
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "ann_dist_threshold",
        ],
        import_item_attr = [
          "ann_dist_list",
        ],
        export_item_attr = [
          "ann_dist",
        ],
        function_name = "AnnCalThresholdValueForDistList",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .filter_by_attr(
        attr_name = "ann_dist",
        remove_if = "<",
        compare_to = "{{ann_dist_threshold}}"
      )

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = [
          "ann_service_name",
          "ann_service_timeout",
          "ann_service_retr_count",
          "ann_service_data_type",
          "ann_service_bucket",
          "_USER_ID_",
          "is_explore_low_active_user",
          "user_embedding_list"
        ],
        item_attrs = [
          "ann_dist_list",
          "ann_dist"
        ],
        for_debug_request_only = True
      )
