from retrieval.retrieval_module import RetrievalModule

class BoostPhotoU2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  @classmethod
  def is_retrieval(cls) -> bool:
    return True
    
  def process(self) -> None:
    self.flow \
      .retrieve_by_ann_embedding(
        space = "ip",
        kess_service = "{{ann_service_name}}",
        timeout_ms = "{{ann_service_timeout}}",
        reason = self.reason,
        items_from_attr = ["_USER_ID_"],
        #embeddings_from_attr = ["augmented_tt_u2i_user_embedding_list"],
        bound_type = {
          "total_limit": "{{ann_service_retr_count}}",
        },
        algo_type = {
          "scann": {},
        },
        src_bucket = "{{ann_service_data_type}}",
        src_data_type = "{{ann_service_data_type}}",
        dest_bucket = "{{ann_service_bucket}}",
        save_distance_to_attr = "ann_dist_list"
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
      .deduplicate(
      ) \
      .filter_by_common_attr(
        common_attr = ["browse_screen__pid_list"]
      ) \
      .sort(score_from_attr = "ann_dist") \
      .limit(size = "{{retrieve_num}}")