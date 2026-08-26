from retrieval.retrieval_module import RetrievalModule

class BoostPhotoI2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "click_list", "like_list", "follow_list", "forward_list", "profile_enter_list", "download_list", "collect_list",
          "comment_list", "click_limit", "like_limit", "follow_limit", "forward_limit", "profile_enter_limit", "download_limit",
          "collect_limit", "comment_limit", "trigger_limit", "hate_list"
        ],
        export_common_attr = [
          "trigger_list",
        ],
        function_name = "GenAnnTriggerList",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .if_("trigger_list == nil or #trigger_list == 0") \
       .return_() \
      .end_() \
      .retrieve_by_ann_embedding(
        space = "cosine",
        reason = self.reason,
        kess_service = "{{service_name}}",
        timeout_ms = "{{service_timeout_ms}}",
        items_from_attr = ["trigger_list"],
        bound_type = {
          "top_k": "{{retrieve_num_per_trigger}}"
        },
        algo_type = {
          "scann": {}
        },
        src_data_type = "{{src_data_type}}",
        src_bucket = "{{src_data_type}}",
        dest_bucket = "{{dest_bucket}}",
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