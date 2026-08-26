from retrieval.retrieval_module import RetrievalModule

class GclI2IAnnRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "global_normal_trigger_list",
          "global_high_value_trigger_list",
          "global_normal_trigger_weight_list",
          "global_high_value_trigger_weight_list",
          {"name": "fountain_gcl_i2i_sample_trigger_num", "as": "sample_trigger_num"},
          {"name": "fountain_gcl_i2i_start_idx", "as": "start_idx"},
        ],
        export_common_attr = [
          {"name": "final_trigger_list", "as": "gcl_triggers"},
        ],
        function_name = "SampleTriggers",
        class_name = "ExploreLightFunctionSetV2",
        skip = "{{skip_fountain_gcl_retr}}"
      ) \
      .retrieve_by_ann_embedding(
        kess_service = "{{fountain_gcl_retr_common_retr_service}}",
        space = "cosine",
        timeout_ms = 50,
        reason = self.reason,
        shard_num = 1,
        items_from_attr = ["gcl_triggers"],
        bound_type = {
          "total_limit": "{{fountain_gcl_retr_request_num}}",
        },
        algo_type = {
          "scann": {},
        },
        src_bucket = "photo",
        dest_bucket = "photo_bucket_splash",
        dest_bucket_item_type = 0,
        skip = "{{skip_fountain_gcl_retr}}"
      )