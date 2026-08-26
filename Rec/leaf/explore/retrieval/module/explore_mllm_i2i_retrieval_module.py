from retrieval import RetrievalModule
class ExploreMllmI2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
    
  def process(self) -> None:
    self.flow.if_("enable_global_trigger ~= nil and enable_global_trigger > 0")
    self._sample_global_triggers("trigger_list", "trigger_weight_list")
    self.flow.end_() \
    .if_("trigger_list == nil or #trigger_list <= 0") \
      .return_() \
    .end_() \
    .get_remote_embedding_lite_v2(
      kess_service = "grpc-MLLM-emb-kess",
      id_converter = {"type_name": "mioEmbeddingIdConverter"},
      shard_num = 2,
      timeout_ms = 15,
      size = 128,
      query_source_type = "common_attr",
      input_attr_name = "trigger_list",
      output_attr_name = "trigger_embedding",
      client_side_shard = True,
      slot = 666,
      is_raw_data = True,
      raw_data_type = "float32"
    )\
    .set_attr_value(
      common_attrs = [
        {
          "name": "emb_size",
          "type": "int",
          "value": 128,
        }
      ],
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "trigger_list", "as": "trigger_list"},
        {"name": "trigger_embedding", "as": "trigger_embedding_list"},
        {"name": "trigger_weight_list", "as": "trigger_weight_list"},
        {"name": "emb_size", "as": "dim"}
      ],
      export_common_attr = [
        {"name": "trigger_list", "as": "trigger_list"}, 
        {"name": "trigger_embedding_list", "as": "trigger_embedding_list"},
        {"name": "trigger_weight_list", "as": "trigger_weight_list"},
      ],
      function_name = "GetValidEmbeddings",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .pack_common_attr(
      input_common_attrs = ["trigger_list"],
      output_common_attr = "trigger_list",
      limit_num = "{{trigger_num}}"
    ) \
    .pack_common_attr(
      input_common_attrs = ["trigger_embedding_list"],
      output_common_attr = "trigger_embedding_list",
      limit_num = "{{return trigger_num * 128}}"
    ) \
    .pack_common_attr(
      input_common_attrs = ["trigger_weight_list"],
      output_common_attr = "trigger_weight_list",
      limit_num = "{{trigger_num}}"
    ) \
    .if_("trigger_list == nil or #trigger_list == 0") \
      .return_() \
    .end_() \
    .delegate_retrieve(
      reason = self.reason,
      kess_service = "{{service_name}}",
      send_common_attrs = ["trigger_list",
                            "trigger_embedding_list", 
                            "top_k", 
                            "browse_screen__pid_list"],
      recv_item_attrs = ["src_id_list", "src_dist_list"],
      timeout_ms = 80,
      request_num = 5000,
      send_common_attrs_in_request = False,
    )\
    .enrich_attr_by_light_function(
      import_common_attr = [
        "ann_dist_threshold",
        "trigger_list",
        "trigger_weight_list"
      ],
      import_item_attr = [
        "src_id_list",
        "src_dist_list"
      ],
      export_item_attr = [
        {"name": "final_score", "as": "final_score"}
      ],
      function_name = "CalcAnnResultFinalScore",
      class_name = "ExploreLightFunctionSetV2"
    ) \
    .sort(
      score_from_attr = "final_score"
    ) \
    .limit("{{result_num}}")