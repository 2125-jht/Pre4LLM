from retrieval import RetrievalModule

class ColossusAnnV2RetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  @classmethod
  def is_retrieval(cls) -> bool:
    return True
    
  def process(self) -> None:
    self.flow \
      .if_("common_trigger_generator__trigger_list == nil or #common_trigger_generator__trigger_list <= 0") \
        .return_() \
      .end_() \
      .copy_attr(
        attrs=[{
          "from_common": "common_trigger_generator__trigger_list",
          "to_common": "trigger_list",
        },
        {
          "from_common": "common_trigger_generator__trigger_weight_list",
          "to_common": "trigger_weight_list",
        }]
      ) \
      .get_remote_embedding_lite(
        kess_service = "{{embedding_service_name}}",
        shard_num = 8,
        timeout_ms = 20,
        id_converter = {
          "type_name": "kuibaEmbeddingIdConverter"
        },
        size = 128,
        input_attr_name = "trigger_list",
        output_attr_name = "colossus_trigger_embedding",
        query_source_type = "common_attr",
        client_side_shard = True
      ) \
      .enrich_attr_by_lua(
        import_common_attr = ["trigger_list", "trigger_weight_list", "colossus_trigger_embedding"],
        export_common_attr = ["trigger_list", "trigger_weight_list", "colossus_trigger_embedding"],
        function_for_common = "calculate",
        lua_script_file = "explore/retrieval/lua/module/colossus_ann__process_embedding.lua"
      ) \
      .retrieve_by_ann_embedding(
        kess_service = "{{ann_service_name}}",
        space = "cosine",
        timeout_ms = 50,
        reason = self.reason,
        shard_num = 1,
        items_from_attr = ["trigger_list"],
        browsed_item_count = 0,
        embeddings_from_attr = ["colossus_trigger_embedding"],
        bound_type = {
          "top_k": "{{ann_topk}}",
        },
        algo_type = {
          "scann": {},
        },
        src_bucket = "photo",
        dest_bucket = "photo",
        save_source_item_to_attr = "src_id_list",
        save_distance_to_attr = "src_dist_list"
      ) \
      .deduplicate(
      ) \
      .filter_by_common_attr(
        common_attr = ["browse_screen__pid_list"]
      ) \
      .enrich_attr_by_lua(
        import_common_attr = ["trigger_list", "trigger_weight_list", "ann_dist_threshold"],
        import_item_attr = ["src_id_list", "src_dist_list"],
        export_item_attr = ["final_score"],
        function_for_common = "calc_trigger_map",
        function_for_item = "calculate",
        lua_script_file = "explore/retrieval/lua/module/colossus_ann__gen_item_score.lua"
      ) \
      .sort(
        score_from_attr = "final_score"
      ) \
      .limit("{{retrieve_num}}")

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = [
          "trigger_list",
          "colossus_trigger_embedding",
          "embedding_service_name"
        ]
      )
