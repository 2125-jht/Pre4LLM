from retrieval import RetrievalModule

class PicMmuI2IRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  @classmethod
  def is_retrieval(cls) -> bool:
    return True
    
  def process(self) -> None:
    ## 从 global trigger 里完成抽取
    self._sample_pic_global_triggers("mmu_i2i_trigger_list", "trigger_weight_")
    self.flow \
      .get_remote_embedding_lite(
        kess_service = "{{embedding_service_name}}",
        shard_num = 4,
        id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
        input_attr_name = "mmu_i2i_trigger_list",
        output_attr_name = "mmu_i2i_trigger_list_emb",
        query_source_type = "common_attr",
        size = 64,
        client_side_shard = True
      ) \
      .set_attr_value(
        common_attrs=[
          {
            "name": "emb_size",
            "type": "int",
            "value": 64
          }
        ],
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "mmu_i2i_trigger_list", "as": "trigger_list"},
          {"name": "mmu_i2i_trigger_list_emb", "as": "trigger_embedding_list"},
          {"name": "emb_size", "as": "dim"}
        ],
        export_common_attr = [
          {"name": "trigger_list", "as": "trigger_list"}, 
          {"name": "trigger_embedding_list", "as": "trigger_emb_list"}
        ],
        function_name = "GetValidEmbeddings",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .if_("trigger_list == nil or #trigger_list == 0") \
        .return_() \
      .end_() \
      .retrieve_by_ann_embedding(
        kess_service = "{{ann_service_name}}",
        space = "cosine",
        timeout_ms = 50,
        reason = self.reason,
        shard_num = 1,
        items_from_attr = ["trigger_list"],
        embeddings_from_attr = ["trigger_emb_list"],
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
           "trigger_list",
          {"name": "trigger_weight_", "as": "trigger_weight_list"},
        ],
        import_item_attr = [
          {"name": "ann_src_list", "as": "src_id_list"},
          {"name": "ann_dist_list", "as": "src_dist_list"}
        ],
        export_item_attr = [
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
      .limit(size = "{{result_num}}")