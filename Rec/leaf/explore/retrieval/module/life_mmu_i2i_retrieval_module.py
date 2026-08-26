from retrieval import RetrievalModule

class LifeMmuI2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  @classmethod
  def is_retrieval(cls) -> bool:
    return True
    
  def process(self) -> None:
    self.flow.pack_common_attr(
        input_common_attrs = [
          "videoPlayingPid", "realtimeClickList", "searchList", "explore_selected_trigger_list"],
        output_common_attr = "seed_photos",
        deduplicate = True
      ) \
      .shuffle_list_attr(
        common_attr= "seed_photos"
      ) \
      .pack_common_attr(
        input_common_attrs = ["seed_photos"],
        output_common_attr = "seed_photos",
        limit = "{{trigger_num}}"
      ) \
      .get_remote_embedding_lite(
        kess_service = "{{embedding_service_name}}",
        shard_num = self.emb_server_shard_num,
        timeout_ms = 20,
        id_converter = {
          "type_name": "kuibaEmbeddingIdConverter"
        },
        size = self.emb_size,
        output_attr_name = "trigger_embedding",
        input_attr_name = "seed_photos",
        query_source_type = "common_attr",
        client_side_shard = True
      ) \
      .set_attr_value(
        common_attrs=[
          {
            "name": "emb_size",
            "type": "int",
            "value": self.emb_size
          }
        ],
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "seed_photos", "as": "trigger_list"},
          {"name": "trigger_embedding", "as": "trigger_embedding_list"},
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
        timeout_ms = 100,
        reason = self.reason,
        shard_num = 1,
        items_from_attr = ["trigger_list"],
        embeddings_from_attr = ["trigger_emb_list"],
        bound_type = {
          "total_limit": "{{result_num}}",
        },
        algo_type = {
          "scann": {},
        },
        src_bucket = "{{ann_src_data_type}}",
        dest_bucket = "{{retr_dest_bucket}}",
      ) \
      .deduplicate()

  @property
  def emb_server_shard_num(self) -> int:
    return self.config.get("emb_server_shard_num", 4)

  @property
  def emb_size(self) -> int:
    return self.config.get("emb_size", 64)