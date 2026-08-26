from retrieval import RetrievalModule

class LifeSearchI2iV2RetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  @classmethod
  def is_retrieval(cls) -> bool:
    return True
    
  def process(self) -> None:
    self.flow \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "search_click_list",
          "search_click_list_timestamps",
          "search_play_list",
          "search_play_list_timestamps",
          "search_play_list_play_duration",
          "search_play_list_video_duration",
          {"name": "life_search_i2i_time_min_thr", "as": "time_min_thr"},
          {"name": "life_search_i2i_boost_time_min_thr", "as": "boost_time_min_thr"},
          {"name": "life_search_i2i_play_time_s_thr", "as": "play_time_s_thr"},
          {"name": "life_search_i2i_trigger_size", "as": "trigger_size"},
        ],
        export_common_attr = [
          {"name": "trigger_list", "as": "search_pid_list"},
          {"name": "boost_list", "as": "search_boost_source_pid_list"}
        ],
        function_name = "GenSearchTriggerList",
        class_name = "ExploreLifeLightFunctionSet",
      ) \
      .if_("search_pid_list == nil or #search_pid_list == 0") \
        .return_() \
      .end_() \
      .get_remote_embedding_lite(
        kess_service = "{{life_search_embedding_service_name}}",
        shard_num = 8,
        timeout_ms = 20,
        id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
        input_attr_name = "search_pid_list",
        output_attr_name = "search_embedding_list",
        query_source_type = "common_attr",
        size = 128,
        client_side_shard = True
      ) \
      .if_("search_embedding_list == nil or #search_embedding_list ~= 128 * #search_pid_list") \
        .return_() \
      .end_() \
      .set_attr_value(
        common_attrs = [
          {
            "name": "search_embedding_dim",
            "type": "int",
            "value": 128
          }
        ]
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "search_pid_list", "as": "trigger_list"},
          {"name": "search_embedding_list", "as": "trigger_embedding_list"},
          {"name": "search_embedding_dim", "as": "dim"}
        ],
        export_common_attr = [
          {"name": "trigger_list", "as": "search_pid_list"},
          {"name": "trigger_embedding_list", "as": "search_embedding_list"}
        ],
        function_name = "GetValidEmbeddings",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .retrieve_by_ann_embedding(
        kess_service = "{{life_search_ann_service_name}}",
        space = "cosine",
        timeout_ms = 50,
        reason = self.reason,
        shard_num = 1,
        items_from_attr = ["search_pid_list"],
        embeddings_from_attr = ["search_embedding_list"],
        bound_type = {
          "total_limit": "{{life_search_retrieval_count}}",
        },
        algo_type = {
          "scann": {},
        },
        src_bucket = "photo",
        dest_bucket = "{{life_search_retrieval_bucket}}",
        save_source_item_to_attr = "src_item_list",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "search_boost_source_pid_list"
        ],
        import_item_attr = [
          "src_item_list"
        ],
        export_item_attr = [
          "is_search_boost"
        ],
        function_name = "IsSearchBoost",
        class_name = "ExploreLifeLightFunctionSet",
      )

  @property
  def emb_server_shard_num(self) -> int:
    return self.config.get("emb_server_shard_num", 8)