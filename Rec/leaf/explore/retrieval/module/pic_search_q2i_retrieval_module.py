from retrieval import RetrievalModule

class PicSearchQ2IRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
    
  def process(self) -> None:
    self.flow \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "search_query_list_keyword",
          "search_query_list_timestamp",
          {"name": "explore_pic_search_q2i_trigger_time_window_min", "as": "time_window_min"},
          {"name": "explore_pic_search_q2i_trigger_recent_time_window_min", "as": "recent_time_window_min"},
          {"name": "explore_pic_search_q2i_trigger_size", "as": "trigger_size"},
        ],
        export_common_attr = [
          "search_query_trigger_list",
          "recent_search_query_trigger_list"
        ],
        function_name = "GetSearchQueryTriggerList",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .retrieve_by_redis(
        reason = self.reason,
        retrieve_num = "{{explore_pic_search_q2i_retr_num}}",
        retrieve_num_per_key = "{{explore_pic_search_q2i_retr_num_per_key}}",
        cluster_name = "explorePicCache",
        timeout_ms = 50,
        key_from_attr = "search_query_trigger_list",
        key_prefix = "{{explore_pic_search_q2i_redis_key_prefix}}",
        item_separator = ",",
        append_src_key_to_attr = "src_query_list"
      ) \
      .deduplicate() \
      .shuffle() \
      .limit(
        size = "{{explore_pic_search_q2i_retr_num_final}}"
      ) \
      .set_attr_value(
        item_attrs = [
          {
            "name": "is_pic_search",
            "type": "int",
            "value": 1
          }
        ]
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "recent_search_query_trigger_list", "as": "attr_list"}
        ],
        import_item_attr = [
          {"name": "src_query_list", "as": "attrs"}
        ],
        export_item_attr = [
          {"name": "is_in_set", "as": "is_pic_recent_search"}
        ],
        function_name = "AttrStringListIsInSet",
        class_name = "ExploreLightFunctionSetV2",
      )
