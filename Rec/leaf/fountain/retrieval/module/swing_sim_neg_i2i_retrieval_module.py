from retrieval import RetrievalModule

class SwingSimNegI2IRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "hate_list",
          "hate_list_timestamps",
          {"name": "trigger_time_gap_day", "as": "time_gap_day"}
        ],
        export_common_attr = [
          "hate_trigger_list",
        ],
        function_name = "GenSwingNegI2IHateList",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .retrieve_by_remote_index(
        kess_service = "{{swing_common_index_service}}",
        timeout_ms = "{{service_timeout_ms}}",
        reason = self.reason,
        common_query = "",
        querys = [{
          "query": "item2itemId_Pic:{{hate_trigger_list}}",
          "max_attr_num" : "{{swing_trigger_hate_list_max_len}}",
          "search_num" : "{{swing_trigger_i2i_max_len}}",
        }],
        save_score_to_attr = "swing_sim_score",
        default_random_search = 0,
        default_total_request_num = "{{swing_retr_total_request_num}}"
      ) \
      .deduplicate() \
      .filter_by_attr(
        attr_name = "swing_sim_score",
        remove_if = "<=",
        compare_to = "{{swing_sim_score_threshold}}",
        remove_if_attr_missing = True,
      ) \
      .copy_item_meta_info(
        save_item_id_to_attr = "photo_id",
      ) \
      .pack_item_attr(
        item_source = {
          "reco_results" : True,
        },
        mappings = [{
          "from_item_attr" : "photo_id", 
          "to_common_attr" : "swing_user_hate_i2i_list",
        }]
      ) \
      .limit(0)