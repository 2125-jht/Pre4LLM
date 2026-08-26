from retrieval.retrieval_module import RetrievalModule  

class SearchSimLrRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .copy_attr(
        attrs = [{
          "from_common": "search_click_list",
          "to_common": "search_click_list_for_search_sim_lr"
        }]
      ) \
      .shuffle_list_attr(
        common_attr = "search_click_list_for_search_sim_lr"
      ) \
      .pack_common_attr(
        input_common_attrs = ["search_click_list_for_search_sim_lr"],
        output_common_attr = "search_click_list_for_search_sim_lr_packed",
        limit_num = "{{trigger_num}}",
        deduplicate = True
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "search_click_list_for_search_sim_lr_packed", "as": "search_click_list"}
        ],
        export_common_attr = [
          {"name": "search_sim_lr_trigger_list", "as": "search_sim_lr_trigger_list"}
        ],
        function_name = "GenSearchSimLrTriggerFromSearchClick",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .if_("(search_sim_lr_trigger_list == nil) or (#search_sim_lr_trigger_list == 0)") \
        .return_() \
      .end_() \
      .fetch_kgnn_neighbors(
        id_from_common_attr = "search_sim_lr_trigger_list",
        save_neighbors_to = "search_sim_lr_items_from_kgnn",
        kess_service = "{{kgnn_service_name}}",
        relation_name = "I2I",
        sample_num = "{{kgnn_sample_num}}",
        timeout_ms = 50,
        sample_type = "topn",
        padding_type = "zero",
        shard_num = "{{kgnn_shard_num}}"
      ) \
      .retrieve_by_common_attr(
        attr = "search_sim_lr_items_from_kgnn",
        reason = self.reason
      ) \
      .limit("{{retrieve_num}}")