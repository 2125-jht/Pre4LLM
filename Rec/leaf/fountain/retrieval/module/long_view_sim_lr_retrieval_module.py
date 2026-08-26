from retrieval.retrieval_module import RetrievalModule  

class LongViewSimLrRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("use_same_hour_seq == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "userInfoPb", "as": "user_info_ptr"},
            "time_before_user_present_time",
            "time_after_user_present_time"
          ],
          export_common_attr = [
            {"name": "user_long_view_same_hour_pid_list", "as": "long_view_list_for_long_view_sim_lr"}
          ],
          function_name = "GenLongViewSameHourPidListFromUserInfo",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .else_() \
        .copy_attr(
          attrs = [{
            "from_common": "user_long_view_pid_list",
            "to_common": "long_view_list_for_long_view_sim_lr"
          }]
        ) \
      .end_() \
      .shuffle_list_attr(
        common_attr = "long_view_list_for_long_view_sim_lr"
      ) \
      .pack_common_attr(
        input_common_attrs = ["long_view_list_for_long_view_sim_lr"],
        output_common_attr = "long_view_list_for_long_view_sim_lr_packed",
        limit_num = "{{trigger_num}}",
        deduplicate = True
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "long_view_list_for_long_view_sim_lr_packed", "as": "search_click_list"}
        ],
        export_common_attr = [
          {"name": "search_sim_lr_trigger_list", "as": "long_view_sim_lr_trigger_list"}
        ],
        function_name = "GenSearchSimLrTriggerFromSearchClick",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .if_("(long_view_sim_lr_trigger_list == nil) or (#long_view_sim_lr_trigger_list == 0)") \
        .return_() \
      .end_() \
      .fetch_kgnn_neighbors(
        id_from_common_attr = "long_view_sim_lr_trigger_list",
        save_neighbors_to = "long_view_sim_lr_items_from_kgnn",
        kess_service = "{{kgnn_service_name}}",
        relation_name = "I2I",
        sample_num = "{{kgnn_sample_num}}",
        timeout_ms = 50,
        sample_type = "topn",
        padding_type = "zero",
        shard_num = "{{kgnn_shard_num}}"
      ) \
      .retrieve_by_common_attr(
        attr = "long_view_sim_lr_items_from_kgnn",
        reason = self.reason
      ) \
      .limit("{{retrieve_num}}")