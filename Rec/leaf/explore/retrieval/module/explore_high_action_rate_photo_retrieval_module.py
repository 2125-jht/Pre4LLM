from retrieval.retrieval_module import RetrievalModule

class ExploreHighActionRatePhotoRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .gsu_common_colossus_resp_retriever(
        colossus_resp_attr = "colossus_resp_v2",
        colossus_service_name = "grpc_colossusSimV2",
        item_key_field = "photo_id",
        item_time_field = "timestamp",
        item_fields = dict(
          play_time = "play_time",
          channel = "channel",
          label = "label"
        ),
        filter_future_items = True,
        filter_future_seconds = "{{filter_future_seconds}}",
        save_result_to_common_attr = "trigger_pid_list"
      ) \
      .filter_by_attr(
        item_list_from_attr = "trigger_pid_list",
        attr_name = "play_time",
        remove_if = "<",
        compare_to = "{{play_time_s_ths}}"
      ) \
      .if_("enable_channel_filter > 0") \
        .filter_by_attr(
          item_list_from_attr = "trigger_pid_list",
          attr_name = "channel",
          remove_if = "!=",
          compare_to = "{{reserve_channel_id}}"
        ) \
      .end_() \
      .enrich_attr_by_light_function(
        item_list_from_attr = "trigger_pid_list",
        import_item_attr = [
          "label",
        ],
        export_item_attr = [
          "is_need_filter_trigger"
        ],
        function_name = "GenHighActionRatePhotoTrigger",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .filter_by_attr(
        item_list_from_attr = "trigger_pid_list",
        attr_name = "is_need_filter_trigger",
        remove_if = "==",
        compare_to = 0
      ) \
      .if_("enable_request_ann_common_retr_server > 0") \
        .delegate_retrieve(
          kess_service = "{{ann_common_retr_server}}",
          timeout_ms = 80,
          reason = self.reason,
          request_type = "default",
          request_num = "{{retrieve_num}}",
          send_browse_set = False,
          send_common_attrs_in_request = False,
          send_common_attrs = [
            "trigger_pid_list",
            "max_trigger_num",
            {"name": "remote_index_search_num", "as": "topk_per_trigger"},
            {"name": "enable_author_shuffle", "as": "enable_trigger_shuffle"},
            "enable_result_shuffle"
          ]
        ) \
        .deduplicate() \
      .else_() \
        .retrieve_by_remote_index(
          kess_service = "{{index_service_name}}",
          timeout_ms = 80,
          reason = self.reason,
          querys = [
            {
              "query": "{{index_term_name}}:{{trigger_pid_list}}",
              "search_num": "{{search_num_per_trigger}}",
              "max_attr_num": "{{max_trigger_num}}",
              "expire_second": "{{index_cache_expire_time_s}}",
              "random_search": 1
            }
          ],
        ) \
        .deduplicate() \
        .shuffle() \
        .limit(
          size = "{{retrieve_num}}"
        ) \
      .end_()
