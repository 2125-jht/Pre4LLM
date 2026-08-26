from retrieval.retrieval_module import RetrievalModule

class LongtermRetargetAuthorRetrievalModule(RetrievalModule):
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
          author_id = "author_id",
          play_time = "play_time",
          channel = "channel",
          duration = "duration",
          label = "label"
        ),
        filter_future_items = True,
        filter_future_seconds = "{{filter_future_seconds}}",
        save_result_to_common_attr = "valid_item_key_list"
      ) \
      .filter_by_attr(
        item_list_from_attr = "valid_item_key_list",
        attr_name = "play_time",
        remove_if = "<",
        compare_to = "{{play_time_s_ths}}"
      ) \
      .filter_by_common_attr(
        item_list_from_attr = "valid_item_key_list",
        common_attr = ["hate_aids"],
        on_item_attr = "author_id"
      ) \
      .if_("enable_svr_filter > 0") \
        .enrich_attr_by_light_function(
          item_list_from_attr = "valid_item_key_list",
          import_common_attr = [
            "svr_threshold"
          ],
          import_item_attr = [
            "duration",
            "play_time"
          ],
          export_item_attr = [
            "need_filter"
          ],
          function_name = "CalcShortPlay",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .filter_by_attr(
          item_list_from_attr = "valid_item_key_list",
          attr_name = "need_filter",
          remove_if = ">",
          compare_to = 0
        ) \
      .end_() \
      .if_("enable_new_photo_filter > 0") \
        .enrich_attr_by_light_function(
          item_list_from_attr = "valid_item_key_list",
          import_common_attr = [
            "lvr_threshold",
            "pic_time_threshold",
            "lvr_duration_threshold"
          ],
          import_item_attr = [
            "duration",
            "play_time",
            "label"
          ],
          export_item_attr = [
            "new_photo_trigger"
          ],
          function_name = "GenNewPhotoTrigger",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .filter_by_attr(
          item_list_from_attr = "valid_item_key_list",
          attr_name = "new_photo_trigger",
          remove_if = "==",
          compare_to = 0
        ) \
      .end_() \
      .if_("enable_channel_filter > 0") \
        .filter_by_attr(
          item_list_from_attr = "valid_item_key_list",
          attr_name = "channel",
          remove_if = "!=",
          compare_to = "{{reserve_channel_id}}"
        ) \
      .end_() \
      .pack_item_attr(
        item_source = {"common_attr": ["valid_item_key_list"]},
        mappings = [{
          "from_item_attr": "author_id",
          "to_common_attr": "trigger_aid_list",
          "dedup_to_common_attr": True,
          "pack_if": "author_id"
        }, {
          "to_common_attr": "trigger_pid_list",
          "dedup_to_common_attr": True
        }]
      ) \
      .if_("enable_request_ann_common_retr_server > 0") \
        .delegate_retrieve(
          kess_service = "{{ann_common_retr_server}}",
          timeout_ms = "{{remote_index_service_timeout_ms}}",
          reason = self.reason,
          request_type = "default",
          request_num = "{{retrieve_num}}",
          send_browse_set = False,
          send_common_attrs_in_request = False,
          send_common_attrs = [
            "trigger_aid_list",
            "trigger_pid_list",
            "max_trigger_num",
            {"name": "remote_index_search_num", "as": "topk_per_trigger"},
            {"name": "enable_author_shuffle", "as": "enable_trigger_shuffle"},
            "enable_result_shuffle",
            "photo_age_threshold"
          ]
        ) \
      .else_() \
        .retrieve_by_remote_index(
          kess_service = "{{remote_index_service_name}}",
          timeout_ms = "{{remote_index_service_timeout_ms}}",
          reason = self.reason, 
          querys = [
            {
              "query": "authorId2PhotoIdOrderByUploadTime:{{trigger_aid_list}}",
              "search_num": "{{remote_index_search_num}}", 
              "max_attr_num": "{{max_trigger_num}}"
            }
          ]
        ) \
        .deduplicate() \
        .shuffle() \
        .limit(
          size = "{{retrieve_num}}"
        ) \
      .end_() 
