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
          author_id = "retarget_author_id",
          play_time = "retarget_play_time"
        ),
        filter_future_items = True,
        filter_future_seconds = "{{filter_future_seconds}}",
        save_result_to_common_attr = "retarget_author_valid_item_key_list"
      ) \
      .filter_by_attr(
        item_list_from_attr = "retarget_author_valid_item_key_list",
        attr_name = "retarget_play_time",
        remove_if = "<",
        compare_to = "{{play_time_s_ths}}"
      ) \
      .filter_by_common_attr(
        item_list_from_attr = "retarget_author_valid_item_key_list",
        common_attr = ["hateAidList"],
        on_item_attr = "retarget_author_id"
      ) \
      .pack_item_attr(
        item_source = {"common_attr": ["retarget_author_valid_item_key_list"]},
        mappings = [{
          "from_item_attr": "retarget_author_id",
          "to_common_attr": "retarget_trigger_aid_list",
          "dedup_to_common_attr": True,
          "pack_if": "retarget_author_id"
        }]
      ) \
      .if_("enable_author_shuffle > 0") \
        .shuffle_list_attr(
          common_attr = "retarget_trigger_aid_list"
        ) \
      .end_() \
      .retrieve_by_remote_index(
        kess_service = "grpc_recoHotOrderedIndexServer",
        timeout_ms = 50,
        reason = self.reason, 
        querys = [
          {
            "query": "authorId2PhotoIdOrderByUploadTime:{{retarget_trigger_aid_list}}",
            "search_num": "{{remote_index_search_num}}", 
            "max_attr_num": "{{max_trigger_num}}"
          }
        ]
      )