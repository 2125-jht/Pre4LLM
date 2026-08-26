from retrieval.retrieval_module import RetrievalModule

class EmbeddingV2U2uMultishardRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .delegate_retrieve(
        partition_service_kconf = "colossus.ann.explore_user_v2_partition_ann",
        request_type = "default",
        timeout_ms = 50,
        recv_item_attrs = ["distance"],
      ) \
      .deduplicate() \
      .sort(score_from_attr="distance") \
      .limit("{{uid_trigger_num}}") \
      .copy_item_meta_info(
        save_item_id_to_attr = "id"
      ) \
      .pack_item_attr(
        item_source = {
          "reco_results": True,
        },
        mappings = [{
          "from_item_attr": "id",
          "to_common_attr": "similar_uId",
        }]
      ) \
      .limit(0) \
      .if_("similar_uId == nil or #similar_uId == 0") \
        .return_() \
      .end_() \
      .retrieve_by_remote_index(
        kess_service = "{{index_service_name}}",
        timeout_ms = 30,
        reason = self.reason,
        querys = [
          {
            "query": "usim:{{similar_uId}}",
            "search_num": "{{search_num}}",
          },
        ],
      ) \
      .limit("{{service_request_num}}")