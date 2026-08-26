from retrieval.retrieval_module import RetrievalModule

class HetuTagV2RetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_remote_index(
        kess_service = "{{fountain_hetu_tag_v2_retrieval_nreason_kess_name}}",
        timeout_ms = 150,
        reason = self.reason,
        reset_item_type = 1,
        common_query = "",
        querys = [{ # 多个 query 有执行先后顺序，从细粒度的开始，召回数量够了以后后续 query 不再执行
          "query": "hetu_tag_v2:{{source_hetu_face_id_v2}}",
          "search_num": "{{fountain_hetu_tag_v2_retrieval_nreason_hetu_face_id_search_num}}"
        },
        {
          "query": "hetu_tag_v2:{{source_hetu_tag_v2}}",
          "search_num": "{{fountain_hetu_tag_v2_retrieval_nreason_hetu_tag_search_num}}"
        },
        {
          "query": "hetu_tag_v2:{{source_hetu_level_four_v2}}",
          "search_num": "{{fountain_hetu_tag_v2_retrieval_nreason_hetu_level_four_search_num}}"
        },
        {
          "query": "hetu_tag_v2:{{source_hetu_level_three_v2}}",
          "search_num": "{{fountain_hetu_tag_v2_retrieval_nreason_hetu_level_three_search_num}}"
        },
        {
          "query": "hetu_tag_v2:{{source_hetu_level_two_v2}}",
          "search_num": "{{fountain_hetu_tag_v2_retrieval_nreason_hetu_level_two_search_num}}"
        }],
        default_search_num = 1000,
        default_random_search = 0,
        default_total_request_num = "{{fountain_hetu_tag_v2_retrieval_nreason_request_num}}",
        skip = "{{skip_fountain_hetu_tag_v2_retrieval_nreason_query_server}}")