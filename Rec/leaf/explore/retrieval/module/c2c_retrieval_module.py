from retrieval.retrieval_module import RetrievalModule

class C2CRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "uOldMmuClusterId300ListList",
          "interest_cid_collaborative_score_map",
          "similar_cid_num"
        ],
        export_common_attr = [
          "interest_cid_similar_cid_list"
        ],
        function_name = "CalInterestCidSimilarCidList",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .retrieve_by_remote_index(
        kess_service = "{{remote_index_service_name}}",
        timeout_ms = 50,
        reason = self.reason, 
        querys = [
          {
            "query": "cid:{{interest_cid_similar_cid_list}}",
            "search_num": "{{remote_index_search_num}}", 
            "max_attr_num": "{{retrieve_num}}"
          }
        ]
      ) \
  