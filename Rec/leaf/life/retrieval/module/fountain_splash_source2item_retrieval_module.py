from retrieval.retrieval_module import RetrievalModule

class FountainSplashSource2ItemRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .switch_("request_type") \
        .case_("fountain_splash_life") \
          .retrieve_by_remote_index(
            kess_service = "{{fountain_splash_source_to_item_retr_index_service}}",
            timeout_ms = 50,
            reason = self.reason,
            querys = [
              {
                "query": "sim:{{featureSourcePId}}",
                "search_num": 500,
                "expire_second": 1,
                "random_search": 1
              }
            ]
          ) \
      .end_()