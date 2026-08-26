from retrieval.retrieval_module import RetrievalModule

class SplashSource2ItemRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
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
        ],
        skip = "{{skip_fountain_splash_source_to_item_retr}}"
      )