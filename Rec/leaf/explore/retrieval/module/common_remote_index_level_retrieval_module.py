from retrieval.retrieval_module import RetrievalModule

class CommonRemoteIndexLevelRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self):
    self.flow \
      .if_("user_risk_level and user_risk_level < user_risk_min") \
        .return_() \
      .end_() \
      .retrieve_by_remote_index(
        kess_service = "{{remote_index_service_name}}",
        timeout_ms = "{{remote_index_service_timeout_ms}}",
        reason = self.reason, 
        querys = [
          {
            "query": self.remote_index_query_term + ":{{user_interest_hetu2}}",
            "random_search": 1,
            "search_num": "{{search_num}}"
          }
        ]
      ) \
      .deduplicate() \
      .limit(
        size = "{{result_num}}"
      )
  
  @property
  def remote_index_query_term(self) -> str:
    assert "remote_index_query_term" in self.config
    return self.config["remote_index_query_term"]