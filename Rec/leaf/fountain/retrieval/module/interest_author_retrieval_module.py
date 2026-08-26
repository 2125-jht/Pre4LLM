from retrieval.retrieval_module import RetrievalModule

class InterestAuthorRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_remote_index(
        kess_service = "{{fountain_interest_author_retr_service_name}}",
        timeout_ms = 50,
        reason = self.reason,
        common_query = "",
        querys = [{
          "query": "{{fountain_interest_author_retrieval_query}}:{{long_term_interest_authors}}",
          "search_num": "{{fountain_interest_author_rete_search_num_per_author}}"
        }],
        default_search_num = 100,
        default_total_request_num = "{{fountain_interest_author_retr_retrieve_num}}"
      )