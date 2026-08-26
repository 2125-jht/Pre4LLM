from retrieval.retrieval_module import RetrievalModule

class NnRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self):
    self.flow \
      .explore_nn_user_embedding_enricher(
        output_user_emb_attr = "nn_user_embedding",
        user_info_attr = "tmp_user_info_ptr",
        kess_service = "{{kess_service}}",
        timeout_ms = 50
      ) \
      .explore_retrieve_by_nn_user_photo(
        user_info_ptr_attr = "tmp_user_info_ptr",
        user_emb_attr = "nn_user_embedding",
        reason = self.reason,
        kess_service = "{{kess_service}}",
        timeout_ms = 50,
        total_limit = self.total_limit
      ) \
      .deduplicate()

  
  @property
  def total_limit(self):
    return self.config.get("total_limit", 2000)