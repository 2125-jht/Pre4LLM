from retrieval.retrieval_module import RetrievalModule

class EmbeddingIcfRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self):
    self.flow \
      .explore_retrieve_by_embedding_icf(
        user_info_attr = "tmp_user_info_ptr",
        reason = self.reason,
        kess_service = "{{kess_service}}",
        service_key = "{{service_key}}",
        use_simple_interface = "{{enable_use_simple_interface}}",
        retrieval_tag = self.retrieval_tag,
        timeout_ms = 50,
        total_limit = self.total_limit
      ) \
      .deduplicate()
  
  @property
  def total_limit(self):
    return self.config.get("total_limit", 2000)

  @property
  def retrieval_tag(self):
    return self.config.get("retrieval_tag", "")