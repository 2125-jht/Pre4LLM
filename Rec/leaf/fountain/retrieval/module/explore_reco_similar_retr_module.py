from retrieval.retrieval_module import RetrievalModule

class ExploreRecoSimilarRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
      self.flow \
        .retrieve_by_explore_reco_similar(
          kess_service = "grpc_LRIcfRetrievalExpV2Filter",
          timeout_ms = 150,
          reason = self.reason,
          total_limit = "{{fountain_retrieval_total_limit_explore_h88_retrieval}}",
          dest_bucket_item_type = 1,
          skip = "{{fountain_retrieval_skip_similar_explore_h88}}")
        