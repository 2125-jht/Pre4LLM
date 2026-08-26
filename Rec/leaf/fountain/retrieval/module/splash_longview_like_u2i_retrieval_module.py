from retrieval.retrieval_module import RetrievalModule

class LongviewRetrSplashFlow(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_("enable_retrieval ~= nil and enable_retrieval == 1") \
        .retrieve_by_ann_embedding(
          kess_service = "{{fountain_long_view_like_retrieval_kess_service_splash}}",
          space = "cosine",
          timeout_ms = 100,
          reason = self.reason,
          items_from_attr = ["_USER_ID_"],
          attr_single_limit = 50,
          bound_type = {
            "total_limit": "{{fountain_long_view_like_retrieval_total_limit_splash}}",
          },
          algo_type = {
            "scann": {}
          },
          src_bucket = "user",
          dest_bucket = "{{topSubdivisionHetuBucket}}",
          dest_bucket_item_type = 1) \
      .end_()
      
  
