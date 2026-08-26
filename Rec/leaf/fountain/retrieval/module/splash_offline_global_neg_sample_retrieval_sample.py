from retrieval.retrieval_module import RetrievalModule

class SplashOfflineGlobalNegativeSampleRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_redis(
        cluster_name = "recoOfflineGlobalNegativeSample",
        item_regex = "(\d+)",
        key_from_attr = "fountain_swing_retr_redis_key",
        reason = self.reason,
        retrieve_num = 10000,
        skip = "{{fountain_skip_swing_retr}}") 
