from retrieval.retrieval_module import RetrievalModule

class SplashGongXianNReasonRetrievalModule(RetrievalModule):
  def __init__(self, 
  name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .retrieve_by_relevance_transfer(
        redis_server = "recoRelevancePhotoTransitionProbability",
        key_source = "featureSourcePId",
        prefix = "relevance_high_ctr",
        reason = self.reason,
        max_size = "{{fountain_gongxian_retrieval_max_size}}",
        version = "{{fountain_gongxian_retrieval_version}}",
        skip = "{{skip_fountain_gongxian_retrieval_splash_nreason}}")