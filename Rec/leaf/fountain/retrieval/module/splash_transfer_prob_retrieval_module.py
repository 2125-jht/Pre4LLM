from retrieval.retrieval_module import RetrievalModule

class SplashTransferProbRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_transfer_prob(
        redis_server = "recoRelevancePhotoTransitionProbability",
        key_source = "featureSourcePId",
        reason = self.reason,
        version = "{{reco_relevance_transfer_prob_retrieval_version}}",
        attrs = [{
          "name": "RelevanceTransferProbAttr",
        }],
        skip = "{{fountain_retrieval_skip_transfer_prob}}")