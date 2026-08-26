from retrieval.retrieval_module import RetrievalModule

class SplashIcfRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_redis(
        cluster_name = "recoUserPreference",
        reason = self.reason,
        key_from_attr = "featureFountainProfileEffViewPidList",
        key_prefix = "{{fountain_splash_icf_retr_key_prefix}}",
        item_separator = ",",
        attr_separator = ":",
        extra_item_attrs = [
          {"name": "redis_score", "type": "double", "as_score": True}
        ],
        retrieve_num = "{{fountain_splash_icf_retr_num}}",
        retrieve_num_per_key = "{{fountain_splash_icf_retr_num_per_key}}",
        skip = "{{return skip_fountain_icf_splash_retr == 1}}",
      )
