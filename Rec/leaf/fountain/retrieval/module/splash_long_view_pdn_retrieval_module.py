from retrieval.retrieval_module import RetrievalModule

class SplashLongViewPdnRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_redis(
        cluster_name = "recoUserPreferAuthor",
        item_separator = ",",
        attr_separator = ":",
        extra_item_attrs = [
          {"name": "redis_score"}
        ],
        key_from_attr = "featureFountainProfileLongViewPidList",
        key_prefix = "{{fountain_longview_pdn_retr_key_prefix}}",
        retrieve_num = "{{fountain_longview_pdn_retr_num}}",
        retrieve_num_per_key = "{{fountain_longview_pdn_retr_num_per_key}}",
        reason = self.reason,
        skip = "{{skip_fountain_longview_pdn_retr}}",
        timeout_ms = 100)