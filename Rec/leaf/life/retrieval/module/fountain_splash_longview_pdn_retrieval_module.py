from retrieval.retrieval_module import RetrievalModule

class FountainSplashLongViewPdnRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .switch_("request_type") \
        .case_("fountain_splash_life") \
          .retrieve_by_redis(
            cluster_name = "recoUserPreferAuthor",
            item_separator = ",",
            attr_separator = ":",
            extra_item_attrs = [
              {"name": "redis_score"}
            ],
            key_from_attr = "featureFountainProfileLongViewPidList",
            key_prefix = "{{fountain_longview_pdn_retr_key_prefix_splash}}",
            retrieve_num = "{{fountain_longview_pdn_retr_num_splash}}",
            retrieve_num_per_key = "{{fountain_longview_pdn_retr_num_per_key_splash}}",
            reason = self.reason,
            timeout_ms = 100
          ) \
          .log_debug_info(
            item_attrs = [
              "redis_score"
            ]
          ) \
      .end_()