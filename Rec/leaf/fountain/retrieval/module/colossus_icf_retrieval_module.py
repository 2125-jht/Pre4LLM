from retrieval.retrieval_module import RetrievalModule

class ColossusIcfRetrievalModule(RetrievalModule):
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
        key_from_attr = "colossusRetrievalTrigger",
        key_prefix = "{{fountain_colossus_icf_retr_key_prefix}}",
        retrieve_num = "{{fountain_colossus_icf_retr_num}}",
        retrieve_num_per_key = "{{fountain_colossus_icf_retr_num_per_key}}",
        reason = self.reason,
        skip = "{{skip_fountain_colossus_icf_retr}}",
        timeout_ms = 100)