from retrieval.retrieval_module import RetrievalModule

class IcfRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_redis(
        cluster_name = "recoUserPreferAuthor",
        item_regex = "(\d+):[0-9]{1,}[.]{0,1}[0-9]*",
        key_from_attr = "featureFountainProfileEffViewPidList",
        key_prefix = "{{fountain_icf_retr_key_prefix}}",
        retrieve_num = "{{fountain_icf_retr_num}}",
        retrieve_num_per_key = "{{fountain_icf_retr_num_per_key}}",
        save_src_key_to_attr = "i2i_trigger_id",
        reason = self.reason,
        timeout_ms = 100,
        skip = "{{skip_fountain_icf_retr}}",
      )
