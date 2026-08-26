from retrieval.retrieval_module import RetrievalModule

class LongViewJingxuanIcfRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_redis(
        cluster_name = "dataScienceExp1",
        item_regex = "(\d+):[0-9]{1,}[.]{0,1}[0-9]*",
        key_from_attr = "featureFountainProfileLongViewPidListSub",
        key_prefix = "{{fountain_longview_jingxuan_icf_retr_key_prefix}}",
        retrieve_num = "{{fountain_longview_jingxuan_icf_retr_num}}",
        retrieve_num_per_key = "{{fountain_longview_jingxuan_icf_retr_num_per_key}}",
        reason = self.reason,
        skip = "{{skip_fountain_longview_jingxuan_icf_retr}}",
        timeout_ms = 100)
