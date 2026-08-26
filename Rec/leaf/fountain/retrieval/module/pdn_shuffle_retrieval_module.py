from retrieval.retrieval_module import RetrievalModule

class PdnShuffleRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow.if_("enable_global_trigger == 1")
    ## 从 global trigger 里完成抽取
    self._sample_global_triggers("pdn_total_triggers")
    self.flow.end_()
    self.flow \
      .retrieve_by_redis(
        cluster_name = "recoUserPreferAuthor",
        item_separator = ",",
        attr_separator = ":",
        extra_item_attrs = [
          {"name": "fountain_pdn_score", "type": "double"}
        ],
        key_from_attr = "pdn_total_triggers",
        key_prefix = "{{fountain_total_pdn_retr_key_prefix}}",
        retrieve_num = "{{fountain_total_pdn_retr_num}}",
        retrieve_num_per_key = "{{fountain_total_pdn_retr_num_per_key}}",
        save_src_key_to_attr = "i2i_trigger_id",
        reason = self.reason,
        timeout_ms = 30
      ) \
      .shuffle(weight_attr = "fountain_pdn_score")
