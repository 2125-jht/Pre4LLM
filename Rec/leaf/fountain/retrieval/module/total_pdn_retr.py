from retrieval.retrieval_module import RetrievalModule

class TotalPdnRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow.if_("enable_global_trigger == 1")
    ## 从 global trigger 里完成抽取
    self._sample_global_triggers("pdn_total_triggers")
    self.flow.end_()
    self.flow \
      .shuffle_list_attr(
        common_attr = "pdn_total_triggers",
        skip = "{{skip_fountain_shuffle_pdn_total_triggers}}") \
      .retrieve_by_redis(
        cluster_name = "recoUserPreferAuthor",
        item_separator = ",",
        attr_separator = ":",
        extra_item_attrs = [
          {"name": "redis_score"}
        ],
        key_from_attr = "pdn_total_triggers",
        key_prefix = "{{fountain_total_pdn_retr_key_prefix}}",
        retrieve_num = "{{fountain_total_pdn_retr_num}}",
        retrieve_num_per_key = "{{fountain_total_pdn_retr_num_per_key}}",
        save_src_key_to_attr = "i2i_trigger_id",
        reason = self.reason,
        timeout_ms = 30,
        skip = "{{skip_fountain_total_pdn_retr}}",
      ) \
      .if_("enable_snake_merge == 1") \
        .deduplicate() \
        .filter_by_browse_set() \
        .cast_attr_type(
          attr_type_cast_configs=[
            {
              "to_type": "int",
              "from_item_attr": "i2i_trigger_id",
              "to_item_attr": "trigger_id"
            }
          ]
        ) \
        .shuffle(
          weight_attr = "redis_score",
        ) \
        .explore_snake_merge(
          cluster_attr_name = "trigger_id",
          max_item_num = "{{result_num}}"
        ) \
      .end_()
