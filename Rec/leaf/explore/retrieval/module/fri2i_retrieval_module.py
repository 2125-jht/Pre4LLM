from retrieval.retrieval_module import RetrievalModule


class FrI2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
      super().__init__(name)

  def process(self) -> None:
    # 从 global trigger 里完成抽取
    self._sample_global_triggers("trigger_list", "trigger_weight_list")
    self.flow \
      .if_("shuffle_triggers == 1") \
        .shuffle_list_attr(
          common_attr="trigger_list"
        ) \
      .end_() \
      .fetch_kgnn_neighbors(
        id_from_common_attr="trigger_list",
        save_neighbors_to="pid_list",
        kess_service="{{service_name}}",
        relation_name="I2I",
        shard_num=1,
        sample_num="{{retrieve_num_per_trigger}}",
        timeout_ms=40,
        sample_type="topn",
        padding_type="no_padding"
      ) \
      .retrieve_by_common_attr(
        attr="pid_list", 
        reason=self.reason
      ) \
      .deduplicate() \
      .filter_by_common_attr(
        common_attr=["browse_screen__pid_list"]
      ) \
      .filter_by_browse_set(
        skip="{{skip_browse_set}}"
      ) \
      .limit(size="{{retrieve_num}}")
