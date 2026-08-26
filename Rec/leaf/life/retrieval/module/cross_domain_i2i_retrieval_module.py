from retrieval.retrieval_module import RetrievalModule

class CrossDomainI2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self):
    self.flow \
      .if_("enable_retrieval == 0") \
        .return_() \
      .end_() \
      .if_("#(colossus_photo_id_list_new_positive or {}) == 0") \
        .return_() \
      .end_() \
      .fetch_kgnn_neighbors(
        id_from_common_attr = "colossus_photo_id_list_new_positive",
        save_neighbors_to = "result_item_id",
        kess_service = "{{life_cross_domain_i2i_kgnn_server_kess}}",
        relation_name = "I2I",
        sample_num = "{{life_cross_domain_i2i_kgnn_sample_num}}",
        timeout_ms = "{{life_cross_domain_i2i_kgnn_timeout}}",
        sample_type = "weight",
        padding_type = "zero",
        sample_without_replacement=True,
        shard_num = 1,
      ) \
      .deduplicate(
        item_list_from_attr = "result_item_id",
      ) \
      .filter_by_common_attr(
        common_attr = ["browse_screen__pid_list"],
        item_list_from_attr = "result_item_id",
        skip = "{{skip_filter_by_browse_screen}}"
      ) \
      .filter_by_browse_set(
        skip = "{{skip_browse_set}}",
        item_list_from_attr = "result_item_id",
      ) \
      .retrieve_by_common_attrs(
        attrs = [
          {
            "name": "result_item_id",
            "reason": self.reason,
            "num_limit": self.retrieve_num
          }
        ]
      ) \
      .limit(
        size = "{{result_num}}"
      )
  
  @property
  def retrieve_num(self) -> int:
    assert "retrieve_num" in self.config
    return self.config["retrieve_num"]
  
  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = ["colossus_photo_id_list_new_positive", "result_item_id"],
        for_debug_request_only = True,
        respect_sample_loggging = True,
      )