from retrieval.retrieval_module import RetrievalModule

class ItemCfRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_explore_fountain_itemcf(
        kess_service = "{{fountain_leaf_itemcf_retr_service_name}}",
        user_info_attr = "userInfoPb",
        reason = self.reason,
        total_limit = "{{fountain_leaf_itemcf_retr_num}}",
        skip = "{{skip_fountain_leaf_itemcf_retr}}",
        timeout_ms = "{{fountain_leaf_itemcf_retr_timeout_ms}}",
        only_fountain_behavior = "{{fountain_leaf_itemcf_only_fountain_behavior}}")