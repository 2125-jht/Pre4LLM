from retrieval.retrieval_module import RetrievalModule

class AddingFollowInnerRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self):
    self.flow \
        .if_("_USER_ID_ > 0") \
          .retrieve_by_redis(
            reason = self.reason,
            cluster_name="recoFollowOfflineFeatureKiwi",
            retrieve_num = "{{adding_follow_retrieval_module_in_explore_num}}",
            key_from_attr = "_USER_ID_",
            key_prefix = "follow_generate_model_",
            timeout_ms = 20,
            item_separator="," 
          ) \
        .end_()