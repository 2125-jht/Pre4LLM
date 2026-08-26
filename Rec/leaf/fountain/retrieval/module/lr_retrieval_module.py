from retrieval.retrieval_module import RetrievalModule

class LrRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  # TODO fountain_model_retrieve 还在 subdivision 目录，需要迁移到 explore 目录下
  def process(self) -> None:
    self.flow \
      .fountain_model_retrieve(
        kess_service = "{{fountain_leaf_lr_retr_service_name}}",
        user_info_attr = "userInfo",
        reason = self.reason,
        total_limit = "{{fountain_leaf_lr_retr_num}}",
        timeout_ms = 150,
        use_simple_user_info = "{{fountain_leaf_lr_use_simple_user_info}}",
        skip = "{{fountain_skip_leaf_lr_retr}}")
