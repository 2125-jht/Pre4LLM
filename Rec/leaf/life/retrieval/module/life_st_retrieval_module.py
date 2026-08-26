from retrieval.retrieval_module import RetrievalModule


class LifeStRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .delegate_retrieve(
        reason=self.reason,
        kess_service="{{service_name}}",
        timeout_ms=50,
        request_type="life_st_retr",
        request_num="{{limit_size}}",
        send_common_attrs_in_request=False,
        send_common_attrs=[],
      )

