from retrieval.retrieval_module import RetrievalModule


class LifeStUnbiasRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .delegate_retrieve(
        reason=self.reason,
        kess_service="{{service_name}}",
        timeout_ms=50,
        request_type="life_st_unbias_retr",
        request_num="{{limit_size}}",
        send_common_attrs_in_request=False,
        send_common_attrs=[
          {"name": "basic_info_gender_v2", "as": "uGender"},
          {"name": "basic_info_age_segment_v2", "as": "uAgeSeg"},
        ],
      )

