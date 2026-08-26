from common.retrieval import RetrievalModule

class CommonRetrievalWithSkipModule(RetrievalModule):
  """
  CommonRetrievalModule 增加 skip 开关
  ------
  - 为和线上开关保持一致，在 delegate_retrieve() 中加 skip 开关
  - 无特殊需求不要使用此 Module
  """
  def __init__(self, name: str, config_dir: str) -> None:
    super().__init__(name, config_dir)
  
  def process(self):

    self.flow \
      .delegate_retrieve(
        kess_service = "{{kess_service_name}}",
        timeout_ms = "{{service_timeout_ms}}",
        reason = self.reason,
        request_type = self.service_request_type,
        request_num = "{{service_request_num}}",
        send_common_attrs_in_request = self.send_common_attrs_in_request, 
        send_common_attrs = self.send_common_attrs,
        skip = "{{skip_kess_service}}"
      ) \
      .deduplicate()

  @property
  def send_common_attrs_in_request(self) -> bool:
    return self.config.get("send_common_attrs_in_request", False)
  
  @property
  def service_request_type(self) -> str:
    return self.config.get("service_request_type", "default")
  
  @property
  def send_common_attrs(self) -> list:
    return self.config.get("send_common_attrs", [])
