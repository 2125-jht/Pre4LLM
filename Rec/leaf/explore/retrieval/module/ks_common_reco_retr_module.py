from retrieval.retrieval_module import RetrievalModule

class KsCommonRecoRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self):
    self.flow \
      .if_("enable_trim_user_info ~= nil and enable_trim_user_info > 0") \
        .explore_custom_trim_user_info(
          user_info_attr = self.user_info_type,
          save_trimed_user_info_to_attr = "trimedUserInfo",
          trim_user_info = self.trim_user_info
        ) \
        .delegate_retrieve(
          kess_service = "{{kess_service_name}}",
          timeout_ms = "{{service_timeout_ms}}",
          reason = self.reason,
          request_type = self.service_request_type,
          request_num = "{{service_request_num}}",
          send_common_attrs_in_request = self.send_common_attrs_in_request, 
          send_common_attrs = [
            {"name": "trimedUserInfo", "as": "user"}
          ]
        ) \
      .else_() \
        .delegate_retrieve(
          kess_service = "{{kess_service_name}}",
          timeout_ms = "{{service_timeout_ms}}",
          reason = self.reason,
          request_type = self.service_request_type,
          request_num = "{{service_request_num}}",
          send_common_attrs_in_request = self.send_common_attrs_in_request, 
          send_common_attrs = [
            {"name": self.user_info_type, "as": "user"}
          ]
        ) \
      .end_() \
      .deduplicate()
  
  @property
  def send_common_attrs_in_request(self) -> bool:
    return self.config.get("send_common_attrs_in_request", False)
  
  @property
  def service_request_type(self) -> str:
    return self.config.get("service_request_type", "default")
  
  @property
  def user_info_type(self) -> str:
    return self.config.get("user_info_type", "userInfo")
  
  @property
  def trim_user_info(self) -> list:
    return self.config.get("trim_user_info", [])