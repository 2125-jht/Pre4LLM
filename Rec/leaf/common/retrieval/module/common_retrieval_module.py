from common.retrieval import RetrievalModule

class CommonRetrievalModule(RetrievalModule):
  def __init__(self, name: str, config_dir: str) -> None:
    super().__init__(name, config_dir)
  
  def process(self):
    send_common_attrs = self.send_common_attrs
    if self.trimed_user_info_config:
      self.flow \
        .explore_custom_trim_user_info(
          user_info_attr = self.user_info_attr,
          save_trimed_user_info_to_attr = "trimed_user_info",
          trim_user_info = self.trimed_user_info_config
        )
      send_common_attrs.append({ "name": "trimed_user_info", "as": "user" })
    else:
      send_common_attrs.append({ "name": self.user_info_attr, "as": "user" })
    
    if self.extra_skip_condition:
      self.flow \
        .if_(self.extra_skip_condition) \
          .return_() \
        .end_()

    self.flow \
      .delegate_retrieve(
        kess_service = "{{kess_service_name}}",
        timeout_ms = "{{service_timeout_ms}}",
        reason = self.reason,
        request_type = self.service_request_type,
        request_num = "{{service_request_num}}",
        send_common_attrs_in_request = self.send_common_attrs_in_request, 
        send_common_attrs = send_common_attrs,
        send_browse_set = self.send_browse_set,
        recv_item_attrs = self.recv_item_attrs,
        recv_common_attrs = self.recv_common_attrs
      ) \
      .deduplicate()

    if (self.set_directly_reach_fullrank):
      self.flow \
        .if_("enable_directly_reach_fullrank ~= 0") \
          .set_attr_value(
            item_attrs=[
              {
                "name": "is_directly_reach_fullrank",
                "type": "int",
                "value": 1
              }
            ]
          ) \
        .end_()
      
    if (self.custom_label_attrs):
      self.flow \
        .set_attr_value(
          item_attrs = [
            {
              "name": label_attr,
              "type": "int",
              "value": 1
            } for label_attr in self.custom_label_attrs
          ],
        )

  @property
  def send_common_attrs_in_request(self) -> bool:
    return self.config.get("send_common_attrs_in_request", False)
  
  @property
  def service_request_type(self) -> str:
    return self.config.get("service_request_type", "default")
  
  @property
  def user_info_attr(self) -> str:
    return self.config.get("user_info_attr", "userInfo")
  
  @property
  def trimed_user_info_config(self) -> list:
    return self.config.get("trimed_user_info")
  
  @property
  def send_common_attrs(self) -> list:
    return self.config.get("send_common_attrs", [])
  
  @property
  def send_browse_set(self) -> bool:
    return self.config.get("send_browse_set", True)
  
  @property
  def extra_skip_condition(self) -> str:
    return self.config.get("extra_skip_condition", None)

  @property
  def recv_item_attrs(self) -> list:
    return self.config.get("recv_item_attrs", [])

  @property
  def recv_common_attrs(self) -> list:
    return self.config.get("recv_common_attrs", [])

  @property
  def set_directly_reach_fullrank(self) -> list:
    return self.config.get("set_directly_reach_fullrank", False)

  @property
  def custom_label_attrs(self) -> list:
    return self.config.get("custom_label_attrs", [])