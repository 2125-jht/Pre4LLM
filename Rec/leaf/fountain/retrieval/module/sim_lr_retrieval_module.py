from retrieval.retrieval_module import RetrievalModule

class SimLrRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self):
    self.flow \
      .explore_custom_trim_user_info(
        user_info_attr = "userInfo",
        save_trimed_user_info_to_attr = "sim_lr_trimed_user_info",
        trim_user_info = self.trimed_user_info_config
    ) \
    .if_("enable_fountain_custom_iic_pdn_retr > 0") \
        .set_attr_value(
          common_attrs=[
            {
              "name": "fountain_retr_custom_iic_pdn_kconf_key",
              "type": "string",
              "value": "reco.fountain.iic_pdn_config_fountain_pdn_on"
            }
          ]
        ) \
    .else_() \
      .set_attr_value(
        common_attrs=[
          {
            "name": "fountain_retr_custom_iic_pdn_kconf_key",
            "type": "string",
            "value": "reco.fountain.iic_pdn_config_fountain_pdn_off"
          }
        ]
      ) \
    .end_() \
    .delegate_retrieve(
      kess_service = "{{kess_service_name}}",
      timeout_ms = "{{service_timeout_ms}}",
      reason = self.reason,
      request_num = "{{service_request_num}}",
      send_common_attrs_in_request = False, 
      send_common_attrs = self.send_common_attrs,
      skip = "{{skip_fountain_sim_lr_retr}}"
    ) \

  @property
  def trimed_user_info_config(self) -> list:
    return self.config.get("trimed_user_info")

  @property
  def send_common_attrs(self) -> list:
    return self.config.get("send_common_attrs", [])