from retrieval.retrieval_module import RetrievalModule


class PicExploredHighInteractionRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .explore_colossus_v2_pic_trigger_enrich(
        colossus_resp_attr="colossus_resp_v2",
        output_colossus_trigger_attr="colossus_trigger_list",
        enable_get_weighted_trigger_list_attr=True,
        picture_trigger_num="{{trigger_num}}",
        picture_trigger_interact_num="{{interact_limit}}",
        colossus_channel_select_str="{{channel_str}}",
      ) \
      .delegate_retrieve(
        kess_service="{{kess_service_name}}",
        timeout_ms="{{service_timeout_ms}}",
        reason=self.reason,
        request_type="default",
        request_num="{{service_request_num}}",
        send_common_attrs_in_request=True,
        send_common_attrs=self.sent_common_attrs
      ) \
      .deduplicate()

  @property
  def user_info_type(self) -> str:
    return self.config.get("user_info_type", "userInfo")

  @property
  def sent_common_attrs(self) -> list:
    return [
      {"name": self.user_info_type, "as": "user"},
      "colossus_trigger_list"
    ]
