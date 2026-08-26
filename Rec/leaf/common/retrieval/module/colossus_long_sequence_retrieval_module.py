from common.retrieval import RetrievalModule

class ColossusLongSequenceRetrievalModule(RetrievalModule):
  def __init__(self, name: str, config_dir: str) -> None:
    super().__init__(name, config_dir)

  def process(self):
    self.flow \
      .explore_custom_trim_user_info(
        user_info_attr = "userInfo",
        save_trimed_user_info_to_attr = "trimedUserInfo",
        trim_user_info = self.trim_user_info
      ) \
      .delegate_retrieve(
        kess_service = "{{service_name}}",
        timeout_ms = "{{service_timeout_ms}}",
        reason = self.reason,
        request_type = "default",
        request_num = "{{retrieve_num}}",
        send_browse_set = False,
        send_common_attrs_in_request = False,
        send_common_attrs = [
          {"name": "trimedUserInfo", "as": "user"},
          "diversity_boost",
          "colossus_photo_id_list",
          "colossus_author_id_list",
          "colossus_channel_list",
          "colossus_play_time_list",
          "colossus_duration_list",
          "colossus_label_list"
        ]
      )

  @property
  def trim_user_info(self) -> list:
    return self.config.get(
      "trim_user_info", 
      [
        "id",
        "device_id",
        "browsed_photo_ids", 
        "slide_browsed_photo_ids"
      ]
    )