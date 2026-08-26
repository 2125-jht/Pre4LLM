from retrieval import CommonModule

class UserInfoColossusModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .if_(self.config["enable_colossus_v2"] + " == 1") \
        .gsu_common_colossusv2_enricher(
          kconf = "colossus.kconf_client.video_item",
          limit = 10000,
          item_fields = dict(
            photo_id = "",
            play_time = "",
            label = "",
            author_id = "",
            channel = "",
            duration = "",
            timestamp = "",
            tag = "",
          ),
          v1_service_name = "grpc_colossusSimV2",
          v1_colossus_resp = "sim_v2_colossus_resp",
        ) \
        .colossus(
          input_colossus_resp_attr = "sim_v2_colossus_resp",
          service_name = "grpc_colossusSimV2",
          client_type = "common_item_client",
          output_attr = "colossus_v2_resp",
          parse_to_pb = False,
        ) \
      .end_() \
      .if_("enable_replace_colossus_v2_from_retr == 1") \
        .copy_attr(
          attrs = [
            {
              "from_common": "colossus_v2_resp",
              "to_common": "colossus_resp_v2",
            },
          ],
        ) \
      .else_() \
        .colossus(
          **self.config["colossus"]
        ) \
      .end_()

    if self.need_colossus_histroy:
      self.flow \
        .if_(self.config["enable_user_colossus"] + " == 1") \
          .explore_user_colossus_history(
            **self.config["explore_user_colossus_history"]
          ) \
        .end_()
  
  @property
  def need_colossus_histroy(self) -> str:
    return self.config.get("need_colossus_histroy", False)
