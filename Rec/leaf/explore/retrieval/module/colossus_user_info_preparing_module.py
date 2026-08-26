from retrieval import CommonModule

class ColossusUserInfoPreparingModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
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
      .gsu_common_colossusv2_enricher(
        kconf = "colossus.kconf_client.video_item",
        limit = 10000,
        item_fields = dict(
          photo_id = "colossus_all_photo_id_list",
          play_time = "colossus_all_play_time_list",
          label = "colossus_all_label_list",
          author_id = "colossus_all_author_id_list",
          channel = "colossus_all_channel_list",
          duration = "colossus_all_duration_list",
          timestamp = "colossus_all_timestamp_list",
          tag = "colossus_all_tag_list",
        ),
      ) \
      .colossus(
        service_name = "grpc_colossusSimV2",
        client_type = "common_item_client",
        input_colossus_resp_attr = "sim_v2_colossus_resp",
        output_attr = "colossus_resp_v2",
        parse_to_pb = False,
      )
