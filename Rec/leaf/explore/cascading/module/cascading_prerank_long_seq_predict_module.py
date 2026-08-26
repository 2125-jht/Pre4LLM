from cascading import CommonModule

class CascadingPrerankLongSeqPredictModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_prerank_long_seq_predict == 1") \
        .delegate_enrich(
          kess_service = "{{explore_prerank_long_seq_service}}",
          recv_item_attrs = [
            {"name": "pctr_retr", "as": "prerank_ctr_long_seq"}
          ],
          timeout_ms = 100,
          send_common_attrs = [
            "uId",
            "colossus_photo_id_list",
            "colossus_author_id_list",
            "colossus_channel_list",
            "colossus_play_time_list",
            "colossus_duration_list",
            "colossus_label_list"
          ],
          request_type = "explore_prerank_ls"
        ) \
      .end_()