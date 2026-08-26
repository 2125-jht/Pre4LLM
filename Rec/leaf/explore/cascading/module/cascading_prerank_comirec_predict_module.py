from cascading import CommonModule

class CascadingPrerankComirecPredictModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_prerank_comirec_predict == 1") \
        .explore_custom_trim_user_info(
          user_info_attr = "userInfo",
          save_trimed_user_info_to_attr = "explore_prerank_comirec_trimmed_user_info",
          trim_user_info = [
            "id",
            "user_profile_v1.follow_list",
            "user_profile_v1.video_playing_stat"
          ]
        ) \
        .delegate_enrich(
          kess_service = "{{explore_prerank_comirec_service}}",
          recv_item_attrs = [
            {"name": "pctr_retr", "as": "prerank_ctr_comirec"}
          ],
          timeout_ms = 100,
          send_common_attrs = [
            "explore_prerank_comirec_trimmed_user_info"
          ],
          request_type = "explore_prerank_comirec"
        ) \
      .end_()
