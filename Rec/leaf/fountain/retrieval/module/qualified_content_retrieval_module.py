from retrieval.retrieval_module import RetrievalModule

class QualifiedContentRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .delegate_retrieve(
        kess_service = "{{fountain_qualified_content_retr_service_name}}",
        request_num = "{{fountain_qualified_content_retr_request_num}}",
        timeout_ms = "{{fountain_qualified_content_retr_timeout_ms}}",
        reason = self.reason,
        send_browse_set = True,
        send_common_attrs_in_request = False,
        send_common_attrs = [
          {"name": "user_browsed_photo_ids", "as": "browsed_photo_ids"},
          {"name": "video_playing_stat_pid_list", "as": "watch_pid_list"},
          {"name": "video_playing_stat_duration_list", "as": "pid_duration_list"},
          {"name": "video_playing_stat_play_time_list", "as": "play_time_list"},
          {"name": "featureSourcePId", "as": "featureSourcePId"},
          {"name": "user_fountain_play_id_list", "as": "user_fountain_play_id_list"},
          {"name": "user_fountain_play_time_list", "as": "user_fountain_play_time_list"},
          {"name": "user_fountain_play_hetu_list", "as": "user_fountain_play_hetu_list"},
          {"name": "colossusTargetItemTrigger", "as": "target_trigger_list"}
        ],
        skip = "{{skip_fountain_qualified_content_retr}}")