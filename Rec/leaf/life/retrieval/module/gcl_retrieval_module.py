from retrieval.retrieval_module import RetrievalModule

class GclRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_retrieval == 0") \
        .return_() \
      .end_() \
      .enrich_with_protobuf(
        from_extra_var = "user_info_ptr",
        attrs = [
          dict(name="user_play_id_list", path="user_profile_v1.video_playing_stat.photo_id"),
          dict(name="user_play_time_list", path="user_profile_v1.video_playing_stat.playing_time"),
          dict(name="user_photo_duration_list", path="user_profile_v1.video_playing_stat.video_duration"),
          dict(name="user_play_author_list", path="user_profile_v1.video_playing_stat.author_id"),
          dict(name="browsed_photo_ids", path="browsed_photo_ids")
        ]
      ) \
      .if_("enable_global_trigger ~= nil and enable_global_trigger > 0")
    ## 从 global trigger 里完成抽取
    self._sample_global_triggers("global_trigger_list", "global_trigger_weight_list")
    self.flow \
      .end_() \
      .delegate_retrieve(
        kess_service = "{{kess_service_name}}",
        timeout_ms = "{{service_timeout_ms}}",
        reason = self.reason,
        request_type = "default",
        request_num = "{{service_request_num}}",
        send_common_attrs_in_request = False, 
        send_common_attrs = [
          {"name": "user_play_id_list", "as": "user_play_id_list"},
          {"name": "user_play_time_list", "as": "user_play_time_list"},
          {"name": "user_photo_duration_list", "as": "user_photo_duration_list"},
          {"name": "user_play_author_list", "as": "user_play_author_list"},
          {"name": "browsed_photo_ids", "as": "browsed_photo_ids"},
          {"name": "colossus_user_info__trigger_id_list", "as": "colossus_trigger_id_list"},
          {"name": "colossus_user_info__trigger_weight_list", "as": "colossus_trigger_play_time_list"},
          {"name": "colossus_user_info__trigger_author_list", "as": "colossus_trigger_author_list"},
          {"name": "colossus_user_info__knowledge_trigger_set", "as": "knowledge_trigger_list"},
          {"name": "knowledge_trigger_expand_cnt", "as": "knowledge_trigger_expand_cnt"},
          {"name": "enable_global_trigger", "as": "enable_global_trigger"},
          {"name": "global_trigger_list", "as": "global_trigger_list"},
          {"name": "global_trigger_weight_list", "as": "global_trigger_weight_list"}
        ],
        recv_item_attrs = [
          "i2i_retr__trigger_pid"
        ]
      ) \
      .deduplicate()