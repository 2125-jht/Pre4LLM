from retrieval.retrieval_module import RetrievalModule

fountain_splash_personal_feature = [
  {"name": "featureUId", "as": "uId"},
  "featureSourcePId",
  "sourcePidAuthorId",
  "sourcePidMmuImgClusterV3",
  "sourcePidMmuImgClusterV4",
  "sourcePidDuration",
  {"name": "source_hetu_level_two",   "as": "SourcePidHetuTagLevel2"},
  {"name": "source_hetu_level_three", "as": "SourcePidHetuTagLevel3"},
  {"name": "source_hetu_level_four", "as": "SourcePidHetuTagLevel4"},
  {"name": "source_hetu_level_five",  "as": "SourcePidHetuTagLevel5"},
  {"name": "videoPlayingPid", "as": "playstat_pids"},
  {"name": "profile_v1_click_trigger_aids", "as": "playstat_aids"},
  {"name": "playstat_playtimes", "as": "playstat_playtimes"},
  {"name": "playstat_durations", "as": "playstat_durations"},
  {"name": "playstat_hetu1s", "as": "playstat_hetu1s"},
  {"name": "playstat_hetu2s", "as": "playstat_hetu2s"},
  {"name": "playstat_hetu3s", "as": "playstat_hetu3s"},
  {"name": "playstat_hetu4s", "as": "playstat_hetu4s"},
  {"name": "userRecentViewTimeListRaw", "as": "playstat_timestamps"},
  {"name": "userRecentViewPageListRaw", "as": "playstat_pages"},
  {"name": "user_fountain_play_id_list", "as": "user_fountain_play_id_list"},
  {"name": "user_fountain_play_aid_list", "as": "user_fountain_play_aid_list"},
  {"name": "user_fountain_play_time_list", "as": "user_fountain_play_time_list"},
  {"name": "user_fountain_play_duration_list", "as": "user_fountain_play_duration_list"},
  {"name": "user_fountain_play_timestamp_list", "as": "user_fountain_play_timestamp_list"},
  {"name": "user_fountain_play_page_list", "as": "user_fountain_play_page_list"},
  {"name": "user_fountain_play_hetu_l1_top1_list", "as": "user_fountain_play_hetu_l1_top1_list"},
  {"name": "user_fountain_play_hetu_l2_top1_list", "as": "user_fountain_play_hetu_l2_top1_list"},
  {"name": "user_fountain_play_hetu_l3_top1_list", "as": "user_fountain_play_hetu_l3_top1_list"},
  {"name": "user_fountain_play_hetu_l4_top1_list", "as": "user_fountain_play_hetu_l4_top1_list"},
]

class FountainSplashPersonalRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .switch_("request_type") \
        .case_("fountain_splash_life") \
          .delegate_retrieve(
            kess_service = "{{fountain_splash_personal_retr_service_name}}",
            timeout_ms = 50,
            reason = self.reason,
            request_type = "default",
            request_num = "{{fountain_splash_personal_retr_retrieve_num}}",
            send_common_attrs = fountain_splash_personal_feature,
            send_common_attrs_in_request = False
          ) \
      .end_()
    
  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = ["user_fountain_play_id_list","user_fountain_play_aid_list","user_fountain_play_time_list", "user_fountain_play_duration_list", "user_fountain_play_timestamp_list", "user_fountain_play_page_list","user_fountain_play_hetu_l1_top1_list", "user_fountain_play_hetu_l4_top1_list"],
        for_debug_request_only = True,
        respect_sample_loggging = True,
      )