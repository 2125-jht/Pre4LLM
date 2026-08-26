from retrieval.retrieval_module import RetrievalModule

class SplashPersonalRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .enrich_with_protobuf(
        from_extra_var = "userInfoPb",
        attrs = [
          dict(name="video_playing_stat_pid_list", path="user_profile_v1.video_playing_stat.photo_id"),
          dict(name="video_playing_stat_aid_list", path="user_profile_v1.video_playing_stat.author_id"),
          dict(name="video_playing_stat_duration_list", path="user_profile_v1.video_playing_stat.video_duration"),
          dict(name="video_playing_stat_play_time_list", path="user_profile_v1.video_playing_stat.playing_time"),
          dict(name="video_playing_stat_timestamp_list", path="user_profile_v1.video_playing_stat.client_timestamp"),
          dict(name="video_playing_stat_page_list", path="user_profile_v1.video_playing_stat.page"),
          dict(name="video_playing_stat_hetu_list", path="user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_one", repeat_limit={"user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_one": 1}, repeat_align=True),
          dict(name="video_playing_stat_hetu2_list", path="user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_two", repeat_limit={"user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_two": 1}, repeat_align=True),
          dict(name="video_playing_stat_hetu3_list", path="user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_three", repeat_limit={"user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_three": 1}, repeat_align=True),
          dict(name="video_playing_stat_hetu4_list", path="user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_four", repeat_limit={"user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_four": 1}, repeat_align=True), 
          dict(name="user_fountain_play_id_list", path="fountain_reco_user_profile.video_play_stat.photo_id"),
          dict(name="user_fountain_play_aid_list", path="fountain_reco_user_profile.video_play_stat.author_id"),
          dict(name="user_fountain_play_time_list", path="fountain_reco_user_profile.video_play_stat.playing_time"),
          dict(name="user_fountain_play_duration_list", path="fountain_reco_user_profile.video_play_stat.video_duration"),
          dict(name="user_fountain_play_timestamp_list", path="fountain_reco_user_profile.video_play_stat.client_timestamp"),
          dict(name="user_fountain_play_page_list", path="fountain_reco_user_profile.video_play_stat.page"),
          dict(name="user_fountain_play_hetu_list", path="fountain_reco_user_profile.video_play_stat.hetu_tag_level_info.hetu_level_one", repeat_limit={"fountain_reco_user_profile.video_play_stat.hetu_tag_level_info.hetu_level_one": 1}, repeat_align=True),
          dict(name="user_fountain_play_hetu2_list", path="fountain_reco_user_profile.video_play_stat.hetu_tag_level_info.hetu_level_two", repeat_limit={"fountain_reco_user_profile.video_play_stat.hetu_tag_level_info.hetu_level_two": 1}, repeat_align=True),
          dict(name="user_fountain_play_hetu3_list", path="fountain_reco_user_profile.video_play_stat.hetu_tag_level_info.hetu_level_three", repeat_limit={"fountain_reco_user_profile.video_play_stat.hetu_tag_level_info.hetu_level_three": 1}, repeat_align=True),
          dict(name="user_fountain_play_hetu4_list", path="fountain_reco_user_profile.video_play_stat.hetu_tag_level_info.hetu_level_four", repeat_limit={"fountain_reco_user_profile.video_play_stat.hetu_tag_level_info.hetu_level_four": 1}, repeat_align=True),
        ]
      ) \
      .delegate_retrieve(
        kess_service = "{{fountain_splash_personal_retr_service_name}}",
        timeout_ms = 50,
        reason = self.reason,
        request_type = "default",
        request_num = "{{fountain_splash_personal_retr_retrieve_num}}",
        send_common_attrs = self.config["fountain_splash_personal_feature"],
        send_common_attrs_in_request = False
      )
    