from retrieval.retrieval_module import RetrievalModule

fountain_multi_target_feature = [
  {"name": "featureUId", "as": "uId"},
  {"name": "upload_count", "as": "uUploadCount"},
  {"name": "follow_count", "as": "uFollowCount"},
  {"name": "fans_count", "as": "uFansCount"},
  {"name": "featureAgeSegment", "as": "uBasicAge"},
  {"name": "location_city_level_v2", "as": "uCityLevelNew"},
  {"name": "upload_rate", "as": "uUploadRate"},
  {"name": "featureUserRequestProvinceId", "as": "uProvinceId"},
  {"name": "featureUserRequestCityId", "as": "uCityId"},
  {"name": "gender", "as": "uGender"},
  {"name": "true_gender", "as": "uTrueGender"},
  {"name": "infer_gender", "as": "uInferGender"},
  {"name": "true_year", "as": "uTrueYear"},
  {"name": "infer_year", "as": "uInferYear"},
  {"name": "featureVisitNet", "as": "uNetwork"},
  {"name": "is_douyin", "as": "uIsDouYin"},
  "featrueUserLongTermHetu1Id",
  "featrueUserLongTermHetu1Score",
  "featrueUserLongTermHetu2Id",
  "featrueUserLongTermHetu2Score",
  "featrueUserLongTermHetu3Id",
  "featrueUserLongTermHetu3Score",
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
  {"name": "click_list", "as": "click_list"},
  "uClickPids",
  "uLikePids",
  "uFollowAids",
  "uLikePidsFountain",
  "uFollowAidsFountain"
]

class FountainU2URetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .switch_("request_type") \
        .case_("fountain_fast_v1_life") \
          .explore_common_user_feature_enricher( # switch逻辑在修改ab后缀策略后统一删除
            user_info_attr = "user_info_ptr",
            user_click_pids_attr = "uClickPids",
            user_like_pids_attr = "uLikePids",
            user_follow_pids_attr = "uFollowAids",
          ) \
          .delegate_retrieve(
            kess_service = "{{fountain_multi_target_u2u_retr_service_name}}",
            timeout_ms = 100,
            reason = self.reason,
            request_type = "default",
            request_num = "{{fountain_multi_target_u2u_retr_retrieve_num}}",
            send_common_attrs = fountain_multi_target_feature,
            send_common_attrs_in_request = False
          ) \
          .deduplicate() \
          .filter_by_common_attr(
            common_attr=["browse_screen__pid_list"]
          ) \
      .end_()
