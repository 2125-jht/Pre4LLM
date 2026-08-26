from retrieval.retrieval_module import RetrievalModule

fountain_memory_feature = [
  "featureUId", 
  {"name": "gender", "as": "featureGender"}, 
  "featureAge",  # RPC
  "featureAgeSegment", 
  "featureProvinceId", 
  "featureCityId",
  "featureClientId", 
  "featureVisitMod", 
  "featureVisitNet", 
  "featureAppSignList",  # RPC
  "featureUserLevel", 
  "featureActiveDays",
  "featureTopDislikeTopic", 
  "featureRiskLevel", 
  "featureLongTermInterestPhotoDnnClusterId", 
  "featureUserRequestProvinceId",
  "featureUserRequestCityId", 
  "featureUserClickCount", # RPC
  "featureUserLikeCount", # RPC
  "featureUserFollowCount", # RPC
  "featureUserLongViewCount", # RPC
  "featureUserCtr", # RPC
  "featureUserLtr", # RPC
  "featureUserWtr", # RPC
  "featureUserFtr", # RPC
  "featureUserLvtr", # RPC
  "featureUserSvtr", # RPC
  "featureUserAvgWatchTime", # RPC
  {"name": "userRecentViewTimeListRaw", "as": "profile_v1_time_list"}, 
  {"name": "videoPlayingPid", "as": "profile_v1_pid_list"}, 
  {"name": "profile_v1_click_trigger_aids", "as": "profile_v1_aid_list"}, 
  {"name": "playstat_durations", "as": "profile_v1_duration_list"}, 
  {"name": "playstat_playtimes", "as": "profile_v1_play_list"}, 
  {"name": "playstat_hetu1s", "as": "profile_v1_hetu_one_list"}, 
  {"name": "playstat_hetu2s", "as": "profile_v1_hetu_two_list"}, 
  {"name": "playstat_hetu3s", "as": "profile_v1_hetu_three_list"}, 
  {"name": "playstat_hetu4s", "as": "profile_v1_hetu_four_list"}, 
  {"name": "userRecentViewPageListRaw", "as": "profile_v1_page_list"}, 
  {"name": "user_fountain_play_timestamp_list", "as": "fountain_time_list"}, 
  {"name": "user_fountain_play_id_list", "as": "fountain_pid_list"}, 
  {"name": "user_fountain_play_aid_list", "as": "fountain_aid_list"}, 
  {"name": "user_fountain_play_duration_list", "as": "fountain_duration_list"}, 
  {"name": "user_fountain_play_time_list", "as": "fountain_play_list"}, 
  {"name": "user_fountain_play_hetu_l1_top1_list", "as": "fountain_hetu_one_list"}, 
  {"name": "user_fountain_play_hetu_l2_top1_list", "as": "fountain_hetu_two_list"}, 
  {"name": "user_fountain_play_hetu_l3_top1_list", "as": "fountain_hetu_three_list"}, 
  {"name": "user_fountain_play_hetu_l4_top1_list", "as": "fountain_hetu_four_list"}, 
  {"name": "user_fountain_play_page_list", "as": "fountain_page_list"}, 
  {"name": "like_list", "as": "featureUserProfileV1LikePidList"}, 
  {"name": "comment_list", "as": "featureUserProfileV1CommentPidList"}, 
  {"name": "follow_list", "as": "featureUserProfileV1FollowPidList"}, 
  {"name": "forward_list", "as": "featureUserProfileV1ForwardPidList"}, 
  {"name": "profile_enter_list", "as": "featureUserProfileV1ProfileEnterPidList"}, 
  {"name": "collect_list", "as": "profile_v1_collect_pid_list"},
  {"name": "uLikePidsFountain", "as": "featureFountainProfileLikePidList"}, 
  {"name": "uCommentPidsFountain", "as": "featureFountainProfileCommentPidList"}, 
  {"name": "uFollowPidsFountain", "as": "featureFountainProfileFollowPidList"}, 
  {"name": "uForwardPidsFountain", "as": "fountain_forward_pid_list"}
]

class FountainMemoryRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .switch_("request_type") \
        .case_("fountain_fast_v1_life") \
          .delegate_retrieve(
            kess_service = "{{fountain_memory_retr_service_name}}",
            timeout_ms = 50,
            reason = self.reason,
            request_type = "fountain_fast_v1",
            request_num = "{{fountain_memory_retr_retrieve_num}}",
            send_common_attrs = fountain_memory_feature,
            send_common_attrs_in_request = False,
            send_browse_set = False
          ) \
          .deduplicate() \
          .filter_by_common_attr(
            common_attr=["browse_screen__pid_list"]
          ) \
      .end_()
