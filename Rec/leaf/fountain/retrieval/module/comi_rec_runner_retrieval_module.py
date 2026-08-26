from retrieval.retrieval_module import RetrievalModule

class ComiRecRunnerRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_("enable_fountain_comi_rec_runner_retr_colossus == 1") \
        .explore_colossus_v2_trigger_enrich(
          colossus_resp_attr = "colossus_resp_v2",
          output_colossus_trigger_attr = "colossus_user_info__trigger_id_list",
          output_colossus_trigger_weight_attr = "colossus_user_info__trigger_weight_list",
          output_colossus_trigger_author_attr = "colossus_user_info__trigger_author_list",
          output_colossus_info_attr = "colossus_user_info__redis_val",
          enable_progressive_trigger = True,
          trigger_select_num = "{{fountain_comi_rec_runner_retr_trigger_select_num}}",
          trigger_select_alpha = "{{fountain_comi_rec_runner_retr_progressive_alpha}}",
          trigger_select_base_num = "{{fountain_comi_rec_runner_retr_progressive_base}}",
          trigger_select_topk = "{{fountain_comi_rec_runner_retr_progressive_topk}}",
          trigger_select_skip_num = "{{fountain_comi_rec_runner_retr_progressive_skip_num}}",
          enable_only_select_fountain = "{{fountain_comi_rec_runner_retr_enable_only_select_fountain}}"
        ) \
      .end_() \
      .explore_custom_trim_user_info(
        user_info_attr = "userInfo",
        save_trimed_user_info_to_attr = "comiRecTrimedUserInfo",
        trim_user_info = [
          "user_profile_v1.like_list",
          "user_profile_v1.follow_list",
          "user_profile_v1.forward_list",
          "user_profile_v1.collect_list",
          "user_profile_v1.comment_list",
          "user_profile_v1.profile_enter_list",
          "user_profile_v1.video_playing_stat",
          "user_profile_v1.search_click_photo_list",
          "user_profile_v1.search_click_author_list",
          "fountain_reco_user_profile.forward_list",
          "fountain_reco_user_profile.follow_list",
          "fountain_reco_user_profile.like_list",
          "fountain_reco_user_profile.comment_list"
        ]
      ) \
      .delegate_retrieve(
        kess_service = "{{fountain_comi_rec_runner_retr_server_name}}",
        timeout_ms = 100,
        reason = self.reason,
        request_num = "{{fountain_comi_rec_runner_retr_request_num}}",
        send_browse_set = True,
        send_common_attrs_in_request = False,
        send_common_attrs = [
          {"name": "fountain_comi_rec_runner_retr_search_num", "as": "search_num"},
          {"name": "fountain_comi_rec_runner_retr_interest_depth", "as": "interest_depth"},
          {"name": "fountain_comi_rec_runner_retr_diversity_boost", "as": "diversity_boost"},
          {"name": "fountain_comi_rec_runner_retr_photo_score_boost", "as": "photo_score_boost"},
          {"name": "fountain_comi_rec_runner_retr_ann_src", "as": "ann_src"},
          {"name": "fountain_comi_rec_runner_retr_ann_dst", "as": "ann_dst"},
          {"name": "featureUId", "as": "uId"},
          {"name": "featureDeviceId", "as": "featureDeviceId"},
          {"name": "featureUserProfileV1FollowAidList", "as": "featureUserProfileV1FollowAidList"},
          {"name": "featureUserProfileV1Play18SPidList", "as": "featureUserProfileV1Play18SPidList"},
          {"name": "featureUserProfileV1Play18SAidList", "as": "featureUserProfileV1Play18SAidList"},
          {"name": "user_fountain_follow_aid_list", "as": "featureFountainProfileFollowAidList"},
          {"name": "featureFountainProfileLongViewPidList", "as": "featureFountainProfileLongViewPidList"},
          {"name": "featureFountainProfileLongViewAidList", "as": "featureFountainProfileLongViewAidList"},
          {"name": "comiRecTrimedUserInfo", "as": "user" },
          "colossusRetrievalTrigger",
          "colossus_user_info__trigger_id_list",
          "colossus_user_info__trigger_author_list"
        ]
      ) 
