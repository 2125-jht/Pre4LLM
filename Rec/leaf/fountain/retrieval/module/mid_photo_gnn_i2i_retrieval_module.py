from retrieval.retrieval_module import RetrievalModule

class MidPhotoGnnI2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .delegate_retrieve(
        kess_service = "{{fountain_mid_photo_gnn_i2i_retr_kess_service}}",
        request_type = "default",
        request_num = "{{fountain_mid_photo_gnn_i2i_retr_request_num}}",
        timeout_ms = "{{fountain_mid_photo_gnn_i2i_retr_timeout_ms}}",
        reason = self.reason,
        send_browse_set = False,
        send_common_attrs_in_request = False,
        send_common_attrs = [
          {"name": "user_browsed_photo_ids", "as": "browsed_photo_ids"},
          {"name": "user_fountain_play_aid_list", "as": "fountain_video_playing_author_list"},
          {"name": "user_fountain_play_time_list", "as": "fountain_video_playing_time_list"},
          {"name": "user_fountain_play_duration_list", "as": "fountain_video_video_duration_list"},
          {"name": "featureFountainProfileLikeAidList", "as": "fountain_like_author_list"},
          {"name": "user_fountain_forward_aid_list", "as": "fountain_forward_author_list"},
          {"name": "featureFountainProfileCommentAidList", "as": "fountain_comment_author_list"},
          {"name": "featureUserProfileV1DownloadAidList", "as": "download_author_list"},
          {"name": "featureUserProfileV1SearchClickAidList", "as": "search_author_list"},
          {"name": "featureUserProfileV1DupClickAidList", "as": "dupclick_author_list"},
          {"name": "video_playing_stat_aid_list", "as": "video_playing_author_list"},
          {"name": "video_playing_stat_play_time_list", "as": "video_playing_time_list"},
          {"name": "video_playing_stat_duration_list", "as": "video_video_duration_list"},
          {"name": "featureUserProfileV1ProfileEnterAidList", "as": "profile_enter_author_list"},
          {"name": "featureUserProfileV1LikeAidList", "as": "like_author_list_limit"},
          {"name": "featureUserProfileV1ForwardAidList", "as": "forward_author_list"},
          {"name": "featureUserProfileV1CommentAidList", "as": "comment_author_list"}
        ],
        skip = "{{skip_fountain_mid_photo_gnn_i2i_retr}}"
      )
