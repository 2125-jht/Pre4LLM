from retrieval.retrieval_module import RetrievalModule

class SineV2RetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .delegate_retrieve(
        kess_service = "{{fountain_sine_v2_retr_server_name}}",
        timeout_ms = "{{fountain_sine_v2_retr_timeout_ms}}",
        reason = self.reason,
        request_num = "{{fountain_sine_v2_retr_request_num}}",
        send_browse_set = True,
        send_common_attrs_in_request = False,
        send_common_attrs = [
          {"name": "fountain_sine_v2_search_num", "as": "search_num"},
          {"name": "fountain_sine_v2_interest_depth", "as": "interest_depth"},
          {"name": "fountain_sine_v2_diversity_boost", "as": "diversity_boost"},
          {"name": "fountain_sine_v2_photo_score_boost", "as": "photo_score_boost"},
          {"name": "featureUId", "as": "uId"},
          {"name": "featureDeviceId", "as": "featureDeviceId"},
          {"name": "featureUserProfileV1LikePidList", "as": "featureUserProfileV1LikePidList"},
          {"name": "featureUserProfileV1Play18SPidList", "as": "featureUserProfileV1Play18SPidList"},
          {"name": "featureUserProfileV1ForwardPidList", "as": "featureUserProfileV1ForwardPidList"},
          {"name": "featureFountainProfileEffViewPidList", "as": "featureFountainProfileEffViewPidList"},
          {"name": "featureUserProfileV1Play18SAidList", "as": "featureUserProfileV1Play18SAidList"},
          {"name": "featureUserProfileV1FollowAidList", "as": "featureUserProfileV1FollowAidList"}
        ],
        skip = "{{skip_fountain_sine_v2_retr_server}}")
