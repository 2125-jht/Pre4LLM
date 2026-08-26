from retrieval.retrieval_module import RetrievalModule

class SineV1RetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .delegate_retrieve(
        kess_service = "{{fountain_sine_v1_retr_server_name}}",
        timeout_ms = 100,
        reason = self.reason,
        request_num = "{{fountain_sine_v1_retr_request_num}}",
        send_browse_set = True,
        send_common_attrs_in_request = False,
        send_common_attrs = [
          {"name": "fountain_sine_v1_search_num", "as": "search_num"},
          {"name": "fountain_sine_v1_interest_depth", "as": "interest_depth"},
          {"name": "fountain_sine_v1_diversity_boost", "as": "diversity_boost"},
          {"name": "fountain_sine_v1_photo_score_boost", "as": "photo_score_boost"},
          {"name": "featureUId", "as": "uId"},
          {"name": "featureDeviceId", "as": "featureDeviceId"},
          {"name": "featureUserProfileV1FollowAidList", "as": "featureUserProfileV1FollowAidList"},
          {"name": "featureUserProfileV1Play18SPidList", "as": "featureUserProfileV1Play18SPidList"},
          {"name": "featureUserProfileV1Play18SAidList", "as": "featureUserProfileV1Play18SAidList"}
        ],
        skip = "{{skip_fountain_sine_v1_retr_server}}"
      )
