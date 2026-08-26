from retrieval.retrieval_module import RetrievalModule

class FountainComirecU2IRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .switch_("request_type") \
        .case_("fountain_fast_v1_life") \
          .delegate_retrieve(
            kess_service = "{{fountain_comi_rec_retr_v2_server_name}}",
            timeout_ms = 150,
            reason = self.reason,
            request_num = "{{fountain_comi_rec_retr_v2_request_num}}",
            send_browse_set = True,
            send_common_attrs_in_request = False,
            send_common_attrs = [
              {"name": "fountain_comi_rec_v2_search_num", "as": "search_num"},
              {"name": "fountain_comi_rec_v2_interest_depth", "as": "interest_depth"},
              {"name": "fountain_comi_rec_v2_diversity_boost", "as": "diversity_boost"},
              {"name": "fountain_comi_rec_v2_photo_score_boost", "as": "photo_score_boost"},
              {"name": "fountain_comi_rec_v2_ann_src", "as": "ann_src"},
              {"name": "fountain_comi_rec_v2_ann_dst", "as": "ann_dst"},
              {"name": "featureUId", "as": "uId"},
              {"name": "did", "as": "featureDeviceId"},
              {"name": "featureUserProfileV1FollowAidList", "as": "featureUserProfileV1FollowAidList"},
              {"name": "featureUserProfileV1Play18SPidList", "as": "featureUserProfileV1Play18SPidList"},
              {"name": "featureUserProfileV1Play18SAidList", "as": "featureUserProfileV1Play18SAidList"}
            ]
          ) \
      .end_()