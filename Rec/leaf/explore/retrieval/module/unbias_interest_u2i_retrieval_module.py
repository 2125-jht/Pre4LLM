from retrieval.retrieval_module import RetrievalModule

class UnbiasInterestU2IRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_ ("user_risk_level and user_risk_level < risk_level_min or mc_u2i_user_embedding_list == nil") \
        .return_() \
      .end_() \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_unbias_interest_u2i_retr_cid_prefix", "as": "key_prefix"},
          "basic_info_age_segment_v2",
          "basic_info_gender_v2",
        ],
        export_common_attr = [
          {"name": "user_age_gender_key", "as": "ann_dst"},
        ],
        function_name = "GetUserAgeGenderKey",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .delegate_retrieve(
        kess_service = "{{service_name}}",
        timeout_ms = "{{service_timeout_ms}}",
        reason = self.reason,
        request_num = "{{retrieve_num}}",
        send_common_attrs_in_request = False,
        send_common_attrs = [
          {"name": "candidate_num", "as": "candidate_num"},
          {"name": "distance_threshold", "as": "distance_threshold"},
          {"name": "search_num", "as": "search_num"},
          {"name": "ann_dst", "as": "ann_dst"},
          {"name": "mc_u2i_user_embedding_list", "as": "src_embedding"},
        ],
        reset_item_type = 0
      ) \
      .filter_by_common_attr(
        common_attr = [
          "browse_screen__pid_list"
        ]
      ) \
      .filter_by_browse_set() \
      .limit(
        size = "{{final_cand_num}}"
      )
