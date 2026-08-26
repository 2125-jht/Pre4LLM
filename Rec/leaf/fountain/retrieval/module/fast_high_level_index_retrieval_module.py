from retrieval.retrieval_module import RetrievalModule

class FastHighLevelIndexRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_("user_risk_level and user_risk_level >= fountain_user_risk_min") \
      .enrich_with_protobuf(
        from_extra_var = "userInfoPb",
        attrs = [
          dict(name="click_list_hetu2", path="fountain_reco_user_profile.click_list.hetu_tag_level_info.hetu_level_two"),
          dict(name="like_list_hetu2", path="fountain_reco_user_profile.like_list.hetu_tag_level_info.hetu_level_two"),
          dict(name="follow_list_hetu2", path="fountain_reco_user_profile.follow_list.hetu_tag_level_info.hetu_level_two"),
          dict(name="forward_list_hetu2", path="fountain_reco_user_profile.forward_list.hetu_tag_level_info.hetu_level_two"),
          dict(name="comment_list_hetu2", path="fountain_reco_user_profile.comment_list.hetu_tag_level_info.hetu_level_two"),
          dict(name="search_click_list_hetu2", path="user_profile_v1.search_click_photo_list.hetu_tag_level_info.hetu_level_two")
        ]
      ) \
      .pack_common_attr(
        input_common_attrs = ["like_list_hetu2", "follow_list_hetu2", "forward_list_hetu2", "comment_list_hetu2", "search_click_list_hetu2"],
        output_common_attr = "profile_v1_interaction_hetu2_list",
        deduplicate = False
      ) \
      .get_kconf_params(
        kconf_configs = [{
          "kconf_key": "reco.hot.hetu2AllSet",
          "export_common_attr": "hetu2_all_set",
          "value_type": "list_int64"
        }]
      ) \
      .shuffle_list_attr(
        common_attr = "hetu2_all_set"
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "profile_v1_interaction_hetu2_list",
          "click_list_hetu2",
          "hetu2_select_num",
          "hetu2_all_set"
        ],
        export_common_attr = [
          "user_interest_hetu2"
        ],
        function_name = "GetUserInterestHetu2",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .retrieve_by_remote_index(
        kess_service = "{{fountain_fast_high_level_index_retr_services}}",
        timeout_ms = 150,
        reason = self.reason,
        common_query = "",
        querys = [{
          "query": "hetuTagLevelTwo:{{user_interest_hetu2}}",
          "random_search": 1,
          "search_num": "{{fountain_fast_high_level_index_search_num}}",
        }],
        default_search_num = 200,
        default_random_search = 1) \
      .end_()