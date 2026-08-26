from retrieval.retrieval_module import RetrievalModule

class PicLongtermInterestHetuRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
    
  def process(self) -> None:
    self.flow \
      .enrich_with_protobuf(
        from_extra_var = "user_info_ptr",
        attrs = [
          dict(name="hetuLevelTwoLongTermId", path="user_interest_profile.hetu_level_two_long_term_id"),
          dict(name="hetuLevelTwoLongTermScore", path="user_interest_profile.hetu_level_two_long_term_score")
        ]
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "hetuLevelTwoLongTermId", "as": "key_list"},
          {"name": "hetuLevelTwoLongTermScore", "as": "value_list"},
          {"name": "pic_hetu_longterm_two_limit_num", "as": "limit_size"},
        ],
        export_common_attr = [
          {"name": "return_key_list", "as": "hetu_longterm_tag_list"},
        ],
        function_name = "SoryKeyListByValueList",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .if_("hetu_longterm_tag_list == nil or #hetu_longterm_tag_list == 0") \
       .return_() \
      .end_() \
      .retrieve_by_remote_index(
        kess_service = "{{remote_index_service_name}}",
        timeout_ms = "{{remote_index_service_timeout_ms}}",
        reason = self.reason,
        common_query = "",
        querys = [{
          "query": "hetuV1LevelTwo:{{hetu_longterm_tag_list}}",
          "random_search": 0,
          "search_num": "{{search_num}}"
        }],
        default_search_num = 50,
      ) \
      .deduplicate() \
      .limit(
        size = "{{result_num}}"
      )
      
  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = [
          "hetu_longterm_tag_list",
          "hetuLevelTwoLongTermId",
          "hetuLevelTwoLongTermScore",
        ]
      )
