from retrieval.retrieval_module import RetrievalModule

class PicFollowAuthorRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self):
    self.flow \
      .enrich_with_protobuf(
        from_extra_var = "user_info_ptr",
        attrs = [
          dict(name="follow_author_list", path="follow_list.user.id"),
         ]
      ) \
      .if_("enable_trigger_shuffle == 1") \
        .shuffle_list_attr(
          common_attr = "follow_author_list"
        ) \
      .end_() \
      .retrieve_by_remote_index(
        kess_service = "{{remote_index_service_name}}",
        timeout_ms = "{{remote_index_service_timeout_ms}}",
        reason = self.reason, 
        querys = [
          {
            "query": "authorId:{{follow_author_list}}",
            "search_num": "{{remote_index_search_num}}", 
            "max_attr_num": "{{author_max_num}}"
          }
        ],
        default_search_num = 30,
        default_total_request_num = 600,
        default_random_search = 1
      ) \
      .deduplicate() \
      .filter_by_common_attr(
        common_attr = ["browse_screen__pid_list"]
      ) \
      .limit(
        size = "{{result_num}}"
      )