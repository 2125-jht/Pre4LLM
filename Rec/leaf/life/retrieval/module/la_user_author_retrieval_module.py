from retrieval.retrieval_module import RetrievalModule

class LaUserAuthorRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self):
    self.flow \
      .copy_attr(
        attrs=[{
          "from_common": "explore_la_long_view_author_list",
          "to_common": "explore_la_long_view_author_list_trigger"
        }]
      ) \
      .if_("enable_trigger_shuffle == 1") \
        .shuffle_list_attr(
          common_attr = "explore_la_long_view_author_list_trigger"
        ) \
      .end_() \
      .retrieve_by_remote_index(
        kess_service = "{{remote_index_service_name}}",
        timeout_ms = 100,
        reason = self.reason,
        querys = [
          {
            "query": "{{retr_index_term}}:{{explore_la_long_view_author_list_trigger}}",
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
