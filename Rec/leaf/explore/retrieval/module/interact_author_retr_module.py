from retrieval.retrieval_module import RetrievalModule

class InteractAuthorRetrModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self):
    self.flow \
      .copy_attr(
        attrs = [{"from_common": "commonTriggerAids", "to_common": "triggerAids"}]
      ) \
      .if_("enable_trigger_shuffle") \
        .shuffle_list_attr(common_attr = "triggerAids") \
      .end_() \
      .retrieve_by_remote_index(
        kess_service = "{{remote_index_service_name}}",
        timeout_ms = "{{remote_index_service_timeout_ms}}",
        reason = self.reason, 
        querys = [
          {
            "query": "{{remote_index_query_term}}:{{triggerAids}}",
            "search_num": "{{remote_index_search_num}}", 
            "max_attr_num": 1000
          }
        ],
        save_score_to_attr = "index_score"
      ) \
      .deduplicate() \
      .filter_by_common_attr(
        common_attr = ["browse_screen__pid_list"],
        skip = "{{skip_filter_by_browse_screen}}"
      ) \
      .filter_by_browse_set(
        skip = "{{skip_browse_set}}"
      ) \
      .sort(
        score_from_attr = "index_score"
      ) \
      .limit(
        size = "{{result_num}}"
      )
  
  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = ["downloadAids", "searchClickAids", "dupClickAids", "longViewAids", "profileEnterAids", "likeAids", "forwardAids", "commentAids", "hateAids", "triggerAids"]
      )