from retrieval import RetrievalModule
class SwingSimNegU2URetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_remote_index(
        kess_service = "{{swing_common_index_service}}",
        timeout_ms = "{{service_timeout_ms}}",
        reason = self.reason,
        common_query = "",
        querys = [{
          "query": "user2userId_V2:{{_USER_ID_}}",
          "max_attr_num" : "{{swing_trigger_hate_list_max_len}}",
          "search_num" : "{{swing_trigger_u2u_max_len}}",
        }],
        save_score_to_attr = "swing_sim_score",
        default_random_search = 0,
        default_total_request_num = "{{swing_retr_total_request_num}}"
      ) \
      .deduplicate() \
      .filter_by_attr(
        attr_name = "swing_sim_score",
        remove_if = "<=",
        compare_to = "{{swing_sim_score_threshold}}",
        remove_if_attr_missing = True,
      ) \
      .copy_item_meta_info(
        save_item_id_to_attr = "swing_u2u_user_id",
      ) \
      .pack_item_attr(
        item_source = {
          "reco_results" : True,
        },
        mappings = [{
          "from_item_attr" : "swing_u2u_user_id", 
          "to_common_attr" : "swing_u2u_list",
        }]
      ) \
      .limit(0) \
      .retrieve_by_redis(
        reason = self.reason,
        retrieve_num = "{{swing_redis_max_retrieve_num}}",
        key_prefix = "hate_item_V2",
        cluster_name = "explorerCache",
        key_from_attr = "swing_u2u_list",
        item_separator = ",",
        extra_item_attrs = [],
        save_result_to_common_attr = "swing_user_hate_u2u2i_list"
      )