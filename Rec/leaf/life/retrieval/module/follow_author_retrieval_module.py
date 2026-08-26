from retrieval.retrieval_module import RetrievalModule

class FollowAuthorRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self):
    self.flow \
      .if_("enable_retrieval == 0") \
        .return_() \
      .end_() \
      .enrich_with_protobuf(
        from_extra_var = "user_info_ptr",
        attrs = [
          dict(name="followAids", path="follow_list.user.id"),
          dict(name="followTimestamps", path="follow_list.time"),
          dict(name="friendAids", path="friend_info_v2.bid_follow_list.friend_id")
        ]
      ) \
      .enrich_attr_by_lua(
        import_common_attr = ["followAids", "friendAids", "followTimestamps", "max_history_s", "skip_filter_bifollow"],
        export_common_attr = ["triggerAids"],
        function_for_common = "calculate",
        lua_script_file = 'life/retrieval/lua/module/follow_author_retr__trigger_filter.lua'
      ) \
      .if_("enable_trigger_shuffle > 0") \
        .shuffle_list_attr(
          common_attr = "triggerAids"
        ) \
      .end_() \
      .if_("enable_author_importance > 0") \
        .fetch_kgnn_neighbors(
          id_from_common_attr = "_USER_ID_",
          save_neighbors_to = "top_follow_author_list",
          save_weight_to = "top_follow_author_weight_list",
          kess_service = "grpc_kgnn_double_user_follow_author_graph-U2U",
          relation_name = "U2U",
          shard_num = 1,
          sample_num = "{{top_author_num}}",
          timeout_ms = 40,
          sample_type = "topn",
          padding_type = "zero"
        ) \
        .enrich_attr_by_lua(
          import_common_attr = ["top_follow_author_list", "top_follow_author_weight_list"],
          export_common_attr = ["top_follow_author_list", "top_follow_author_weight_list"],
          function_for_common = "generate_valid_top_author",
          lua_script_file = 'life/retrieval/lua/module/follow_author_retr__trigger_filter.lua'
        ) \
        .filter_by_common_attr(
          common_attr = ["top_follow_author_list"],
          item_list_from_attr = "triggerAids"
        ) \
        .gen_common_attr_by_lua(
          attr_map = {
            "top_author_each_num": "math.max(math.min(math.floor(result_num / (#(top_follow_author_list or {}) + 1) ) , 200), 5)",
            "author_each_num": "math.max(math.min(math.floor(result_num * 1.5 / (#(triggerAids or {}) + 1) ) , 100), 2)",
            "tmp_result_num": "math.floor(result_num * 1.5)"
          }
        ) \
        .retrieve_by_remote_index(
          kess_service = "{{remote_index_service_name}}",
          timeout_ms = "{{remote_index_service_timeout_ms}}",
          reason = self.reason, 
          querys = [
            {
              "query": "{{retr_index_term}}:{{top_follow_author_list}}",
              "search_num": "{{top_author_each_num}}", 
              "max_attr_num": "{{author_max_num}}"
            },{
              "query": "{{retr_index_term}}:{{triggerAids}}",
              "search_num": "{{author_each_num}}", 
              "max_attr_num": "{{author_max_num}}"
            }
          ],
          save_score_to_attr = "index_score",
          default_total_request_num = "{{tmp_result_num}}",
          browsed_item_count = "{{filter_browsed_num}}"
        ) \
      .else_() \
        .retrieve_by_remote_index(
          kess_service = "{{remote_index_service_name}}",
          timeout_ms = "{{remote_index_service_timeout_ms}}",
          reason = self.reason, 
          querys = [
            {
              "query": "{{retr_index_term}}:{{triggerAids}}",
              "search_num": "{{remote_index_search_num}}", 
              "max_attr_num": "{{author_max_num}}"
            }
          ],
          save_score_to_attr = "index_score"
        ) \
      .end_() \
      .deduplicate() \
      .sort(
        score_from_attr = "index_score"
      ) \
      .limit(
        size = "{{result_num}}"
      )
  
  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = ["triggerAids", "top_follow_author_weight_list", "top_follow_author_list"]
      )
      