from retrieval.retrieval_module import RetrievalModule

class PdnRetrModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow.if_("enable_global_trigger ~= nil and enable_global_trigger > 0")
    ## 从 global trigger 里完成抽取
    self._sample_global_triggers("trigger_list", "trigger_weight_list")
    self.flow \
      .else_() \
        .copy_attr(
          attrs=[{
            "from_common": "colossus_user_info__trigger_id_list",
            "to_common": "colossus_trigger_id_list"
          },
          {
            "from_common": "videoPlayingPid",
            "to_common": "trigger_list"
          }]
        ) \
        .limit(
          size = 40,
          item_list_from_attr = "trigger_list"
        ) \
        .shuffle_list_attr(
          common_attr = "colossus_trigger_id_list"
        ) \
        .pack_common_attr(
          input_common_attrs = ["colossus_trigger_id_list"],
          output_common_attr = "extra_trigger_list",
          limit_num = "{{colossus_trigger_max_num}}"
        ) \
        .pack_common_attr(
          input_common_attrs = ["trigger_list", "extra_trigger_list"],
          output_common_attr = "trigger_list",
          deduplicate = True
        ) \
        .if_("enable_knowledge_trigger > 0 and colossus_user_info__knowledge_trigger_set ~= nil and #colossus_user_info__knowledge_trigger_set > 0") \
          .enrich_attr_by_lua(
            import_common_attr = ["trigger_list", "colossus_user_info__knowledge_trigger_set", "knowledge_trigger_max_num"],
            export_common_attr = ["trigger_list"],
            function_for_common = "add_knowledge_trigger",
            lua_script_file = "explore/retrieval/lua/module/colossus_ann__add_knowledge_trigger.lua"
          ) \
        .end_() \
      .end_() \
      .if_("trigger_list == nil or #trigger_list == 0") \
        .return_() \
      .end_() \
      .if_("enable_snake_merge == 1") \
        .retrieve_by_remote_index(
          kess_service = "{{index_service_name}}",
          timeout_ms = "{{index_service_timeout_ms}}",
          reason = self.reason,
          querys = [
            {
              "query": "{{index_term_name}}:{{trigger_list}}",
              "search_num": "{{search_num_per_trigger}}",
              "expire_second": "{{index_cache_expire_time_s}}",
              "random_search": 1
            }
          ],
          save_score_to_attr = "index_score",
          save_query_index_to_attr = "query_index",
          exclude_items_in_attr = "trigger_list"
        ) \
        .deduplicate() \
        .filter_by_browse_set() \
        .filter_by_common_attr(
          common_attr = ["browse_screen__pid_list"]
        ) \
        .shuffle(
          weight_attr = "index_score",
        ) \
        .explore_snake_merge(
          cluster_attr_name = "query_index",
          max_item_num = "{{cand_num}}"
        ) \
      .else_() \
        .retrieve_by_remote_index(
          kess_service = "{{index_service_name}}",
          timeout_ms = "{{index_service_timeout_ms}}",
          reason = 1,
          querys = [
            {
              "query": "{{index_term_name}}:{{trigger_list}}",
              "search_num": "{{search_num_per_trigger}}",
              "expire_second": "{{index_cache_expire_time_s}}",
              "random_search": 1
            }
          ],
          save_score_to_attr = "index_score",
          save_query_index_to_attr = "query_index",
          exclude_items_in_attr = "trigger_list",
          save_result_to_common_attr = "result_item_id",
        ) \
        .deduplicate(
          item_list_from_attr = "result_item_id",
        ) \
        .if_("skip_filter_by_browse_screen == 0") \
          .filter_by_common_attr(
            common_attr = ["browse_screen__pid_list"],
            item_list_from_attr = "result_item_id",
          ) \
        .end_() \
        .if_("skip_browse_set == 0") \
          .filter_by_browse_set(
            item_list_from_attr = "result_item_id",
          ) \
        .end_() \
        .explore_add_inverted_index_weighted_score(
          item_list_from_attr = "result_item_id",
          score_attr = "index_score",
          query_index_attr = "query_index",
          query_index_weight_list = "{{trigger_weight_list}}",
          query_index_trigger_list = "{{trigger_list}}",
          save_weighted_score_to_attr = "weighted_score",
          save_trigger_to_attr = "i2i_retr__trigger_pid"
        ) \
        .sort(
          item_list_from_attr = "result_item_id",
          score_from_attr = "weighted_score",
        ) \
        .retrieve_by_common_attrs(
          attrs = [
            {
              "name": "result_item_id",
              "reason": self.reason,
              "num_limit": self.retrieve_num
            }
          ]
        ) \
        .limit(size = "{{cand_num}}") \
      .end_()
  
  @property
  def retrieve_num(self) -> int:
    assert "retrieve_num" in self.config
    return self.config["retrieve_num"]