from retrieval.retrieval_module import RetrievalModule

class LongtermInterestAuthorRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self):
    self.flow \
      .enrich_with_protobuf(
        from_extra_var = "user_info_ptr",
        attrs = [
          dict(name="history_author_triggers", path="history_triggers.history_long_view_triggers"), 
          dict(name="hateAids", path="user_profile_v1.hate_list.author_id")
        ]
      ) \
      .pack_common_attr(
        input_common_attrs = ["hateAids", "browse_screen__aid_list"],
        output_common_attr = "hateAids",
        deduplicate = True
      ) \
      .if_("use_slide_longterm_author == 1") \
        .retrieve_by_redis(
          reason = 0,
          retrieve_num = "{{longterm_author_max_size}}",
          cluster_name = "recoEyeshotFollow",
          timeout_ms = 10,
          key_from_attr = "_USER_ID_", 
          key_prefix = self.key_prefix_from_attr,
          item_separator = ",",
          save_result_to_common_attr = "long_term_auhtor_dpage__list"
        ) \
        .pack_common_attr(
          input_common_attrs = ["long_term_auhtor_dpage__list", "history_author_triggers"],
          output_common_attr = "history_author_triggers",
          deduplicate = True
        ) \
      .end_() \
      .enrich_attr_by_lua(
        import_common_attr = ["history_author_triggers", "hateAids", "author_max_num"],
        export_common_attr = ["triggerAids"],
        function_for_common = "calculate",
        lua_script_file = "explore/retrieval/lua/module/longterm_interest_author_retr__trigger_filter.lua"
      ) \
      .if_("enable_similar_author == 1") \
        .retrieve_by_redis(
          reason = 2,
          retrieve_num = "{{retr_num_limit}}",
          retrieve_num_per_key = "{{retr_num_each}}",
          cluster_name = "recoHotLauRank",
          timeout_ms = 18,
          key_from_attr = "triggerAids",
          item_separator = ",",
          save_result_to_common_attr = "simTriggerAids"
        ) \
        .copy_attr(
          attrs = [{"from_common": "simTriggerAids", "to_common": "triggerAids"}]
        ) \
      .else_if_("enable_homogenous_author == 1") \
        .retrieve_by_redis(
          reason = 2,
          retrieve_num = "{{retr_num_limit}}",
          retrieve_num_per_key = "{{retr_num_each}}",
          cluster_name = "recoUserPreference",
          timeout_ms = 18,
          key_from_attr = "triggerAids",
          item_separator = ",",
          save_result_to_common_attr = "simTriggerAids"
        ) \
        .copy_attr(
          attrs = [{"from_common": "simTriggerAids", "to_common": "triggerAids"}]
        ) \
      .end_() \
      .retrieve_by_remote_index(
        kess_service = "{{remote_index_service_name}}",
        timeout_ms = 72,
        reason = self.reason, 
        querys = [
          {
            "query": "authorId2PhotoIdOrderByUploadTime:{{triggerAids}}",
            "search_num": "{{remote_index_search_num}}", 
            "max_attr_num": 1000
          }
        ],
        save_score_to_attr = "index_score",
        save_query_index_to_attr = "query_index"
      ) \
      .deduplicate() \
      .if_("use_browset_filter == 1") \
        .filter_by_browse_set() \
      .end_() \
      .if_("use_rank_neg_filter == 1") \
        .split_string(
          input_common_attr = "rank_neg_photo_id_list_str",
          output_common_attr = "longterm_author_rank_neg_photo_id_filter_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
        .filter_by_common_attr(
          common_attr=["longterm_author_rank_neg_photo_id_filter_list"],
        ) \
      .end_() \
      .if_("enable_snake_merge == 1") \
        .shuffle(
          weight_attr = "index_score"
        ) \
        .explore_snake_merge(
          cluster_attr_name = "query_index",
          max_item_num = "{{result_num}}"
        ) \
      .else_() \
        .sort(
          score_from_attr = "index_score"
        ) \
        .limit(
          size = "{{result_num}}"
        ) \
      .end_()
    
  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = ["history_author_triggers", "hateAids", "triggerAids"]
      )

  @property
  def key_prefix_from_attr(self) -> str:
    return self.config.get("key_prefix_from_attr", "")