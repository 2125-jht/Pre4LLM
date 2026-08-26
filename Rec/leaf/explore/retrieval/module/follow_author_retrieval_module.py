from retrieval.retrieval_module import RetrievalModule

class FollowAuthorRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self):
    self.flow \
      .enrich_with_protobuf(
        from_extra_var = "user_info_ptr",
        attrs = [
          dict(name="followTimestamps", path="follow_list.time"),
        ]
      ) \
      .enrich_attr_by_lua(
        import_common_attr = ["followAids", "friendAids", "followTimestamps", "max_history_s", "skip_filter_bifollow"],
        export_common_attr = ["triggerAids"],
        function_for_common = "calculate",
        lua_script_file = 'explore/retrieval/lua/module/follow_author_retr__trigger_filter.lua'
      ) \
      .if_("enable_trigger_shuffle > 0") \
        .shuffle_list_attr(
          common_attr = "triggerAids"
        ) \
      .end_() \
      .if_("enable_high_value_ua == 1") \
        .if_("enable_high_value_ua_emp == 1") \
          .retrieve_by_redis(
            cluster_name = "recoAuthorPopularity",
            item_separator = ",",
            key_from_attr = "featureUId",
            key_prefix = "highvalue_new_info_KUAISHOU_",
            retrieve_num = "{{author_max_num}}",
            retrieve_num_per_key = "{{author_max_num}}",
            save_result_to_common_attr = "followed_ua_author",
            timeout_ms = 20
          ) \
        .else_() \
          .retrieve_by_redis(
            cluster_name = "recoFollowOfflineFeatureKiwi",
            item_separator = ",",
            attr_separator = ":",
            extra_item_attrs = [
              {"name": "redis_score"}
            ],
            key_from_attr = "featureUId",
            key_prefix = "hv_ua_score_120d_KUAISHOU_",
            retrieve_num = "{{author_max_num}}",
            retrieve_num_per_key = "{{author_max_num}}",
            save_result_to_common_attr = "followed_ua_author",
            timeout_ms = 20
          ) \
        .end_() \
        .if_("enable_trigger_merge == 1 and #(followed_ua_author or {}) < author_max_num") \
          .pack_common_attr(
            input_common_attrs = ["followed_ua_author", "triggerAids"],
            output_common_attr = "triggerAids",
            limit_num = "{{author_max_num}}",
            deduplicate = True
          ) \
        .else_() \
          .if_("#(followed_ua_author or {}) == 0") \
            .return_() \
          .end_() \
          .copy_attr(
            attrs = [{
              "from_common": "followed_ua_author",
              "to_common": "triggerAids"
            }]
          ) \
        .end_() \
      .end_() \
      .retrieve_by_remote_index(
        kess_service = "{{remote_index_service_name}}",
        timeout_ms = 80,
        reason = self.reason, 
        querys = [
          {
            "query": "{{retr_index_term}}:{{triggerAids}}",
            "search_num": "{{remote_index_search_num}}", 
            "max_attr_num": "{{author_max_num}}",
            "random_search" : "{{enable_random_search}}"
          }
        ],
        save_score_to_attr = "index_score",
        save_query_index_to_attr = "followed_photo_query_index"
      ) \
      .deduplicate() \
      .shuffle(
        weight_attr = "index_score",
        skip = "{{skip_result_shuffle}}"
      ) \
      .if_("enable_snake_merge == 1") \
        .explore_snake_merge(
          cluster_attr_name = "followed_photo_query_index",
          max_item_num = "{{result_num}}"
        ) \
      .else_() \
        .sort(
          score_from_attr = "index_score"
        ) \
        .limit(
          size = "{{result_num}}"
        ) \
      .end_() \
  
  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        item_attrs = ["redis_score@followed_ua_author"]
      )
