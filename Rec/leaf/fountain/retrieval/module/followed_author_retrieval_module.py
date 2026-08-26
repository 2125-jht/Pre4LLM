from retrieval.retrieval_module import RetrievalModule

class FollowedAuthorRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  # 关注作者召回
  # 在老代码中 v2 是关闭的
  def process(self) -> None:
    self.flow \
      .copy_attr(
        attrs = [{
          "from_common": "followAuthors",
          "to_common": "followTriggers"
        }]
      ) \
      .if_("enable_fountain_followed_author_retrieval_high_value_ua == 1") \
        .if_("enable_fountain_followed_author_retrieval_high_value_ua_emp == 1") \
          .retrieve_by_redis(
            cluster_name = "recoAuthorPopularity",
            item_separator = ",",
            key_from_attr = "featureUId",
            key_prefix = "highvalue_new_info_KUAISHOU_",
            retrieve_num = "{{fountain_followed_author_trigger_num}}",
            retrieve_num_per_key = "{{fountain_followed_author_trigger_num}}",
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
            retrieve_num = "{{fountain_followed_author_trigger_num}}",
            retrieve_num_per_key = "{{fountain_followed_author_trigger_num}}",
            save_result_to_common_attr = "followed_ua_author",
            timeout_ms = 20
          ) \
        .end_() \
        .gen_common_attr_by_lua(
          attr_map={
            "followed_ua_author": "followed_ua_author or {}",
          }
        ) \
        .if_("enable_fountain_followed_author_retrieval_trigger_merge == 1 and #(followed_ua_author or {}) < fountain_followed_author_trigger_num") \
          .shuffle_list_attr(
            common_attr="followTriggers"
          ) \
          .pack_common_attr(
            input_common_attrs = ["followed_ua_author", "followTriggers"],
            output_common_attr = "followTriggers",
            limit_num = "{{fountain_followed_author_trigger_num}}",
            deduplicate = True
          ) \
        .else_() \
          .copy_attr(
            attrs = [{
              "from_common": "followed_ua_author",
              "to_common": "followTriggers"
            }]
          ) \
        .end_() \
      .end_() \
      .retrieve_by_remote_index(
        # 关注作者
        kess_service = "{{fountain_followed_author_retrieval_query_server}}",
        timeout_ms = 30,
        reason = self.reason,
        reset_item_type = 1,
        common_query = "",
        querys = [{
          "query": "{{fountain_followed_author_retrieval_query}}:{{followTriggers}}",
          "search_num": "{{fountain_followed_author_search_num}}"
        }],
        default_search_num = 100,
        default_total_request_num = "{{fountain_followed_author_search_total_num}}",
        save_score_to_attr = "followed_photo_score",
        save_query_index_to_attr = "followed_photo_query_index",
        skip = "{{skip_fountain_followed_author_retrieval_query_server}}"
      ) \
      .shuffle(
        weight_attr = "followed_photo_score",
        skip = "{{skip_fountain_followed_author_retr_shuffle}}"
      ) \
      .explore_snake_merge(
        cluster_attr_name = "followed_photo_query_index",
        max_item_num = "{{fountain_followed_author_snake_merge_limit}}",
        skip = "{{skip_fountain_followed_author_retrieval_snake_merge}}"
      ) \
      .fetch_kgnn_neighbors(
        id_from_common_attr = "_USER_ID_",
        save_neighbors_to = "top_follow_author_list",
        save_weight_to = "top_follow_author_weight_list",
        kess_service = "grpc_kgnn_double_user_follow_author_graph-U2U",
        relation_name = "U2U",
        shard_num = 1,
        sample_num = "{{fountain_followed_author_retr_v3_top_author_num}}",
        timeout_ms = 20,
        sample_type = "topn",
        padding_type = "zero",
        skip = "{{skip_fountain_followed_author_importance_apply}}"
      ) \
      .enrich_attr_by_lua(
        import_common_attr = ["top_follow_author_list", "top_follow_author_weight_list"],
        export_common_attr = ["top_follow_author_list", "top_follow_author_weight_list"],
        function_for_common = "generate_valid_top_author",
        lua_script_file = "fountain/retrieval/lua/module/followed_author_retr__trigger_filter.lua",
        skip = "{{skip_fountain_followed_author_retr_v3}}"
      ) \
      .copy_attr(
        attrs = [{
          "from_common": "followAuthors",
          "to_common": "followAuthorTriggers"
        }],
        skip = "{{skip_fountain_followed_author_retr_v3}}"
      ) \
      .filter_by_common_attr(
        common_attr = ["top_follow_author_list"],
        item_list_from_attr = "followAuthorTriggers",
        skip = "{{skip_fountain_followed_author_retr_v3}}"
      ) \
      .gen_common_attr_by_lua(
        attr_map = {
          "follow_author_retr_num_per_author": "math.max(math.floor(fountain_followed_author_retr_v3_total_num * 1.2 / (#(followAuthorTriggers or {}) + 1)), 2)"
        },
        skip = "{{skip_fountain_followed_author_retr_v3}}"
      ) \
      .retrieve_by_remote_index(
        # 关注作者 v3
        kess_service = "{{fountain_followed_author_retrieval_query_server}}",
        timeout_ms = 30,
        reason = self.reason,
        reset_item_type = 1,
        common_query = "",
        querys = [{
          "query": "authorId2PhotoIdOrderByUploadTime:{{followAuthorTriggers}}",
          "search_num": "{{follow_author_retr_num_per_author}}",
          "max_attr_num": 1000
        }],
        default_search_num = 100,
        browsed_item_count = 0,
        default_total_request_num = "{{fountain_followed_author_retr_v3_total_num}}",
        save_score_to_attr = "followed_photo_score",
        skip = "{{skip_fountain_followed_author_retr_v3}}"
      ) \
      .sort(
        score_from_attr = "followed_photo_score",
        skip = "{{skip_fountain_follow_retr_post_sort_truncate}}"
      ) \
      .limit(
        size = "{{fountain_follow_retr_post_truncate_size}}",
        skip = "{{skip_fountain_follow_retr_post_sort_truncate}}"
      )