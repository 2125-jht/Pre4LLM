from retrieval.retrieval_module import RetrievalModule

class PdnActionRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "search_num_per_trigger", "as": "origin_size"},
          {"name": "increase_quota_status", "as": "increase_quota_status"},
          {"name": "increase_quota_factor", "as": "factor"}
        ],
        export_common_attr = [
          {"name": "final_size", "as": "search_num_per_trigger"}
        ],
        function_name = "IncreaseQuotaProcess",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .if_("enable_global_trigger ~= nil and enable_global_trigger > 0")
    ## 从 global trigger 里完成抽取
    self._sample_global_triggers("trigger_list", "trigger_weight_list")
    self.flow \
      .else_() \
        .enrich_with_protobuf(
          from_extra_var = "user_info_ptr",
          attrs = [
            dict(name="like_list", path="user_profile_v1.like_list.photo_id"),
            dict(name="follow_list", path="user_profile_v1.follow_list.photo_id"),
            dict(name="forward_list", path="user_profile_v1.forward_list.photo_id"),
            dict(name="comment_list", path="user_profile_v1.comment_list.photo_id"),
            dict(name="collect_list", path="user_profile_v1.collect_list.photo_id"),
            dict(name="download_list", path="user_profile_v1.download_video_list.photo_id"),
            dict(name="search_click_list", path="user_profile_v1.search_click_photo_list.photo_id"),
            dict(name="fountain_like_list", path="fountain_reco_user_profile.like_list.photo_id"),
            dict(name="fountain_follow_list", path="fountain_reco_user_profile.follow_list.photo_id"),
            dict(name="fountain_forward_list", path="fountain_reco_user_profile.forward_list.photo_id"),
            dict(name="fountain_comment_list", path="fountain_reco_user_profile.comment_list.photo_id"),
          ]
        ) \
        .split_string(
          input_common_attr = "action_weight_str",
          output_common_attr = "action_weight_list",
          delimiters = ",",
          trim_spaces = True,
          parse_to_double = True
        ) \
        .enrich_attr_by_lua(
          import_common_attr = ["like_list", "follow_list", "forward_list", "comment_list", "collect_list", "download_list", "search_click_list", "action_weight_list"],
          export_common_attr = ["trigger_index_list", "trigger_list", "trigger_weight_list"],
          function_for_common = "generate_raw_trigger",
          lua_script_file = "explore/retrieval/lua/module/pdn_action_retr__gen_trigger_weight.lua"
        ) \
        .shuffle_list_attr(
          common_attr = "trigger_index_list"
        ) \
        .enrich_attr_by_lua(
          import_common_attr = ["trigger_index_list", "trigger_list", "trigger_weight_list", "trigger_num"],
          export_common_attr = ["trigger_list", "trigger_weight_list"],
          function_for_common = "generate_final_trigger",
          lua_script_file = "explore/retrieval/lua/module/pdn_action_retr__gen_trigger_weight.lua"
        ) \
      .end_() \
      .if_("trigger_list == nil or #trigger_list == 0") \
        .return_() \
      .end_() \
      .pack_common_attr(
        input_common_attrs = ["trigger_list", "browse_screen__pid_list"],
        output_common_attr = "filter_list",
        deduplicate = True
      ) \
      .retrieve_by_remote_index(
        kess_service = "{{index_service_name}}",
        timeout_ms = "{{index_service_timeout_ms}}",
        reason = 1,
        querys = [
          {
            "query": "{{index_term_name}}:{{trigger_list}}",
            "search_num": "{{search_num_per_trigger}}",
            "expire_second": "{{index_cache_expire_time_s}}",
            "random_search": "{{index_random_search}}"
          }
        ],
        save_score_to_attr = "index_score",
        save_query_index_to_attr = "query_index",
        exclude_items_in_attr = "filter_list",
        save_result_to_common_attr = "result_item_id",
      ) \
      .deduplicate(
        item_list_from_attr = "result_item_id",
      ) \
      .filter_by_common_attr(
        common_attr = ["browse_screen__pid_list"],
        item_list_from_attr = "result_item_id",
        skip = "{{skip_filter_by_browse_screen}}"
      ) \
      .filter_by_browse_set(
        skip = "{{skip_browse_set}}",
        item_list_from_attr = "result_item_id",
      ) \
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
      .limit(size = "{{cand_num}}")
  
  @property
  def retrieve_num(self) -> int:
    assert "retrieve_num" in self.config
    return self.config["retrieve_num"]