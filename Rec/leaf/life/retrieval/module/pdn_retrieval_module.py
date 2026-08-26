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
        .if_("enable_scatter_play_list > 0") \
          .enrich_with_protobuf(
            from_extra_var = "user_info_ptr",
            attrs = [
              dict(name="raw_playlist", path="user_profile_v1.video_playing_stat.photo_id"),
              dict(name="play_hetu_list", path="user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_two", repeat_limit={"user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_two": 1}, repeat_align=True)
            ],
          ) \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "raw_playlist", "as": "trigger_list"},
              {"name": "play_hetu_list", "as": "trigger_scatter_attr_list"},
              {"name": "play_list_trigger_max_num", "as": "total_limit"},
              {"name": "play_list_scatter_each_limit", "as": "each_limit"}
            ],
            export_common_attr = [
              {"name": "final_trigger_list", "as": "trigger_list"}
            ],
            function_name = "ScatterTriggers",
            class_name = "ExploreLightFunctionSetV2"
          ) \
        .else_() \
          .enrich_with_protobuf(
            from_extra_var = "user_info_ptr",
            attrs = [
              dict(name="trigger_list", path="user_profile_v1.video_playing_stat.photo_id", repeat_limit={"user_profile_v1.video_playing_stat": 40})
            ]
          ) \
        .end_() \
        .if_("enable_scatter_trigger > 0") \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "colossus_user_info__trigger_id_list", "as": "trigger_list"},
              {"name": "colossus_user_info__trigger_tag_list", "as": "trigger_scatter_attr_list"},
              {"name": "hetu_low_level_tags", "as": "scatter_map_keys"},
              {"name": "hetu_level2_tags", "as": "scatter_map_values"},
              {"name": "colossus_trigger_max_num", "as": "total_limit"},
              {"name": "trigger_scatter_each_limit", "as": "each_limit"}
            ],
            export_common_attr = [
              {"name": "final_trigger_list", "as": "extra_trigger_list"}
            ],
            function_name = "ScatterTriggers",
            class_name = "ExploreLightFunctionSetV2"
          ) \
        .else_() \
          .copy_attr(
            attrs=[{
              "from_common": "colossus_user_info__trigger_id_list",
              "to_common": "colossus_trigger_id_list"
            }]
          ) \
          .shuffle_list_attr(
            common_attr = "colossus_trigger_id_list"
          ) \
          .pack_common_attr(
            input_common_attrs = ["colossus_trigger_id_list"],
            output_common_attr = "extra_trigger_list",
            limit_num = "{{colossus_trigger_max_num}}"
          ) \
        .end_() \
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
            lua_script_file = "life/retrieval/lua/module/colossus_ann__add_knowledge_trigger.lua"
          ) \
        .end_() \
        .if_("interest_explore_trigger_expand_cnt ~= nil and interest_explore_trigger_expand_cnt > 0") \
          .pack_common_attr(
            input_common_attrs = ["trigger_list", "colossus_user_info__interest_explore_trigger_set"],
            output_common_attr = "trigger_list",
            limit_num = "{{return #(trigger_list or {}) + interest_explore_trigger_expand_cnt}}",
            deduplicate = True
          ) \
        .end_() \
        .if_("mc_enable_opt_card_trigger == 1") \
          .if_("optcard_like_trigger_id_list ~= nil") \
            .pack_common_attr(
              # TODO(liucong03): 2月份删除 mc_enable_opt_card_trigger 部分逻辑
              input_common_attrs = ["trigger_list", "optcard_like_trigger_id_list"],
              output_common_attr = "trigger_list",
              deduplicate = True
            ) \
          .end_() \
          .if_("optcard_dislike_trigger_id_list ~= nil") \
            .filter_by_common_attr(
              item_list_from_attr = "trigger_list",
              common_attr = ["optcard_dislike_trigger_id_list"]
            ) \
          .end_() \
        .end_() \
      .end_() \
      .if_("trigger_list == nil or #trigger_list == 0") \
        .return_() \
      .end_() \
      .retrieve_by_remote_index(
        kess_service = "{{index_service_name}}",
        timeout_ms = "{{index_service_timeout_ms}}",
        reason = 1,
        querys = [
          {
            "query": "sim:{{trigger_list}}",
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