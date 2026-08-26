from retrieval import RetrievalModule

class ColossusAnnRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  @classmethod
  def is_retrieval(cls) -> bool:
    return True
    
  def process(self) -> None:
    self.flow.if_("enable_global_trigger ~= nil and enable_global_trigger > 0")
    ## 从 global trigger 里完成抽取
    self._sample_global_triggers("trigger_list", "trigger_weight_list")
    self.flow \
      .else_() \
        .if_("enable_scatter_trigger > 0", to_be_delete = "date=2023-11-16;committer=liyunhao") \
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
              {"name": "final_trigger_list", "as": "trigger_list"}
            ],
            function_name = "ScatterTriggers",
            class_name = "ExploreLightFunctionSetV2"
          ) \
        .else_() \
          .pack_common_attr(
            input_common_attrs = ["colossus_user_info__trigger_id_list"],
            output_common_attr = "raw_triggers"
          ) \
          .shuffle_list_attr(
            common_attr = "raw_triggers"
          ) \
          .pack_common_attr(
            input_common_attrs = ["raw_triggers"],
            output_common_attr = "trigger_list",
            limit_num = "{{colossus_trigger_max_num}}"
          ) \
        .end_() \
        .if_("enable_knowledge_trigger > 0 and colossus_user_info__knowledge_trigger_set ~= nil and #colossus_user_info__knowledge_trigger_set > 0", to_be_delete = "date=2024-05-29;committer=liyunhao") \
          .enrich_attr_by_lua(
            import_common_attr = ["trigger_list", "colossus_user_info__knowledge_trigger_set", "knowledge_trigger_max_num"],
            export_common_attr = ["trigger_list"],
            function_for_common = "add_knowledge_trigger",
            lua_script_file = "explore/retrieval/lua/module/colossus_ann__add_knowledge_trigger.lua"
          ) \
        .end_() \
        .if_("enable_interaction_trigger > 0 and profile_v1_interaction_trigger_list ~= nil and #profile_v1_interaction_trigger_list > 0", to_be_delete = "date=2024-05-29;committer=liyunhao") \
          .copy_attr(
            attrs=[{
              "from_common": "profile_v1_interaction_trigger_list",
              "to_common": "interaction_trigger_list",
            }]
          ) \
          .limit(
            item_list_from_attr = "interaction_trigger_list",
            size = "{{interaction_trigger_max_num}}"
          ) \
        .end_() \
        .if_("enable_scatter_play_list > 0", to_be_delete = "date=2023-11-16;committer=liyunhao") \
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
              {"name": "final_trigger_list", "as": "playlist"}
            ],
            function_name = "ScatterTriggers",
            class_name = "ExploreLightFunctionSetV2"
          ) \
        .else_() \
          .if_("enable_shuffle_play_list > 0", to_be_delete = "date=2023-11-16;committer=liyunhao") \
            .enrich_with_protobuf(
              from_extra_var = "user_info_ptr",
              attrs = [
                dict(name="playlist", path="user_profile_v1.video_playing_stat.photo_id"),
              ],
            ) \
            .shuffle_list_attr(
              common_attr = "playlist"
            ) \
            .limit(
              item_list_from_attr = "playlist",
              size = "{{play_list_trigger_max_num}}"
            ) \
          .else_() \
            .enrich_with_protobuf(
              from_extra_var = "user_info_ptr",
              attrs = [
                dict(name="playlist", path="user_profile_v1.video_playing_stat.photo_id", repeat_limit={"user_profile_v1.video_playing_stat": 80})
              ]
            ) \
          .end_() \
        .end_() \
        .enrich_attr_by_lua(
          import_common_attr = ["trigger_list", "colossus_user_info__trigger_id_list", "colossus_user_info__trigger_weight_list", "playlist",
                                "interaction_trigger_list", "enable_interaction_trigger"],
          export_common_attr = ["trigger_list", "trigger_weight_list"],
          function_for_common = "calculate",
          lua_script_file = "explore/retrieval/lua/module/colossus_ann__gen_trigger_weight.lua"
        ) \
      .end_() \
      .if_("trigger_list == nil or #trigger_list <= 0") \
        .return_() \
      .end_() \
      .get_remote_embedding_lite(
        kess_service = "{{embedding_service_name}}",
        shard_num = 8,
        timeout_ms = 20,
        id_converter = {
          "type_name": "kuibaEmbeddingIdConverter"
        },
        size = 128,
        input_attr_name = "trigger_list",
        output_attr_name = "colossus_trigger_embedding",
        query_source_type = "common_attr",
        client_side_shard = True
      ) \
      .enrich_attr_by_lua(
        import_common_attr = ["trigger_list", "trigger_weight_list", "colossus_trigger_embedding"],
        export_common_attr = ["trigger_list", "trigger_weight_list", "colossus_trigger_embedding"],
        function_for_common = "calculate",
        lua_script_file = "explore/retrieval/lua/module/colossus_ann__process_embedding.lua"
      ) \
      .if_("enable_tnu_new_bucket == 1 and uIsExploreTnuCrowdUser == 1") \
        .retrieve_by_ann_embedding(
          kess_service = "{{ann_service_name}}",
          space = "cosine",
          timeout_ms = 50,
          reason = self.reason,
          shard_num = 1,
          items_from_attr = ["trigger_list"],
          embeddings_from_attr = ["colossus_trigger_embedding"],
          bound_type = {
            "top_k": "{{ann_topk}}",
          },
          algo_type = {
            "scann": {},
          },
          src_bucket = "photo",
          dest_bucket = "photo_tnu",
          save_source_item_to_attr = "src_id_list",
          save_distance_to_attr = "src_dist_list"
        ) \
      .else_() \
        .retrieve_by_ann_embedding(
          kess_service = "{{ann_service_name}}",
          space = "cosine",
          timeout_ms = 50,
          reason = self.reason,
          shard_num = 1,
          items_from_attr = ["trigger_list"],
          embeddings_from_attr = ["colossus_trigger_embedding"],
          bound_type = {
            "top_k": "{{ann_topk}}",
          },
          algo_type = {
            "scann": {},
          },
          src_bucket = "photo",
          dest_bucket = "photo",
          save_source_item_to_attr = "src_id_list",
          save_distance_to_attr = "src_dist_list"
        ) \
      .end_() \
      .deduplicate(
      ) \
      .filter_by_common_attr(
        common_attr = ["browse_screen__pid_list"]
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "ann_dist_threshold",
          "trigger_list",
          "trigger_weight_list"
        ],
        import_item_attr = [
          "src_id_list",
          "src_dist_list"
        ],
        export_item_attr = [
          {"name": "src_id", "as": "i2i_retr__trigger_pid"},
          {"name": "final_score", "as": "final_score"}
        ],
        function_name = "CalcAnnResultFinalScore",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .filter_by_browse_set(
        skip = "{{skip_browse_set}}"
      ) \
      .sort(
        score_from_attr = "final_score"
      ) \
      .limit("{{retrieve_num}}")

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = [
          "trigger_list",
          "colossus_trigger_embedding",
          "embedding_service_name"
        ]
      )
