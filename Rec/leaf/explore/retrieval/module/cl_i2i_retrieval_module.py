from retrieval.retrieval_module import RetrievalModule

class ClI2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow.if_("enable_global_trigger ~= nil and enable_global_trigger > 0")
    ## 从 global trigger 里完成抽取
    self._sample_global_triggers("trigger_list", "trigger_weight_list")
    self.flow \
      .else_() \
        .if_("enable_cluster_trigger > 0", to_be_delete = "date=2024-05-29;committer=liyunhao") \
          .explore_select_action_list_trigger_enricher(
            user_info_ptr_attr = "user_info_ptr",
            max_trigger_len = "{{cluster_trigger_max_trigger_len}}",
            max_candidate_len = "{{cluster_trigger_max_candidate_len}}",
            enable_click_list = "{{enable_click_list}}",
            trigger_list_name = "trigger_list",
          ) \
        .else_() \
          .enrich_with_protobuf(
            from_extra_var = "user_info_ptr",
            attrs = [
              dict(name="click_list", path="user_profile_v1.click_list.photo_id"),
              dict(name="like_list", path="user_profile_v1.like_list.photo_id"),
              dict(name="follow_list", path="user_profile_v1.follow_list.photo_id"),
              dict(name="forward_list", path="user_profile_v1.forward_list.photo_id"),
              dict(name="profile_enter_list", path="user_profile_v1.profile_enter_list.photo_id"),
              dict(name="download_list", path="user_profile_v1.download_video_list.photo_id"),
              dict(name="collect_list", path="user_profile_v1.collect_list.photo_id"),
              dict(name="hate_list", path="user_profile_v1.hate_list.photo_id")
            ]
          ) \
          .enrich_attr_by_lua(
            import_common_attr = [
              "click_list", "like_list", "follow_list", "forward_list", "profile_enter_list", "download_list", "collect_list",
              "click_limit", "like_limit", "follow_limit", "forward_limit", "profile_enter_limit", "download_limit", "collect_limit",
              "trigger_limit", "hate_list"
            ], 
            export_common_attr = ["trigger_list"],
            function_for_common = "gen_trigger",
            lua_script_file = "explore/retrieval/lua/module/cl_i2i_retr__trigger_generator.lua"
          ) \
        .end_() \
        .if_("knowledge_trigger_expand_cnt ~= nil and knowledge_trigger_expand_cnt > 0", to_be_delete = "date=2024-05-29;committer=liyunhao") \
          .pack_common_attr(
            input_common_attrs = ["trigger_list", "colossus_user_info__knowledge_trigger_set"],
            output_common_attr = "trigger_list",
            limit_num = "{{return #(trigger_list or {}) + knowledge_trigger_expand_cnt}}",
            deduplicate = True
          ) \
        .end_() \
      .end_() \
      .if_("trigger_list == nil or #trigger_list == 0") \
       .return_() \
      .end_() \
      .retrieve_by_ann_embedding(
        reason = self.reason,
        kess_service = "{{service_name}}",
        space = "cosine",
        timeout_ms = "{{service_timeout_ms}}",
        items_from_attr = ["trigger_list"],
        bound_type = {
          "top_k": "{{retrieve_num_per_trigger}}"
        },
        algo_type = {
          "faiss": {}
        },
        src_data_type = "{{src_data_type}}",
        src_bucket = "{{src_data_type}}",
        dest_bucket = "{{dest_bucket}}",
        save_distance_to_attr = "ann_dist_list",
        save_source_item_to_attr = "ann_src_list"
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "ann_dist_threshold", 
          "trigger_list",
          "trigger_weight_list"
        ],
        import_item_attr = [
          {"name": "ann_src_list", "as": "src_id_list"},
          {"name": "ann_dist_list", "as": "src_dist_list"}
        ],
        export_item_attr = [
          {"name": "src_id", "as": "i2i_retr__trigger_pid"},
          {"name": "final_score", "as": "ann_dist"}
        ],
        function_name = "CalcAnnResultFinalScore",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .deduplicate(
        skip = "{{skip_deduplicate}}"
      ) \
      .filter_by_attr(
        attr_name = "ann_dist",
        remove_if = "<",
        compare_to = "{{ann_dist_threshold}}"
      ) \
      .filter_by_common_attr(
        common_attr = ["browse_screen__pid_list"]
      ) \
      .filter_by_browse_set(
        skip = "{{skip_browse_set}}"
      ) \
      .sort(score_from_attr = "ann_dist") \
      .limit(size = "{{retrieve_num}}")
