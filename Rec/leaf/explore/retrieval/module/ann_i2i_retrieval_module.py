from retrieval.retrieval_module import RetrievalModule

class AnnI2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_("enable_global_trigger ~= nil and enable_global_trigger > 0")
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
            trigger_list_name = "interact_pids",
          ) \
          .copy_attr(
            attrs=[{
              "from_common": "colossus_user_info__trigger_id_list",
              "to_common": "colossus_list"
            }]
          ) \
          .shuffle_list_attr(
            common_attr="colossus_list"
          ) \
          .truncate(
            size_limit = "{{colossus_num}}",
            item_list_from_attr = "colossus_list"
          ) \
          .pack_common_attr(
            input_common_attrs = ["interact_pids", "colossus_list"],
            output_common_attr = "trigger_list",
            deduplicate = True
          ) \
        .else_() \
          .enrich_with_protobuf(
            from_extra_var = "user_info_ptr",
            attrs = [
              dict(name="click_list", path="user_profile_v1.video_playing_stat.photo_id"),
              dict(name="like_list", path="user_profile_v1.like_list.photo_id"),
              dict(name="follow_list", path="user_profile_v1.follow_list.photo_id"),
              dict(name="forward_list", path="user_profile_v1.forward_list.photo_id"),
              dict(name="comment_list", path="user_profile_v1.comment_list.photo_id"),
              dict(name="profile_enter_list", path="user_profile_v1.profile_enter_list.photo_id"),
              dict(name="download_list", path="user_profile_v1.download_video_list.photo_id"),
              dict(name="collect_list", path="user_profile_v1.collect_list.photo_id"),
              dict(name="hate_list", path="user_profile_v1.hate_list.photo_id")
            ]
          ) \
          .truncate(
            size_limit = "{{like_num}}",
            item_list_from_attr = "like_list"
          ) \
          .truncate(
            size_limit = "{{follow_num}}",
            item_list_from_attr = "follow_list"
          ) \
          .truncate(
            size_limit = "{{forward_num}}",
            item_list_from_attr = "forward_list"
          ) \
          .truncate(
            size_limit = "{{download_num}}",
            item_list_from_attr = "download_list"
          ) \
          .truncate(
            size_limit = "{{comment_num}}",
            item_list_from_attr = "comment_list"
          ) \
          .truncate(
            size_limit = "{{profile_enter_num}}",
            item_list_from_attr = "profile_enter_list"
          ) \
          .truncate(
            size_limit = "{{collect_num}}",
            item_list_from_attr = "collect_list"
          ) \
          .pack_common_attr(
            input_common_attrs = [
              "like_list", "follow_list", "forward_list", "comment_list",
              "profile_enter_list", "download_list", "collect_list"
            ],
            output_common_attr = "interact_pids",
            deduplicate = True
          ) \
          .shuffle_list_attr(
            common_attr= "interact_pids"
          ) \
          .truncate(
            size_limit = "{{interact_num}}",
            item_list_from_attr = "interact_pids"
          ) \
          .copy_attr(
            attrs=[{
              "from_common": "click_list",
              "to_common": "realtime_list"
            }]
          ) \
          .truncate(
            size_limit = "{{realtime_num}}",
            item_list_from_attr = "realtime_list"
          ) \
          .shuffle_list_attr(
            common_attr= "click_list"
          ) \
          .truncate(
            size_limit = "{{click_num}}",
            item_list_from_attr = "click_list"
          ) \
          .copy_attr(
            attrs=[{
              "from_common": "colossus_user_info__trigger_id_list",
              "to_common": "colossus_list"
            }]
          ) \
          .shuffle_list_attr(
            common_attr="colossus_list"
          ) \
          .truncate(
            size_limit = "{{colossus_num}}",
            item_list_from_attr = "colossus_list"
          ) \
          .pack_common_attr(
            input_common_attrs = ["realtime_list", "interact_pids", "colossus_list", "click_list"],
            output_common_attr = "trigger_list",
            deduplicate = True
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
        .filter_by_common_attr(
          item_list_from_attr = "trigger_list",
          common_attr = ["hate_list"]
        ) \
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
        embeddings_from_attr = ["trigger_embedding_list"],
        bound_type = {
          "top_k": "{{retrieve_num_per_trigger}}"
        },
        algo_type = {
          "scann": {}
        },
        src_data_type = "{{src_data_type}}",
        src_bucket = "{{src_data_type}}",
        dest_bucket = "{{dest_bucket}}",
        save_source_item_to_attr = "src_id_list",
        save_distance_to_attr = "src_dist_list"
      ) \
      .deduplicate() \
      .filter_by_common_attr(
        common_attr = ["browse_screen__pid_list"]
      ) \
      .filter_by_browse_set(
        skip = "{{skip_browse_set}}"
      ) \
      .if_("enable_ann_result_sort ~= nil and enable_ann_result_sort > 0", to_be_delete = "date=2023-11-16;committer=shaolei") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "trigger_list",
            "trigger_weight_list"
          ],
          import_item_attr = [
            "src_id_list",
            "src_dist_list"
          ],
          export_item_attr = [
            {"name": "src_id", "as": "i2i_retr__trigger_pid"},
            {"name": "final_score", "as": "ann_dist"}
          ],
          function_name = "CalcAnnResultFinalScore",
          class_name = "ExploreLightFunctionSetV2"
        ) \
        .sort(score_from_attr = "ann_dist") \
      .else_() \
        .enrich_attr_by_light_function(
          import_item_attr = [
            {"name": "src_id_list", "as": "extract_hetu_tag_list"}
          ],
          export_item_attr = [
            {"name": "first_hetu_tag", "as": "i2i_retr__trigger_pid"}
          ],
          function_name = "ExtractFirstHetuTag",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_() \
      .limit(size = "{{retrieve_num}}")
  
  def post_process(self):
    self.flow \
      .log_debug_info(
        item_attrs = ["i2i_retr__trigger_pid"]
      )