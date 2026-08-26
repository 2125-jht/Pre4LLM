from retrieval.retrieval_module import RetrievalModule

class AugmentedTtI2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
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
          dict(name="hate_list", path="user_profile_v1.hate_list.photo_id"),
          dict(name="click_page_type_list", path="user_profile_v1.click_list.page_type")
        ]
      ) \
      .if_("enable_trigger_with_weight > 0") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "click_list", "like_list", "follow_list", "forward_list", "profile_enter_list", "download_list", "collect_list",
            "click_weight", "like_weight", "like_weight", "forward_weight", "profile_enter_weight", "download_weight", "collect_weight",
            "is_confidence_filter", "confidence_weight",
            "trigger_limit", "hate_list", "click_page_type_list"
          ],
          export_common_attr = [
            "trigger_list",
            "trigger_list_weight"
          ],
          function_name = "GenAnnTriggerListByWeight",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .else_() \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "click_list", "like_list", "follow_list", "forward_list", "profile_enter_list", "download_list", "collect_list",
            "click_weight", "like_weight", "like_weight", "forward_weight", "profile_enter_weight", "download_weight", "collect_weight",
            "is_confidence_filter", "confidence_weight",
            "trigger_limit", "hate_list", "click_page_type_list"
          ],
          export_common_attr = [
            "trigger_list",
          ],
          function_name = "GenAnnTriggerList",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("trigger_list == nil or #trigger_list == 0") \
       .return_() \
      .end_() \
      .retrieve_by_ann_embedding(
        reason = self.reason,
        kess_service = "{{service_name}}",
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
        save_distance_to_attr = "ann_dist_list"
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "ann_dist_threshold",
        ],
        import_item_attr = [
          "ann_dist_list",
        ],
        export_item_attr = [
          "ann_dist",
        ],
        function_name = "AnnCalThresholdValueForDistList",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .filter_by_attr(
        attr_name = "ann_dist",
        remove_if = "<",
        compare_to = "{{ann_dist_threshold}}"
      ) \
      .filter_by_common_attr(
        common_attr = ["browse_screen__pid_list"]
      ) \
      .sort(score_from_attr = "ann_dist") \
      .limit(size = "{{retrieve_num}}")

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = [
          "click_list",
          "click_page_type_list"
        ],
        for_debug_request_only = True
      )