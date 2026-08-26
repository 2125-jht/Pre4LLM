from retrieval.retrieval_module import RetrievalModule

class PdnInteractRetrModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow.if_("enable_global_trigger ~= nil and enable_global_trigger > 0")
    ## 从 global trigger 里完成抽取
    self._sample_global_triggers("trigger_list", "trigger_weight_list_tmp")
    self.flow \
      .else_() \
        .enrich_with_protobuf(
          from_extra_var = "user_info_ptr",
          attrs = [
            dict(name="trigger_list", path="user_profile_v1.video_playing_stat.photo_id", repeat_limit={"user_profile_v1.video_playing_stat": 200})
          ]
        ) \
      .end_() \
      .if_("trigger_list == nil or #trigger_list == 0") \
        .return_() \
      .end_() \
      .if_("enable_global_trigger == nil or enable_global_trigger == 0") \
        .explore_pdn_trigger_weight_enricher(
          user_info_ptr_attr = "user_info_ptr",
          output_trigger_weight_attr = "trigger_weight_list",
          request_service = "{{infer_service_name}}",
          request_layer = "{{infer_layer_name}}",
          enable_extra_user_attr = "{{enable_extra_user_attr}}"
        ) \
      .else_() \
        .copy_attr(
          attrs = [{"from_common": "trigger_weight_list_tmp", "to_common": "trigger_weight_list"}]
        ) \
      .end_() \
      .retrieve_by_remote_index(
        kess_service = "{{index_service_name}}",
        timeout_ms = "{{index_service_timeout_ms}}",
        consistent_hash = False,
        reason = 1,
        querys = [
          {
            "query": "sim:{{trigger_list}}",
            "search_num": "{{search_num_per_trigger}}",
            "expire_second": "{{index_cache_expire_time_s}}",
            "random_search": 0
          }
        ],
        save_score_to_attr = "index_score",
        save_query_index_to_attr = "query_index",
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
        weight_mode = "log_sum",
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
      .set_attr_value(
        common_attrs=[
          {
            "name": "reason",
            "type": "int",
            "value": self.reason
          }
        ],
        skip = "{{skip_personal_quota}}"
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "cand_num", "as": "retr_size"},
          "reason_ratio_map_attr",
          "reason"
        ],
        export_common_attr = [
          {"name": "retr_size", "as": "cand_num"}
        ],
        function_name = "DynamicRetrQuota",
        class_name = "ExploreLightFunctionSetV2",
        skip = "{{skip_personal_quota}}"
      ) \
      .limit(size = "{{cand_num}}")
  
  @property
  def retrieve_num(self) -> int:
    assert "retrieve_num" in self.config
    return self.config["retrieve_num"]
  