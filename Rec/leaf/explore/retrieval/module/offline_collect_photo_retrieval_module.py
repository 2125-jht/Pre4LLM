from retrieval.retrieval_module import RetrievalModule

class OfflineCollectPhotoRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self):
    self.flow \
      .enrich_with_protobuf(
        from_extra_var = "user_info_ptr",
        attrs = [
          dict(name="collectPids", path="user_profile_v1.collect_list.photo_id")
        ]
      ) \
      .shuffle_list_attr(
        common_attr = "collectPids"
      ) \
      .pack_common_attr(
        input_common_attrs = ["collectPids"],
        output_common_attr = "triggerPids",
        limit_num = "{{trigger_num}}",
        deduplicate = True
      ) \
      .enrich_attr_by_lua(
        import_common_attr = ["triggerPids", "redis_retrieval_num", "trigger_num"],
        export_common_attr = ["redis_retrieval_num_per_key"],
        function_for_common = "calculate",
        lua_script_file = "explore/retrieval/lua/module/offline_collect_photo_retr__key_generator.lua"
      ) \
      .retrieve_by_redis(
        reason = self.reason,
        retrieve_num = "{{redis_retrieval_num}}",
        cluster_name = "recoSubdivisionCache",
        timeout_ms = 30,
        key_from_attr = "triggerPids", 
        key_prefix = "hot_collect_v0_",
        retrieve_num_per_key = "{{redis_retrieval_num_per_key}}",
        item_regex = "(\d+)"
      ) \
      .deduplicate() \
      .shuffle() \
      .limit(
        size = "{{redis_retrieval_num}}"
      )
  
  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = ["triggerPids"]
      )
