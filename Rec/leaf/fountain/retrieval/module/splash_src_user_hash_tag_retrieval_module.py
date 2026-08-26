from retrieval import RetrievalModule

class SplashSourceUserHashTagRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "source_user_hash_tag_id", "as": "int_list"},
          {"name": "redis_key_prefix", "as": "prefix"},
        ],
        export_common_attr = [
          {"name": "final_string_list", "as": "redis_keys"}
        ],
        function_name = "AddPrefixForIntList",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .get_common_attr_from_redis(
        cluster_name = "recoExploreNegPhoto",
        is_async = True,
        redis_params = [
          {
            "redis_key": "{{redis_keys}}",
            "output_attr_type": "string_list",
            "output_attr_name": "fountain_splash_src_user_hash_tag_photo_ids_str_list"
          }
        ]
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_splash_src_user_hash_tag_photo_ids_str_list", "as": "source_list"},
          {"name": "redis_value_delimiter", "as": "delimiter"},
          {"name": "redis_value_max_size", "as": "max_size"}, # 最多取出多少个元素
        ],
        export_common_attr = [
          {"name": "final_list", "as": "fountain_splash_src_user_hash_tag_photo_id_list"}
        ],
        function_name = "SplitStringListToIntList",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .retrieve_by_common_attr(
        attr = "fountain_splash_src_user_hash_tag_photo_id_list",
        reason = self.reason
      ) \
      .filter_by_common_attr(
        common_attr = [
          "browse_screen__pid_list"
        ]
      ) \
      .deduplicate() \
      .shuffle() \
      .limit(
        size = "{{cand_num}}"
      )