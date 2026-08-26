from retrieval import RetrievalModule

class PicHotContentRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
    
  def process(self) -> None:
    self.flow \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_pic_hot_content_redis_key_prefix", "as": "key_prefix"},
        "basic_info_age_segment_v2",
        "basic_info_gender_v2",
        "location_city_level_v2"
      ],
      export_common_attr = [
        "hot_pic_user_city_age_gender_key",
      ],
      function_name = "GetPicAgeGenderRedisKey",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .if_("hot_pic_user_city_age_gender_key == nil") \
      .return_() \
    .end_() \
    .retrieve_by_redis(
      reason = self.reason,
      cluster_name = "recoAnalysis",
      timeout_ms = 20,
      retrieve_num = "{{explore_pic_hot_content_retr_num}}",
      key_from_attr = "hot_pic_user_city_age_gender_key",
      item_separator = ",",
    ) \
    .deduplicate() \
    .shuffle() \
    .limit(size = "{{explore_pic_hot_content_retr_num_final}}") \
    .set_attr_value(
      item_attrs = [
        {
          "name": "is_pic_hot_content",
          "type": "int",
          "value": 1
        }
      ]
    )