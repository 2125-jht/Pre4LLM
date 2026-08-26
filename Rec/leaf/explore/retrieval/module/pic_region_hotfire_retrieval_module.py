from retrieval import RetrievalModule

class PicRegionHotfireRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
    
  def process(self) -> None:
    self.flow \
      .if_("enable_location_ad_code == 1") \
        .gen_common_attr_by_lua(
          attr_map={
            "current_location_city_ad_code": "current_location_ad_code and math.floor(current_location_ad_code / 100)",
            "hometown_city_ad_code": "hometown_ad_code and math.floor(hometown_ad_code / 100)",
            "freq_city_ad_code": "freq_ad_code and math.floor(freq_ad_code / 100)"
          }
        ) \
        .pack_common_attr(
          input_common_attrs = ["current_location_city_ad_code", "hometown_city_ad_code", "freq_city_ad_code"],
          output_common_attr = "user_city_id_list",
          deduplicate = True
        ) \
      .else_() \
        .pack_common_attr(
          input_common_attrs = ["featureCityId", "featureUserRequestCityId"],
          output_common_attr = "user_city_id_list",
          deduplicate = True
        ) \
      .end_() \
      .retrieve_by_redis(
        reason = self.reason,
        retrieve_num = "{{explore_pic_region_hotfire_retr_num}}",
        retrieve_num_per_key = "{{explore_pic_region_hotfire_retr_num_per_key}}",
        cluster_name = "recoPersonalCem",
        timeout_ms = 50,
        key_from_attr = "user_city_id_list",
        key_prefix = "{{explore_pic_region_hotfire_retr_redis_key_prefix}}",
        item_separator = ","
      ) \
      .deduplicate() \
      .if_("enable_filter_browse_set == 1") \
        .filter_by_common_attr(
          common_attr = ["browse_screen__pid_list"]
        ) \
      .end_() \
      .shuffle() \
      .limit(
        size = "{{explore_pic_region_hotfire_retr_num_final}}"
      )
