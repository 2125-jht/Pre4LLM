from retrieval.retrieval_module import RetrievalModule

class LbsRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self):
    self.flow \
      .gen_common_attr_by_lua(
        attr_map = {
          "current_location_city_ad_code": "math.floor(current_location_ad_code / 100)",
          "hometown_city_ad_code": "math.floor(hometown_ad_code / 100)",
          "freq_city_ad_code":  "math.floor(freq_ad_code / 100)"
        }
      ) \
      .pack_common_attr(
        input_common_attrs = ["current_location_city_ad_code", "hometown_city_ad_code", "freq_city_ad_code"],
        output_common_attr = "user_city_ad_code_list"
      ) \
      .retrieve_by_redis(
        retrieve_num = 3000,
        cluster_name = "recoExploreLlmHetu",
        key_prefix = "city_adcode_",
        key_from_attr = "user_city_ad_code_list",
        item_separator = ",",
        retrieve_num_per_key = "{{city_search_num}}",
        reason = self.reason
      ) \
      .deduplicate() \
      .shuffle() \
      .filter_by_common_attr(
        common_attr = ["browse_screen__pid_list"]
      ) \
      .limit(
        size = "{{request_num}}"
      ) \
      .set_attr_value(
        item_attrs = [
          {
            "name": "lbs_item",
            "type": "int",
            "value": 1
          }
        ]
      )
