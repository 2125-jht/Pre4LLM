from retrieval.retrieval_module import RetrievalModule

class HetuMemoryRankRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self):
    if self.redis_key_from_attr:
      self.flow \
        .retrieve_by_redis(
          reason = self.reason,
          retrieve_num = "{{redis_retrieval_num}}",
          cluster_name = self.redis_cluster_name,
          timeout_ms = self.redis_timeout_ms,
          key_from_attr = self.redis_key_from_attr, 
          key_prefix = "{{redis_key_prefix}}",
          item_separator = self.redis_item_separator,
          attr_separator = self.redis_attr_separator,
          extra_item_attrs = self.item_attrs
        )

    elif self.key:
      self.flow \
        .retrieve_by_redis(
          reason = self.reason,
          retrieve_num = "{{redis_retrieval_num}}",
          cluster_name = self.redis_cluster_name,
          timeout_ms = self.redis_timeout_ms,
          key = self.key,
          item_separator = self.redis_item_separator,
          attr_separator = self.redis_attr_separator,
          extra_item_attrs = self.item_attrs
        )

    else:
      self.flow \
        .retrieve_by_redis(
          reason = self.reason,
          retrieve_num = "{{redis_retrieval_num}}",
          cluster_name = self.redis_cluster_name,
          timeout_ms = self.redis_timeout_ms,
          key_from_attr = "_USER_ID_", 
          key_prefix = "{{redis_key_prefix}}",
          item_separator = self.redis_item_separator,
          attr_separator = self.redis_attr_separator,
          extra_item_attrs = self.item_attrs
        )

    self.flow \
      .count_reco_result(
        save_count_to="hetu_memory_item_num"
      ) \
      .if_("enable_reserve_retr_result == 1 and hetu_memory_item_num <= 0") \
        .set_attr_value(
          no_overwrite=True,
          common_attrs=[
              {
              "name": "hetu_level2_id_list",
              "type": "int_list",
              "value": [
                  110,111,113,115,118,119,126,128,129,130,136,147,153,154,155,160,
                  161,162,163,167,168,169,170,179,195,201,203,204,213,214,220,223,224,232,
                  233,235,263,264,265,266,267,268,269,270,271,272,273,274,292,314,317,318,
                  319,323,324,325,327,330,341,347,348,352,354,366,368,372,373,374,386,387,
                  389,399,402,418,430,431,432,545,548,561,563,564,565,581,593,628,630,631,
                  632,661,663,665,666,667,668,670,671,672,673,674,675,677,678,679,680,681,
                  682,683,684,685,686,687,688,689,690,691,692,693,694,695,696,697,698,699,
                  700,701,702,703,704,705,706,707,708,710,711,712,713,714,715,716,717,718,
                  719,720,721,722,723,724,725,726,727,728,729,730,731,732,733,734,735,736,
                  737,738,739,740,741,742,743,744,745,746,747,748,749,750
              ]
              }
          ]
        ) \
        .retrieve_by_redis(
          reason = self.reason,
          retrieve_num = "{{reserve_retrieve_num}}",
          retrieve_num_per_key = "{{reserve_retrieve_num_per_key}}",
          cluster_name = "recoHotLauRank",
          timeout_ms = 30,
          key_from_attr = "hetu_level2_id_list",
          key_prefix = "la_hetu_retrival_",
          item_separator = ",",
        ) \
      .end_() \
      .deduplicate() \
      .if_("enable_shuffle_retr_result == 1 or hetu_memory_item_num <= 0") \
        .shuffle() \
      .end_() \
      .if_("enable_browset_filter == 1 or hetu_memory_item_num <= 0") \
        .filter_by_browse_set() \
      .end_() \
      .if_("hetu_memory_item_num > 0") \
        .if_("result_num ~= nil") \
          .limit(
            size = "{{result_num}}"
          ) \
        .end_() \
      .else_() \
        .if_("reserve_result_num ~= nil") \
          .limit(
            size="{{reserve_result_num}}"
          ) \
        .end_() \
      .end_() \
      .set_attr_value(
        item_attrs = [{
          "name": "is_hetu_memory_rank_retrieval",
          "type": "int",
          "value": 1
        }]
      )
        
   
  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        item_attrs = self.unused_item_attrs,
        for_debug_request_only = True
      )

  @property
  def redis_cluster_name(self) -> str:
    assert "redis_cluster_name" in self.config
    return self.config["redis_cluster_name"]
  
  @property
  def redis_timeout_ms(self) -> int:
    return self.config.get("redis_timeout_ms", 10)

  @property
  def redis_item_separator(self) -> str:
    return self.config.get("redis_item_separator", ",")

  @property
  def redis_attr_separator(self) -> str:
    return self.config.get("redis_attr_separator", "-")
  
  @property
  def key(self) -> str:
    return self.config.get("key")

  @property
  def item_attrs(self) -> str:
    return self.config.get("item_attrs", [])

  @property
  def unused_item_attrs(self) -> str:
    return self.config.get("unused_item_attrs", [])

  @property
  def redis_key_from_attr(self) -> str:
    return self.config.get("redis_key_from_attr")