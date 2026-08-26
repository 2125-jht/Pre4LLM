from retrieval.retrieval_module import RetrievalModule

class RedisPhotoRetrievalModule(RetrievalModule):
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
      .deduplicate() \
      .if_("enable_browset_filter == 1") \
        .filter_by_browse_set() \
      .end_() \
      .if_("enable_rank_neg_filter == 1") \
        .split_string(
          input_common_attr = "rank_neg_photo_id_list_str",
          output_common_attr = "redis_rank_neg_photo_id_filter_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
        .filter_by_common_attr(
          common_attr=["redis_rank_neg_photo_id_filter_list"],
        ) \
      .end_() \
      .limit(
        size = "{{result_num}}",
        skip = "{{return result_num == nil}}"
      ) \
   
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