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
      .if_("enable_shuffle_retr_result ~= nil and enable_shuffle_retr_result == 1") \
        .shuffle() \
      .end_() \
      .if_("enable_inner_sort == 1") \
        .sort(
            score_from_attr = self.sort_attr,
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

  @property
  def sort_attr(self) -> str:
    return self.config.get("sort_attr", "")