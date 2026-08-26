from retrieval.retrieval_module import RetrievalModule

class RedisShardPhotoRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self):
    self.flow \
      .gen_common_attr_by_lua(
        attr_map = {
          "shard_id": "util.GetTimestamp() % shard_num",
        },
      ) \
      .retrieve_by_redis(
        reason = self.reason,
        retrieve_num = "{{redis_retrieval_num}}",
        cluster_name = self.redis_cluster_name,
        timeout_ms = self.redis_timeout_ms,
        key_from_attr = "shard_id", 
        key_prefix = "{{redis_key_prefix}}",
        item_separator = self.redis_item_separator,
      ) \

    self.flow \
      .deduplicate() \
      .shuffle() \
      .limit(
        size = "{{result_num}}",
        skip = "{{return result_num == nil}}"
      ) \
  
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