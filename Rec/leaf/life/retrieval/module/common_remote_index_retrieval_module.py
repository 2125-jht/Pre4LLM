from retrieval.retrieval_module import RetrievalModule

class CommonRemoteIndexRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self):
    if self.key:
      self.flow \
        .retrieve_by_redis(
          reason = 0,
          retrieve_num = "{{redis_retrieval_num}}",
          cluster_name = self.redis_cluster_name,
          timeout_ms = self.redis_timeout_ms,
          key = self.key, 
          item_separator = self.redis_item_separator,
          save_result_to_common_attr = self.save_to_common_attr_name
        )
    else:
      self.flow \
        .retrieve_by_redis(
          reason = 0,
          retrieve_num = "{{redis_retrieval_num}}",
          cluster_name = self.redis_cluster_name,
          timeout_ms = self.redis_timeout_ms,
          key_from_attr = "_USER_ID_", 
          key_prefix = "{{redis_key_prefix}}",
          item_separator = self.redis_item_separator,
          save_result_to_common_attr = self.save_to_common_attr_name
        )

    self.flow \
      .if_("enable_shuffle_retr_result ~= nil and enable_shuffle_retr_result == 1") \
        .deduplicate(item_list_from_attr = self.save_to_common_attr_name) \
        .shuffle_list_attr(common_attr=self.save_to_common_attr_name) \
        .limit(
          size = "{{max_trigger_num}}",
          item_list_from_attr = self.save_to_common_attr_name
        ) \
      .end_() \
      .retrieve_by_remote_index(
        kess_service = "{{remote_index_service_name}}",
        timeout_ms = "{{remote_index_service_timeout_ms}}",
        reason = self.reason, 
        querys = [
          {
            "query": "{{remote_index_query_term}}" + ":{{" + self.remote_index_query_trigger_attr_name + "}}",
            "random_search": 0,
            "search_num": self.search_num
          }
        ],
        save_score_to_attr = "index_score"
      ) \
      .deduplicate() \
      .filter_by_common_attr(
        common_attr = ["browse_screen__pid_list"],
        skip = "{{skip_filter_by_browse_screen}}"
      ) \
      .filter_by_browse_set(
        skip = "{{skip_browse_set}}"
      ) \
      .sort(
        score_from_attr = "index_score"
      ) \
      .limit(
        size = "{{result_num}}"
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
  def save_to_common_attr_name(self) -> str:
    assert "save_to_common_attr_name" in self.config
    return self.config["save_to_common_attr_name"]
  
  @property
  def remote_index_query_trigger_attr_name(self) -> str:
    assert "remote_index_query_trigger_attr_name" in self.config
    return self.config["remote_index_query_trigger_attr_name"]
  
  @property
  def search_num(self) -> int:
    return self.config.get("search_num", 11)
  
  @property
  def key(self) -> str:
    return self.config.get("key")