from retrieval.retrieval_module import RetrievalModule

class RecommendByFriendsModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self):
    self.flow \
      .explore_redis_recommend_by_friend_retriever(
        reason = self.reason,
        cluster_name = self.redis_cluster_name,
        retrieve_num = "{{redis_retrieval_num}}",
        key_prefix = "{{key_prefix}}",
        key_attr = "_USER_ID_",
        record_attr = "is_recommend_by_friend",
        save_friends_to_attr = "recommend_friend_list",
        timeout_ms = self.redis_timeout_ms,
      ) \
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

  @property
  def redis_cluster_name(self) -> str:
    assert "redis_cluster_name" in self.config
    return self.config["redis_cluster_name"]

  @property
  def redis_timeout_ms(self) -> int:
    return self.config.get("redis_timeout_ms", 20)