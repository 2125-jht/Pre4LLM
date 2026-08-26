from retrieval.retrieval_module import RetrievalModule

class MemoryRankRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .retrieve_by_redis(
          reason = self.reason,
          retrieve_num = "{{fountain_memory_rank_retrieval_num}}",
          cluster_name = "recoEyeshotFollow",
          timeout_ms = 30,
          key_from_attr = "_USER_ID_",
          key_prefix = "{{redis_key_prefix}}",
          item_separator = ",",
          attr_separator = "-",
          extra_item_attrs =  [{"name": "memory_rank_score", "type": "double"}]
      ) \
      .deduplicate() \
      .filter_by_browse_set() \
      .if_("enable_cache_rank_prefilter == 1") \
        .split_string(
          input_common_attr = "rank_neg_photo_id_list_str",
          output_common_attr = "nearline_rank_neg_photo_id_filter_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
        .filter_by_common_attr(
          common_attr=["nearline_rank_neg_photo_id_filter_list"],
        ) \
      .end_() \
      .if_("enable_memory_rank_inner_sort == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .sort(score_from_attr = "memory_rank_score") \
      .end_() \
      .limit(
        size = "{{fountain_memory_rank_retrieval_limit_num}}",
      )