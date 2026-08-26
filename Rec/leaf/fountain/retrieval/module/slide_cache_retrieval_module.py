from retrieval.retrieval_module import RetrievalModule

class SlideCacheRetrievalModule(RetrievalModule):
  def __init__(self, name=str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
    .str_format(
      format_string="slide_cache_retr_%s",
      input_attrs=["_DEVICE_ID_"],
      output_attr="slide_cache_retr_redis_key",
    ) \
    .get_common_attr_from_redis(
      cluster_name = "recoSSSCelebration",
      redis_params = [
        {
          "redis_key": "{{slide_cache_retr_redis_key}}",
          "output_attr_name": "slide_cache_retr_result_str",
          "output_attr_type": "string"
        }
      ]
    ) \
    .gen_common_attr_by_lua(
      attr_map={
        "cache_hit": "(slide_cache_retr_result_str == nil) and 0 or 1"
      }
    ) \
    .perflog_attr_value(
      check_point="slide_cache_retrieval",
      common_attrs=["cache_hit"],
      aggregator="avg"
    ) \
    .if_("slide_cache_retr_result_str ~= nil") \
      .explore_slide_cache_retriever(
        cache_str_attr_name = "slide_cache_retr_result_str",
        slide_cache_result_expire_s = "{{slide_cache_result_expire_s}}",
        reserve_reason = "{{reserve_reason}}",
        reason = self.reason
      ) \
      .deduplicate() \
      .shuffle() \
      .limit(
        size = "{{retrieve_num}}"
      ) \
    .end_()