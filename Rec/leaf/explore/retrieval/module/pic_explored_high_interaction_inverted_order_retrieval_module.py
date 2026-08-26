from retrieval.retrieval_module import RetrievalModule


class PicExploredHighInteractionInvertedOrderRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .enrich_attr_by_light_function(
        import_common_attr=[
          "basic_info_gender_v2",
          "basic_info_age_segment_v2",
          "location_city_level_v2",
          "score_type"
        ],
        export_common_attr=["inverted_order_pool_redis_key"],
        function_name="GenPicInvertedOrderPoolRetrRedisKey",
        class_name="ExploreLightFunctionSetV2"
      ) \
      .if_('inverted_order_pool_redis_key ~= nil and inverted_order_pool_redis_key ~= ""') \
        .retrieve_by_redis(
          reason=self.reason,
          retrieve_num="{{retr_total_limit}}",
          cluster_name="recoEyeshotFollow",
          timeout_ms="{{service_timeout_ms}}",
          key_from_attr="inverted_order_pool_redis_key",
          item_separator=","
        ) \
      .end_if_()
