from retrieval import CommonModule

class SplashFetchRerankTopPhotoModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("fountain_enable_fetch_rerank_top_photo == 1") \
        .gen_common_attr_by_lua(
          attr_map={
            "fountain_rerank_top_photo_redis_key": "fountain_rerank_top_photo_redis_key_prefix .. tostring(featureSourcePId)"
          }
        ) \
        .get_common_attr_from_redis(
          cluster_name = "recoExploreNegPhoto",
          redis_params = [
            {
              "redis_key": "{{fountain_rerank_top_photo_redis_key}}",
              "output_attr_name": "fountain_rerank_top_photo_id_list_str"
            }
          ]
        ) \
        .split_string(
          input_common_attr = "fountain_rerank_top_photo_id_list_str",
          output_common_attr = "fountain_rerank_top_photo_id_retrieval_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
      .end_()