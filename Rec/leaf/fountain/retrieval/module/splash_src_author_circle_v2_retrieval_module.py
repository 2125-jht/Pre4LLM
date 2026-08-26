from retrieval import RetrievalModule

class SplashSourceAuthorCircleV2RetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .get_common_attr_from_redis(
        cluster_name = "recoExploreNegPhoto",
        is_async = True,
        redis_params = [
          {
            "redis_key": "{{source_author_circle_v2}}",
            "key_prefix": "{{redis_key_prefix}}",
            "output_attr_name": "fountain_splash_src_author_circle_v2_photo_ids_str"
          }
        ]
      ) \
      .split_string(
        input_common_attr = "fountain_splash_src_author_circle_v2_photo_ids_str",
        output_common_attr = "fountain_splash_src_author_circle_v2_photo_id_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .retrieve_by_common_attr(
        attr = "fountain_splash_src_author_circle_v2_photo_id_list",
        reason = self.reason
      ) \
      .filter_by_common_attr(
        common_attr = [
          "browse_screen__pid_list"
        ]
      ) \
      .deduplicate() \
      .shuffle() \
      .limit(
        size = "{{cand_num}}"
      )