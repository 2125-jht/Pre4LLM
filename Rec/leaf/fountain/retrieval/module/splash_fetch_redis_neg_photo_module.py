from retrieval import CommonModule

class SplashFetchRedisNegPhotoModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .if_("fountain_enable_fetch_rank_neg_photo == 1", to_be_delete = "date=2024-05-29;committer=liuhao07") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_rank_neg_photo_key_prefix", "as": "string_a"},
            {"name": "_DEVICE_ID_", "as": "string_b"}
          ],
          export_common_attr = [
            {"name": "final_string", "as": "fountain_rank_neg_photo_redis_key"}
          ],
          function_name = "ConcatString",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .get_common_attr_from_redis( #上一刷精排结果过滤
          cluster_name = "recoExploreNegPhoto",
          redis_params = [
            {
              "redis_key": "{{fountain_rank_neg_photo_redis_key}}",
              "output_attr_name": "rank_neg_photo_id_list_str"
            }
          ]
        ) \
      .end_()
        