from retrieval import CommonModule

class FetchRedisNegPhotoModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .if_("fountain_enable_fetch_rank_neg_photo == 1") \
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
      .end_() \
      .if_("fountain_enable_fetch_rerank_neg_photo == 1", to_be_delete = "date=2024-05-29;committer=liuhao07") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_rerank_neg_photo_key_prefix", "as": "string_a"},
            {"name": "_DEVICE_ID_", "as": "string_b"}
          ],
          export_common_attr = [
            {"name": "final_string", "as": "fountain_rerank_neg_photo_redis_key"}
          ],
          function_name = "ConcatString",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .get_common_attr_from_redis( #上一刷精排结果过滤
          cluster_name = "recoExploreNegPhoto",
          redis_params = [
            {
              "redis_key": "{{fountain_rerank_neg_photo_redis_key}}",
              "output_attr_name": "rerank_neg_photo_id_list_str"
            }
          ]
        ) \
      .end_() \
      .if_("fountain_enable_fetch_mc_neg_photo == 1", to_be_delete = "date=2024-05-29;committer=shaohua") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_mc_neg_photo_key_prefix", "as": "string_a"},
            {"name": "_DEVICE_ID_", "as": "string_b"}
          ],
          export_common_attr = [
            {"name": "final_string", "as": "fountain_mc_neg_photo_redis_key"}
          ],
          function_name = "ConcatString",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .get_common_attr_from_redis( #上一刷精排结果过滤
          cluster_name = "recoExploreNegPhoto",
          redis_params = [
            {
              "redis_key": "{{fountain_mc_neg_photo_redis_key}}",
              "output_attr_name": "mc_neg_photo_id_list_str"
            }
          ]
        ) \
      .end_() \
      .if_("fountain_enable_fetch_explore_rank_pos_photo == 1", to_be_delete = "date=2024-05-29;committer=liuhao07") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rank_pos_photo_key_prefix", "as": "string_a"},
            {"name": "_DEVICE_ID_", "as": "string_b"}
          ],
          export_common_attr = [
            {"name": "final_string", "as": "explore_rank_pos_photo_redis_key"}
          ],
          function_name = "ConcatString",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .get_common_attr_from_redis(
          cluster_name = "recoExploreNegPhoto",
          redis_params = [
            {
              "redis_key": "{{explore_rank_pos_photo_redis_key}}",
              "output_attr_name": "explore_rank_pos_photo_id_list_str"
            }
          ]
        ) \
        .split_string(
          input_common_attr = "explore_rank_pos_photo_id_list_str",
          output_common_attr = "explore_rank_pos_photo_id_retrieval_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
      .end_() \
      .if_("fountain_enable_fetch_fountain_rank_pos_photo == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_rank_pos_photo_key_prefix", "as": "string_a"},
            {"name": "_DEVICE_ID_", "as": "string_b"}
          ],
          export_common_attr = [
            {"name": "final_string", "as": "fountain_rank_pos_photo_redis_key"}
          ],
          function_name = "ConcatString",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .get_common_attr_from_redis(
          cluster_name = "recoExploreNegPhoto",
          redis_params = [
            {
              "redis_key": "{{fountain_rank_pos_photo_redis_key}}",
              "output_attr_name": "fountain_rank_pos_photo_id_list_str"
            }
          ]
        ) \
        .split_string(
          input_common_attr = "fountain_rank_pos_photo_id_list_str",
          output_common_attr = "fountain_rank_pos_photo_id_retrieval_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
      .end_() \
      .if_("fountain_enable_fetch_fountain_rerank_pos_photo == 1") \
        .gen_common_attr_by_lua(
          attr_map={
            "fountain_rerank_pos_photo_redis_key": "fountain_rerank_pos_photo_key_prefix .. tostring(_USER_ID_)"
          }
        ) \
        .get_common_attr_from_redis(
          cluster_name = "recoExploreNegPhoto",
          redis_params = [
            {
              "redis_key": "{{fountain_rerank_pos_photo_redis_key}}",
              "output_attr_name": "fountain_rerank_pos_photo_id_list_str"
            }
          ]
        ) \
        .split_string(
          input_common_attr = "fountain_rerank_pos_photo_id_list_str",
          output_common_attr = "fountain_rerank_pos_photo_id_retrieval_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
        .filter_by_browse_set(
          item_list_from_attr = "fountain_rerank_pos_photo_id_retrieval_list"
        ) \
      .end_()
        
