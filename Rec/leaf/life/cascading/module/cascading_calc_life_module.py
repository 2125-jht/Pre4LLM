from cascading import CommonModule

class CascadingCalcLifeModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow\
    .if_("enable_cascading_calc_life == 1") \
      .if_("enable_common_index == 1 or enable_young_photo_get_from_common_index == 1") \
        .get_item_attr_by_remote_index(  # 后续用于生活页添加特定字段的收口
          kess_service="{{explore_common_index_service}}",
          max_value_bytes=1048576,
          timeout_ms=100,
          partition_size="{{explore_common_index_part_size}}",
          attrs=[
            "is_young_photo",
            "da_young_18_30_vv_rate",
            "da_1_2_city_vv_rate",
            "young_photo_18_23_prob",
            "young_photo_24_30_prob"
          ],
        ) \
      .end_() \
    .end_()\
    .if_("enable_cascading_calc_hetu_reshape == 1") \
      .enrich_attr_by_light_function(
        import_item_attr = [
          {"name": "hetu_tag_level_info__hetu_level_one", "as": "extract_hetu_tag_list"},
        ],
        export_item_attr = [
          {"name": "first_hetu_tag", "as": "hetu_level_one_top1"},
        ],
        function_name = "ExtractFirstHetuTag",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .enrich_attr_by_light_function(
        import_item_attr = [
          "hetu_level_one_top1",
        ],
        export_item_attr = [
          {"name": "hetu_level_one_ratio", "as": "life_mc_s1_hetu_level_one_ratio"},
        ],
        function_name = "CalHetuOneRatio",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .enrich_attr_by_light_function(
        import_item_attr = [
          "hetu_level_one_top1",
        ],
        import_common_attr = [
          "colossus_hetu_one_list",
        ],
        export_item_attr = [
          "colossus_hetu_one_ratio",
        ],
        function_name = "CalcHistoryHetuRatio",
        class_name = "ExploreLifeLightFunctionSet",
      ) \
      .item_attr_operation(
        item_attr_a="life_mc_s1_hetu_level_one_ratio",
        common_attr_b="{{life_s1_hetu_reshape_coef}}",
        operator="+",
        output_attr="life_s1_hetu_one_ratio",
      )\
      .item_attr_operation(
        item_attr_a="colossus_hetu_one_ratio",
        item_attr_b="life_s1_hetu_one_ratio",
        operator="/",
        output_attr="life_s1_hetu_one_reshaping_score",
      )\
      .log_debug_info(
        common_attrs = [
          "colossus_hetu_one_list",
          # "colossus_tag_list"
        ],
        item_attrs = [
          "hetu_level_one_top1",
          "colossus_hetu_one_ratio",
          # "colossus_hetu_one_ratio3",
          "life_mc_s1_hetu_level_one_ratio",
          "hetu_tag_level_info__hetu_level_one",
          "life_s1_hetu_one_ratio",
          "life_s1_hetu_one_reshaping_score",
          ],
        for_debug_request_only = True,
        respect_sample_logging = True,
      ) \
    .end_()\
      