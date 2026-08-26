from ranking import CommonModule

class RankingLightCalcScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_fr_light_calc_score == 1") \
        .enrich_attr_by_light_function(
        import_item_attr = [
          "hetu_level_one_top1",
        ],
        export_item_attr = [
          {"name": "hetu_level_one_ratio", "as": "life_fr_hetu_level_one_ratio"},
        ],
        function_name = "CalHetuOneRatio",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .item_attr_operation(
        item_attr_a="life_fr_hetu_level_one_ratio",
        common_attr_b="{{life_fr_hetu_reshape_coef}}",
        operator="+",
        output_attr="life_fr_hetu_one_ratio",
      )\
      .item_attr_operation(
        item_attr_a="colossus_hetu_one_ratio",
        item_attr_b="life_fr_hetu_one_ratio",
        operator="/",
        output_attr="life_fr_hetu_one_reshaping_score",
      )\
      .end_()\
      .log_debug_info(
        common_attrs = [
          "colossus_hetu_one_list",
          # "colossus_tag_list"
        ],
        item_attrs = [
          "hetu_level_one_top1",
          "colossus_hetu_one_ratio",
          # "colossus_hetu_one_ratio3",
          "life_fr_hetu_level_one_ratio",
          "hetu_tag_level_info__hetu_level_one",
          "life_fr_hetu_one_ratio",
          "life_fr_hetu_one_reshaping_score",
          ],
        for_debug_request_only = True,
        respect_sample_logging = True,
      ) \
        