from cascading import CommonModule

class CascadingPrerankFountainEfficiencyModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("explore_fountain_efficiency_score == 1", to_be_delete = "date=2024-05-29;committer=guohao") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "user_age_segment",
            "fountain_efficiency_vv_0",
            "fountain_efficiency_vv_1",
            "fountain_efficiency_vv_2",
            "fountain_efficiency_vv_3",
            "fountain_efficiency_vv_4",
            "fountain_efficiency_vv_5",
            "fountain_efficiency_vv_6",
            "fountain_efficiency_vv_default",
            "fountain_efficiency_vv_hetu",
            {"name": "explore_fountain_efficiency_vv_default_value", "as": "global_default_value"},
          ],
          import_item_attr = [
            "hetu_tag_level_info_v2__hetu_level_one",
          ],
          export_item_attr = [
            {"name": "output_score", "as": "fountain_efficiency_vv"},
          ],
          function_name = "GetFountainEfficiencyScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()
