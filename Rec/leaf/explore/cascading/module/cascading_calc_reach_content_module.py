from cascading import CommonModule

class CascadingCalcReachContentModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_reach_content_channel == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "uStandardExploreRealshowLabelList", "as": "uStandardExploreRealshowLabelList"},
            {"name": "uStandardExploreRealshowTimestampList", "as": "uStandardExploreRealshowTimestampList"},
            {"name": "explore_reach_content_label_count_threshold", "as": "label_count_threshold"},
            {"name": "explore_reach_content_time_window_minutes", "as": "time_window_minutes"},
          ],
          export_common_attr = [
            {"name": "exceed_label_threshold", "as": "is_reach_content_too_much"},
          ],
          function_name = "CalcExploreReachContentLabelCount",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .if_("is_reach_content_too_much == 0") \
          .set_attr_value(
            item_attrs = [
              {
                "name": "reach_content",
                "type": "int",
                "value": 1
              }
            ],
            target_item = {
              "lbs_item": 1,
            }
          ) \
        .end_() \
      .end_()

