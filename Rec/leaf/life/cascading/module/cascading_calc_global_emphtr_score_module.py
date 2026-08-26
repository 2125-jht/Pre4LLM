from cascading import CommonModule

class CascadingCalcGlobalEmphtrScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_mc_cal_global_emphtr_score == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_hate_like_rate_report_weight", "as": "report_weight"},
            {"name": "explore_global_realshow_thres_for_emphtr_score", "as": "global_realshow_thres"},
          ],
          import_item_attr = [
            "explore_stat__real_show_count",
            "explore_stat__negative_count",
            "explore_stat__like_count",
            "explore_stat__report_count",
          ],
          export_item_attr = [
            "global_emphtr_score"
          ],
          function_name = "CalcGlobalEmphtrScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()\
      .if_("life_enable_mc_cal_global_empwtr_score == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "life_global_realshow_thres_for_empwtr_score", "as": "global_realshow_thres"},
          ],
          import_item_attr = [
            "explore_stat__real_show_count",
            "explore_stat__click_count",
            "explore_stat__view_length_sum",
            "duration_ms",
            "explore_stat__long_play_count",
            "explore_stat__short_play_count",
            "is_picture",
            "photo_id",
          ],
          export_item_attr = [
            "global_empwtr_score"
          ],
          function_name = "CalcGlobalEmpwtrScore",
          class_name = "ExploreLifeLightFunctionSet",
        ) \
      .end_()