from ranking import CommonModule

class RankingCalcSvtrAdaptWtdScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_fr_cal_svtr_adapt_wtd_score == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "svtr_adapt_wtd_threshold",  "as": "threshold"},
            {"name": "svtr_adapt_wtd_beta",  "as": "beta"},
          ],
          import_item_attr = [
            {"name": "psvr",  "as": "svtr"},
            {"name": "awesome_wtd",  "as": "wtd"},
          ],
          export_item_attr = [
            {"name": "output_score",  "as": "svtr_adapt_wtd_score"},
          ],
          function_name = "CalcSvtrAdaptWtdScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()