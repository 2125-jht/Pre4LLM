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
      .end_() \
      .if_("enable_mc_cal_emp_report_rate_score == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_vv_thres_for_emp_report_score", "as": "vv_thres"},
          ],
          import_item_attr = [
            "explore_stat__click_count",
            "explore_stat__report_count",
            "fountain_stats__real_show_count",
            "fountain_stats__report_count",
          ],
          export_item_attr = [
            "emp_report_score"
          ],
          function_name = "CalcEmpReportScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_mc_cal_emp_cancel_like_score == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_like_thres_for_emp_cancel_like_score", "as": "like_thres"},
          ],
          import_item_attr = [
            "explore_stat__like_count",
            "explore_stat__cancel_like_count"
          ],
          export_item_attr = [
            "emp_cancel_like_score"
          ],
          function_name = "CalcEmpCancelLikeScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .calc_weighted_sum(
          channels = [
            {"name": "emp_cancel_like_score", "weight": "{{explore_raw_cancel_like_weight_for_emp_cancel_like_score}}"},
            {"name": "empirical_watch_time", "weight": "{{explore_watch_time_weight_for_emp_cancel_like_score}}"},
          ],
          output_item_attr = "emp_cancel_like_score",
        ) \
      .end_()
