from cascading_v2 import CommonModule

class CascadingHtrFilterModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("explore_enable_mc_phtr_filter_v2 == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            { "name": "hate_list_timestamps", "as": "hate_ts_list" },
            "base_phtr_filter_threshold",
            "min_phtr_filter_threshold",
            "recent_minute_for_high_freq_hate",
            "phtr_thrshold_temperature",
            "phtr_thrshold_smooth",
            "mc_htr_filter_ltr_threshold",
            "mc_htr_filter_wtr_threshold",
          ],
          import_item_attr = [
            "cascade_phtr",
            "mc_ensemble_pltr",
            "mc_ensemble_pwtr", 
          ],
          export_item_attr = [
            "mc_need_htr_filter",
          ],
          function_name = "NeedHtrFilter",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .filter_by_attr(
          name = "explore_mc_stage1_htr_filter",
          traceback = True,
          attr_name = "mc_need_htr_filter",
          remove_if = "==",
          compare_to = 1,
          cancel_num = 1500,
        ) \
      .end_()
