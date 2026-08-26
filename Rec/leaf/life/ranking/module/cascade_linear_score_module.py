from ranking import CommonModule

class CasCadeLinearScore(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_("enable_interact_feff_score == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "fr_cascade_linear_score_ctr_weight",
            "fr_cascade_linear_score_ltr_weight",
            "fr_cascade_linear_score_wtr_weight",
            "fr_cascade_linear_score_ftr_weight",
            "fr_cascade_linear_score_cmtr_weight",
            "fr_cascade_linear_score_pptr_weight",
            "fr_cascade_linear_score_pepstr_weight",
            "fr_cascade_linear_score_cltr_weight",
            "fr_cascade_linear_score_svr_weight",
            "fr_cascade_linear_score_lvtr_weight",
            "fr_cascade_linear_score_lvtr2_weight",
            "fr_cascade_linear_score_watchtime_weight",
          ],
          import_item_attr = [
            "cascade_pctr",
            "cascade_pltr",
            "cascade_pwtr",
            "cascade_pftr",
            "cascade_plvtr",
            "cascade_plvtr2",
            "cascade_psvtr",
            "cascade_ptr",
            "cascade_pwatch_time",
            "cascade_pepstr",
            "cascade_pcmtr",
            "cascade_pcltr",
          ],
          export_item_attr = [
            "cascade_linear_score",
          ],
          function_name = "CalCascadeLinearScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()
  
  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        item_attrs = [
          "cascade_linear_score",
        ],
        for_debug_request_only = True
      )
