from ranking import CommonModule

class ClickCostGainScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_("enable_use_click_cost_gain_score == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "click_cost_score_use_kelly_function",
            "click_cost_fr_score2_version",
            "fr_click_cost_score_cltr_weight",
            "fr_click_cost_score_ctr_weight",
            "fr_click_cost_score_ltr_weight",
            "fr_click_cost_score_wtr_weight",
            "fr_click_cost_score_ftr_weight",
            "fr_click_cost_score_cmtr_weight",
            "fr_click_cost_score_cmef_weight",
            "fr_click_cost_score_pptr_weight",
            "fr_click_cost_score_pepstr_weight",
            "fr_click_cost_score_fetr_weight",
            "fr_click_cost_score_feff_weight",
            "fr_click_cost_score_fr_score1_weight",
            "fr_click_cost_score_fr_score2_weight",
          ],
          import_item_attr = [
            "corr_pctr",
            "pltr",
            "pwtr",
            "pftr",
            "pptr",
            "pepstr",
            "pcltr",
            "pcmtr",
            "pcmef",
            "fetr",
            "fountain_eff",
            "fr_score1",
            "fr_score2",
          ],
          export_item_attr = [
            "click_cost_score",
          ],
          function_name = "CalClickCostGainScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()
  
  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        item_attrs = [
          "click_cost_score",
        ],
        for_debug_request_only = True
      )
