from cascading import CommonModule

class CascadingCalcPicCommentQualityScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_calc_pic_mc_comment_quality_score == 1") \
        .calc_by_formula1(
          kconf_key = "formula.scenarioKey43.explore_pic_mc_comment_quality_value",
          target_item = {
            "is_picture": 1
          },
          import_item_attr = [
            "pic_good_comment_score",
            "mc_ensemble_pctr",
            "mc_ensemble_pltr",
            "mc_ensemble_pwtr",
            "mc_ensemble_pcltr",
            "mc_ensemble_pcmtr",
            "mc_ensemble_pftr",
            "cascade_pctr",
            "cascade_pltr",
            "cascade_pwtr",
            "cascade_pftr",
            "cascade_pcltr",
            "cascade_pcmtr"
          ],
          export_formula_value = [
            {"name": "pic_mc_ensemble_comment_quality_score", "as": "pic_mc_comment_quality_score"}
          ],
          abtest_biz_name = "KUAISHOU_APPS",
          perf_tag = "{{explore_pic_mc_comment_quality_refactor_f1_perf_tag}}"
        ) \
      .end_() 
