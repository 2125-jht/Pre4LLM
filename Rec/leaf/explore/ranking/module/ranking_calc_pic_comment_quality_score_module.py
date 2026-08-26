from ranking import CommonModule

class RankingCalcPicCommentQualityScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_calc_pic_fr_comment_quality_score == 1") \
        .calc_by_formula1(
          kconf_key = "formula.scenarioKey18.explore_pic_fr_comment_quality_value",
          target_item = {
            "is_picture": 1
          },
          import_item_attr = [
            "pic_good_comment_score",
            "pic_ltr_for_good_comment",
            "pctr",
            "pltr",
            "pwtr",
            "pcltr",
            "pcmtr",
            "pftr",
            "pcmef",
            "pic_ltr_lvtr"
          ],
          export_formula_value = [
            {"name": "fr_pic_comment_quality_score", "as": "pic_fr_comment_quality_score"}
          ],
          abtest_biz_name = "KUAISHOU_APPS",
          perf_tag = "{{explore_pic_fr_comment_quality_refactor_f1_perf_tag}}"
      ) \
      .end_() 
    

