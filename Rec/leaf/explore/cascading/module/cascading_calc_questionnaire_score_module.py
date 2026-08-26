from cascading import CommonModule

class CascadingCalcQuestionnaireScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("explore_mc_enable_calc_questionnaire_score == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_questionnaire_score_min_total_count", "as": "fountain_questionnaire_score_min_total_count"},
          {"name": "explore_questionnaire_score_pos_threshold", "as": "fountain_questionnaire_score_pos_threshold"},
          {"name": "explore_questionnaire_score_neg_threshold", "as": "fountain_questionnaire_score_neg_threshold"},
          {"name": "explore_questionnaire_score_unsure_threshold", "as": "fountain_questionnaire_score_unsure_threshold"},
          {"name": "explore_questionnaire_score_use_global", "as": "fountain_questionnaire_score_use_global"},
        {"name": "explore_questionnaire_score_enable_topk_or_audit_valid", "as": "questionnaire_score_enable_topk_or_audit_valid"},
        {"name": "explore_questionnaire_score_topk_level_threshold", "as": "questionnaire_score_topk_level_threshold"},
        {"name": "explore_questionnaire_score_audit_level_threshold", "as": "questionnaire_score_audit_level_threshold"},
        ],
        import_item_attr = [
          "questionnaire_info__positive_count",
          "questionnaire_info__negative_count",
          "questionnaire_info__unsure_count",
          "explore_questionnaire_info__negative_count",
          "explore_questionnaire_info__positive_count",
          "explore_questionnaire_info__unsure_count",
          "topk_audit_level",
          "audit_hot_high_tag_level",
        ],
        export_item_attr = [
          "questionnaire_score"
        ],
        function_name = "CalcQuestionnaireScore",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .end_()
