from ranking import CommonModule

class RankingCalcHetuLtScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_calculate_hetu_lt_score == 1 and user_age_segment > 0 and user_age_segment <= explore_hetu_lt_score_user_age_threshold") \
        .enrich_attr_by_light_function(
          import_common_attr=[
            "user_info_ptr",
          ],
          import_item_attr=[
            "hetu_tag_level_info_v2__hetu_level_two",
          ],
          export_item_attr=[
            "hetu_lt_score",
          ],
          function_name="CalcFrHetuLtScore",
          class_name="ExploreLightFunctionSetV2",
        ) \
      .end_()