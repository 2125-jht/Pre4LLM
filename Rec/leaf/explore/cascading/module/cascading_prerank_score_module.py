from cascading import CommonModule

class CascadingPrerankScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("explore_enable_cal_user_age_interest_tagnex_tgi == 1") \
        .cal_user_age_interest_tagnex_tgi() \
      .end_() \
      .if_("explore_enable_cal_user_stage_interest_tagnex_tgi == 1") \
        .if_("explore_enable_user_stage_interest_tagnex_tgi_divide_active_degree == 0 or find_user_active_degree == 1") \
          .cal_user_stage_interest_tagnex_tgi() \
        .end_() \
      .end_() \
      .if_("explore_enable_cal_user_career_interest_tagnex_tgi == 1") \
        .if_("explore_enable_user_career_interest_tagnex_tgi_divide_active_degree == 0 or find_user_active_degree == 1") \
          .cal_user_career_interest_tagnex_tgi() \
        .end_() \
      .end_() \
      .if_("explore_enable_cal_user_no_bias_interest_tagnex_tgi == 1") \
        .cal_user_no_bias_interest_tagnex_tgi() \
      .end_() \
      .if_("explore_enable_cal_interest_score_history_coef == 1 and interest_score_based_valid_user == 1") \
        .interest_score_history_coef_calculator() \
      .end_()
      
