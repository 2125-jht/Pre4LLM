from retrieval import CommonModule

class ExploreInterestHetuTagsEnricherModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
    .if_("enable_interest_hetu_tags_enricher == 1") \
      .explore_interest_hetu_tags_enricher(
        colossus_v2_attr_name = "colossus_resp_v2",
        user_info_ptr_attr = "user_info_ptr",
        shortterm_interest_count_limit = "{{explore_shortterm_interest_count_limit}}",
        shortterm_interest_score_limit = "{{explore_shortterm_interest_score_limit}}",
        shortterm_interest_expired_gap_second = "{{explore_shortterm_interest_expired_gap_second}}",
        shortterm_interest_realshow_weight = "{{explore_shortterm_interest_realshow_weight}}",
        shortterm_interest_click_weight = "{{explore_shortterm_interest_click_weight}}",
        shortterm_interest_like_weight = "{{explore_shortterm_interest_like_weight}}",
        shortterm_interest_follow_weight = "{{explore_shortterm_interest_follow_weight}}",
        shortterm_interest_forward_weight = "{{explore_shortterm_interest_forward_weight}}",
        longterm_interest_count_limit_str = "{{explore_longterm_interest_count_limit_str}}",
        longterm_interest_recent_hours = "{{explore_longterm_interest_recent_hours}}",
        longterm_interest_finish_rate_threshold = "{{explore_longterm_interest_finish_rate_threshold}}",
        longterm_interest_stat_day_upper = "{{explore_longterm_interest_stat_day_upper}}",
        longterm_interest_stat_day_lower = "{{explore_longterm_interest_stat_day_lower}}",
        longterm_interest_click_weight = "{{explore_longterm_interest_click_weight}}",
        longterm_interest_playtime_weight = "{{explore_longterm_interest_playtime_weight}}",
        export_short_term_interest_attr = "interest_explore_shortterm_hetu",
        export_long_term_interest_one_attr = "interest_explore_longterm_hetu_one",
        export_long_term_interest_two_attr = "interest_explore_longterm_hetu_two",
        export_long_term_interest_three_attr = "interest_explore_longterm_hetu_three"
      ) \
    .end_()

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = [
          "interest_explore_shortterm_hetu",
          "interest_explore_longterm_hetu_one",
          "interest_explore_longterm_hetu_two",
          "interest_explore_longterm_hetu_three"
        ],
      )
