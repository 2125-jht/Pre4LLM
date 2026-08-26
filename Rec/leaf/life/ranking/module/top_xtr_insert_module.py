from ranking import CommonModule

class TopXtrInsertModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_("enable_explore_rank_top_pwatchtime_photo_insert == 1") \
        .switch_("explore_rank_top_pwatchtime_score_switch") \
          .case_(2) \
            .force_insert_position_enricher(
              insert_type = "top_xtr",
              xtr_score_attr = "awesome_wtd",
              ctr_score_attr = "pctr",
              top_xtr_insert_position_limit = "{{explore_rank_top_pwatchtime_photo_insert_limit}}",
              top_xtr_insert_photo_ratio = "{{explore_rank_top_pwatchtime_photo_insert_photo_ratio}}",
              force_insert_position_attr = "rank_top_pwatchtime_photo_insert_position",
            ) \
          .default_() \
            .force_insert_position_enricher(
              insert_type = "top_xtr",
              xtr_score_attr = "fr_score2",
              ctr_score_attr = "pctr",
              top_xtr_insert_position_limit = "{{explore_rank_top_pwatchtime_photo_insert_limit}}",
              top_xtr_insert_photo_ratio = "{{explore_rank_top_pwatchtime_photo_insert_photo_ratio}}",
              force_insert_position_attr = "rank_top_pwatchtime_photo_insert_position",
            ) \
          .end_() \
        .force_insert(
          position_from_attr = "rank_top_pwatchtime_photo_insert_position",
        ) \
      .end_()
  
  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        item_attrs = [
          "rank_top_pwatchtime_photo_insert_position",
          "fr_score2",
          "awesome_wtd",
          "pctr",
        ],
        for_debug_request_only = True,
        item_num_limit = 100,
      )
