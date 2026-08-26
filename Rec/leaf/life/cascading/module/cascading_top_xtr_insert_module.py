from cascading import CommonModule

class CascadingTopXtrInsertModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_("enable_explore_cascade_top_pwatchtime_photo_insert == 1") \
        .force_insert_position_enricher(
          insert_type = "top_xtr",
          xtr_score_attr = "mc_ensemble_pwatch_time",
          ctr_score_attr = "cascade_pctr",
          top_xtr_insert_position_limit = "{{explore_cascade_top_pwatchtime_photo_insert_limit}}",
          top_xtr_insert_photo_ratio = "{{explore_cascade_top_pwatchtime_photo_insert_photo_ratio}}",
          force_insert_position_attr = "cascade_top_pwatchtime_photo_insert_position",
        ) \
        .force_insert(
          position_from_attr = "cascade_top_pwatchtime_photo_insert_position",
        ) \
      .end_()\
  
  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        item_attrs = [
          "mc_ensemble_pwatch_time",
          "cascade_top_pwatchtime_photo_insert_position",
          "cascade_pctr"
        ],
        for_debug_request_only = True,
        item_num_limit = 520,
      )
