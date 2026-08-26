from ranking import CommonModule

class TopXtrInsertModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    """leave empty function by AutoDelete"""
  
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
