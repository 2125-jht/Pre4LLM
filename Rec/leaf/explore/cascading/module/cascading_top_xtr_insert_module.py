from cascading import CommonModule

class CascadingTopXtrInsertModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    """leave empty function by AutoDelete"""
  
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
