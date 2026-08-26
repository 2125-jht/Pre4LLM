from ranking import CommonModule

class ClickCostGainScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    """leave empty function by AutoDelete"""
  
  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        item_attrs = [
          "click_cost_score",
        ],
        for_debug_request_only = True
      )
