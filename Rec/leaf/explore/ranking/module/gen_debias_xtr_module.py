from ranking import CommonModule

class GenDebiasXtrModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    """leave empty function by AutoDelete"""
  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        item_attrs = [
          "pctr_debias_hetu",
          "pltr_debias_hetu",
          "pwtr_debias_hetu",
          "pftr_debias_hetu",
          "pcmtr_debias_hetu",
          "pptr_debias_hetu",
          "fr_score2_debias_duration",
          "awesome_wtd_debias_v2"
        ],
        for_debug_request_only = True,
      )
