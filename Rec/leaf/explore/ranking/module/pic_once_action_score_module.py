from ranking import CommonModule

class PicOnceActionScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    """leave empty function by AutoDelete"""
  
  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        item_attrs = [
          "pic_interact_fusion_score",
          "pic_watch_time_fusion_score"
        ],
        for_debug_request_only = True,
        target_item = {
          "is_picture" : 1
        }
      )
