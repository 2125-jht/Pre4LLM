from cascading import CommonModule

class CascadingS1PhotoTypeModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("explore_enable_cover_sense_view_score_trans == 1") \
        .explore_cover_sense_view_score_trans() \
      .end_()
