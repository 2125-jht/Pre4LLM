from retrieval import CommonModule

class ExploreInterestHetuTagsEnricherModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    """leave empty function by AutoDelete"""

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
