from retrieval import CommonModule

class ExploreExploreHetuTagsEnricherModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    """leave empty function by AutoDelete"""

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = [
          "colossus_explore_hetu_tags",
          "recent_top_show_hetu",
        ],
      )
