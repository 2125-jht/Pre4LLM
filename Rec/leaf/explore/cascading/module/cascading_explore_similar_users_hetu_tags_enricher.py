from cascading import CommonModule

class CascadingExploreSimilarUsersHetuTagsEnricher(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
    .pack_common_attr(
      input_common_attrs = ["colossus_explore_hetu_tags", "similar_user_explore_hetu_tags"],
      output_common_attr = "explore_hetu_tags",
      deduplicate = True,
    )

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = [
          "explore_hetu_tags",
        ],
      )
