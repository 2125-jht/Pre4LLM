from rerank import CommonModule

class ForceInsertModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_follow_author_photo_insert == 1") \
        .force_insert_position_enricher(
          user_info_ptr_attr = "user_info_ptr",
          page_type = "EXPLORE",
          follow_author_insert_position_limit = "{{follow_author_photo_insert_position_limit}}",
          is_follow_author_photo_attr = "is_follow_author",
          force_insert_position_attr = "photo_insert_position",
        ) \
        .force_insert(
          position_from_attr = "photo_insert_position",
        ) \
      .end_()
