from cascading import CommonModule

class CascadingWhiteAuthorModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .switch_("support_and_big_v_author_change") \
      .case_(1, to_be_delete = "date=2023-11-16;committer=liuhu") \
        .copy_attr(
          attrs=[{
            "from_item": "is_big_v_white_author_photo",
            "to_item": "is_white_author",
          }]
        ) \
      .case_(2, to_be_delete = "date=2023-11-16;committer=liuhu") \
        .copy_attr(
          attrs=[{
            "from_item": "is_support_author",
            "to_item": "is_white_author",
          }]
        ) \
      .end_()
