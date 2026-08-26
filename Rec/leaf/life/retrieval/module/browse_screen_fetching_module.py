from retrieval import CommonModule

class BrowseScreenFetchingModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .explore_browse_screen_enrich(
        user_info_ptr_attr = "user_info_ptr",
        latest_screen_count = "{{explore_browse_screen_count}}",
        latest_screen_time_ms = "{{explore_browse_screen_time_ms}}",
        save_photo_ids_to_attr = "browse_screen__pid_list",
        save_author_ids_to_attr = "browse_screen__aid_list",
      )
