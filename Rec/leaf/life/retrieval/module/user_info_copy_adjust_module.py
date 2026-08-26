from retrieval import CommonModule

class UserInfoCopyAdjustModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self):
    self.flow \
      .explore_copy_adjust_user_info_enrich(
        user_info_ptr_attr = "user_info_ptr",
        output_user_info_ptr_attr = "tmp_user_info_ptr",
        output_user_info_attr = "tmpUserInfo"
      ) \
      .log_debug_info(
        common_attrs = [
          "tmp_user_info_ptr"
        ]
      )