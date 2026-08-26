from retrieval import CommonModule

class LifeRealtimeUserItemModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .if_("enable_life_get_realshow_item == 1") \
        .explore_life_browse_set_enricher(
          realshow_list_size = "{{life_realshow_list_size}}",
          time_gap_s = "{{life_realshow_time_gap_s}}",
          user_info_ptr_attr = "user_info_ptr",
          output_click_list_attr = "life_click_common_list",
          output_realshow_list_attr = "life_realshow_common_list",
          output_realshow_timestamp_list_attr = "life_realshow_timestamp_common_list",
        ) \
      .end_()
