from retrieval import CommonModule

class ExploreRealtimeUserItemModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
    .if_("enable_explore_realtime_user_item == 1") \
      .gen_realtime_browse_set(
        enable_fountain_browse = "{{enable_explore_fountain_browse_set}}",
        enable_hot_browse = "{{enable_explore_explore_browse_set}}",
        realtime_hot_bs_size = "{{explore_realtime_explore_browse_set_size}}",
        realtime_fountain_bs_size = "{{explore_realtime_fountain_browse_set_size}}",
        profile_time_threshold = "{{explore_profile_time_threshold}}",
        enable_fix_real_show_list = "{{enable_explore_fix_real_show_list}}",
        user_info_ptr_attr = "user_info_ptr",
        output_common_attr = "explore_recent_play_list"
      ) \
    .end_() \
    .if_("enable_explore_get_realshow_click_item == 1") \
      .gen_realtime_browse_set(
        enable_fountain_browse = "{{enable_explore_fountain_realshow_click_browse_set}}",
        enable_hot_browse = "{{enable_explore_explore_realshow_click_browse_set}}",
        enable_hot_click_list = "{{enable_explore_explore_realshow_click_click_set}}",
        realtime_hot_bs_size = "{{explore_realtime_explore_realshow_click_browse_set_size}}",
        realtime_fountain_bs_size = "{{explore_realtime_fountain_realshow_click_browse_set_size}}",
        profile_time_threshold = "{{explore_profile_time_realshow_click_threshold}}",
        enable_fix_real_show_list = "{{enable_explore_fix_real_show_click_list}}",
        user_info_ptr_attr = "user_info_ptr",
        output_click_common_attr = "explore_click_common_list",
        output_common_attr = "explore_realshow_click_common_list",
        output_timestamp_common_attr = "explore_realshow_click_timestamp_common_list",
        output_hetu_five_common_attr = "explore_realshow_hetu_five_common_list",
      ) \
      .aggregate_list_attr(
        for_common = True,
        mappings = [{
          "from_attr": "explore_realshow_click_timestamp_common_list",
          "to_attr": "explore_realshow_max_timestamp",
          "aggregator": "max"
        }]
      ) \
    .end_()
