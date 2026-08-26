from cascading import CommonModule

class CascadingUserHistoryCidsStatModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_user_history_cids_stat == 1") \
        .user_history_cids_stat_enricher(
          cluster_id_attr = "hetu_sim_cluster_id",
          recent_realshow_items_attr = "explore_realshow_click_common_list",
          recent_realshow_top_ratio = "{{explore_user_history_cids_recent_realshow_top_ratio}}",
          recent_realshow_min_count = "{{explore_user_history_cids_recent_realshow_min_count}}",
          save_recent_realshow_cids_attr = "user_recent_realshow_cids"
        ) \
      .end_()
