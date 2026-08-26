from ranking import CommonModule

class PicSelectModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_generate_item_key_for_pic_interest_expand == 1 and util.Random() < explore_pic_interest_expand_replace_ratio"
        + " and (explore_pic_interest_expand_enable_limit_page_index == 0 or page_index <= explore_pic_interest_expand_max_page_index)"
        + " and (explore_pic_interest_expand_enable_limit_refresh_times == 0 or refreshTimes <= explore_pic_interest_expand_max_refresh_times)"
      ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "enable_explore_pic_valid_interest_cluster_replace", "as": "enable_valid_interest_cluster"},
            {"name": "enable_explore_pic_long_interest_cluster_replace", "as": "enable_long_interest_cluster"},
            {"name": "enable_explore_pic_recent_search_replace", "as": "enable_recent_search"},
            {"name": "explore_pic_interest_cluster_replace_pctr_thresh", "as": "pctr_thresh"},
            {"name": "explore_pic_interest_cluster_replace_candicate_topk", "as": "candicate_topk"},
          ],
          import_item_attr = [
            "is_pic_valid_interest_cluster",
            "is_pic_long_interest_cluster",
            "is_pic_recent_search",
            "corr_pctr"
          ],
          export_common_attr = [
            "item_key_for_pic_interest_expand"
          ],
          function_name = "GenItemKeyForPicInterestExpand",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture": 1
          }
        ) \
      .end_()

