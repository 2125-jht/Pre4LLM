from cascading_v2 import CommonModule

class CascadingPrerankCommonScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self._calc_is_diversity_hetu1_degraded()
    self._calc_pic_search_boost_user_degree()
    self._calc_pic_double_outside_valid_interest_num()

  def _calc_is_diversity_hetu1_degraded(self) -> None:
    self.flow \
      .if_("explore_enable_gen_is_diversity_degraded == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_diversity_degraded_min_per_hetu1", "as": "min_per_hetu1"},
            {"name": "explore_diversity_degraded_hetu1_min", "as": "hetu1_min"},
          ],
          import_item_attr = [
            "hetu_tag_level_info_v2__hetu_level_one"
          ],
          export_common_attr = [
            "is_diversity_hetu1_degraded"
          ],
          function_name = "GenDiversityDegraded",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()

  def _calc_pic_search_boost_user_degree(self) -> None:
    self.flow \
      .if_("enable_explore_calc_pic_search_boost_user_degree == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "search_click_list",
            "search_click_list_timestamps",
            {"name": "uStandardClickPicAllIdList", "as": "pic_click_list"},
            {"name": "explore_search_click_pic_time_gap_min", "as": "time_gap_min"},
            {"name": "uDoubleOutsideValidPicClusterCnt7dKV", "as": "user_cluster_cnt"},
            {"name": "explore_pic_search_boost_user_cluster_thresh", "as": "user_cluster_thresh"},
          ],
          export_common_attr = [
            {"name": "search_degree", "as": "pic_search_boost_user_degree"},
          ],
          function_name = "CalcPicSearchBoostUserDegree",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()

  def _calc_pic_double_outside_valid_interest_num(self) -> None:
    self.flow \
      .if_("enable_explore_calc_pic_double_outside_valid_interest_num == 1") \
        .gen_common_attr_by_lua(
          attr_map = {
            "pic_double_outside_valid_interest_num": "#(uDoubleOutsideValidPicCluster7dList or {})",
          }
        ) \
      .end_()
