from cascading import CommonModule

class CascadingCalcTrendScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("fetch_short_window_photo_info == 1") \
        .get_item_attr_by_distributed_common_index(
          photo_store_kconf_key = "reco.distributedIndex.hotShortWindowInfoCommonIndex",
          use_dynamic_photo_store = True,
          attrs = [
            "rc3h",
            "rc12h",
            "pc3h",
            "pc12h",
            "lc3h",
            "lc12h",
            "lpc3h",
            "lpc12h",
            "nc3h",
            "nc12h"
          ]
        ) \
      .end_() \
      .if_("enable_explore_calc_trend_score == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_trend_score_lc_weight", "as": "lc_weight"},
            {"name": "explore_trend_score_lpc_weight", "as": "lpc_weight"},
            {"name": "explore_trend_score_nc_weight", "as": "nc_weight"},
          ],
          import_item_attr = [
            {"name": "pc3h", "as": "pc"},
            {"name": "lc3h", "as": "lc"},
            {"name": "lpc3h", "as": "lpc"},
            {"name": "nc3h", "as": "nc"},
          ],
          export_item_attr = [
            {"name": "short_window_score", "as": "short_window_score_3h"}
          ],
          function_name = "CalcShortWindowScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_trend_score_lc_weight", "as": "lc_weight"},
            {"name": "explore_trend_score_lpc_weight", "as": "lpc_weight"},
            {"name": "explore_trend_score_nc_weight", "as": "nc_weight"},
          ],
          import_item_attr = [
            {"name": "pc12h", "as": "pc"},
            {"name": "lc12h", "as": "lc"},
            {"name": "lpc12h", "as": "lpc"},
            {"name": "nc12h", "as": "nc"},
          ],
          export_item_attr = [
            {"name": "short_window_score", "as": "short_window_score_12h"}
          ],
          function_name = "CalcShortWindowScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_trend_score_lc_weight", "as": "lc_weight"},
            {"name": "explore_trend_score_lpc_weight", "as": "lpc_weight"},
            {"name": "explore_trend_score_nc_weight", "as": "nc_weight"},
          ],
          import_item_attr = [
            {"name": "explore_stat__click_count", "as": "pc"},
            {"name": "explore_stat__like_count", "as": "lc"},
            {"name": "explore_stat__long_play_count", "as": "lpc"},
            {"name": "explore_stat__negative_count", "as": "nc"},
          ],
          export_item_attr = [
            {"name": "short_window_score", "as": "short_window_score_all"}
          ],
          function_name = "CalcShortWindowScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .enrich_attr_by_light_function(
          import_item_attr = [
            "short_window_score_3h",
            "short_window_score_12h",
            "short_window_score_all",
            "upload_time"
          ],
          export_item_attr = [
            {"name": "trend_score", "as": "explore_photo_trend_score"}
          ],
          function_name = "CalcTrendScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()