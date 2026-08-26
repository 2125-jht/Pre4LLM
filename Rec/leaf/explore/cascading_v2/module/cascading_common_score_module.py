from cascading_v2 import CommonModule

class CascadingCommonScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self._calc_fountain_view_weight()
    self._calc_user_pic_recent_ctr_score()

  def _calc_fountain_view_weight(self) -> None:
    self.flow \
      .if_("explore_enable_mc_cal_ef_view_weight == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "colossus_channel_list",
            "explore_min_explore_view_cnt",
            "explore_min_fountain_view_cnt",
            "explore_ef_weight_alpha",
            "explore_ef_weight_beta",
            "explore_ef_weight_min",
            "explore_ef_weight_max",
          ],
          export_common_attr = [
            "explore_fountain_view_weight",
          ],
          function_name = "CalcExploreFountainViewWeight",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()

  def _calc_user_pic_recent_ctr_score(self) -> None:
    self.flow \
      .if_("explore_pic_quota_enable_recent_realshow_decay == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "expl_pic_recent_realshow_time_gap_min", "as": "recent_time_gap_min"},
            {"name": "uStandardRealShowPicAllIdList", "as": "pic_realshow_pids"},
            {"name": "uStandardClickPicAllIdList", "as": "pic_click_pids"},
            "user_info_ptr",
          ],
          export_common_attr = [
            "pic_recent_realshow_not_click_cnt",
          ],
          function_name = "ProccessPicActionList",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "expl_pic_recent_realshow_not_click_max", "as": "realshow_not_click_max"},
            {"name": "expl_pic_recent_realshow_not_click_min", "as": "realshow_not_click_min"},
            {"name": "expl_pic_recent_realshow_ctr_base", "as": "ctr_base"},
            {"name": "pic_recent_realshow_not_click_cnt", "as": "realshow_not_click_cnt"},
            "pic_da_user_pref_ptr",
            "basic_info_age_segment_v2",
            "uIsPicDeep",
          ],
          export_common_attr = [
            "user_pic_recent_ctr_score",
          ],
          function_name = "PicCtrByRealshow",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()
