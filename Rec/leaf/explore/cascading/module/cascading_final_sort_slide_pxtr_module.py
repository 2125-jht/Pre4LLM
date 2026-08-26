from cascading import CommonModule

class CascadingFinalSortSlidePxtrModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def get_slide_pxtr_params_with_sorted_ctr(self):
    slide_pxtr_params_with_sorted_ctr = [
      ("cascade_slide_pctr", "play_realshow_count", "pc3h", "rc3h", "cascade_pctr"),
      ("cascade_slide_pltr", "play_realshow_count", "pc3h", "rc3h", "cascade_pltr"),
      ("cascade_slide_pwtr", "play_realshow_count", "pc3h", "rc3h", "cascade_pwtr"),
      ("cascade_slide_pcltr", "play_realshow_count", "pc3h", "rc3h", "cascade_pcltr"),
      ("cascade_slide_pftr", "play_realshow_count", "pc3h", "rc3h", "cascade_pftr"),
      ("cascade_slide_awesome_wtd", "play_realshow_count", "pc3h", "rc3h", "cascade_pwtd_inverse"),
    ]

    return slide_pxtr_params_with_sorted_ctr  

  def get_slide_pxtr_params_with_sorted_mc_s2_ratio(self):
    slide_pxtr_params_with_sorted_mc_s2_ratio = [
      ("cascade_sort_ratio_slide_pctr", "distribution_ratio", "explore_mc_s2_hetu_level_one_ratio", "explore_prerank_hetu_level_one_ratio", "cascade_pctr"),
      ("cascade_sort_ratio_slide_pltr", "distribution_ratio", "explore_mc_s2_hetu_level_one_ratio", "explore_prerank_hetu_level_one_ratio", "cascade_pltr"),
      ("cascade_sort_ratio_slide_pwtr", "distribution_ratio", "explore_mc_s2_hetu_level_one_ratio", "explore_prerank_hetu_level_one_ratio", "cascade_pwtr"),
      ("cascade_sort_ratio_slide_pcmtr", "distribution_ratio", "explore_mc_s2_hetu_level_one_ratio", "explore_prerank_hetu_level_one_ratio", "cascade_pcmtr"),
      ("cascade_sort_ratio_slide_pcltr", "distribution_ratio", "explore_mc_s2_hetu_level_one_ratio", "explore_prerank_hetu_level_one_ratio", "cascade_pcltr"),
      ("cascade_sort_ratio_slide_pftr", "distribution_ratio", "explore_mc_s2_hetu_level_one_ratio", "explore_prerank_hetu_level_one_ratio", "cascade_pftr"),
      ("cascade_sort_ratio_slide_awesome_wtd", "distribution_ratio", "explore_mc_s2_hetu_level_one_ratio", "explore_prerank_hetu_level_one_ratio", "cascade_pwtd_inverse"),
    ]

    return slide_pxtr_params_with_sorted_mc_s2_ratio 

  def process(self) -> None:
    self.flow \
      .if_("enable_cal_mc_s2_adjust_diversity_distribution == 1") \
        .enrich_attr_by_light_function(
          import_item_attr = [
            "hetu_level_one_top1",
          ],
          export_item_attr = [
            {"name": "hetu_level_one_ratio", "as": "explore_mc_s2_hetu_level_one_ratio"},
          ],
          function_name = "CalHetuOneRatio",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_calc_mc_s2_slide_pctr_score == 1") \
        .sort_and_cal_slide_pxtr(self.get_slide_pxtr_params_with_sorted_ctr()) \
      .end_() \
      .if_("enable_calc_slide_pxtr_score_with_sorted_mc_s2_ratio == 1") \
        .sort_and_cal_slide_pxtr(self.get_slide_pxtr_params_with_sorted_mc_s2_ratio()) \
      .end_()