from ranking import CommonModule

class RankingCalSlidePctrScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def get_slide_pxtr_params_with_sorted_ctr(self):
    slide_pxtr_params_with_sorted_ctr = [
      ("fr_slide_pctr", "play_realshow_count", "pc3h", "rc3h", "count", "corr_pctr"),
      ("fr_slide_awesome_wtd", "play_realshow_count", "pc3h", "rc3h", "count", "awesome_wtd"),
    ]

    return slide_pxtr_params_with_sorted_ctr  

  def get_slide_pxtr_params_with_sorted_fr_s2_ratio(self):
    slide_pxtr_params_with_sorted_fr_s2_ratio = [
      ("fr_sort_ratio_slide_pctr", "distribution_ratio", "explore_rank_hetu_level_one_ratio", "explore_prerank_hetu_level_one_ratio", "ratio", "corr_pctr"),
      ("fr_sort_ratio_slide_pltr", "distribution_ratio", "explore_rank_hetu_level_one_ratio", "explore_prerank_hetu_level_one_ratio", "ratio", "pltr"),
      ("fr_sort_ratio_slide_pwtr", "distribution_ratio", "explore_rank_hetu_level_one_ratio", "explore_prerank_hetu_level_one_ratio", "ratio", "pwtr"),
      ("fr_sort_ratio_slide_pcmtr", "distribution_ratio", "explore_rank_hetu_level_one_ratio", "explore_prerank_hetu_level_one_ratio", "ratio", "pcmtr"),
      ("fr_sort_ratio_slide_pcltr", "distribution_ratio", "explore_rank_hetu_level_one_ratio", "explore_prerank_hetu_level_one_ratio", "ratio", "pcltr"),
      ("fr_sort_ratio_slide_pftr", "distribution_ratio", "explore_rank_hetu_level_one_ratio", "explore_prerank_hetu_level_one_ratio", "ratio", "pftr"),
      ("fr_sort_ratio_slide_awesome_wtd", "distribution_ratio", "explore_rank_hetu_level_one_ratio", "explore_prerank_hetu_level_one_ratio", "ratio", "awesome_wtd"),
    ]

    return slide_pxtr_params_with_sorted_fr_s2_ratio 

  def get_slide_pxtr_params_with_sorted_diversity_fr(self):
    slide_pxtr_params_with_sorted_diversity_fr = [
      ("fr_sort_diversity_slide_pctr", "diversity_score", "diversity_fr_ranking", "ratio", "corr_pctr"),
      ("fr_sort_diversity_slide_pltr", "diversity_score", "diversity_fr_ranking", "ratio", "pltr"),
      ("fr_sort_diversity_slide_pcmtr", "diversity_score", "diversity_fr_ranking", "ratio", "pcmtr"),
      ("fr_sort_diversity_slide_pcltr", "diversity_score", "diversity_fr_ranking", "ratio", "pcltr"),
      ("fr_sort_diversity_slide_awesome_wtd", "diversity_score", "diversity_fr_ranking", "ratio", "awesome_wtd"),
    ]

    return slide_pxtr_params_with_sorted_diversity_fr 

  def process(self) -> None:
    self.flow \
    .if_("enable_cal_fr_adjust_diversity_distribution == 1") \
      .enrich_attr_by_light_function(
        import_item_attr = [
          "hetu_level_one_top1",
        ],
        export_item_attr = [
          {"name": "hetu_level_one_ratio", "as": "explore_rank_hetu_level_one_ratio"},
        ],
        function_name = "CalHetuOneRatio",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .if_("enable_ranking_calc_slide_pctr_score == 1") \
      .sort_ratio_and_cal_slide_pxtr(self.get_slide_pxtr_params_with_sorted_ctr()) \
    .end_() \
    .if_("enable_ranking_calc_slide_pxtr_score_with_sorted_fr_s2_ratio == 1") \
      .sort_ratio_and_cal_slide_pxtr(self.get_slide_pxtr_params_with_sorted_fr_s2_ratio()) \
    .end_() \
    .if_("enable_ranking_calc_slide_pxtr_score_with_sorted_diversity_fr == 1") \
      .sort_individual_score_and_cal_slide_pxtr(self.get_slide_pxtr_params_with_sorted_diversity_fr()) \
    .end_()