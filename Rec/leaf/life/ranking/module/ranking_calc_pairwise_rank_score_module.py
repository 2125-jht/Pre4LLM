from ranking import CommonModule

class RankingCalcPairwiseRankScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_fr_cal_pairwise_rank_score == 1") \
        .explore_calc_pairwise_rank_score_enricher(
          pairwise_score_helper_conf_path = "reco.exploreRank.recoHotFrWtdPairScoreMap",
          wtd_attr = "awesome_wtd",
          smooth = "{{explore_fr_cal_pairwise_rank_score_smooth}}",
          pairwise_rank_score_attr = "pairwise_rank_score",
          pairwise_rank_raw_score_attr = "pairwise_rank_raw_score" 
        ) \
      .end_() \
      .if_("enable_fr_cal_ctr_pairwise_rank_score == 1") \
        .enrich_attr_by_light_function(
          import_item_attr = [
            {"name": "score_pctr", "as": "score"},
          ],
          import_common_attr = [
            {"name": "explore_fr_cal_ctr_pairwise_rank_score_alpha", "as": "boost_discount_coeff"}
          ],
          export_item_attr = [
            {"name": "score", "as": "ctr_for_pairwise_rank_score"},
          ],
          function_name = "BoostOrDiscountV2",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .explore_calc_pairwise_rank_score_enricher(
          pairwise_score_helper_conf_path = "reco.exploreRank.recoHotFrCtrPairScoreMap",
          wtd_attr = "ctr_for_pairwise_rank_score",
          smooth = "{{explore_fr_cal_ctr_pairwise_rank_score_smooth}}",
          pairwise_rank_score_attr = "ctr_pairwise_rank_score",
          pairwise_rank_raw_score_attr = "ctr_pairwise_rank_raw_score" 
        ) \
      .end_()