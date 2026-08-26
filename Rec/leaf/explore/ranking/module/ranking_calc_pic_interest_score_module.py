from ranking import CommonModule

class RankingCalcPicInterestScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_calc_fr_pic_search_score == 1") \
        .pack_item_attr(
          item_source = {
            "reco_results": True,
          },
          mappings = [
            {
              "aggregator": "avg",
              "from_item_attr": "pctr",
              "to_common_attr": "fr_pic_pctr_avg"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "pltr",
              "to_common_attr": "fr_pic_pltr_avg"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "pwtr",
              "to_common_attr": "fr_pic_pwtr_avg"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "pcmtr",
              "to_common_attr": "fr_pic_pcmtr_avg"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "pcltr",
              "to_common_attr": "fr_pic_pcltr_avg"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "pftr",
              "to_common_attr": "fr_pic_pftr_avg"
            },
          ],
          target_item = {
            "is_picture" : 1
          },
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_fr_pic_search_cluster_score_thresh", "as": "search_cluster_score_thresh"},
            {"name": "explore_fr_pic_search_cluster_score_max", "as": "search_cluster_score_max"},
            {"name": "explore_fr_pic_search_cluster_score_weight", "as": "search_cluster_score_weight"},
            {"name": "explore_fr_pic_search_score_pxtr_score_max", "as": "pxtr_score_max"},
            {"name": "explore_fr_pic_search_score_pctr_thresh", "as": "pctr_thresh"},
            {"name": "fr_pic_pctr_avg", "as": "pctr_avg"},
            {"name": "fr_pic_pltr_avg", "as": "pltr_avg"},
            {"name": "fr_pic_pwtr_avg", "as": "pwtr_avg"},
            {"name": "fr_pic_pcmtr_avg", "as": "pcmtr_avg"},
            {"name": "fr_pic_pcltr_avg", "as": "pcltr_avg"},
            {"name": "fr_pic_pftr_avg", "as": "pftr_avg"},
            {"name": "explore_fr_pic_search_score_pctr_alpha", "as": "pctr_alpha"},
            {"name": "explore_fr_pic_search_score_pltr_alpha", "as": "pltr_alpha"},
            {"name": "explore_fr_pic_search_score_pwtr_alpha", "as": "pwtr_alpha"},
            {"name": "explore_fr_pic_search_score_pcltr_alpha", "as": "pcltr_alpha"},
            {"name": "explore_fr_pic_search_score_pcmtr_alpha", "as": "pcmtr_alpha"},
            {"name": "explore_fr_pic_search_score_pftr_alpha", "as": "pftr_alpha"},
            {"name": "explore_fr_pic_search_score_pctr_beta", "as": "pctr_beta"},
            {"name": "explore_fr_pic_search_score_pltr_beta", "as": "pltr_beta"},
            {"name": "explore_fr_pic_search_score_pwtr_beta", "as": "pwtr_beta"},
            {"name": "explore_fr_pic_search_score_pcltr_beta", "as": "pcltr_beta"},
            {"name": "explore_fr_pic_search_score_pcmtr_beta", "as": "pcmtr_beta"},
            {"name": "explore_fr_pic_search_score_pftr_beta", "as": "pftr_beta"},
          ],
          import_item_attr = [
            {"name": "pic_search_interest_cluster_score", "as": "search_cluster_score"},
            "pctr",
            "pltr",
            "pwtr",
            "pcltr",
            "pcmtr",
            "pftr",
          ],
          export_item_attr = [
            {"name": "search_interest_score", "as": "fr_pic_search_score"}
          ],
          function_name = "CaclPicSearchInterestScore",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture" : 1
          },
        ) \
      .end_()

