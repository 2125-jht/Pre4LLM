from ranking import CommonModule

class RankingCalcSharpeRatioScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_fr_cal_sharpe_ratio_score == 1 and basic_info_age_segment_v2 ~= explore_fr_cal_sharpe_ratio_score_skip_age") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_fr_cal_sharpe_ratio_score_wtd_weight", "as": "fr_cascade_linear_score_watchtime_weight"},
            {"name": "explore_fr_cal_sharpe_ratio_score_lvtr_weight", "as": "fr_cascade_linear_score_lvtr_weight"},
            {"name": "explore_fr_cal_sharpe_ratio_score_fr_score1_weight", "as": "fr_cascade_linear_score_lvtr2_weight"},
            {"name": "explore_fr_cal_sharpe_ratio_score_fr_score2_weight", "as": "fr_cascade_linear_score_pepstr_weight"},
            {"name": "explore_fr_cal_sharpe_ratio_score_cpr_weight", "as": "fr_cascade_linear_score_pptr_weight"},
            {"name": "explore_fr_cal_sharpe_ratio_score_ltr_weight", "as": "fr_cascade_linear_score_ltr_weight"},
            {"name": "explore_fr_cal_sharpe_ratio_score_wtr_weight", "as": "fr_cascade_linear_score_wtr_weight"},
            {"name": "explore_fr_cal_sharpe_ratio_score_cmtr_weight", "as": "fr_cascade_linear_score_cmtr_weight"},
            {"name": "explore_fr_cal_sharpe_ratio_score_cltr_weight", "as": "fr_cascade_linear_score_cltr_weight"},
            {"name": "explore_fr_cal_sharpe_ratio_score_ftr_weight", "as": "fr_cascade_linear_score_ftr_weight"},
          ],
          import_item_attr = [
            {"name": "awesome_wtd", "as": "cascade_pwatch_time"},
            {"name": "plvtr", "as": "cascade_plvtr"},
            {"name": "fr_score1", "as": "cascade_plvtr2"},
            {"name": "fr_score2", "as": "cascade_pepstr"},
            {"name": "cpr", "as": "cascade_ptr"},
            {"name": "pltr", "as": "cascade_pltr"},
            {"name": "pwtr", "as": "cascade_pwtr"},
            {"name": "pcmtr", "as": "cascade_pcmtr"},
            {"name": "pcltr", "as": "cascade_pcltr"},
            {"name": "fetr", "as": "cascade_pftr"},
          ],
          export_item_attr = [
            {"name": "cascade_linear_score", "as": "linear_score_for_sharpe_ratio"},
          ],
          function_name = "CalCascadeLinearScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .enrich_attr_by_light_function(
          import_item_attr = [
            {"name": "score_pctr", "as": "boost_discount_coeff"},
            {"name": "linear_score_for_sharpe_ratio", "as": "score"},
          ],
          export_item_attr = [
            {"name": "score", "as": "ctr_multy_linear_score"},
          ],
          function_name = "BoostOrDiscountByItemCoeff",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .pack_item_attr(
          item_source = {
            "reco_results": True,
          },
          mappings = [
            {
              "aggregator": "avg",
              "from_item_attr": "ctr_multy_linear_score",
              "to_common_attr": "ctr_multy_linear_score_avg"
            },
          ]
        ) \
        .explore_calc_sharpe_ratio_score_enricher(
          mean_conf_path = "reco.exploreRank.recoHotFrCtrMultyWtdMeanMap",
          std_conf_path = "reco.exploreRank.recoHotFrCtrMultyWtdStdMap",
          ctr_attr = "score_pctr",
          xtr_attr = "linear_score_for_sharpe_ratio",
          risk_free_attr = "{{explore_fr_cal_sharpe_ratio_score_risk_free}}",
          request_risk_free_attr = "{{ctr_multy_linear_score_avg}}",
          std_beta_attr = "{{explore_fr_cal_sharpe_ratio_score_std_beta}}",
          use_raw_attr = "{{explore_fr_cal_sharpe_ratio_score_use_raw}}",
          global_rf_weight_attr = "{{explore_fr_cal_sharpe_ratio_score_global_risk_free_weight}}",
          request_rf_weight_attr = "{{explore_fr_cal_sharpe_ratio_score_request_risk_free_weight}}",
          sharpe_ratio_score_attr = "ctr_multy_wtd_sharpe_ratio_score"
        ) \
      .end_()
