from cascading import CommonModule

class CascadingCalcSharpeScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_mc_cal_emp_sharpe_score == 1") \
        .calc_weighted_sum(
          channels = [
            {"name": "empirical_watch_time", "weight": "{{explore_mc_cal_emp_sharpe_score_wtd_weight}}"},
            {"name": "empirical_lvtr", "weight": "{{explore_mc_cal_emp_sharpe_score_lvtr_weight}}"},
          ],
          output_item_attr = "emp_linear_score",
        ) \
        .item_attr_operation(
          item_attr_a = "empirical_ctr",
          item_attr_b = "emp_linear_score",
          operator = "*",
          output_attr = "emp_ctr_multy_emp_linear_score"
        ) \
        .pack_item_attr(
          item_source = {
            "reco_results": True,
          },
          mappings = [
            {
              "aggregator": "avg",
              "from_item_attr": "emp_ctr_multy_emp_linear_score",
              "to_common_attr": "emp_ctr_multy_emp_linear_score_avg"
            },
          ]
        ) \
        .explore_calc_sharpe_ratio_score_enricher(
          mean_conf_path = "reco.exploreRank.recoHotFrCtrMultyWtdMeanMap",
          std_conf_path = "reco.exploreRank.recoHotFrCtrMultyWtdStdMap",
          ctr_attr = "empirical_ctr",
          xtr_attr = "emp_linear_score",
          risk_free_attr = "{{explore_mc_cal_emp_sharpe_score_risk_free}}",
          request_risk_free_attr = "{{emp_ctr_multy_emp_linear_score_avg}}",
          std_beta_attr = "{{explore_mc_cal_emp_sharpe_score_std_beta}}",
          use_raw_attr = "{{explore_mc_cal_emp_sharpe_score_use_raw}}",
          global_rf_weight_attr = "{{explore_mc_cal_emp_sharpe_score_global_risk_free_weight}}",
          request_rf_weight_attr = "{{explore_mc_cal_emp_sharpe_score_request_risk_free_weight}}",
          sharpe_ratio_score_attr = "emp_sharpe_score"
        ) \
      .end_()
