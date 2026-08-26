from ranking import CommonModule

class PureValueScoreModule(CommonModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)
    
    def rank_queues(self):
      queues = [
        {
          "name": "corr_pctr",
          "use_mapping": "explore_pure_value_pctr_use_mapping",
          "weight_attr": "explore_pure_value_pctr_weight"
        },
        {
          "name": "pltr",
          "use_mapping": "explore_pure_value_pltr_use_mapping",
          "weight_attr": "explore_pure_value_pltr_weight"
        },
        {
          "name": "pwtr",
          "use_mapping": "explore_pure_value_pwtr_use_mapping",
          "weight_attr": "explore_pure_value_pwtr_weight"
        },
        {
          "name": "pftr",
          "use_mapping": "explore_pure_value_pftr_use_mapping",
          "weight_attr": "explore_pure_value_pftr_weight"
        },
        {
          "name": "pcmtr",
          "use_mapping": "explore_pure_value_pcmtr_use_mapping",
          "weight_attr": "explore_pure_value_pcmtr_weight"
        },
        {
          "name": "pcmef",
          "use_mapping": "explore_pure_value_pcmef_use_mapping",
          "weight_attr": "explore_pure_value_pcmef_weight"
        },
        {
          "name": "pptr",
          "use_mapping": "explore_pure_value_pptr_use_mapping",
          "weight_attr": "explore_pure_value_pptr_weight"
        },
        {
          "name": "pepstr",
          "use_mapping": "explore_pure_value_pepstr_use_mapping",
          "weight_attr": "explore_pure_value_pepstr_weight"
        },
        {
          "name": "pcltr",
          "use_mapping": "explore_pure_value_pcltr_use_mapping",
          "weight_attr": "explore_pure_value_pcltr_weight"
        },
        {
          "name": "fetr",
          "use_mapping": "explore_pure_value_fetr_use_mapping",
          "weight_attr": "explore_pure_value_fetr_weight"
        },
        {
          "name": "fountain_eff",
          "use_mapping": "explore_pure_value_feff_use_mapping",
          "weight_attr": "explore_pure_value_feff_weight"
        },
        {
          "name": "awesome_wtd",
          "use_mapping": "explore_pure_value_wtd_use_mapping",
          "weight_attr": "explore_pure_value_wtd_weight"
        },
        {
          "name": "plvtr",
          "use_mapping": "explore_pure_value_plvtr_use_mapping",
          "weight_attr": "explore_pure_value_plvtr_weight"
        },
        {
          "name": "psvr",
          "use_mapping": "explore_pure_value_psvr_use_mapping",
          "weight_attr": "explore_pure_value_psvr_weight"
        },
        {
          "name": "fr_score1",
          "use_mapping": "explore_pure_value_fr_score1_use_mapping",
          "weight_attr": "explore_pure_value_fr_score1_weight"
        },
        {
          "name": "fr_score2",
          "use_mapping": "explore_pure_value_fr_score2_use_mapping",
          "weight_attr": "explore_pure_value_fr_score2_weight"
        },
        {
          "name": "pevtr",
          "use_mapping": "explore_pure_value_pevtr_use_mapping",
          "weight_attr": "explore_pure_value_pevtr_weight"
        },
        {
          "name": "pvtr",
          "use_mapping": "explore_pure_value_pvtr_use_mapping",
          "weight_attr": "explore_pure_value_pvtr_weight"
        },
      ]
      return queues

    
    def process(self) -> None:
      self.flow \
        .explore_ensemble_score_calc_pure_value_enricher(
          skip = "{{skip_explore_calc_ensemble_score_pure_value}}",
          save_score_to_attr = "explore_fullrank_pure_value_score",
          user_power_calc = "{{explore_fullrank_pure_value_score_use_power_calc}}",
          queues = self.rank_queues(),
        )
    
    def post_process(self) -> None:
      self.flow \
        .log_debug_info(
          item_attrs = [
            "explore_fullrank_pure_value_score"
          ],
          for_debug_request_only = True
        )