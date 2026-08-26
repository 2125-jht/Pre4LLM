from ranking import CommonModule

class RankingEnsembleAbsXtrScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
    .if_("explore_rank_absolute_xtr_score_que == 1") \
      .explore_absolute_xtr_score_que_enricher(
        explore_absolute_xtr_boost_threshold = "{{explore_absolute_xtr_boost_threshold}}",
        explore_absolute_xtr_boost_weight = "{{explore_absolute_xtr_boost_weight}}",
        enable_explore_absolute_xtr_cliff = "{{enable_explore_absolute_xtr_cliff}}",
        pxtr_fractile_kconf_path = "reco.offline.consumeTimeLtrPxtrFractile",
        absolute_xtr_score_que_attr = "absolute_xtr_score_que",
        queues = [
        {
          "name": "pctr",
          "xtr_attr": "pctr",
          "weight_attr": "explore_absolute_xtr_ensemble_ctr_wgt",
          "fractile_value_attr": "fr_ensemble_pctr_fractile",
        },
        {
          "name": "pltr",
          "xtr_attr": "pltr",
          "weight_attr": "explore_absolute_xtr_ensemble_ltr_wgt",
          "fractile_value_attr": "fr_ensemble_pltr_fractile",
        },
        {
          "name": "pwtr",
          "xtr_attr": "pwtr",
          "weight_attr": "explore_absolute_xtr_ensemble_wtr_wgt",
          "fractile_value_attr": "fr_ensemble_pwtr_fractile",
        },
        {
          "name": "pftr",
          "xtr_attr": "pftr",
          "weight_attr": "explore_absolute_xtr_ensemble_ftr_wgt",
          "fractile_value_attr": "fr_ensemble_pftr_fractile",
        },
        {
          "name": "pptr",
          "xtr_attr": "pptr",
          "weight_attr": "explore_absolute_xtr_ensemble_ptr_wgt",
          "fractile_value_attr": "fr_ensemble_pptr_fractile",
        },
        {
          "name": "pcmtr",
          "xtr_attr": "pcmtr",
          "weight_attr": "explore_absolute_xtr_ensemble_cmtr_wgt",
          "fractile_value_attr": "fr_ensemble_pcmtr_fractile",
        },
        {
          "name": "pcltr",
          "xtr_attr": "pcltr",
          "weight_attr": "explore_absolute_xtr_ensemble_cltr_wgt",
          "fractile_value_attr": "fr_ensemble_pcltr_fractile",
        },
        {
          "name": "fetr",
          "xtr_attr": "fetr",
          "weight_attr": "explore_absolute_xtr_ensemble_fetr_wgt",
          "fractile_value_attr": "fr_ensemble_pfetr_fractile",
        },
        {
          "name": "fountain_eff",
          "xtr_attr": "fountain_eff",
          "weight_attr": "explore_absolute_xtr_ensemble_fountain_eff_wgt",
          "fractile_value_attr": "fr_ensemble_fountain_eff_fractile",
        },
        {
          "name": "fr_score1",
          "xtr_attr": "fr_score1",
          "weight_attr": "explore_absolute_xtr_ensemble_score1_wgt",
          "fractile_value_attr": "fr_ensemble_fr_score1_fractile",
        },
        {
          "name": "fr_score2",
          "xtr_attr": "fr_score2",
          "weight_attr": "explore_absolute_xtr_ensemble_score2_wgt",
          "fractile_value_attr": "fr_ensemble_fr_score2_fractile",
        },
        {
          "name": "awesome_wtd",
          "xtr_attr": "awesome_wtd",
          "weight_attr": "explore_absolute_xtr_ensemble_wtd_wgt",
          "fractile_value_attr": "fr_ensemble_wtd_fractile",
        },
        ],
      ) \
    .end_() \
    .if_("enable_highorder_interact_score == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "highorder_interact_score_weight_str_v2",
          "highorder_interact_cost_score_alpha",
          "highorder_interact_cost_score_beta",
        ],
        import_item_attr = [
          {"name": "pctr", "as": "pctr_input"},
          {"name": "pltr", "as": "pltr_input"},
          {"name": "pwtr", "as": "pwtr_input"},
          {"name": "pftr", "as": "pftr_input"},
          {"name": "pcmtr", "as": "pcmtr_input"},
          {"name": "pptr", "as": "pptr_input"},
          {"name": "pcltr", "as": "pcltr_input"},
        ],
        export_item_attr = [
          {"name": "highorder_interact_score_output", "as": "highorder_interact_score"},
        ],
        function_name = "CalcHighOrderIntrXtrScore",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_()
